"""Regression checks for run isolation, persistence, and launch failure handling."""
import json
import sys
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from data import TokenDatasetConfig
from model_config import LMConfig
from model_io import load_model, save_model
from fixtures import mock_wandb
from train import TrainConfig, train, training_run_name


def test_optimizer(model, optimizer_name, learning_rate, weight_decay, beta1, beta2,
                   optimizer_epsilon=1e-8):
    """Test-only factory for verifying isolation of custom optimizer arguments."""
    from optimizers import build_optimizer
    optimizer = build_optimizer(model, optimizer_name, learning_rate, weight_decay, beta1, beta2)
    for group in optimizer.param_groups:
        group['eps'] = optimizer_epsilon
    return optimizer


class ReleaseSafetyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        rows = np.arange(160, dtype=np.uint16).reshape(20, 8) % 32
        for split in ('train', 'val'):
            path = self.root / split
            path.mkdir()
            rows.tofile(path / 'tokens.bin')
            (path / 'metadata.json').write_text(json.dumps(dict(
                dtype='uint16', num_sequences=len(rows), seq_len=8)))
        self.config = TrainConfig(
            model_config=LMConfig('tiny', 32, 8, 16, 32, 1, 2, 2, head_dim=8),
            train_dataset=TokenDatasetConfig('train', 'test', 8, path=str(self.root/'train')),
            val_dataset=TokenDatasetConfig('val', 'test', 8, path=str(self.root/'val')),
            num_train_sequences=20, batch_size=4, precision='fp32',
            num_evals=1, data_seed=None, warmup_percent=.4,
            wandb_online=False, metric_loggers=(), save_model=False,
            model_dir=str(self.root/'models'))

    def run_cpu(self, config):
        with patch('torch.cuda.device_count', return_value=0):
            return train(config)

    def test_model_names_and_explicit_suffixes_isolate_custom_runs(self):
        c = self.config
        variants = [c,
            replace(c, model_config=replace(c.model_config, name='tiny-wide',
                                            hidden_size=32, num_attention_heads=4)),
            replace(c, model_config=replace(c.model_config, name='tiny-deep', num_hidden_layers=2)),
            replace(c, model_builder='example:build', run_name_suffix='custom-model'),
            replace(c, model_builder_kwargs={'scale': .25}, run_name_suffix='scale-025'),
            replace(c, optimizer_kwargs={'optimizer_epsilon': 1e-4}, run_name_suffix='eps-1e-4')]
        self.assertEqual(len({training_run_name(v) for v in variants}), len(variants))
        self.assertTrue(training_run_name(variants[-1]).endswith('-eps-1e-4'))
        self.assertEqual(training_run_name(c), training_run_name(replace(c, model_dir='/elsewhere')))

    def test_completed_run_does_not_skip_another_optimizer_configuration(self):
        c = replace(self.config, save_model=True,
                    optimizer_builder='test_a2_release_safety:test_optimizer')
        first = self.run_cpu(c)
        with patch('test_a2_release_safety.test_optimizer', wraps=test_optimizer) as factory:
            second = self.run_cpu(replace(c, optimizer_kwargs={'optimizer_epsilon': 1.},
                                          run_name_suffix='eps-1'))
            factory.assert_called_once()
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters())))
        self.assertEqual(len(list((self.root/'models').glob('*/model.pt'))), 2)

    def test_readout_scale_round_trips_final_model_and_training_checkpoint(self):
        c = replace(self.config, save_model=True,
                    model_builder="fixtures:ScaledTestLM", model_builder_kwargs={"scale": .25})
        model = self.run_cpu(c).eval()
        x = torch.randint(0, 32, (2, 8))
        directory = self.root/'models'/training_run_name(c)
        for path in (directory, directory/'latest.pt'):
            restored = load_model(path, device='cpu', precision='fp32')
            self.assertEqual(restored.scale, .25)
            torch.testing.assert_close(model(x), restored(x), rtol=0, atol=0)
        save_model(model, self.root/'scaled', metadata={
            'model_builder': c.model_builder, 'model_builder_kwargs': c.model_builder_kwargs})
        restored = load_model(self.root/'scaled', device='cpu', precision='fp32')
        torch.testing.assert_close(model(x), restored(x), rtol=0, atol=0)

    def test_resume_preserves_scaled_model_and_optimizer_trajectory(self):
        c = replace(self.config, save_model=True, keep_checkpoint_steps=(2,),
                    model_builder="fixtures:ScaledTestLM", model_builder_kwargs={"scale": .25})
        expected = self.run_cpu(c)
        source = self.root/'models'/training_run_name(c)
        resumed_c = replace(c, model_dir=str(self.root/'resumed'))
        destination = self.root/'resumed'/training_run_name(c)
        destination.mkdir(parents=True)
        shutil.copyfile(source/'step_2.pt', destination/'latest.pt')
        resumed = self.run_cpu(resumed_c)
        self.assertEqual(resumed.scale, .25)
        for a, b in zip(expected.parameters(), resumed.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_laptop_stress_launcher_routes_arguments_without_gpu_execution(self):
        from experiments.a2 import modal_stress
        remote = Mock()
        remote.with_options.return_value = remote
        with patch.object(modal_stress, 'MODAL_ENVIRONMENT', 'cs312-test'), \
             patch.object(modal_stress.app, 'run'), \
             patch.object(modal_stress, 'secrets', return_value=['test-secret']) as secrets, \
             patch.object(modal_stress, '_run', remote):
            modal_stress.main(['--lr', '.001', '--width', '640', '--output', 'probe.json'])
            secrets.assert_called_once_with(include_wandb=True)
            remote.with_options.assert_called_once_with(secrets=['test-secret'])
        args = remote.remote.call_args.args[0]
        self.assertIn('/root/shared_data/datasets/dclm_9p6m_ctx1024/train', args)
        self.assertIn('/root/data/a2-stress/probe.json', args)
        self.assertEqual(args[args.index('--width') + 1], '640')

    def test_wandb_records_applied_lr_without_changing_schedule(self):
        wandb = Mock()
        wandb.init.return_value = SimpleNamespace(id='offline-test', url='offline-test')
        applied = []
        original_step = torch.optim.AdamW.step
        def record(opt, *args, **kwargs):
            applied.append(opt.param_groups[0]['lr'])
            return original_step(opt, *args, **kwargs)
        with patch.dict(sys.modules, {'wandb': wandb}), patch.object(torch.optim.AdamW, 'step', record):
            self.run_cpu(replace(self.config, wandb_online=True, wandb_entity='test',
                                 wandb_project='test', learning_rate=.1))
        logged = [c.args[0]['learning_rate'] for c in wandb.log.call_args_list
                  if 'learning_rate' in c.args[0]]
        self.assertEqual(logged, applied)
        np.testing.assert_allclose(applied, [0., .05, .1, .1*2/3, .1/3])

    def test_gradient_diagnostics_are_read_only_and_explain_clipping(self):
        from experiments.a2.probes import GradientLogger
        from metric_logging import AFTER_BACKWARD, MetricLogger
        logger = GradientLogger()
        c = replace(self.config, wandb_online=True,
                    metric_loggers=(MetricLogger(AFTER_BACKWARD, logger),))
        with patch.dict(sys.modules, {'wandb': mock_wandb()}):
            measured = self.run_cpu(c)
            plain = self.run_cpu(replace(c, metric_loggers=()))
        for a, b in zip(measured.parameters(), plain.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        records = [json.loads(line) for line in logger.path.read_text().splitlines()]
        self.assertEqual(len(records), 5)
        self.assertTrue(all(0 < row['clip_coefficient'] <= 1 for row in records))
        self.assertTrue(all(0 <= row['embedding_fraction_squared_norm'] <= 1 for row in records))


if __name__ == '__main__':
    unittest.main()
