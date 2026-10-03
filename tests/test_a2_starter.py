"""Infrastructure checks only; no implementations of the assignment exercises."""
import json
import sys
from fixtures import mock_wandb
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from dataclasses import replace
import numpy as np
import torch
from data import TokenDatasetConfig, PreprocessedTokenDataset
from experiments.a2.baseline import config
from experiments.a2.data import stage_prefix
from model_config import LMConfig
from modeling import LlamaRotaryEmbedding
from metric_logging import MetricLogger, AFTER_TRAIN_STEP
from optimizers import build_optimizer
from lr_schedules import build_scheduler
from train import build_model, causal_lm_loss, train


def write_split(path, rows):
    path.mkdir()
    rows.tofile(path/'tokens.bin')
    (path/'metadata.json').write_text(json.dumps(dict(
        dtype='uint16', num_sequences=len(rows), seq_len=rows.shape[1])))


def custom_optimizer(model, optimizer_name, **kwargs):
    if optimizer_name != 'custom':
        raise ValueError(optimizer_name)
    return build_optimizer(model, 'adamw', **kwargs)


class Steps:
    def setup(self, ctx):
        self.values = [ctx.step]

    def __call__(self, ctx):
        self.values.append(ctx.step + 1)
        return {}


class StarterTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        wandb_patch = patch.dict(sys.modules, {'wandb': mock_wandb()})
        wandb_patch.start()
        self.addCleanup(wandb_patch.stop)

    def test_baseline_uses_unscaled_rope(self):
        c = config('train', 'val')
        restored = LMConfig.from_dict(c.model_config.to_dict())
        self.assertEqual(restored, c.model_config)
        self.assertNotIn("rope_scaling", restored.to_dict())
        rope = LlamaRotaryEmbedding(restored)
        expected = 1.0 / (500000.0 ** (torch.arange(0, restored.head_dim, 2,
                                      dtype=torch.float32) / restored.head_dim))
        torch.testing.assert_close(rope.inv_freq, expected, rtol=0, atol=0)
        model = build_model(replace(c, model_config=replace(restored,
            hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=2, head_dim=8),
            tie_word_embeddings=True, dropout=.1, precision='fp32'), 'cpu')
        self.assertIs(model.lm_head.weight, model.model.embed_tokens.weight)
        self.assertEqual(model.dropout, .1)

    def test_global_prefix_matches_shared_loader_without_double_shuffle(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            rows = np.arange(48, dtype=np.uint16).reshape(12, 4)
            write_split(root/'source', rows)
            target = stage_prefix(root/'source', root/'target', 7)
            data = PreprocessedTokenDataset(target)
            np.testing.assert_array_equal(data.tokens,
                rows[np.random.default_rng(42).permutation(len(rows))[:7]])
            a1 = PreprocessedTokenDataset(root/"source", sequence_limit=7).shuffle(seed=42)
            np.testing.assert_array_equal(data.tokens, a1.tokens)
            self.assertIs(data.shuffle(seed=42), data)
            with self.assertRaises(FileExistsError):
                stage_prefix(root/'source', target, 7)

    def check_accumulation_and_logs(self, count, groups, micro_batches=3):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            rows = np.arange(count * 8, dtype=np.uint16).reshape(count, 8) % 32
            for split in ('train', 'val'):
                write_split(root/split, rows)
            steps = Steps()
            c = config(root/'train', root/'val')
            c = replace(c, model_config=LMConfig('tiny', 32, 8, 16, 32, 2, 2, 2, head_dim=8),
                train_dataset=TokenDatasetConfig('train', 'test', 8, path=str(root/'train')),
                val_dataset=TokenDatasetConfig('val', 'test', 8, path=str(root/'val')),
                num_train_sequences=count, data_seed=None, batch_size=6, num_micro_batches=micro_batches,
                optimizer_name='custom', optimizer_builder='test_a2_starter:custom_optimizer',
                warmup_percent=0., precision='fp32', grad_norm=None, num_evals=2,
                wandb_online=True, save_model=False, model_dir=str(root/'models'),
                metric_loggers=(MetricLogger(AFTER_TRAIN_STEP, steps),))
            expected = build_model(c, 'cpu')
            opt = build_optimizer(expected, 'adamw', c.learning_rate,
                                  c.weight_decay, c.beta1, c.beta2, **c.optimizer_kwargs)
            scheduler = build_scheduler(opt, c.lr_schedule, 0, len(groups))
            for group in groups:
                for start, stop in group:
                    tokens = torch.tensor(rows[start:stop].astype(np.int64))
                    (causal_lm_loss(expected(tokens), tokens) / micro_batches).backward()
                opt.step(); scheduler.step(); opt.zero_grad()
            with patch('torch.cuda.device_count', return_value=0):
                actual = train(c)
            for p, q in zip(actual.parameters(), expected.parameters()):
                torch.testing.assert_close(p, q, rtol=2e-5, atol=2e-7)
            self.assertEqual(steps.values, list(range(len(groups) + 1)))

    def test_drops_incomplete_accumulation_group(self):
        self.check_accumulation_and_logs(10, [[(0, 2), (2, 4), (4, 6)]])

    def test_weights_partial_microbatch_equally(self):
        self.check_accumulation_and_logs(11,
            [[(0, 2), (2, 4), (4, 6)], [(6, 8), (8, 10), (10, 11)]])

    def test_keeps_partial_batch_without_accumulation(self):
        self.check_accumulation_and_logs(10, [[(0, 6)], [(6, 10)]], micro_batches=1)

    def test_default_probes_preserve_training_and_write_both_logs(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);rows=np.arange(352,dtype=np.uint16).reshape(44,8)%32
            for split in ('train','val'):write_split(root/split,rows)
            c=config(root/'train',root/'val')
            self.assertEqual(len(c.metric_loggers),4)
            c=replace(c,model_config=LMConfig('tiny',32,8,16,32,1,2,2,head_dim=8),
                train_dataset=TokenDatasetConfig('train','test',8,path=str(root/'train')),
                val_dataset=TokenDatasetConfig('val','test',8,path=str(root/'val')),
                num_train_sequences=44,data_seed=None,batch_size=6,num_micro_batches=3,
                warmup_percent=0.,precision='fp32',grad_norm=None,num_evals=2,
                wandb_online=True,save_model=False,model_dir=str(root/'measured'))
            with patch('torch.cuda.device_count',return_value=0):
                measured=train(c)
                plain=train(replace(c,metric_loggers=(),model_dir=str(root/'plain')))
            for p,q in zip(measured.parameters(),plain.parameters()):
                torch.testing.assert_close(p,q,rtol=0,atol=0)
            from experiments.a2 import alignment,probes
            records=[json.loads(t) for t in alignment._logger.path.read_text().splitlines()]
            self.assertEqual({r['step'] for r in records},set(range(1,8)))
            for name in ('features.jsonl', 'gradients.jsonl'):
                path = next((root/'measured').rglob(name))
                self.assertEqual(json.loads(path.read_text().splitlines()[-1])['step'], 7)
            self.assertTrue(any(r['status']=='finite' for r in records))
            import sys
            wb = sys.modules['wandb']
            logged = [call.args[0] for call in wb.log.call_args_list]
            update_rows = {row['logging/alignment/step']: row for row in logged
                           if 'logging/alignment/step' in row}
            for record in records:
                if record['alpha'] is not None:
                    self.assertEqual(update_rows[record['step']]['logging/alignment/alpha/' + record['parameter']], record['alpha'])
            feature_rows = [json.loads(t) for t in probes.features.path.read_text().splitlines()]
            readout_rows = {row['logging/readout_alignment/step']: row for row in logged
                           if 'logging/readout_alignment/step' in row}
            for record in feature_rows:
                for kind, value in record['readout_alignment'].items():
                    if value['omega'] is not None:
                        self.assertEqual(readout_rows[record['step']]['logging/readout_alignment/' + kind + '_omega'], value['omega'])
            self.assertTrue(any('alignment_charts/update_alpha' in row for row in logged))
            self.assertTrue(any('alignment_charts/readout_omega' in row for row in logged))
            self.assertEqual(len(list((root/'measured').rglob('features.jsonl'))),1)
            with patch('torch.cuda.device_count',return_value=0):
                disabled=train(replace(c,wandb_online=False,model_dir=str(root/'disabled')))
            for p,q in zip(measured.parameters(),disabled.parameters()):
                torch.testing.assert_close(p,q,rtol=0,atol=0)
            self.assertEqual(list((root/'disabled').rglob('features.jsonl')),[])


if __name__ == '__main__':
    unittest.main()
