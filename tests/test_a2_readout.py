"""Readout probe correctness and training-trajectory checks."""
import json
import sys
from fixtures import mock_wandb
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from data import TokenDatasetConfig
from model_config import LMConfig
from metric_logging import AFTER_TRAIN_STEP, MetricLogger
from train import TrainConfig, train, build_model
from experiments.a2.probes import FeatureLogger
from experiments.a2.readout import measure_readout, readout_alignment


class ReadoutTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        wandb_patch = patch.dict(sys.modules, {'wandb': mock_wandb()})
        wandb_patch.start()
        self.addCleanup(wandb_patch.stop)

    def test_analytical_cases_and_zero_movement(self):
        n = 16
        x = torch.ones(3, n)
        # Rows each select a coordinate: exact sqrt(n), without sampling noise.
        no = measure_readout(torch.eye(n), x, tile_rows=3)
        self.assertAlmostEqual(no['ratio'], math.sqrt(n))
        self.assertAlmostEqual(no['omega'], .5)
        full = measure_readout(torch.ones(7, n), x, tile_rows=3)
        self.assertAlmostEqual(full['ratio'], n)
        self.assertAlmostEqual(full['omega'], 1.)
        self.assertEqual(measure_readout(torch.ones(7, n), torch.zeros_like(x))['status'],
                         'undefined_denominator')

    def test_tiled_action_matches_full_and_ignores_autocast(self):
        g = torch.Generator().manual_seed(12)
        w = torch.randn(13, 16, generator=g)
        h = torch.randn(2, 3, 16, generator=g)
        expected = (h @ w.T).double().square().mean().sqrt() / (
            w.double().square().mean().sqrt() * h.double().square().mean().sqrt())
        with torch.autocast('cpu', dtype=torch.bfloat16):
            actual = measure_readout(w, h, tile_rows=3)
        self.assertAlmostEqual(actual['ratio'], expected.item(), places=7)
        paired = readout_alignment(w, h, w, h)
        self.assertIsNone(paired['movement']['omega'])
        self.assertEqual(paired['current']['fan_in'], 16)

    def test_feature_logger_training_parity_and_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            tokens = np.arange(96, dtype=np.uint16).reshape(12, 8) % 32
            for split in ['train', 'val']:
                d = root / split; d.mkdir(); tokens.tofile(d / 'tokens.bin')
                (d / 'metadata.json').write_text(json.dumps(dict(
                    dtype='uint16', num_sequences=12, seq_len=8)))
            c = TrainConfig(model_config=LMConfig('test', 32, 8, 16, 32, 2, 2, 2, head_dim=8),
                precision='fp32', wandb_online=True, save_model=False,
                num_train_sequences=12, batch_size=4, num_micro_batches=2,
                num_evals=1, warmup_percent=0., data_seed=None,
                model_dir=str(root / 'plain'),
                train_dataset=TokenDatasetConfig('train', 'test', 8, path=str(root/'train')),
                val_dataset=TokenDatasetConfig('val', 'test', 8, path=str(root/'val')))
            logger = FeatureLogger()
            measured_c = replace(c, model_dir=str(root/'measured'),
                metric_loggers=(MetricLogger(AFTER_TRAIN_STEP, logger),))
            with patch('torch.cuda.device_count', return_value=0):
                plain = train(c)
                measured = train(measured_c)
            for a, b in zip(plain.parameters(), measured.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            rows = [json.loads(s) for s in logger.path.read_text().splitlines()]
            self.assertEqual([r['step'] for r in rows], [0,1,2,3])
            self.assertIsNone(rows[0]['readout_alignment']['movement']['omega'])
            for row in rows[1:]:
                for key in ['current', 'movement']:
                    self.assertTrue(math.isfinite(row['readout_alignment'][key]['omega']))
            # A resume must use the original readout, not the checkpoint readout.
            resumed = FeatureLogger()
            ctx = SimpleNamespace(model=measured, config=measured_c, step=3,
                val_batches={'val': [logger.batch]})
            resumed.setup(ctx)
            torch.testing.assert_close(resumed.initial['readout_weight'],
                logger.initial['readout_weight'], rtol=0, atol=0)
            last = json.loads(resumed.path.read_text().splitlines()[-1])
            self.assertEqual(last['readout_alignment'], rows[-1]['readout_alignment'])
            self.assertTrue(measured.training)


if __name__ == '__main__':
    unittest.main()
