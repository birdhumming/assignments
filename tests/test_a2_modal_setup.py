"""Offline checks for A2 configuration and data preparation."""
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from data import DEFAULT_DATASET_DIR_NAME, PreprocessedTokenDataset
from experiments.a2.modal_launcher import config, prepare_prefixes, prefix_path


class ModalSetupTests(unittest.TestCase):
    def test_p1_configs_use_course_mounts_and_current_recipe(self):
        from experiments.a2.p1_learning_rate import runs
        RUNS = runs((.0015, .003))
        self.assertEqual(len(RUNS), 3)
        self.assertEqual({c.learning_rate for c in RUNS}, {.0015, .003, .006})
        for c in RUNS:
            self.assertEqual(c.train_dataset.path, '/root/data/datasets/a2-global-prefixes/n4800000-seed42/train')
            self.assertEqual(c.val_dataset.path, '/root/shared_data/datasets/dclm_9p6m_ctx1024/val')
            self.assertEqual(c.num_train_sequences, 4800000)
            self.assertEqual(c.lr_schedule, 'linear')
            self.assertEqual(len(c.metric_loggers), 4)
        for batch in (8, 16, 32, 64, 128, 256):
            c = config(tokens=153_600_000, batch=batch)
            self.assertEqual(c.num_micro_batches, max(1, batch // 64))
            self.assertEqual(c.batch_size // c.num_micro_batches, min(batch, 64))
            self.assertIsNone(c.optimizer_builder)
            self.assertEqual(c.optimizer_kwargs, {})
        with self.assertRaisesRegex(ValueError, 'multiples of 64'):
            config(batch=96)
        self.assertEqual(config(diagnostics=False).metric_loggers, ())

    def test_prefixes_match_global_order_and_reuse_safely(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);shared=root/'shared';user=root/'user'
            source=shared/DEFAULT_DATASET_DIR_NAME/'train';source.mkdir(parents=True)
            rows=np.arange(8*1024,dtype=np.uint16).reshape(8,1024)
            rows.tofile(source/'tokens.bin')
            (source/'metadata.json').write_text(json.dumps(dict(dtype='uint16',num_sequences=8,seq_len=1024)))
            other=user/'unrelated-dataset'/'train'
            other.mkdir(parents=True)
            (other/'tokens.bin').write_bytes(b'unrelated cached data')
            prepare_prefixes(shared,user,[5,3,5])
            self.assertEqual((other/'tokens.bin').read_bytes(),b'unrelated cached data')
            dst=Path(prefix_path(user,5));before=(dst/'tokens.bin').stat().st_mtime_ns
            data=PreprocessedTokenDataset(dst)
            np.testing.assert_array_equal(data.tokens,rows[np.random.default_rng(42).permutation(len(rows))[:5]])
            smaller=PreprocessedTokenDataset(prefix_path(user,3))
            np.testing.assert_array_equal(smaller.tokens,data.tokens[:3])
            self.assertIs(data.shuffle(42),data)
            prepare_prefixes(shared,user,[3,5])
            self.assertEqual(before,(dst/'tokens.bin').stat().st_mtime_ns)
            meta=json.loads((dst/'metadata.json').read_text());meta['shuffle']='prefix_then_permutation'
            (dst/'metadata.json').write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError,'does not match'):
                prepare_prefixes(shared,user,[5])

    def test_interrupted_preparation_does_not_publish_partial_cache(self):
        from experiments.a2 import modal_launcher as launcher
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root/'shared'/DEFAULT_DATASET_DIR_NAME/'train'
            source.mkdir(parents=True)
            np.zeros((8, 1024), dtype=np.uint16).tofile(source/'tokens.bin')
            (source/'metadata.json').write_text(json.dumps(dict(dtype='uint16', num_sequences=8, seq_len=1024)))
            def interrupted(source, output, *args, **kwargs):
                output.mkdir()
                (output/'tokens.bin').write_bytes(b'partial')
                raise RuntimeError('interrupted')
            with patch.object(launcher, 'stage_prefix', side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                    prepare_prefixes(root/'shared', root/'user', [5])
            self.assertFalse(Path(prefix_path(root/'user', 5)).exists())
            prepare_prefixes(root/'shared', root/'user', [5])
            self.assertEqual(PreprocessedTokenDataset(prefix_path(root/'user', 5)).metadata['num_sequences'], 5)

    def test_target_launch_is_explicit_and_deduplicates_predictions(self):
        from experiments.a2 import p1_learning_rate as p1
        with patch('experiments.a2.modal_launcher.launch_training_jobs') as launch:
            p1.main(['--predicted-lrs', '.002', '.004'])
            launch.assert_not_called()
            p1.main(['--predicted-lrs', '.0015', '.003', '--execute'])
            launch.assert_called_once()
            self.assertEqual(len(launch.call_args.args[0]), 3)
            self.assertTrue(all(c.num_train_sequences == 4800000 for c in launch.call_args.args[0]))

    def test_provided_sources_exclude_target_and_have_complete_grids(self):
        from experiments.a2.provided_sweeps import load, reference_diagnostics
        parts = {part: load(part) for part in ['P1a', 'P1b', 'P2a']}
        self.assertEqual([len(parts[p]) for p in parts], [9, 9, 36])
        unique = {r['run_id']: r for rows in parts.values() for r in rows}
        self.assertEqual(len(unique), 45)
        self.assertTrue(all(r['tokens'] < 4_915_200_000 for r in unique.values()))
        for tokens in {r['tokens'] for r in parts['P2a']}:
            rows = [r for r in parts['P2a'] if r['tokens'] == tokens]
            self.assertEqual(len({(r['learning_rate'], r['weight_decay']) for r in rows}), 9)
        probes = reference_diagnostics()
        self.assertEqual(probes['width'], 512)
        self.assertTrue(probes['alignment'])
        self.assertTrue(any(r['kind'] == 'fixed_batch' for r in probes['diagnostics']))

    def test_new_provided_sources_are_separate_complete_small_budget_sweeps(self):
        from experiments.a2.provided_sweeps import load
        budgets = {153_600_000, 307_200_000, 614_400_000}
        cases = [('P1e', 'adamw', 'cos', .1, {.0015, .003, .006}),
                 ('P1d', 'adamh', 'linear', 0.,
                  {.0003, .001, .0015, .003, .006, .01, .015, .03})]
        for part, optimizer, schedule, wd, lrs in cases:
            rows = load(part)
            self.assertEqual({r['tokens'] for r in rows}, budgets)
            self.assertEqual(len(rows), len(budgets) * len(lrs))
            self.assertEqual(len({r['run_id'] for r in rows}), len(rows))
            for tokens in budgets:
                self.assertEqual({r['learning_rate'] for r in rows if r['tokens'] == tokens}, lrs)
            self.assertTrue(all(r['optimizer'] == optimizer and r['lr_schedule'] == schedule
                                and r['weight_decay'] == wd and r['parts'] == part for r in rows))
            self.assertTrue(all(r['final_val_loss'] > 0 for r in rows))
        self.assertTrue(all(not r['run_url'] or r['run_url'].startswith('https://wandb.ai/')
                            for r in load('P1d')))
        self.assertTrue(all(r['run_url'].startswith('https://wandb.ai/') for r in load('P1e')))


if __name__ == '__main__':unittest.main()
