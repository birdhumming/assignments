"""Check chart values, optimizer-step axes, and undefined alignment handling."""
import unittest
from unittest.mock import Mock, MagicMock, patch
from experiments.a2.wandb_diagnostics import log_records, update_metrics, readout_metrics


class WandbDiagnosticsTests(unittest.TestCase):
    def test_saved_records_keep_real_steps_and_skip_undefined_values(self):
        wb = Mock()
        updates = [dict(step=1, parameter='q.weight', alpha=None),
                   dict(step=2, parameter='q.weight', alpha=.7),
                   dict(step=9, parameter='q.weight', alpha=.8)]
        features = [dict(step=0, readout_alignment={'current': {'omega': .5, 'ratio': 4.},
                                                   'movement': {'omega': None, 'ratio': None}}),
                    dict(step=9, readout_alignment={'movement': {'omega': .9, 'ratio': 12.}})]
        log_records(wb, updates, features)
        rows = [call.args[0] for call in wb.log.call_args_list]
        self.assertEqual([r.get('logging/alignment/step') for r in rows[:4]], [None, 1, 2, 9])
        self.assertNotIn('logging/alignment/alpha/q.weight', rows[1])
        self.assertNotIn('logging/readout_alignment/movement_omega', rows[0])
        self.assertEqual(rows[3]['logging/alignment/alpha/q.weight'], .8)
        self.assertEqual(rows[3]['logging/readout_alignment/movement_omega'], .9)
        self.assertTrue(all(not call.kwargs for call in wb.log.call_args_list))
        wb.define_metric.assert_any_call('logging/alignment/*', step_metric='logging/alignment/step', step_sync=False)
        update_plot = wb.plot.line_series.call_args_list[0].kwargs
        self.assertEqual(update_plot['xs'], [[2, 9]])
        self.assertEqual(update_plot['ys'], [[.7, .8]])
        self.assertEqual(set(rows[-1]), {'alignment_charts/update_alpha', 'alignment_charts/readout_omega'})

    def test_nonfinite_values_are_not_charted(self):
        self.assertEqual(update_metrics([dict(parameter='q', alpha=float('nan'))], 1), {'alignment/step': 1})
        self.assertEqual(readout_metrics(dict(step=0, readout_alignment={'movement': {'omega': None, 'ratio': float('inf')}})), {'readout_alignment/step': 0})

    def test_five_step_entrypoint_logs_by_default_with_explicit_offline_option(self):
        import sys
        from experiments.a2 import p31_student
        wb = MagicMock()
        wb.init.return_value.__enter__.return_value.url = 'https://wandb.test/run'
        result = dict(config={}, parameter_groups=[], data_sha256={}, alignment=[],
                      history=[dict(step=0, val_loss=3., readout_alignment={'current': {'omega': .5}})])
        args = ['--train-path', 'train', '--val-path', 'val', '--output', 'result.json', '--lr', '.001']
        with patch.dict(sys.modules, {'wandb': wb}), \
             patch.object(p31_student, 'load_tokens'), patch.object(p31_student, 'run', return_value=result):
            p31_student.main(args)
            wb.init.assert_called_once()
            self.assertTrue(any('logging/readout_alignment/current_omega' in call.args[0] for call in wb.log.call_args_list))
            wb.reset_mock()
            p31_student.main(args + ['--no-wandb'])
            wb.init.assert_not_called()
            wb.log.assert_not_called()
