"""Small synthetic plumbing tests, not assignment parameterization recipes."""
from dataclasses import replace
from pathlib import Path
import tempfile
import json
import numpy as np
import unittest
from unittest.mock import patch
import torch

from experiments.a2.stress import StressConfig, StressTransformer, load_tokens, run


def fixture_initialize(model):
    # Deliberately width-independent arbitrary test values, not a scaling rule.
    for p in model.parameters():
        p.uniform_(.01, .05)


def fixture_groups(model, lr):
    return [dict(params=list(model.parameters()), lr=lr, eps=1e-6)]


class StressTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.c = StressConfig(width=8, depth=2, head_dim=4, vocab=16,
            context=4, batch=4, steps=5, microbatch=2, probe_sequences=2,
            checkpoint_blocks=1)
        g = torch.Generator().manual_seed(91)
        self.train = torch.randint(0, 16, (20, 4), generator=g)
        self.val = torch.randint(0, 16, (4, 4), generator=g)

    def run_fixture(self, **kw):
        return run(self.c, self.train, self.val, base_lr=.001,
                   initialize_fn=fixture_initialize, groups_fn=fixture_groups,
                   device='cpu', **kw)

    def test_missing_student_work_fails_before_model_allocation(self):
        with self.assertRaises(NotImplementedError):
            run(self.c, self.train, self.val, base_lr=.001, device='cpu')

    def test_causal_gated_silu_architecture_and_uninitialized_guard(self):
        previous = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            m = StressTransformer(self.c)
        finally:
            torch.set_default_dtype(previous)
        self.assertTrue(all(p.dtype == torch.float32 for p in m.parameters()))
        self.assertTrue(torch.isnan(m.embed.weight).all())
        self.assertIsNot(m.embed.weight, m.head.weight)
        with torch.no_grad():
            fixture_initialize(m)
            m.eval()
            x = self.val.clone(); y = x.clone(); y[:, -1] = (y[:, -1] + 1) % 16
            torch.testing.assert_close(m(x)[:, :-1], m(y)[:, :-1], rtol=0, atol=0)
        self.assertEqual(m.blocks[0].gate.weight.shape, (28, 8))
        self.assertEqual(m.blocks[0].up.weight.shape, (28, 8))
        self.assertEqual(m.blocks[0].down.weight.shape, (8, 28))
        def partial(model):
            model.embed.weight.fill_(1.)
        with self.assertRaisesRegex(ValueError, 'fully initialized'):
            run(self.c, self.train, self.val, base_lr=.001, initialize_fn=partial,
                groups_fn=fixture_groups, device='cpu')

    def test_five_updates_all_matrices_and_read_only_alignment(self):
        state = torch.random.get_rng_state().clone()
        result = self.run_fixture()
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        plain = self.run_fixture(alignment=False)
        self.assertEqual(result['history'], plain['history'])
        self.assertEqual([r['step'] for r in result['history']], list(range(6)))
        expected = {f'blocks.{i}.{n}.weight' for i in range(2)
                    for n in ['q', 'k', 'v', 'o', 'gate', 'up', 'down']} | {'head.weight'}
        self.assertEqual(len(result['alignment']), 75)
        for step in range(1, 6):
            rows = [r for r in result['alignment'] if r['step'] == step]
            self.assertEqual({r['parameter'] for r in rows}, expected)
            self.assertTrue(all(r['input_step'] == step - 1 for r in rows))
            self.assertTrue(all(r['fan_in'] == (28 if '.down.' in r['parameter'] else 8)
                                for r in rows))
        self.assertEqual(result['history'][0]['features']['final_norm']['movement'], 0.)
        self.assertIsNone(result['history'][0]['readout_alignment']['movement']['omega'])
        for row in result['history'][1:]:
            for kind in ['current', 'movement']:
                self.assertEqual(row['readout_alignment'][kind]['status'], 'finite')
        self.assertEqual(set(result['history'][5]['unscaled_branch_rms']),
                         {'blocks.0.o', 'blocks.0.down', 'blocks.1.o', 'blocks.1.down'})

    def test_checkpoint_and_microbatch_do_not_change_effective_batch(self):
        a = self.run_fixture(alignment=False)
        b = run(replace(self.c, microbatch=4, checkpoint_blocks=2),
                self.train, self.val, base_lr=.001, initialize_fn=fixture_initialize,
                groups_fn=fixture_groups, device='cpu', alignment=False)
        for x, y in zip(a['history'], b['history']):
            self.assertAlmostEqual(x['val_loss'], y['val_loss'], places=6)

    def test_stress_block_matches_course_forward_and_gradients(self):
        from model_config import LMConfig
        from modeling import LlamaDecoderLayer
        from experiments.a2.stress import GatedSiLUBlock
        c = self.c
        reference = LlamaDecoderLayer(LMConfig('test', c.vocab, c.context, c.width,
            28, c.depth, 2, 2, head_dim=c.head_dim), 0, qk_norm=True)
        actual = GatedSiLUBlock(c)
        mapping = {'q': reference.self_attn.q_proj, 'k': reference.self_attn.k_proj,
            'v': reference.self_attn.v_proj, 'o': reference.self_attn.o_proj,
            'gate': reference.mlp.gate_proj, 'up': reference.mlp.up_proj,
            'down': reference.mlp.down_proj, 'norm1': reference.input_layernorm,
            'norm2': reference.post_attention_layernorm,
            'qnorm': reference.self_attn.q_norm, 'knorm': reference.self_attn.k_norm}
        with torch.no_grad():
            for name, module in mapping.items():
                getattr(actual, name).weight.copy_(module.weight)
        x = torch.randn(2, c.context, c.width, requires_grad=True)
        y = x.detach().clone().requires_grad_()
        model = StressTransformer(c)
        pos = model.rope(x, torch.arange(c.context)[None, :])
        a = actual(x, *pos)
        b = reference(y, pos)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        a.square().sum().backward(); b.square().sum().backward()
        torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
        for name, module in mapping.items():
            torch.testing.assert_close(getattr(actual, name).weight.grad,
                                       module.weight.grad, rtol=0, atol=0)

    def test_depth_autocast_preserves_fp32_residuals_and_adam_state(self):
        c = replace(self.c, precision='mp')
        m = StressTransformer(c)
        with torch.no_grad():
            fixture_initialize(m)
        observed = []
        handles = [b.register_forward_hook(lambda _m, _a, out: observed.append(out.dtype))
                   for b in m.blocks]
        try:
            with torch.autocast('cpu', dtype=torch.bfloat16):
                logits = m(self.val)
            self.assertEqual(logits.dtype, torch.bfloat16)
            self.assertTrue(observed and all(t == torch.float32 for t in observed))
        finally:
            for h in handles:
                h.remove()
        original = torch.optim.Adam.step
        checked = []
        def check_state(opt, *args, **kwargs):
            result = original(opt, *args, **kwargs)
            for group in opt.param_groups:
                for p in group['params']:
                    self.assertEqual(p.dtype, torch.float32)
                    self.assertEqual(p.grad.dtype, torch.float32)
                    for name in ['exp_avg', 'exp_avg_sq']:
                        self.assertEqual(opt.state[p][name].dtype, torch.float32)
            checked.append(True)
            return result
        with patch.object(torch.optim.Adam, 'step', check_state):
            result = run(c, self.train, self.val, base_lr=.001,
                initialize_fn=fixture_initialize, groups_fn=fixture_groups, device='cpu')
        self.assertEqual(len(checked), 5)
        self.assertEqual(result['precision'], 'bfloat16_autocast_fp32_residual')

    def test_loader_rechunks_a_fixed_prefix(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = np.arange(32, dtype=np.uint16).reshape(4, 8)
            rows.tofile(root / 'tokens.bin')
            (root / 'metadata.json').write_text(json.dumps(dict(
                dtype='uint16', num_sequences=4, seq_len=8)))
            tokens = load_tokens(root, 3, 4)
            torch.testing.assert_close(tokens, torch.arange(12).reshape(3, 4))
            with self.assertRaises(ValueError):
                load_tokens(root, 20, 4)

    def test_no_overwrite_and_no_missing_optimizer_parameters(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'result.json'
            self.run_fixture(output=path)
            with self.assertRaises(FileExistsError):
                self.run_fixture(output=path)
        def missing(model, lr):
            return [dict(params=[model.head.weight], lr=lr, eps=1e-6)]
        with self.assertRaisesRegex(ValueError, 'every model parameter exactly once'):
            run(self.c, self.train, self.val, base_lr=.001, initialize_fn=fixture_initialize,
                groups_fn=missing, device='cpu')


if __name__ == '__main__':
    unittest.main()
