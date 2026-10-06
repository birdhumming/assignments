"""P4.1 parameterizations for the five-step stress test (width and depth).

Width (depth 2, head dim 64, reference width n0 = 512, m = n / n0):
  kaiming : hidden var 1/k, readout var 1/n,  readout mult 1,   hidden LR eta
  mup     : hidden var 1/k, readout var 1/n0, readout mult 1/m, hidden LR eta/m
Depth (width 64, one head, reference depth 2, r = L / 2, on top of mup width rules):
  mup       : c_L = 1,        block LR x1,        block eps 1e-8
  depth_mup : c_L = r^-1/2,   block LR x r^-1/2,  block eps 1e-8 r^-1/2
  completep : c_L = r^-1,     block LR x1,        block eps 1e-8 r^-1
Embeddings use variance 1, normalization gains 1; embedding/readout/final-norm LR eta, eps 1e-8.
"""
import math

import torch

EPS = 1e-8
WIDTH_POLICIES = ('kaiming', 'mup')
DEPTH_POLICIES = ('mup', 'depth_mup', 'completep')


def _is_block_param(name):
    return name.startswith('blocks.')


def _is_hidden_matrix(name):
    return _is_block_param(name) and name.split('.')[-2] in ('q', 'k', 'v', 'o', 'gate', 'up', 'down')


def depth_multipliers(policy, depth, reference_depth):
    r = depth / reference_depth
    if policy in ('kaiming', 'mup'):
        return {'residual': 1., 'block_lr': 1., 'block_eps': 1.}
    if policy == 'depth_mup':
        return {'residual': r ** -.5, 'block_lr': r ** -.5, 'block_eps': r ** -.5}
    if policy == 'completep':
        return {'residual': 1. / r, 'block_lr': 1., 'block_eps': 1. / r}
    raise ValueError(policy)


def make_policy(policy, *, reference_width=512, reference_depth=2):
    """Return (initialize, parameter_groups) callbacks for stress.run."""
    if policy not in set(WIDTH_POLICIES) | set(DEPTH_POLICIES):
        raise ValueError(policy)
    mup_width = policy != 'kaiming'

    def initialize(model):
        n = model.config.width
        m = n / reference_width
        for name, p in model.named_parameters():
            if name == 'embed.weight':
                p.normal_(0., 1.)
            elif name == 'head.weight':
                p.normal_(0., 1. / math.sqrt(reference_width if mup_width else n))
            elif name.endswith(('norm.weight', 'norm1.weight', 'norm2.weight')):
                p.fill_(1.)
            elif _is_hidden_matrix(name):
                p.normal_(0., 1. / math.sqrt(p.shape[1]))  # fan-in = input dim
            else:
                raise ValueError(f'Unhandled parameter {name}')
        model.output_multiplier = 1. / m if mup_width else 1.
        mult = depth_multipliers(policy, model.config.depth, reference_depth)
        for block in model.blocks:
            block.residual_multiplier = mult['residual']

    def parameter_groups(model, base_lr):
        n = model.config.width
        m = n / reference_width
        mult = depth_multipliers(policy, model.config.depth, reference_depth)
        hidden_lr = base_lr / m if mup_width else base_lr
        groups = {}
        for name, p in model.named_parameters():
            if _is_hidden_matrix(name):
                key = (hidden_lr * mult['block_lr'], EPS * mult['block_eps'])
            elif _is_block_param(name):  # within-block norms
                key = (base_lr * mult['block_lr'], EPS * mult['block_eps'])
            else:  # embedding, readout, final norm
                key = (base_lr, EPS)
            groups.setdefault(key, []).append(p)
        return [{'params': ps, 'lr': lr, 'eps': eps} for (lr, eps), ps in groups.items()]

    return initialize, parameter_groups


if __name__ == '__main__':
    # CPU smoke test on random tokens: every policy must initialize, group, and take five steps.
    import argparse

    from experiments.a2.stress import StressConfig, run
    p = argparse.ArgumentParser()
    p.add_argument('--width', type=int, default=64); p.add_argument('--depth', type=int, default=2)
    p.add_argument('--context', type=int, default=16); p.add_argument('--batch', type=int, default=4)
    a = p.parse_args()
    for pol in ('kaiming', 'mup', 'depth_mup', 'completep'):
        c = StressConfig(width=a.width, depth=a.depth, head_dim=64, batch=a.batch, microbatch=a.batch,
                         context=a.context, probe_sequences=2, precision='fp32')
        g = torch.Generator().manual_seed(0)
        train = torch.randint(0, c.vocab, (c.steps * c.batch, c.context), generator=g)
        val = torch.randint(0, c.vocab, (c.batch, c.context), generator=g)
        init, groups = make_policy(pol)
        r = run(c, train, val, base_lr=1e-3, initialize_fn=init, groups_fn=groups, device='cpu', alignment=True)
        lrs = sorted({(round(x['lr'], 10), x['eps']) for x in r['parameter_groups']})
        print(pol, 'val', [round(h['val_loss'], 3) for h in r['history']], 'groups', lrs,
              'out_mult', r['output_multiplier'], 'res_mult', r['residual_multipliers'][0])
