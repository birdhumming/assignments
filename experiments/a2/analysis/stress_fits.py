"""P4.1 analysis: loss-vs-LR fits, feature movement, update alignment, readout alignment from stress JSONs.

Fetch first:  uv run modal volume get <user volume> a2-stress/ ~/a2/stress/ --env cs312-aleyang
Then:         uv run python -m experiments.a2.analysis.stress_fits
"""
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

DATA = Path(os.environ.get('A2_DATA_DIR', Path.home() / 'a2'))
STRESS = DATA / 'stress'
OUT = DATA / 'tables'
NAME = re.compile(r'p41-(?P<test>width|depth)-(?P<policy>[a-z_]+)-w(?P<width>\d+)-d(?P<depth>\d+)-lr(?P<lr>[0-9.e-]+)-(?P<precision>\w+)')


def load():
    rows = []
    for f in sorted(STRESS.glob('p41-*.json')):
        m = NAME.match(f.stem)
        if not m:
            continue
        r = json.loads(f.read_text())
        meta = m.groupdict(); meta['width'] = int(meta['width']); meta['depth'] = int(meta['depth']); meta['lr'] = float(meta['lr'])
        rows.append(dict(**meta, result=r, final=r['history'][-1]['val_loss'], history=r['history'], alignment=r['alignment']))
    return rows


def fit_curve(lrs, losses):
    """Quadratic in log LR around the sampled minimum (3-5 points). Returns lr*, L*, best sampled lr."""
    lrs, losses = np.array(lrs), np.array(losses)
    o = np.argsort(lrs); lrs, losses = lrs[o], losses[o]
    i = int(np.argmin(losses)); lo, hi = max(0, i - 2), min(len(lrs), i + 3)
    x, y = np.log(lrs[lo:hi]), losses[lo:hi]
    if len(x) >= 3:
        a, b, c = np.polyfit(x, y, 2)
        if a > 0:
            xs = float(np.clip(-b / (2 * a), x.min(), x.max()))
            return float(np.exp(xs)), float(np.polyval([a, b, c], xs)), float(lrs[i])
    return float(lrs[i]), float(losses[i]), float(lrs[i])


def size_of(r):
    return r['width'] if r['test'] == 'width' else r['depth']


def curves(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r['test'], r['policy'], size_of(r))].append(r)
    table = []
    for (test, policy, size), rs in sorted(groups.items()):
        lr_s, L_s, best = fit_curve([r['lr'] for r in rs], [r['final'] for r in rs])
        table.append({'test': test, 'policy': policy, 'size': size, 'n': len(rs), 'lr_star': lr_s, 'loss_star': L_s, 'best_lr': best,
                          'losses': {r['lr']: r['final'] for r in sorted(rs, key=lambda r: r['lr'])}})
    return table


def probes(r):
    """Per-step logit RMS, final-norm RMS/movement, omega_move, residual/branch RMS, alpha per matrix."""
    out = []
    for h in r['history']:
        fn = h['features'].get('final_norm', {})
        ra = h.get('readout_alignment', {})
        out.append({'step': h['step'], 'val_loss': h['val_loss'], 'logit_rms': h['logit_rms'],
                        'final_rms': fn.get('rms'), 'final_move': fn.get('movement'), 'final_rel_move': fn.get('relative_movement'),
                        'omega_move': (ra.get('movement') or {}).get('omega'),
                        'residual_rms': {k: v for k, v in h['residual_rms'].items()},
                        'branch_rms': {k: v for k, v in h['unscaled_branch_rms'].items()},
                        'features': {k: v for k, v in h['features'].items()}})
    alpha = defaultdict(dict)
    for a in r['alignment']:
        alpha[a['step']][a['parameter']] = a.get('alpha')
    return out, alpha


def md_table(rows, cols, fmt=None):
    fmt = fmt or {}
    head = '| ' + ' | '.join(cols) + ' |\n|' + '---|' * len(cols) + '\n'
    body = ''
    for r in rows:
        body += '| ' + ' | '.join(fmt.get(c, lambda v: f'{v}')(r[c]) if r.get(c) is not None else '' for c in cols) + ' |\n'
    return head + body


def main():
    rows = load()
    print(len(rows), 'stress results')
    OUT.mkdir(parents=True, exist_ok=True)
    table = curves(rows)
    g = lambda v: f'{v:.4g}'
    md = '# P4.1 loss-vs-LR fits (val loss after update 5)\n\n'
    md += md_table(table, ['test', 'policy', 'size', 'n', 'best_lr', 'lr_star', 'loss_star'], {'best_lr': g, 'lr_star': g, 'loss_star': g})
    md += '\n## Sampled losses\n\n'
    for t in table:
        md += f"- {t['test']} {t['policy']} {t['size']}: " + ', '.join(f'{lr:g}: {L:.4f}' for lr, L in t['losses'].items()) + '\n'
    # probes at each group's best sampled LR
    best = {(t['test'], t['policy'], t['size']): t['best_lr'] for t in table}
    md += '\n# Probes at best sampled LR\n\n'
    for r in sorted(rows, key=lambda r: (r['test'], r['policy'], size_of(r))):
        key = (r['test'], r['policy'], size_of(r))
        if abs(r['lr'] - best[key]) > 1e-12:
            continue
        p, alpha = probes(r)
        md += f"\n## {r['test']} {r['policy']} size {size_of(r)} lr {r['lr']:g}\n\n"
        md += md_table(p, ['step', 'val_loss', 'logit_rms', 'final_rms', 'final_move', 'omega_move'],
                       {k: g for k in ('val_loss', 'logit_rms', 'final_rms', 'final_move', 'omega_move')})
        md += '\nupdate alignment alpha (step 1 / step 5):\n\n'
        names = sorted(alpha[1]) if 1 in alpha else []
        arows = [{'parameter': n, 'a1': alpha[1].get(n), 'a5': alpha.get(5, {}).get(n)} for n in names
                 if n == 'head.weight' or n.startswith(('blocks.0.', f"blocks.{(r['depth'] - 1)}."))]
        md += md_table(arows, ['parameter', 'a1', 'a5'], {'a1': g, 'a5': g})
        md += '\nresidual RMS before norm / unscaled branch RMS at step 5:\n\n'
        last = p[-1]
        md += ', '.join(f'{k}: {v:.3g}' for k, v in last['residual_rms'].items()) + '\n\n'
        md += ', '.join(f'{k}: {v:.3g}' for k, v in last['branch_rms'].items()) + '\n'
    (OUT / 'p41_stress.md').write_text(md)
    json.dump(table, (OUT / 'p41_fits.json').open('w'), indent=1)
    print('wrote', OUT / 'p41_stress.md')


if __name__ == '__main__':
    main()
