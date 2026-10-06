"""Plots + markdown table for the P3.1 noisy-quadratic results (reads nqm_results.json)."""
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

DATA = Path(os.environ.get('A2_DATA_DIR', Path.home() / 'a2'))
FIG = Path(__file__).resolve().parents[1] / 'figures'
FIG.mkdir(exist_ok=True)
res = json.loads((DATA / 'nqm_results.json').read_text())


def curve(ax, r, label, color):
    src, tgt = r['source'], r['target']
    Bs = np.array(sorted(int(b) for b in src)); Bt = np.array(sorted(int(b) for b in tgt))
    ax.loglog(Bs, [src[str(b)]['lr_star'] for b in Bs], 'o-', color=color, label=f"{label} (p={r['exponent']:.2f})")
    ax.loglog(Bt, [tgt[str(b)]['lr_star'] for b in Bt], 's', color=color, mfc='none', ms=9)
    xs = np.array([1, 512]); ax.loglog(xs, r['coeff'] * xs ** r['exponent'], '--', color=color, alpha=.5)
    ax.plot([256], [r['pred_lr_256']], 'x', color=color, ms=10)


fig, ax = plt.subplots(figsize=(6, 4.5))
for key, c in (('sgd', 'C0'), ('rmsprop', 'C1'), ('adam', 'C2')):
    curve(ax, res[key], res[key]['label'], c)
ax.plot([], [], 'ko', label='source (B<=64) fitted lr*'); ax.plot([], [], 'ks', mfc='none', label='target measured lr*')
ax.plot([], [], 'kx', label='predicted lr*(256)')
ax.set_xlabel('batch size B'); ax.set_ylabel('optimal learning rate'); ax.set_title('NQM: optimal LR vs batch size (sigma=1)')
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIG / 'p31_lr_vs_batch.png', dpi=130)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
for ax, (kind, title) in zip(axes, (('sigma', 'two-dimensional H=diag(1,10)'), ('scalar', 'scalar H=1'))):
    for i, sigma in enumerate((1, 10, 100, 300)):
        for opt, ls in (('rmsprop', '-'), ('adam', '--')):
            r = res[kind][f'{opt}_sigma{sigma}']
            src = r['source']; Bs = np.array(sorted(int(b) for b in src))
            ax.loglog(Bs, [src[str(b)]['lr_star'] for b in Bs], ls, marker='o', ms=3, color=f'C{i}',
                      label=f"{opt} sigma={sigma} (p={r['exponent']:.2f})")
    ax.set_title(title); ax.set_xlabel('batch size B'); ax.legend(fontsize=7)
axes[0].set_ylabel('optimal learning rate'); fig.tight_layout(); fig.savefig(FIG / 'p31_sigma.png', dpi=130)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, key in zip(axes, ('sgd', 'rmsprop', 'adam')):
    for b, c in zip((1, 8, 64, 256), ('C0', 'C1', 'C2', 'C3')):
        d = (res[key]['source'] if b <= 64 else res[key]['target'])[str(b)]
        ax.loglog(d['lrs'], d['losses'], 'o-', ms=3, color=c, label=f'B={b}')
    ax.set_title(res[key]['label']); ax.set_xlabel('learning rate'); ax.legend(fontsize=8)
axes[0].set_ylabel('final loss'); fig.tight_layout(); fig.savefig(FIG / 'p31_loss_curves.png', dpi=130)

rows = ['| setting | exponent p | predicted lr*(256) | measured lr*(256) | loss at predicted | tuned loss | gap |', '|---|---|---|---|---|---|---|']
def row(name, r):
    t = r['target']['256']
    rows.append(f"| {name} | {r['exponent']:.2f} | {r['pred_lr_256']:.3g} | {t['lr_star']:.3g} | {r['loss_at_pred_256']:.3g} | {t['loss_star']:.3g} | {r['gap_256']:+.2g} |")
for k in ('sgd', 'rmsprop', 'adam'):
    row(res[k]['label'], res[k])
for kind in ('sigma', 'scalar'):
    for k, r in res[kind].items():
        row(f'{kind} {k}', r)
rows += ['', '| momentum setting | lr* | loss* |', '|---|---|---|']
for k, r in res['momentum'].items():
    rows.append(f"| {k} | {r['lr_star']:.3g} | {r['loss_star']:.3g} |")
(DATA / 'tables').mkdir(exist_ok=True)
(DATA / 'tables' / 'p31_nqm.md').write_text('\n'.join(rows) + '\n')
print('\n'.join(rows))
