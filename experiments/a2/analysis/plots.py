"""P4.1/P4.2 figures: fitted optimal base LR and fitted minimum loss versus width/depth, plus loss-LR curves."""
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

TABLES = Path('/home/ubuntu/a2/tables')
FIG = Path(__file__).resolve().parents[1] / 'figures'
FIG.mkdir(exist_ok=True)
LABEL = {'kaiming': 'Kaiming', 'baseline': 'baseline (Kaiming-style)', 'mup': 'µP', 'depth_mup': 'Depth-µP', 'completep': 'CompleteP', 'supplied': 'source (width 512, depth 8)'}


def p41():
    fits = json.loads((TABLES / 'p41_fits.json').read_text())
    for test, xlabel in (('width', 'width'), ('depth', 'depth')):
        rows = [f for f in fits if f['test'] == test]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        for pol in dict.fromkeys(f['policy'] for f in rows):
            sub = sorted((f for f in rows if f['policy'] == pol), key=lambda f: f['size'])
            xs = [f['size'] for f in sub]
            axes[0].plot(xs, [f['lr_star'] for f in sub], 'o-', label=LABEL[pol])
            axes[1].plot(xs, [f['loss_star'] for f in sub], 'o-', label=LABEL[pol])
            for f in sub:
                lrs = sorted(float(k) for k in f['losses'])
                axes[2].plot(lrs, [f['losses'][k] for k in sorted(f['losses'], key=float)], 'o-', ms=3, label=f"{LABEL[pol]} {f['size']}")
        axes[0].set(xscale='log', yscale='log', xlabel=xlabel, ylabel='fitted optimal base LR', title=f'P4.1 {test}: fitted lr*')
        axes[1].set(xscale='log', xlabel=xlabel, ylabel='fitted min val loss (5 steps)', title=f'P4.1 {test}: fitted loss*')
        axes[2].set(xscale='log', xlabel='base LR', ylabel='val loss after 5 updates', title=f'P4.1 {test}: loss-LR curves')
        axes[2].set_ylim(top=min(12, axes[2].get_ylim()[1]))
        axes[0].legend(); axes[2].legend(fontsize=6, ncol=2)
        fig.tight_layout(); fig.savefig(FIG / f'p41_{test}.png', dpi=130); plt.close(fig)


def p42():
    fits = json.loads((TABLES / 'p42_fits.json').read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    wrows = [f for f in fits if f['depth'] == 8]
    for pol in ('baseline', 'mup'):
        sub = sorted((f for f in wrows if f['policy'] in (pol, 'supplied')), key=lambda f: f['width'])
        xs = [f['width'] for f in sub]
        axes[0].plot(xs, [f['lr_star'] for f in sub], 'o-', label=LABEL[pol])
        axes[1].plot(xs, [f['loss_star'] for f in sub], 'o-', label=f'{LABEL[pol]} tuned (fit)')
        axes[1].plot(xs, [f['loss_at_source_lr'] for f in sub], 'x--', label=f'{LABEL[pol]} at transferred LR 0.003')
    axes[0].axhline(.003, color='gray', ls=':', label='source LR 0.003')
    axes[0].set(xscale='log', yscale='log', xlabel='width', ylabel='fitted optimal LR', title='P4.2 width: fitted lr*')
    axes[1].set(xscale='log', xlabel='width', ylabel='val loss at 153.6M tokens', title='P4.2 width: loss')
    axes[0].legend(); axes[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / 'p42_width.png', dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    drows = [f for f in fits if f['width'] == 512]
    for pol in ('mup', 'depth_mup', 'completep'):
        sub = sorted((f for f in drows if f['policy'] in (pol, 'supplied')), key=lambda f: f['depth'])
        xs = [f['depth'] for f in sub]
        axes[0].plot(xs, [f['lr_star'] for f in sub], 'o-', label=LABEL[pol])
        axes[1].plot(xs, [f['loss_star'] for f in sub], 'o-', label=f'{LABEL[pol]} tuned (fit)')
        axes[1].plot(xs, [f['loss_at_source_lr'] for f in sub], 'x--', label=f'{LABEL[pol]} at LR 0.003')
    axes[0].axhline(.003, color='gray', ls=':', label='source LR 0.003')
    axes[0].set(xscale='log', yscale='log', xlabel='depth', ylabel='fitted optimal LR', title='P4.2 depth: fitted lr*')
    axes[1].set(xscale='log', xlabel='depth', ylabel='val loss at 153.6M tokens', title='P4.2 depth: loss')
    axes[0].legend(); axes[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / 'p42_depth.png', dpi=130); plt.close(fig)


if __name__ == '__main__':
    p41(); p42()
    print('wrote', sorted(p.name for p in FIG.glob('p4*.png')))
