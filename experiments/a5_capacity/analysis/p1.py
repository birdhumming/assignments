"""Problem 1 analysis from the local manifest copy (results/a5_capacity/reference).

    uv run python -m experiments.a5_capacity.analysis.p1
"""
import gzip
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ..p1_depth_width import DEPTHS, RATIOS, EXTREMES, training_flops, plot_loss_vs_parameters
from modeling import LMConfig

ROOT = Path('results/a5_capacity/reference')
PLOTS = Path('plots/a5_capacity')
STYLE = {'font.family': 'serif', 'font.serif': ['Palatino', 'P052', 'DejaVu Serif'], 'font.size': 11,
         'axes.spines.top': False, 'axes.spines.right': False}
COLORS = ('#5178D5', '#7030A0', '#2B8D6E', '#F8961E', '#C7A020')


def load():
    release = json.loads((ROOT/'manifest.json').read_text())
    rows = [r for r in release['runs'] if r['val_loss'] is not None and math.isfinite(r['val_loss'])
            and r['tokens'] == 614_400_000]
    for r in rows:
        r['flops'] = training_flops(LMConfig.from_dict(r['architecture']))
    return release, rows


def best_by_shape(rows, seed=42):
    out = {}
    for r in rows:
        if r['seed'] != seed:
            continue
        k = r['depth'], r['width']
        if k not in out or r['val_loss'] < out[k]['val_loss']:
            out[k] = r
    return out


def fit_reference(sel):
    """L = Linf + A (C/C0)^-alpha on the d4..d10 default ladder (ratio 64), C0 = d8 cost."""
    ladder = [sel[d, 64*d] for d in DEPTHS]
    C0 = sel[8, 512]['flops']
    c = np.array([r['flops']/C0 for r in ladder]); L = np.array([r['val_loss'] for r in ladder])
    best = None
    for a in np.linspace(.05, 2., 1951):           # linear in (Linf, A) at fixed alpha
        X = np.stack([np.ones_like(c), c**-a], 1)
        coef, *_ = np.linalg.lstsq(X, L, rcond=None)
        err = ((X@coef-L)**2).sum()
        if coef[1] > 0 and (best is None or err < best[0]):
            best = err, coef[0], coef[1], a
    _, Linf, A, a = best
    return dict(Linf=Linf, A=A, alpha=a, C0=C0, ladder=[(r['depth'], r['flops'], r['val_loss']) for r in ladder])


def plot_loss_vs_flops(rows, sel, fit, out=PLOTS/'p1_loss_vs_flops.png'):
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 5))
        for ratio, color in zip(RATIOS, COLORS):
            trials = [r for r in rows if r['width'] == ratio*r['depth'] and r['depth'] in DEPTHS]
            pts = [sel[d, ratio*d] for d in DEPTHS if (d, ratio*d) in sel]
            ax.scatter([r['flops']/1e15 for r in trials], [r['val_loss'] for r in trials], color=color, s=16, alpha=.15)
            ax.plot([r['flops']/1e15 for r in pts], [r['val_loss'] for r in pts], 'o-', color=color, lw=1.5, ms=5,
                    label=f'W/D = {ratio}')
            for r in pts:
                ax.annotate(f"d{r['depth']}", (r['flops']/1e15, r['val_loss']), fontsize=6, alpha=.6,
                            xytext=(3, 3), textcoords='offset points')
        xs = np.logspace(np.log10(min(r['flops'] for r in rows))-15, np.log10(max(r['flops'] for r in rows))-15, 100)
        ax.plot(xs, fit['Linf'] + fit['A']*(xs*1e15/fit['C0'])**-fit['alpha'], 'k--', lw=1, alpha=.6,
                label=f"ladder fit: {fit['Linf']:.2f} + {fit['A']:.2f}(C/C0)^-{fit['alpha']:.2f}")
        for key, (d, w) in EXTREMES.items():
            r = sel.get((d, w))
            if r:
                ax.plot(r['flops']/1e15, r['val_loss'], 'k*' if d > 8 else 'kP', ms=10, label=f'{key} d{d} w{w}')
        ax.set(xscale='log', xlabel='Estimated training FLOPs (PFLOP)', ylabel='Validation loss (nats / next token)',
               title='Loss vs estimated compute at 614.4M training positions')
        ax.grid(True, which='major', ls=':', alpha=.3); ax.legend(frameon=False, fontsize=8)
        out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=300, bbox_inches='tight'); plt.close(fig)


def plot_directions(sel, out=PLOTS/'p1_directions.png'):
    """From d8: deeper at width 512 vs wider at depth 8, vs the ratio-64 ladder."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
        ax = axes[0]
        series = {'only depth (W=512)': [(d, 512) for d in (3, 4, 5, 6, 8, 10, 12, 16, 20, 22)],
                  'only width (D=8)': [(8, w) for w in (128, 256, 320, 384, 448, 512, 576, 640, 1024, 2048)],
                  'ratio 64 ladder': [(d, 64*d) for d in DEPTHS]}
        for (name, shapes), color in zip(series.items(), COLORS):
            pts = [sel[s] for s in shapes if s in sel]
            ax.plot([r['flops']/1e15 for r in pts], [r['val_loss'] for r in pts], 'o-', color=color, label=name, ms=4)
            for r in pts:
                ax.annotate(f"d{r['depth']}w{r['width']}", (r['flops']/1e15, r['val_loss']), fontsize=5.5, alpha=.7,
                            xytext=(3, 2), textcoords='offset points')
        ax.set(xscale='log', xlabel='Estimated training FLOPs (PFLOP)', ylabel='Validation loss',
               title='Spending compute on depth only or width only'); ax.grid(True, ls=':', alpha=.3); ax.legend(frameon=False, fontsize=8)
        ax = axes[1]
        for d, color in zip(DEPTHS, plt.cm.viridis(np.linspace(0, 1, len(DEPTHS)))):
            pts = [sel[d, ratio*d] for ratio in RATIOS if (d, ratio*d) in sel]
            ax.plot([r['width']/r['depth'] for r in pts], [r['val_loss'] for r in pts], 'o-', color=color, ms=4, label=f'd{d}')
        ax.set(xscale='log', xlabel='W/D', ylabel='Validation loss', title='Loss vs ratio at fixed depth')
        ax.set_xticks(RATIOS, [str(r) for r in RATIOS]); ax.grid(True, ls=':', alpha=.3); ax.legend(frameon=False, fontsize=8, ncol=2)
        fig.savefig(out, dpi=300, bbox_inches='tight'); plt.close(fig)


def plot_iso_budget(sel, fit, out=PLOTS/'p1_iso_budget.png'):
    """Best ratio at several budgets: loss vs ratio for shapes grouped by FLOP band; efficiency vs ratio."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
        grid = [sel[d, ratio*d] for d in DEPTHS for ratio in RATIOS if (d, ratio*d) in sel]
        for r in grid:
            r['eff'] = fit['C0']*(fit['A']/max(r['val_loss']-fit['Linf'], 1e-6))**(1/fit['alpha'])/r['flops']
        ax = axes[0]
        sc = ax.scatter([r['width']/r['depth'] for r in grid], [r['eff'] for r in grid], c=np.log10([r['flops'] for r in grid]),
                        cmap='plasma', s=30)
        for r in grid:
            ax.annotate(f"d{r['depth']}", (r['width']/r['depth'], r['eff']), fontsize=5.5, alpha=.7, xytext=(3, 2), textcoords='offset points')
        ax.axhline(1, color='k', lw=.8, ls='--')
        ax.set(xscale='log', xlabel='W/D', ylabel='efficiency vs ratio-64 ladder fit', title='Dense-equivalent efficiency by ratio')
        ax.set_xticks(RATIOS, [str(r) for r in RATIOS]); ax.grid(True, ls=':', alpha=.3)
        plt.colorbar(sc, ax=ax, label='log10 training FLOPs')
        ax = axes[1]
        for ratio, color in zip(RATIOS, COLORS):
            pts = sorted([r for r in grid if r['width'] == ratio*r['depth']], key=lambda r: r['flops'])
            ax.plot([r['flops']/1e15 for r in pts], [r['eff'] for r in pts], 'o-', color=color, ms=4, label=f'W/D = {ratio}')
        ax.axhline(1, color='k', lw=.8, ls='--')
        ax.set(xscale='log', xlabel='Estimated training FLOPs (PFLOP)', ylabel='efficiency', title='Efficiency vs budget per ratio')
        ax.grid(True, ls=':', alpha=.3); ax.legend(frameon=False, fontsize=8)
        fig.savefig(out, dpi=300, bbox_inches='tight'); plt.close(fig)
    return grid


def history(r):
    with gzip.open(ROOT/r['history'], 'rt') as f:
        return json.load(f)


def plot_extremes(rows, out=PLOTS/'p1_extremes.png'):
    picks = {'d8 w512 (lr .003)': next(r for r in rows if (r['depth'], r['width'], r['lr'], r['seed']) == (8, 512, .003, 42))}
    for key, (d, w) in EXTREMES.items():
        picks[f'{key}: d{d} w{w} (lr .003)'] = next(r for r in rows if (r['depth'], r['width']) == (d, w))
    table = {}
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
        for (name, r), color in zip(picks.items(), COLORS):
            h = history(r)
            tr = [(e['optimizer_step'], e['train_loss']) for e in h if e.get('train_loss') is not None]
            va = [(e['optimizer_step'], e['val_loss']) for e in h if e.get('val_loss') is not None]
            axes[0].plot(*zip(*tr), color=color, lw=.6, alpha=.6)
            axes[0].plot(*zip(*va), 'o-', color=color, ms=3, label=f"{name}: {r['val_loss']:.3f}")
            axes[1].plot(*zip(*va), 'o-', color=color, ms=3, label=name)
            table[name] = dict(val_loss=r['val_loss'], parameters=r['parameters'], flops=r['flops'], lr=r['lr'],
                               runtime=h[-1].get('_runtime'), run=r['run_name'])
        axes[0].set(ylim=(2.8, 5.), xlabel='optimizer step', ylabel='loss', title='Parameter-matched extremes vs d8 (train faint, val dots)')
        axes[1].set(ylim=(2.95, 3.4), xlim=(3000, 9500), xlabel='optimizer step', ylabel='val loss', title='Late training')
        for ax in axes:
            ax.grid(True, ls=':', alpha=.3); ax.legend(frameon=False, fontsize=8)
        fig.savefig(out, dpi=300, bbox_inches='tight'); plt.close(fig)
    return table


def main():
    release, rows = load()
    sel = best_by_shape(rows)
    plot_loss_vs_parameters(release, output=str(PLOTS/'p1_loss_vs_parameters.png'))
    fit = fit_reference(sel)
    plot_loss_vs_flops(rows, sel, fit)
    plot_directions(sel)
    grid = plot_iso_budget(sel, fit)
    extremes = plot_extremes(rows)
    counts = {}
    for r in rows:
        counts[r['depth'], r['width']] = counts.get((r['depth'], r['width']), 0) + 1
    table = [dict(depth=d, width=w, ratio=w/d, val_loss=r['val_loss'], lr=r['lr'], trials=counts[d, w],
                  parameters=r['parameters'], flops=r['flops'], eff=r.get('eff')) for (d, w), r in sorted(sel.items())]
    summary = dict(published_at=release['published_at'], run_count=release['run_count'], fit=fit, shapes=table, extremes=extremes)
    Path('results/a5_capacity/p1_summary.json').write_text(json.dumps(summary, indent=1, default=float))
    print('fit', {k: v for k, v in fit.items() if k != 'ladder'})
    print(f"{'shape':10s} {'ratio':>5s} {'loss':>6s} {'lr':>7s} {'n':>2s} {'params(M)':>9s} {'PFLOP':>7s} {'eff':>5s}")
    for t in table:
        print(f"d{t['depth']}w{t['width']:<5d} {t['ratio']:5.0f} {t['val_loss']:6.3f} {t['lr']:7.4f} {t['trials']:2d} "
              f"{t['parameters']/1e6:9.1f} {t['flops']/1e15:7.1f} {t['eff'] or float('nan'):5.2f}")
    for k, v in extremes.items():
        print(k, v)


if __name__ == '__main__':
    main()
