"""Figures for the supplied-sweep analyses (P1a/b/d, P2a/b) and the P3.1 momentum sweep."""
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiments.a2.analysis.sources import by_budget, p2_joint, powerlaw, quad_opt
from experiments.a2.provided_sweeps import load

DATA = Path(os.environ.get('A2_DATA_DIR', Path.home() / 'a2'))
FIG = Path(__file__).resolve().parents[1] / 'figures'
B = 1e9


def _opt(rows):
    return {t: quad_opt(v) for t, v in by_budget(rows).items()}


def p1_sources():
    p1a, p1b, p1d = load('P1a'), load('P1b'), load('P1d')
    oa, ob, od = _opt(p1a), _opt(p1b), _opt(p1d)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    ax = axes[0]
    for t, pts in sorted(by_budget(p1a + p1b).items()):
        ax.semilogx([p[0] for p in pts], [p[1] for p in pts], 'o-', ms=4, label=f'{t/B:.4g}B')
    ax.set_xlabel('peak learning rate'); ax.set_ylabel('final validation loss')
    ax.set_title('P1(a,b): AdamW loss vs LR, supplied'); ax.legend(fontsize=7)
    ax = axes[1]
    ts = np.array(sorted(oa)); tb = np.array(sorted(ob))
    ax.loglog(ts, [oa[t][0] for t in ts], 'o', color='C0', label='fitted lr*, small budgets (a)')
    ax.loglog(tb, [ob[t][0] for t in tb], 's', color='C3', label='fitted lr*, large budgets (b)')
    grid = np.logspace(np.log10(1.2e8), np.log10(6e9), 50)
    for sub, lab, c, ls in ((oa, 'small-3 rule', 'C0', '--'), ({**oa, **ob}, 'all-six rule', 'C2', ':'), (ob, 'larger-3 rule', 'C3', '-.')):
        tt = np.array(sorted(sub)); cc, p, r2 = powerlaw(tt, np.array([sub[t][0] for t in tt]))
        ax.loglog(grid, cc * grid ** p, ls, color=c, alpha=.8, label=f'{lab}: D^{p:+.2f}, R²={r2:.2f}')
    ax.axvline(4.9152e9, color='k', lw=.5); ax.set_ylim(1e-3, 1e-2)
    ax.set_xlabel('training tokens'); ax.set_ylabel('optimal peak LR'); ax.set_title('P1(a–c): fitted optima and rules'); ax.legend(fontsize=7)
    ax = axes[2]
    td = np.array(sorted(od))
    ax.loglog(td, [od[t][0] for t in td], 'o', color='C1', label='Hyperball fitted lr*')
    cc, p, r2 = powerlaw(td, np.array([od[t][0] for t in td]))
    g2 = np.logspace(np.log10(1.2e8), np.log10(1.5e9), 30)
    ax.loglog(g2, cc * g2 ** p, '--', color='C1', label=f'Hyperball rule D^{p:+.2f}, R²={r2:.2f}')
    ax.plot([1.2288e9], [cc * 1.2288e9 ** p], 'x', color='C1', ms=10, label='prediction 0.0084')
    ax.plot([1.2288e9], [0.0102], 's', mfc='none', color='C1', ms=9, label='measured lr* 0.0102')
    ax.loglog(ts, [oa[t][0] for t in ts], 'o', color='C0', label='AdamW fitted lr*')
    cc, p, r2 = powerlaw(ts, np.array([oa[t][0] for t in ts]))
    ax.loglog(g2, cc * g2 ** p, '--', color='C0', label=f'AdamW rule D^{p:+.2f}, R²={r2:.2f}')
    ax.set_xlabel('training tokens'); ax.set_ylabel('optimal peak LR'); ax.set_title('P1(d): Hyperball vs AdamW'); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / 'p1_sources.png', dpi=130); plt.close(fig)


def _joint_fit(rs):
    x = np.log([r['learning_rate'] for r in rs]); y = np.log([r['weight_decay'] for r in rs]); z = np.array([r['final_val_loss'] for r in rs])
    X = np.column_stack([np.ones_like(x), x, y, x * x, x * y, y * y])
    coef, *_ = np.linalg.lstsq(X, z, rcond=None)
    return coef, x, y, z


def _eval(coef, x, y):
    a, b, c, d, e, f = coef
    return a + b * x + c * y + d * x * x + e * x * y + f * y * y


def p2():
    rows = load('P2a')
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        joint = p2_joint(rows)
    ts = sorted(joint)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, t in zip(axes, ts):
        rs = [r for r in rows if r['tokens'] == t]
        coef, x, y, z = _joint_fit(rs)
        gx = np.linspace(x.min() - .7, x.max() + .7, 80); gy = np.linspace(y.min() - .7, y.max() + .7, 80)
        GX, GY = np.meshgrid(gx, gy)
        Z = _eval(coef, GX, GY)
        cs = ax.contour(np.exp(GX), np.exp(GY), Z, levels=14, cmap='viridis')
        ax.clabel(cs, fontsize=6, fmt='%.3f')
        sc = ax.scatter(np.exp(x), np.exp(y), c=z, cmap='viridis', edgecolors='k', s=45, zorder=3)
        lr_s, wd_s = joint[t][0], joint[t][1]
        ax.plot([lr_s], [wd_s], 'r*', ms=13, zorder=4, label=f'fitted optimum ({lr_s:.4f}, {wd_s:.2f})')
        lrs = np.exp(gx); ax.plot(lrs, lr_s * wd_s / lrs, 'r--', lw=1, label='lr·wd = lr*·wd*')
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(np.exp(gx[0]), np.exp(gx[-1])); ax.set_ylim(np.exp(gy[0]), np.exp(gy[-1]))
        ax.set_title(f'{t/B:.4g}B tokens'); ax.set_xlabel('peak LR'); ax.legend(fontsize=6, loc='lower left')
    axes[0].set_ylabel('weight decay')
    fig.suptitle('P2(a,b): quadratic fits in (log LR, log WD); the long axis follows lr·wd = const', fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / 'p2_contours.png', dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    T = np.array(ts, float)
    grid = np.logspace(np.log10(1.2e8), np.log10(3e9), 40)
    for name, vals, c in (('lr*', [joint[t][0] for t in ts], 'C0'), ('wd*', [joint[t][1] for t in ts], 'C1'), ('lr*·wd*', [joint[t][0] * joint[t][1] for t in ts], 'C2')):
        cc, p, r2 = powerlaw(T, np.array(vals))
        ax.loglog(T, vals, 'o', color=c); ax.loglog(grid, cc * grid ** p, '--', color=c, label=f'{name}: D^{p:+.2f}, R²={r2:.3f}')
    ax.plot([2.4576e9], [2.782e-4], 'x', color='C2', ms=10, label='P2(c) predicted product')
    ax.plot([2.4576e9], [.003 * .096], 's', mfc='none', color='C2', ms=9, label='P2(c) measured (lr .003 × wd* .096)')
    ax.set_xlabel('training tokens'); ax.set_title('P2(b): fitted optima vs tokens'); ax.legend(fontsize=7)
    ax = axes[1]
    p1 = load('P1a') + load('P1b')
    p1best = [min(r['final_val_loss'] for r in p1 if r['tokens'] == t) for t in ts]
    p2best = [joint[t][3]['final_val_loss'] for t in ts]
    ax.semilogx(T, p1best, 'o-', label='P1 best (WD fixed 0.1)'); ax.semilogx(T, p2best, 's-', label='P2 best (LR, WD jointly tuned)')
    ax2 = ax.twinx(); ax2.semilogx(T, np.array(p1best) - np.array(p2best), 'k--', marker='^', label='difference')
    ax2.set_ylabel('P1 − P2 loss'); ax2.set_ylim(0, .06)
    ax.set_xlabel('training tokens'); ax.set_ylabel('best measured validation loss'); ax.set_title('P2(a): benefit of tuning weight decay')
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / 'p2_trends.png', dpi=130); plt.close(fig)


def p31_momentum():
    res = json.loads((DATA / 'nqm_results.json').read_text())['momentum']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    betas = (0., .5, .9, .95, .98, .99)
    for ax, Bsz in zip(axes, (16, 256)):
        ax.semilogy(betas, [res[f'adam_B{Bsz}_beta1_{b}']['loss_star'] for b in betas], 'o-', label='Adam, LR tuned per β₁')
        for mu, c in ((0.0, 'C1'), (0.9, 'C3')):
            r = res[f'sgd_B{Bsz}_mu{mu}']; ax.axhline(r['loss_star'], color=c, ls='--', label=f'SGD µ={mu}, tuned (lr*={r["lr_star"]:.3g})')
        ax.set_title(f'P3.1(d): B = {Bsz} ({8192 // Bsz} updates)'); ax.set_xlabel('β₁'); ax.legend(fontsize=7)
    axes[0].set_ylabel('best final loss after tuning LR')
    fig.tight_layout(); fig.savefig(FIG / 'p31_momentum.png', dpi=130); plt.close(fig)


if __name__ == '__main__':
    p1_sources(); p2(); p31_momentum()
    print('wrote', sorted(p.name for p in FIG.glob('p1_*.png')) + sorted(p.name for p in FIG.glob('p2_*.png')) + ['p31_momentum.png'])
