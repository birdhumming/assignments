"""P3 MoE: loss table with router diagnostics, loss curves, and own-router comparison."""
import json
from .common import load, run, final_loss, curve, fmt, save, plt, tag

CENTRAL = [('dense', 'dense FFN'), ('moe-random', 'random top-1'), ('moe-frozen', 'frozen top-1'),
           ('moe-learned', 'learned top-1'), ('moe-unit', 'unit top-1 (matched combine)'), ('moe-full', 'all four experts')]
TRICKS = [('moe-balanced', 'balance .01'), ('moe-balanced001', 'balance .001'), ('moe-balanced1', 'balance .1'),
          ('moe-norm', 'normalized router'), ('moe-noisy', 'noise .05'), ('moe-noisy02', 'noise .2'),
          ('moe-cloned', 'shared init'), ('moe-quantile', 'quantile routing'),
          ('moe-norm-balanced', 'norm + balance .01'), ('moe-balanced-noisy', 'balance .01 + noise .05'),
          ('moe-cloned-balanced', 'shared init + balance .01'), ('moe-all', 'norm+balance+noise+shared init')]
OWN = [('moe-hash', 'hash router (own)'), ('moe-st', 'straight-through router (own)'), ('moe-zloss', 'router z-loss 1e-3 (own)')]


def row(runs, meas, key, label):
    r = run(runs, key)
    m = meas.get(f'a5-{key}-d8-w512-lr0.003', {})
    rt = m.get('routers') or {}
    return dict(label=tag(label, r), key=key, loss=final_loss(r), state=(r or {}).get('state', 'missing'),
                load_cv=rt.get('load_cv'), entropy=rt.get('entropy'), max_p=rt.get('mean_max_probability'),
                params=m.get('parameters'))


def table(rows, caption, label):
    lines = [r'\begin{table}[H]\centering\small', r'\begin{tabular}{lrrrr}\toprule',
             r'Variant & val loss & max layer load CV & norm. entropy & mean max $p$ \\ \midrule']
    for r in rows:
        lines.append(f"{r['label']} & {fmt(r['loss'])} & {fmt(r['load_cv'], 3)} & {fmt(r['entropy'], 3)} & {fmt(r['max_p'], 3)} \\\\")
    lines += [r'\bottomrule\end{tabular}', f'\\caption{{{caption}}}\\label{{{label}}}', r'\end{table}']
    return '\n'.join(lines)


def main():
    runs, meas = load()
    central = [row(runs, meas, *c) for c in CENTRAL]
    tricks = [row(runs, meas, *c) for c in TRICKS]
    own = [row(runs, meas, *c) for c in OWN]
    for r in central + tricks + own:
        print(f"{r['label']:34s} {r['state']:9s} loss {fmt(r['loss'])} cv {fmt(r['load_cv'],3)} H {fmt(r['entropy'],3)} maxp {fmt(r['max_p'],3)}")
    tex = table(central, 'P3(a--b): routing comparison, d8w512, LR .003, 9,375 updates. Diagnostics on 8 held-out sequences of 128 tokens.', 'tab:p3central')
    tex += '\n' + table(tricks, 'P3(c--d): balancing and router conditioning on learned top-1 routing.', 'tab:p3tricks')
    tex += '\n' + table(own, 'P3(e): own router variants; control is learned top-1 (balance 0).', 'tab:p3own')
    open('experiments/a5_capacity/latex/p3_tables.tex', 'w').write(tex)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for key, label in CENTRAL:
        s, l = curve(run(runs, key))
        if s: axes[0].plot(s, l, label=label)
    axes[0].set(xlabel='update', ylabel='val loss', ylim=(2.8, 4.0), title='Routing comparison')
    axes[0].legend(fontsize=7)
    for key, label in [('moe-learned', 'learned top-1')] + TRICKS[:8]:
        s, l = curve(run(runs, key))
        if s: axes[1].plot(s, l, label=label, lw=1)
    axes[1].set(xlabel='update', ylabel='val loss', ylim=(2.8, 4.0), title='Balancing / conditioning')
    axes[1].legend(fontsize=7)
    save(fig, 'p3_curves.png')

    fig, ax = plt.subplots(figsize=(6, 3.6))
    pts = [r for r in central + tricks + own if r['loss'] is not None and r['load_cv'] is not None]
    for r in pts:
        ax.scatter(r['load_cv'], r['loss'], s=18)
        ax.annotate(r['label'], (r['load_cv'], r['loss']), fontsize=6, xytext=(3, 2), textcoords='offset points')
    ax.set(xlabel='max layer load CV (0 = perfect balance)', ylabel='final val loss', title='Loss vs load imbalance')
    save(fig, 'p3_loss_vs_cv.png')
    json.dump(dict(central=central, tricks=tricks, own=own), open('results/a5_capacity/p3_summary.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
