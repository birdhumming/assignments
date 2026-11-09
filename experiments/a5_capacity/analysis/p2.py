"""P2 readouts: rank ladders per method, suppressed gradient energy, SpamLang."""
import json, math
from .common import load, run, final_loss, curve, fmt, save, plt

RANKS = [32, 64, 128, 256, None]
METHODS = [('softmax', 'softmax'), ('mos', 'MoS (4 components)'), ('sigsoftmax', 'Sigsoftmax'), ('cplm', 'CPLM')]


def key(method, rank):
    if rank is None:
        return 'dense' if method == 'softmax' else method
    return f'{method}-r{rank}'


def main():
    runs, meas = load()
    M = lambda k: meas.get(f'a5-{k}-d8-w512-lr0.003', {})
    grid = {}
    for method, _ in METHODS:
        for rank in RANKS:
            k = key(method, rank); r = run(runs, k)
            ge = (M(k).get('gradient_energy') or {})
            grid[k] = dict(method=method, rank=rank, loss=final_loss(r), state=(r or {}).get('state', 'missing'),
                           suppressed=ge.get('suppressed_fraction'), eff_rank=ge.get('rank'))
            print(f"{k:16s} {grid[k]['state']:9s} loss {fmt(grid[k]['loss'])} suppressed {fmt(grid[k]['suppressed'],3)} ref {fmt(None if rank is None else 1-rank/4096,3)}")
    lines = [r'\begin{table}[H]\centering\small\begin{tabular}{lrrrrr}\toprule',
             r'Method & rank 32 & rank 64 & rank 128 & rank 256 & full (512) \\ \midrule']
    for method, label in METHODS:
        lines.append(label + ' & ' + ' & '.join(fmt(grid[key(method, r)]['loss']) for r in RANKS) + r' \\')
    lines += [r'\midrule softmax: suppressed gradient energy & ' + ' & '.join(fmt(grid[key('softmax', r)]['suppressed'], 3) for r in RANKS) + r' \\',
              r'independent-subspace reference $1-r/4096$ & ' + ' & '.join(f'{1-(r or 512)/4096:.3f}' for r in RANKS) + r' \\',
              r'\bottomrule\end{tabular}\caption{P2(a,d): final validation loss by head rank and output method, d8w512, LR .003. Suppressed energy is the fraction of $p-\mathrm{onehot}(y)$ gradient energy outside the head column space, measured on 8 held-out sequences.}\label{tab:p2grid}\end{table}']
    open('experiments/a5_capacity/latex/p2_tables.tex', 'w').write('\n'.join(lines))

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for method, label in METHODS:
        pts = [(r or 512, grid[key(method, r)]['loss']) for r in RANKS if grid[key(method, r)]['loss'] is not None]
        if pts: axes[0].plot(*zip(*pts), 'o-', ms=4, label=label)
    axes[0].set(xscale='log', xlabel='head rank (512 = full)', ylabel='final val loss', title='Loss vs readout rank'); axes[0].legend(fontsize=8)
    axes[0].set_xticks([32, 64, 128, 256, 512]); axes[0].set_xticklabels(['32', '64', '128', '256', 'full'])
    pts = [(r, grid[key('softmax', r)]['suppressed']) for r in RANKS if r and grid[key('softmax', r)]['suppressed'] is not None]
    if pts: axes[1].plot(*zip(*pts), 'o-', label='measured (softmax head)')
    rs = [32, 64, 128, 256, 512]; axes[1].plot(rs, [1 - r/4096 for r in rs], 'k--', label=r'$1-r/4096$ random subspace')
    axes[1].set(xscale='log', xlabel='head rank', ylabel='gradient energy outside col(W)', title='Suppressed gradient energy'); axes[1].legend(fontsize=8)
    axes[1].set_xticks(rs); axes[1].set_xticklabels([str(r) for r in rs])
    save(fig, 'p2_rank.png')

    # SpamLang: 64 repeats of one symbol -> the only uncertainty is the first token.
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    spam = {}
    for v in (64, 256, 1024, 4096):
        r = runs.get(f'a5-spam-v{v}-w64-lr0.001'); spam[f'v{v}-w64'] = (final_loss(r), (r or {}).get('state', 'missing'))
        s, l = curve(r)
        if s: axes[0].plot(s, l, label=f'V={v}, w=64 (floor log V/63 = {math.log(v)/63:.3f})')
    axes[0].set(xlabel='update', ylabel='val loss (nats/token)', yscale='log', title='SpamLang: vocabulary at width 64'); axes[0].legend(fontsize=7)
    for w in (64, 128, 256, 512):
        r = runs.get(f'a5-spam-v4096-w{w}-lr0.001'); spam[f'v4096-w{w}'] = (final_loss(r), (r or {}).get('state', 'missing'))
        s, l = curve(r)
        if s: axes[1].plot(s, l, label=f'V=4096, w={w}')
    axes[1].set(xlabel='update', ylabel='val loss (nats/token)', yscale='log', title='SpamLang: width at V=4096'); axes[1].legend(fontsize=7)
    save(fig, 'p2_spam.png')
    for k, v in spam.items(): print('spam', k, v)
    json.dump(dict(grid=grid, spam=spam), open('results/a5_capacity/p2_summary.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
