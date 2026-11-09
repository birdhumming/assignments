"""Reference-value table and the list of every run, for the end of the report."""
import json
from .common import load, fmt, final_loss

VARIANT_LABELS = {}  # run name -> short label; default is the variant string from the run name

CUT = {'cplm-r64', 'mos-r64', 'softmax-r256', 'sigsoftmax-r128', 'loop-1block-r1', 'moe-zloss', 'moe-st',
       'moe-balanced-noisy', 'moe-cloned-balanced', 'mos-r256', 'cplm-r256', 'loop-1block-r9', 'loop-1block-r18-norm-recall'}


def parse(name):
    """a5-{variant}-d8-w512-lr{lr}[-ms1] -> (variant, lr); a5-spam-vV-wW-lr -> (spam vV wW, lr)."""
    body = name[3:]
    lr = float(body.rsplit('-lr', 1)[1].split('-')[0])
    variant = body.rsplit('-lr', 1)[0]
    if variant.startswith('spam-'):
        return variant.replace('spam-', 'SpamLang ').replace('-w', ', W=').replace('v', 'V='), lr
    return variant.replace('-d8-w512', ''), lr


def status(name, r, meas):
    loss = final_loss(r)
    if r is None or r.get('state') == 'missing':
        return 'lost' if name in meas else 'cut'
    if loss is None:
        return 'cut' if r.get('state') in ('cancelled', 'finished', 'crashed') else r.get('state', '--')
    s = f'{loss:.4f}' if loss > 1e-2 else f'{loss:.1e}'
    if loss > 6.8:
        s = 'plateau'
    return s + (r'$^\dagger$' if r.get('from_ckpt') else '')


def runs_table(runs, meas):
    rows = []
    for name in sorted(runs):
        if not name.startswith('a5-'):
            continue
        r = runs[name]
        variant, lr = parse(name)
        mins = r.get('runtime')
        st = status(name, r, meas)
        rows.append((variant, f'{lr:g}', st, '--' if not mins or st in ('cut', 'lost') else f'{max(1, mins/60):.0f}'))
    half = (len(rows) + 1) // 2
    lines = [r'\begin{center}\scriptsize\begin{tabular}{llrr @{\hspace{2em}} llrr}\toprule',
             r'Variant & LR & Status & Min & Variant & LR & Status & Min \\ \midrule']
    for i in range(half):
        a = rows[i]
        b = rows[i + half] if i + half < len(rows) else ('', '', '', '')
        lines.append(' & '.join(x.replace('_', r'\_') for x in a + b) + r' \\')
    lines += [r'\bottomrule\end{tabular}\end{center}']
    return '\n'.join(lines), len([r for r in rows if r[2] not in ('cut', 'lost')])


def ref_table(runs):
    L = lambda k: final_loss(runs.get(f'a5-{k}-d8-w512-lr0.003'))
    f = lambda *ks: ', '.join('--' if L(k) is None else f'{L(k):.3f}' for k in ks)
    d = lambda *ks: ', '.join('--' if L(k) is None or L('dense') is None else f'{L(k)-L("dense"):+.3f}' for k in ks)
    rows = [
        ('$L$ of d8', 'our run, supplied run at LR 0.003, seed-1 rerun', f('dense') + ', 2.932, ' + f('dense-seed1')),
        ('Default ladder', 'd4, d6, d8, d10 ($W=64D$)', '3.272, 3.056, 2.932, 2.846'),
        ('$\\epsilon$ of width-only from d8', 'd8w1024 ($3.5\\times$ cost), d8w2048 ($13\\times$)', '0.74, 0.57'),
        ('$L$ at 34M parameters', 'd1w1280, d62w192, d8w512', '3.432, 3.017, 2.932'),
        ('$\\Delta L$ of a low-rank softmax head', 'rank 32, 64, 128, 256', d('softmax-r32', 'softmax-r64', 'softmax-r128', 'softmax-r256')),
        ('Suppressed gradient energy $E_r$', 'rank 32, 128, 512 (reference $1-r/4096$)', '0.994, 0.972, 0.878 (0.992, 0.969, 0.875)'),
        ('$\\Delta L$ of MoS, CPLM, Sigsoftmax', 'at rank 32, relative to the rank-32 softmax', ', '.join('--' if L(k) is None else f'{L(k)-L("softmax-r32"):+.3f}' for k in ('mos-r32', 'cplm-r32', 'sigsoftmax-r32'))),
        ('$\\Delta L$ of MoS, CPLM, Sigsoftmax', 'at full rank, relative to softmax', d('mos', 'cplm', 'sigsoftmax')),
        ('SpamLang loss at width 64', '$V=64, 256, 1024, 4096$, 800 updates', '$3\\times10^{-7}$, $8\\times10^{-6}$, $1\\times10^{-3}$, 0.32'),
        ('SpamLang loss at $V=4096$', '$W=64, 128, 256, 512$', '0.32, $5\\times10^{-3}$, $2\\times10^{-5}$, $9\\times10^{-7}$'),
        ('$\\Delta L$ of four-expert MoE', 'random, frozen, learned, unit combine, all four', d('moe-random', 'moe-frozen', 'moe-learned', 'moe-unit', 'moe-full')),
        ('$\\Delta L$ of router tricks', 'balance .001, .01, .1; noise .05; norm; shared init', d('moe-balanced001', 'moe-balanced', 'moe-balanced1', 'moe-noisy', 'moe-norm', 'moe-cloned')),
        ('Worst-layer load CV', 'learned, balance .01, quantile, unit, random', '1.73, 0.13, 0.17, 0.82, 0.08'),
        ('$\\Delta L$ of our routers', 'hash, straight-through, z-loss', d('moe-hash', 'moe-st', 'moe-zloss')),
        ('$\\Delta L$ of looping at 8 applications', '3, 2, 1 distinct core blocks', d('loop-r2', 'loop-r3', 'loop-r6')),
        ('$L$ of one block applied $R$ times', '$R=2, 3, 6, 12, 18$', f('loop-1block-r2', 'loop-1block-r3', 'loop-r6', 'loop-1block-r12', 'loop-1block-r18')),
        ('$L$ of boundary maps on the 6-pass loop', 'plain, norm, recall 1, recall .5, norm+recall', f('loop-r6', 'loop-boundary-norm', 'loop-recall', 'loop-recall05', 'loop-norm-recall')),
        ('Loss at $4\\times$ the trained passes', 'plain loop, norm+recall loop', '4.21, 3.03'),
        ('Switching at mid-training', 'unique$\\to$tied, tied$\\to$unique, tied reset, unique reset', f('unique-to-loop') + ', 2.991$^\\dagger$, ' + f('loop-reset', 'unique-reset')),
        ('$\\Delta L$ from a new seed', 'd8, learned top-1 MoE', ', '.join('--' if L(a) is None or L(b) is None else f'{L(a)-L(b):+.4f}' for a, b in (('dense-seed1', 'dense'), ('moe-learned-seed1', 'moe-learned')))),
    ]
    lines = [r'\begin{table}[H]\centering\small\caption{Reference values. $L$ is a validation loss, $\Delta L$ a difference from the dense d8 (2.927) unless stated.}\label{tab:ref}',
             r'\begin{tabular}{p{2.1in}p{2.45in}p{1.85in}}\toprule Quantity & Setting & Value \\ \midrule']
    lines += [f'{a} & {b} & {c} \\\\' for a, b, c in rows]
    lines += [r'\bottomrule\end{tabular}\end{table}']
    return '\n'.join(lines)


def main():
    runs, meas = load()
    tex, n = runs_table(runs, meas)
    open('experiments/a5_capacity/latex/runs_table.tex', 'w').write(tex)
    open('experiments/a5_capacity/latex/ref_table.tex', 'w').write(ref_table(runs))
    print('runs with a result:', n)
    rep = 'experiments/a5_capacity/latex/report.tex'
    s = open(rep).read().replace('%NRUNS%', str(n))
    open(rep, 'w').write(s)


if __name__ == '__main__':
    main()
