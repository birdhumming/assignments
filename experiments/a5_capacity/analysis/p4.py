"""P4 looping: ladders, signals per pass, inference loop-count sweeps, switching."""
import json
from .common import tag, load, run, final_loss, curve, fmt, save, plt

LADDER8 = [('dense', 8, 6), ('loop-r1', 8, 6), ('loop-r2', 8, 3), ('loop-r3', 8, 2), ('loop-r6', 8, 1)]  # (name, apps, distinct core blocks)
LADDER1 = [('loop-r6' if r == 6 else 'loop-1block-r%d' % r, r + 2, 1) for r in (1, 2, 3, 4, 6, 9, 12, 18)]
BOUNDARY = [('loop-r6', 'plain loop'), ('loop-boundary-norm', '+ boundary norm'), ('loop-recall', '+ recall 1.0'),
            ('loop-norm-recall', '+ norm + recall'), ('loop-recall05', '+ recall 0.5'),
            ('loop-1block-r18-norm-recall', '1 block, 18 passes, norm+recall')]
SWITCH = [('loop-to-unique', 'tied -> unique at 4688'), ('unique-to-loop', 'unique -> tied at 4688'),
          ('loop-reset', 'tied, Adam reset at 4688'), ('unique-reset', 'unique, Adam reset at 4688'),
          ('loop-r6', 'tied throughout'), ('dense', 'unique throughout')]


def main():
    runs, meas = load()
    M = lambda k: meas.get(f'a5-{k}-d8-w512-lr0.003', {})
    rows = []
    for name, apps, blocks in LADDER8 + LADDER1:
        r = run(runs, name)
        rows.append(dict(name=tag(name, r), apps=apps, blocks=blocks, loss=final_loss(r), state=(r or {}).get('state', 'missing'),
                         params=M(name).get('parameters'), runtime=(r or {}).get('runtime')))
        print(f"{name:16s} apps {apps:2d} blocks {blocks} {rows[-1]['state']:9s} loss {fmt(rows[-1]['loss'])} params {rows[-1]['params']} s {rows[-1]['runtime']}")
    lines = [r'\begin{table}[H]\centering\small\begin{tabular}{lrrrrr}\toprule',
             r'Model & block apps & distinct core blocks & params (M) & val loss & wall-clock (min) \\ \midrule']
    for r in rows:
        p = '--' if r['params'] is None else f"{r['params']/1e6:.1f}"
        t = '--' if r['runtime'] is None else f"{r['runtime']/60:.0f}"
        lines.append(f"{r['name']} & {r['apps']} & {r['blocks']} & {p} & {fmt(r['loss'])} & {t} \\\\")
    lines += [r'\bottomrule\end{tabular}', r'\caption{P4(a--b,d): looping ladders at d8w512, LR .003. Prelude and coda are always one block each; the 8-application ladder keeps compute fixed, the 1-block ladder keeps parameters fixed.}\label{tab:p4ladder}\end{table}']
    brows = []
    for name, label in BOUNDARY:
        r = run(runs, name); brows.append((tag(label, r), final_loss(r)))
    lines += [r'\begin{table}[H]\centering\small\begin{tabular}{lr}\toprule Variant (6 passes, 1 core block unless noted) & val loss \\ \midrule']
    lines += [f'{l} & {fmt(v)} \\\\' for l, v in brows]
    lines += [r'\bottomrule\end{tabular}\caption{P4(c): boundary normalization and prelude injection (recall).}\label{tab:p4boundary}\end{table}']
    srows = []
    for name, label in SWITCH:
        r = run(runs, name) or {}
        final = final_loss(r)
        if final is None and M(name).get('nll') is not None:  # W&B log lost; checkpoint diagnostic NLL instead
            final, label = M(name)['nll'], label + r'$^\dagger$'
        srows.append((label, r.get('switch_before'), r.get('switch_after'), final))
    lines += [r'\begin{table}[H]\centering\small\begin{tabular}{lrrr}\toprule Schedule & before switch & after switch & final \\ \midrule']
    lines += [f'{l} & {fmt(b)} & {fmt(a)} & {fmt(f)} \\\\' for l, b, a, f in srows]
    lines += [r'\bottomrule\end{tabular}\caption{P4(e): changing sharing mid-training (update 4,688 of 9,375) with matched Adam-reset controls. $^\dagger$: W\&B log lost; final column is the checkpoint held-out NLL on the diagnostic set (dense model 2.935, tied loop 3.034 on that set). The unique Adam-reset value is the W\&B summary at pull time; that record was later overwritten by a run-id collision and its checkpoint was not retained, so it could not be re-verified.}\label{tab:p4switch}\end{table}']
    open('experiments/a5_capacity/latex/p4_tables.tex', 'w').write('\n'.join(lines))

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for name, apps, blocks in LADDER8:
        s, l = curve(run(runs, name))
        if s: axes[0].plot(s, l, label=f'{name} ({blocks} core blocks)')
    axes[0].set(title='8 block applications, fewer distinct blocks', xlabel='update', ylabel='val loss', ylim=(2.85, 3.6)); axes[0].legend(fontsize=7)
    xs = [(r['apps'], r['loss']) for r in rows if r['blocks'] == 1 and r['loss'] is not None and r['name'].startswith('loop-1block')]
    if xs:
        axes[1].plot(*zip(*sorted(xs)), 'o-', label='1 core block, R passes')
    d = next((r['loss'] for r in rows if r['name'] == 'dense'), None)
    if d: axes[1].axhline(d, color='k', ls='--', lw=1, label='dense d8')
    axes[1].set(title='One shared core block: loss vs executed depth', xlabel='block applications (R+2)', ylabel='val loss'); axes[1].legend(fontsize=7)
    for name, label in BOUNDARY:
        s, l = curve(run(runs, name))
        if s: axes[2].plot(s, l, label=label, lw=1)
    axes[2].set(title='Boundary map variants', xlabel='update', ylabel='val loss', ylim=(2.85, 3.6)); axes[2].legend(fontsize=7)
    save(fig, 'p4_ladders.png')

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    for name, label in BOUNDARY[:4] + [('dense', 'dense (per block)')]:
        sig = M(name).get('signals')
        if not sig: continue
        entries = sig if isinstance(sig, list) else sig.get('passes', sig)
        try:
            act = [e['activation_rms'] for e in entries]; grad = [e['gradient_rms'] for e in entries]; upd = [e['relative_update'] for e in entries]
        except (TypeError, KeyError):
            print('signal format', name, str(sig)[:200]); continue
        x = range(len(act))
        axes[0].plot(x, act, 'o-', label=label, ms=3); axes[1].plot(x, grad, 'o-', label=label, ms=3); axes[2].plot(x, upd, 'o-', label=label, ms=3)
    axes[0].set(title='activation RMS', xlabel='block application', yscale='log'); axes[1].set(title='state-gradient RMS', xlabel='block application', yscale='log'); axes[2].set(title='relative update per application', xlabel='block application', yscale='log')
    axes[0].legend(fontsize=7)
    save(fig, 'p4_signals.png')

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for name in ['loop-r6', 'loop-r3', 'loop-norm-recall', 'loop-1block-r4', 'loop-1block-r6', 'loop-1block-r18', 'loop-1block-r18-norm-recall']:
        sw = M(name).get('loop_sweep')
        if not sw: continue
        ps = sorted(int(k) for k in sw); ax.plot(ps, [sw[str(p)] for p in ps], 'o-', ms=3, label=f"{name} (trained {M(name).get('trained_passes')})")
    ax.set(xlabel='inference loop count', ylabel='held-out NLL (64 seq x 512 tok)', ylim=(2.8, 7), title='Evaluating at unseen loop counts'); ax.legend(fontsize=7)
    save(fig, 'p4_eval_passes.png')

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for name, label in SWITCH:
        s, l = curve(run(runs, name))
        if s: ax.plot(s, l, label=label, lw=1)
    ax.axvline(4688, color='gray', ls=':', lw=1); ax.set(xlabel='update', ylabel='val loss', ylim=(2.85, 3.5), title='Switching sharing mid-training'); ax.legend(fontsize=7)
    save(fig, 'p4_switch.png')
    json.dump(dict(ladder=rows, boundary=brows, switch=srows), open('results/a5_capacity/p4_summary.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
