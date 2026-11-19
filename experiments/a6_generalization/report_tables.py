"""Turn results/a6_generalization/runs.json into LaTeX fragments and P4 plots for the report."""
import json, pathlib, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from .analysis import power_law, predict
T = pathlib.Path('experiments/a6_generalization/latex'); P = pathlib.Path('plots/a6_generalization')
RUNS = json.loads(pathlib.Path('results/a6_generalization/runs.json').read_text())
BASE = 1.2627  # supplied d8 32-pass baseline
def L(name):
    r = RUNS.get(name); return r['final'] if r and r.get('final') is not None else None
def fmt(x, d=4): return '--' if x is None else f'{x:.{d}f}'
def delta(x): return '--' if x is None else f'{x - BASE:+.4f}'
def table(caption, label, header, rows, spec):
    return (r'\begin{table}[H]\centering\small\caption{' + caption + r'}\label{' + label + '}' + '\n' + r'\begin{tabular}{' + spec + r'}\toprule ' + header + r' \\ \midrule' + '\n'
            + '\n'.join(r + r' \\' for r in rows) + '\n' + r'\bottomrule\end{tabular}\end{table}' + '\n')
def state(name):
    r = RUNS.get(name); 
    if not r: return 'not started'
    if r.get('final') is not None: return 'finished'
    last = r.get('last') or {}; return f"running ({last.get('epochs', 0):.1f} passes, {last.get('validation_nll', float('nan')):.4f})"

def p1():
    rows = []
    for arch, lab in (('moe', 'MoE, 4 experts, learned top-1, quantile balance'), ('loop', 'Loop, prelude + 6 passes of a shared block + coda, RMS + recall')):
        ls = {lr: L(f'{arch}-lr{lr:g}') for lr in (.001, .003, .01)}
        best = min((v for v in ls.values() if v is not None), default=None)
        rows.append(f"{lab} & {fmt(ls[.001])} & {fmt(ls[.003])} & {fmt(ls[.01])} & {fmt(best)} & {delta(best)}")
    rows.append(f"Dense d8 (supplied) & -- & {BASE:.4f} & -- & {BASE:.4f} & 0")
    (T / 'p1_ours.tex').write_text(r'\subsection*{Parts (b, c). MoE and looping from Assignment 5}' + '\n\n' + r'\input{p1_ours_text}' + '\n\n' +
        table('Problems 1(b,c): A5 architectures retrained on the 30k pool for 32 passes at WD 3.2, three LRs, seed 42, data order paired with the supplied baseline. Unfinished cells are marked.', 'tab:p1ours',
              r'Architecture & LR .001 & LR .003 & LR .01 & best & $\Delta L$ vs d8', rows, 'lrrrrr'))
    if not (T / 'p1_ours_text.tex').exists(): (T / 'p1_ours_text.tex').write_text('% written by hand\n')

def p2():
    hb = {lr: L(f'hyperball-lr{lr:g}') for lr in (.001, .003, .01)}
    rows = [f"AdamW WD .1 (supplied) & -- & 1.4057 & {1.4057 - BASE:+.4f}", f"AdamW WD 3.2 (supplied) & -- & {BASE:.4f} & 0"] + \
           [f"Hyperball WD 0 & {lr:g} & {fmt(v)} & {delta(v)}" for lr, v in hb.items()]
    (T / 'p2_hyperball.tex').write_text('\n' + table('Problem 2(a)(ii): Hyperball (Frobenius-norm projection of every Linear weight, plain Adam elsewhere, WD 0) against the supplied AdamW controls, 32 passes.', 'tab:hb',
              r'Optimizer & LR & $L$ & $\Delta L$ vs WD 3.2', rows, 'llrr') + r'\input{p2_hyperball_text}' + '\n')
    dr = {d: L(f'dropout{d:g}') for d in (.05, .15, .25)}
    rows = [f"0 (supplied baseline) & {BASE:.4f} & 0", f"0 (our seed 43) & {fmt(L('baseline-seed43'))} & {delta(L('baseline-seed43'))}"] + [f"{d:g} & {fmt(v)} & {delta(v)}" for d, v in dr.items()] + \
           [f".15, LR .01 & {fmt(L('dropout0.15-lr0.01'))} & {delta(L('dropout0.15-lr0.01'))}", f".15, seed 43 & {fmt(L('dropout0.15-seed43'))} & {delta(L('dropout0.15-seed43'))}", f".15, seed 44 & {fmt(L('dropout0.15-seed44'))} & {delta(L('dropout0.15-seed44'))}"]
    (T / 'p2_dropout.tex').write_text(r'\lead{(i) Dropout alone.}' + ' ' + r'\input{p2_dropout_text}' + '\n' + table('Problem 2(b)(i): attention-probability dropout at the baseline recipe (LR .003, WD 3.2, 32 passes), dropout off at evaluation.', 'tab:drop', r'Attention dropout & $L$ & $\Delta L$', rows, 'lrr'))
    for f in ('p2_hyperball_text', 'p2_dropout_text', 'p2_composition', 'p2_synthesis', 'p3_synthesis'):
        if not (T / f'{f}.tex').exists(): (T / f'{f}.tex').write_text('% written by hand\n')

def p3():
    rows = [f"AdamW (supplied baseline) & {BASE:.4f} & 0", f"AdamW seed 43 & {fmt(L('baseline-seed43'))} & {delta(L('baseline-seed43'))}", f"SAM $\\rho=.05$ on AdamW & {fmt(L('sam-rho0.05'))} & {delta(L('sam-rho0.05'))}"]
    (T / 'p3_ours.tex').write_text(r'\subsection*{Part (c). SAM against AdamW}' + '\n\n' + r'\input{p3_ours_text}' + '\n' + table('Problem 3(c): SAM (one ascent step of radius $\\rho$ per update, same LR, WD and schedule) against AdamW, 32 passes, seed 42.', 'tab:sam', r'Optimizer & $L$ & $\Delta L$', rows, 'lrr'))
    if not (T / 'p3_ours_text.tex').exists(): (T / 'p3_ours_text.tex').write_text('% written by hand\n')

INITS = ('scratch', 'related', 'dclm', 'noise'); LABEL = dict(scratch='scratch', related='related DNA', dclm='DCLM text', noise='nucleotide noise')
def ft(init, passes, lr, n=30, wd=None, adapt=None):
    name = f"ft-{init}-n{n}k-p{passes}-lr{lr:g}"
    if wd is not None: name += f"-wd{float(wd)}"
    if adapt: name += f"-adaptation{adapt}"
    return L(name)
LRS = {'scratch': ((.003, None), (.001, None)), 'related': ((.003, None), (.001, None), (3e-4, 0.), (1e-4, 0.), (1e-5, 0.))}
LRS['dclm'] = LRS['noise'] = LRS['related']
def best_ft(init, passes, n=30):
    cands = {(lr, wd): ft(init, passes, lr, n, wd) for lr, wd in LRS[init]}
    cands = {k: v for k, v in cands.items() if v is not None}
    if not cands: return None, None
    k = min(cands, key=cands.get); return cands[k], k

def p4():
    # (a) passes table and plot
    rows = []; fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    for init in INITS:
        cells = []; ys = []
        for p in (1, 4, 16):
            v, k = best_ft(init, p); ys.append(v)
            cells.append('--' if v is None else f"{v:.4f} ({k[0]:g}{'' if k[1] is None else '/wd0'})")
        rows.append(f"{LABEL[init]} & " + ' & '.join(cells))
        if any(y is not None for y in ys): ax[0].plot([p for p, y in zip((1, 4, 16), ys) if y is not None], [y for y in ys if y is not None], 'o-', label=LABEL[init])
    ax[0].axhline(BASE, color='gray', ls=':', label='scratch, 32 passes (supplied)'); ax[0].set(xscale='log', xlabel='passes over 30k chunks', ylabel='validation NLL', title='4(a): best LR per initialization and pass count'); ax[0].legend(fontsize=7)
    # (b) sample scaling at 16 passes
    fits = {}
    for init in INITS:
        ns, ys = [], []
        for n in (1, 3, 10, 30):
            v, k = best_ft(init, 16, n)
            if v is not None: ns.append(n * 1000); ys.append(v)
        if len(ns) >= 2: ax[1].plot(ns, ys, 'o-', label=LABEL[init])
        if len(ns) >= 3:
            try: fits[init] = power_law(np.array(ns), np.array(ys))
            except Exception: pass
    ax[1].set(xscale='log', xlabel='fine-tuning chunks', ylabel='validation NLL', title='4(b): 16 passes (related 1k = 3.19, off scale)', ylim=(1.24, 1.42)); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(P / 'p4.png', dpi=160)
    brows = []
    for init in INITS:
        cells = [fmt(best_ft(init, 16, n)[0]) for n in (1, 3, 10, 30)]
        a, b = best_ft(init, 16, 10)[0], best_ft(init, 16, 30)[0]
        brows.append(f"{LABEL[init]} & " + ' & '.join(cells) + (f" & {a - b:.4f}" if a is not None and b is not None else ' & --'))
    # (c) adaptation on related
    crows = []
    for lab, lr, wd, adapt in (('full, LR 3e-4, WD 0', 3e-4, 0., None), ('full, LR 1e-4, WD 0', 1e-4, 0., None), ('full, LR .001, WD 3.2', .001, None, None), ('full, LR .003, WD 3.2', .003, None, None), ('full, LR .003, WD .1', .003, .1, None),
                               ('decay toward init, LR 3e-4 (cut at pass 2: 1.2309)', 3e-4, 0., 'toward_initialization'), ('decay toward init, LR .003', .003, None, 'toward_initialization'),
                               ('frozen head, LR .003', .003, None, 'frozen_head'), ('frozen body, LR .003', .003, None, 'frozen_body')):
        v = ft('related', 16, lr, 30, wd, adapt); crows.append(f"{lab} & {fmt(v)}")
    (T / 'p4.tex').write_text(r'\input{p4_text_a}' + '\n' + r'\begin{figure}[H]\centering\includegraphics[width=.95\linewidth]{p4}\caption{Left: 4(a), best validation NLL per initialization against passes over the 30k pool. Right: 4(b), 16 passes against fine-tuning pool size, log axis.}\label{fig:p4}\end{figure}' + '\n'
        + table('Problem 4(a): best validation NLL per initialization and pass count on 30k chunks; parentheses give the selected LR (and WD 0 where used). LRs tried: .003/WD 3.2 and .001/WD 3.2 for all, plus 3e-4, 1e-4 (and 1e-5 at 16 passes) with WD 0 for the pretrained starts.', 'tab:p4a', r'Initialization & 1 pass & 4 passes & 16 passes', rows, 'lrrr')
        + r'\input{p4_text_b}' + '\n' + table('Problem 4(b): 16 passes over pools of 1k to 30k chunks, best LR per point; last column is the gain from tripling the pool from 10k to 30k.', 'tab:p4b', r'Initialization & 1k & 3k & 10k & 30k & gain 10k$\to$30k', brows, 'lrrrrr')
        + r'\input{p4_text_c}' + '\n' + table('Problem 4(c): adapting the related-DNA initialization, 16 passes on 30k chunks. The LR 3e-4 adaptation variants were queued last and cut.', 'tab:p4c', r'Variant & $L$', crows, 'lr') + r'\input{p4_text_d}' + '\n')
    for f in ('p4_text_a', 'p4_text_b', 'p4_text_c', 'p4_text_d'):
        if not (T / f'{f}.tex').exists(): (T / f'{f}.tex').write_text('% written by hand\n')

def runs_table():
    rows = []
    for n in sorted(RUNS):
        r = RUNS[n]; s = state(n); esc = n.replace('_', '\\_'); rows.append(f"\\texttt{{{esc}}} & {fmt(r.get('final'))} & {s}")
    (T / 'runs_table.tex').write_text(r'\small Names encode the recipe: \texttt{ft-<init>-n<pool>k-p<passes>-lr<LR>[-wd<WD>][-adaptation<variant>]} for fine-tuning; everything else is a 32-pass scratch run of d8 at WD 3.2 unless the name says otherwise. All seeds are 42 (model and data order) unless named.' + '\n\n' +
        r'\begin{longtable}{lrl}\toprule Run & $L$ & state \\ \midrule' + '\n' + '\n'.join(r + r' \\' for r in rows) + '\n' + r'\bottomrule\end{longtable}' + '\n')
    rows = [f"Dense d8, 32 passes (supplied) & {BASE:.4f}", f"Supplied suite best, d9 & 1.2594", f"5-member ensemble (supplied) & 1.2543", f"AdamW WD .1, 32 passes (supplied) & 1.4057"]
    for lab, n in (('Hyperball best', None), ('Dropout .15', 'dropout0.15'), ('SAM', 'sam-rho0.05'), ('MoE LR .003', 'moe-lr0.003'), ('Loop LR .003', 'loop-lr0.003')):
        v = max((L(f'hyperball-lr{lr:g}') or 9 for lr in (.001, .003, .01)), default=None) if n is None else L(n)
        if n is None: v = min((L(f'hyperball-lr{lr:g}') for lr in (.001, .003, .01) if L(f'hyperball-lr{lr:g}') is not None), default=None)
        rows.append(f"{lab} & {fmt(v)}")
    for init in INITS:
        v, k = best_ft(init, 16); rows.append(f"Fine-tune from {LABEL[init]}, 16 passes, 30k & {fmt(v)}")
    (T / 'ref_table.tex').write_text(table('Validation NLL of the main configurations in this report.', 'tab:ref', r'Configuration & $L$', rows, 'lr'))

if __name__ == '__main__':
    p1(); p2(); p3(); p4(); runs_table(); print('ok', len(RUNS), 'runs,', sum(1 for r in RUNS.values() if r.get('final') is not None), 'finished')
