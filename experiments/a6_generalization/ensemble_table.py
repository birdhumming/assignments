"""Write latex/p2_composition.tex from results/a6_generalization/ensembles.json (P2(b)(iii))."""
import json, pathlib
T = pathlib.Path('experiments/a6_generalization/latex'); f = pathlib.Path('results/a6_generalization/ensembles.json')
if not f.exists():
    (T / 'p2_composition.tex').write_text(r'\lead{(iii) Composition.} The dropout-member and mixed ensembles were evaluated on Modal CPU from the saved checkpoints; that job had not returned at the time of writing, so this part is reported without numbers.' + '\n'); raise SystemExit
R = json.load(open(f))
def row(lab, d): return lab + ' & ' + ' & '.join(f"{d[str(k)]:.4f}" if str(k) in d else '--' for k in range(1, 6))
rows = [row('no dropout, supplied bank (seeds)', R['no_dropout']),
        row('dropout members .05, .15, .25', R['dropout_members']),
        row('dropout .05 then bank members', R['mixed_first_dropout']),
        row('all three dropout then bank', R['mixed_all_dropout_then_bank'])]
pairs = [f"{k.replace('pair_', '').replace('_', ' + ').replace('bank0', 'bank 1').replace('bank1', 'bank 2')} & {v['2']:.4f}" for k, v in R.items() if k.startswith('pair_')]
tex = (r'\lead{(iii) Composition.} \input{p2_composition_text}' + '\n'
       + r'\begin{table}[H]\centering\small\caption{Problem 2(b)(iii): validation NLL of probability-averaged ensembles of $K$ members, by composition. Single dropout members: '
       + ', '.join(f"{n.replace('dropout', '')}: {v:.4f}" for n, v in R['single'].items()) + r'.}\label{tab:comp}' + '\n'
       + r'\begin{tabular}{lrrrrr}\toprule Members in order & $K=1$ & 2 & 3 & 4 & 5 \\ \midrule' + '\n' + '\n'.join(r + r' \\' for r in rows)
       + '\n' + r'\midrule \multicolumn{6}{l}{Pairs ($K=2$)} \\' + '\n' + '\n'.join(p + r' \\' for p in pairs) + '\n' + r'\bottomrule\end{tabular}\end{table}' + '\n')
(T / 'p2_composition.tex').write_text(tex)
if not (T / 'p2_composition_text.tex').exists(): (T / 'p2_composition_text.tex').write_text('% by hand\n')
print('ok')
