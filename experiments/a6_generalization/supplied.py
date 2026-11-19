"""Supplied-data analysis for A6: P1(a) parameter scaling, P2(a) WD controls, P2(b)(ii) ensembles, P3(a) schedules, P3(b) batch."""
import json, gzip, pathlib, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from .analysis import power_law, predict, chunk_bootstrap
from . import p2_regularization as p2mod
R = pathlib.Path('experiments/a6_generalization/references'); P = pathlib.Path('plots/a6_generalization'); P.mkdir(parents=True, exist_ok=True)
T = pathlib.Path('experiments/a6_generalization/latex'); T.mkdir(exist_ok=True)
J = lambda f: json.loads((R / f).read_text())
out = {}

def p1():
    runs = J('parameter_scaling.json')['runs']
    N = np.array([r['parameters'] for r in runs]); L = np.array([r['validation_nll'] for r in runs])
    fit = power_law(N, L); out['p1_fit'] = fit
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    ax[0].plot(N, L, 'o'); xs = np.geomspace(N.min(), N.max() * 2, 100); ax[0].plot(xs, predict(fit, xs), '--', label=f"fit: {fit['offset']:.3f} + {fit['amplitude']:.3g} N^-{fit['exponent']:.2f}")
    for r in runs: ax[0].annotate(f"d{r['depth']} ({r['epochs']}ep)", (r['parameters'], r['validation_nll']), fontsize=7, xytext=(3, 3), textcoords='offset points')
    ax[0].set(xscale='log', xlabel='parameters', ylabel='validation NLL', title='Tuned dense suite, 30k chunks'); ax[0].legend(fontsize=8)
    ax[1].plot(N, [r['training_flops'] for r in runs], 'o-'); ax[1].set(xscale='log', yscale='log', xlabel='parameters', ylabel='training FLOPs', title='Selected compute')
    fig.tight_layout(); fig.savefig(P / 'p1_parameters.png', dpi=160)
    rows = [f"d{r['depth']} & {r['width']} & {r['parameters']/1e6:.1f} & {r['lr']:g} & {r['wd']:g} & {r['epochs']} & {r['training_flops']/1e15:.0f} & {r['validation_nll']:.4f} \\\\" for r in runs]
    (T / 'p1_table.tex').write_text(r'\begin{table}[H]\centering\small\caption{Problem 1(a): the supplied tuned dense suite (seed 42, linear decay). LR, WD and epochs were selected on validation data per size.}\label{tab:p1}' + '\n' +
        r'\begin{tabular}{lrrrrrrr}\toprule Model & width & params (M) & LR & WD & epochs & PFLOP & val NLL \\ \midrule' + '\n' + '\n'.join(rows) + '\n' + r'\bottomrule\end{tabular}\end{table}' + '\n')
    ep = J('epoch_scaling.json')['runs']; out['epochs'] = [(r['epochs'], r['lr'], r['wd'], r['validation_nll']) for r in ep]

def p2():
    wd = J('wd_tuning.json')['runs']; out['wd_controls'] = {r['wd']: r['validation_nll'] for r in wd}
    bank = p2mod.ensemble_curve(); curve = bank['ensemble_validation_nll']
    L = np.array(curve, dtype=float); K = np.arange(1, len(L) + 1)
    fit = power_law(K, L); out['ensemble'] = dict(K=K.tolist(), L=L.tolist(), fit=fit)
    fits = {}
    for lo, hi in ((1, 5), (2, 5), (1, 4)):
        m = (K >= lo) & (K <= hi); fits[f'{lo}-{hi}'] = power_law(K[m], L[m]) if m.sum() >= 3 else None
    out['ensemble_fit_ranges'] = fits
    chunks = bank.get('chunk_losses')
    if chunks is not None:
        arr = np.array(chunks)  # [K, chunks]
        rng = np.random.default_rng(0); offs = []
        for _ in range(200):
            idx = rng.integers(0, arr.shape[1], arr.shape[1]); Lb = arr[:, idx].mean(1)
            try: offs.append(power_law(K[:len(Lb)], Lb)['offset'])
            except Exception: pass
        out['ensemble_offset_ci'] = [float(np.percentile(offs, 2.5)), float(np.percentile(offs, 97.5))] if offs else None
    fig, ax = plt.subplots(figsize=(5, 3.4)); ax.plot(K, L, 'o'); xs = np.linspace(1, 8, 50); ax.plot(xs, predict(fit, xs), '--', label=f"{fit['offset']:.4f} + {fit['amplitude']:.3g} K^-{fit['exponent']:.2f}")
    ax.set(xlabel='ensemble size K', ylabel='validation NLL', title='No-dropout ensembles (supplied members)'); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(P / 'p2_ensemble.png', dpi=160)

def p3():
    s = J('schedules.json')['runs']; out['schedules'] = {k: sorted((r['epochs'], r['lr'], r['validation_nll']) for r in v) for k, v in s.items()}
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    for k, v in s.items():
        v = sorted(v, key=lambda r: r['epochs']); ax[0].plot([r['epochs'] for r in v], [r['validation_nll'] for r in v], 'o-', label=k)
        for r in v:
            h = r.get('history') or []
            if h and r['epochs'] in (32, 256): ax[1].plot([p.get('epochs', p.get('step')) for p in h], [p.get('validation_nll', p.get('val_nll')) for p in h], label=f"{k} {r['epochs']}ep", alpha=.8)
    ax[0].set(xscale='log', xlabel='epochs (horizon)', ylabel='final validation NLL', title='Schedule vs horizon'); ax[0].legend(fontsize=8)
    ax[1].set(xlabel='epochs', ylabel='validation NLL', ylim=(1.24, 1.40), title='Loss over training'); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(P / 'p3_schedules.png', dpi=160)
    b = J('batch_scaling.json')['runs']; out['batch'] = [(r['batch'], r['lr'], r['wd'], r['validation_nll'], ((r.get('sharpness') or {}).get('directional') or {}).get('0.05', {}).get('symmetric_curvature')) for r in b]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4)); B = [r['batch'] for r in b]
    ax[0].plot(B, [r['validation_nll'] for r in b], 'o-'); ax[0].set(xscale='log', xlabel='batch size', ylabel='final validation NLL', title='Batch size (sqrt LR/WD scaling, 32 epochs)')
    sh = [((r.get('sharpness') or {}).get('directional') or {}).get('0.05', {}).get('symmetric_curvature') for r in b]
    if any(x is not None for x in sh):
        B2 = [bb for bb, x in zip(B, sh) if x is not None]; sh = [x for x in sh if x is not None]; B = B2
        ax[1].plot(B, sh, 'o-'); ax[1].set(xscale='log', xlabel='batch size', ylabel='directional sharpness', title='Supplied sharpness probe')
    fig.tight_layout(); fig.savefig(P / 'p3_batch.png', dpi=160)

if __name__ == '__main__':
    p1(); p2(); p3()
    pathlib.Path('results/a6_generalization/supplied.json').write_text(json.dumps(out, indent=1, default=str) + '\n')
    print(json.dumps(out, indent=0, default=str)[:3000])
