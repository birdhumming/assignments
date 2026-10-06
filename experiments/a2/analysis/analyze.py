"""A2 analysis of our W&B runs (after pull_wandb.py): tables for P1(c,d), P2(c), P3.2, P4.2.

Writes markdown tables to $A2_DATA_DIR/tables/*.md and figures to experiments/a2/figures/.
"""
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiments.a2.analysis.sources import quad_opt

DATA = Path(os.environ.get('A2_DATA_DIR', Path.home() / 'a2'))
TABLES = DATA / 'tables'
FIG = Path(__file__).resolve().parents[1] / 'figures'
PROVIDED = Path(__file__).resolve().parents[3] / 'worksheets/hparam_invariants/data/provided_sweeps.csv'


def load_runs():
    runs = []
    for f in sorted((DATA / 'runs').glob('*.json')):
        r = json.loads(f.read_text())
        c = r['config']
        mc = c.get('model_config') or {}
        runs.append({
            'name': r['name'], 'id': r['id'], 'tags': set(r['tags']), 'state': r['state'],
            'lr': c.get('learning_rate'), 'wd': c.get('weight_decay'), 'batch': c.get('batch_size'), 'beta1': c.get('beta1'),
            'tokens': (c.get('num_train_sequences') or 0) * 1024, 'optimizer': c.get('optimizer_name'),
            'width': mc.get('hidden_size'), 'depth': mc.get('num_hidden_layers'),
            'policy': ((c.get('model_builder_kwargs') or {}).get('policy')),
            'final': r['summary'].get('val_loss'), 'runtime_min': (r['summary'].get('_runtime') or 0) / 60,
            'history': r['history']})
    return runs


def provided():
    with open(PROVIDED) as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in ('tokens', 'learning_rate', 'weight_decay', 'final_val_loss', 'batch_size', 'beta1'):
            r[k] = float(r[k])
    return rows


def with_tag(runs, tag):
    return [r for r in runs if tag in r['tags'] and r['state'] == 'finished' and r['final'] is not None]


def fit_lr_curve(pairs):
    """pairs: [(lr, loss)] -> (lr*, L*, best sampled lr, best sampled loss)."""
    pairs = sorted(pairs)
    i = int(np.argmin([p[1] for p in pairs]))
    lo, hi = max(0, i - 2), min(len(pairs), i + 3)
    sub = pairs[lo:hi]
    if len(sub) >= 3:
        x, L, a = quad_opt(sub)
        if a > 0 and min(p[0] for p in sub) <= x <= max(p[0] for p in sub):
            return x, L, pairs[i][0], pairs[i][1]
    return pairs[i][0], pairs[i][1], pairs[i][0], pairs[i][1]


def md(rows, cols, fmt='{:.4g}'):
    out = '| ' + ' | '.join(cols) + ' |\n|' + '---|' * len(cols) + '\n'
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c)
            cells.append('' if v is None else (fmt.format(v) if isinstance(v, float) else str(v)))
        out += '| ' + ' | '.join(cells) + ' |\n'
    return out


def lr_table(runs, extra=()):
    rows = sorted(runs, key=lambda r: r['lr'])
    return md([dict(lr=r['lr'], **{k: r[k] for k in extra}, val_loss=r['final'], minutes=r['runtime_min']) for r in rows],
              ['lr', *extra, 'val_loss', 'minutes'])


def p1c(runs, out):
    rs = with_tag(runs, 'a2-p1c')
    out.append('# P1(c): 4.9152B-token AdamW target\n\nPredictions recorded before launch: all-six fit 0.00373, larger-three fit 0.00223; '
               'best sampled 2.4576B LR 0.003, next 0.0015 and 0.006.\n\n' + lr_table(rs))
    if len(rs) >= 3:
        x, L, bx, bL = fit_lr_curve([(r['lr'], r['final']) for r in rs])
        out.append(f'Fitted optimum: lr* = {x:.4g}, loss* = {L:.4f}; best sampled lr {bx:g} ({bL:.4f}).\n')
        for lab, p in (('all-six', .00373), ('larger-three', .00223), ('transferred 2.46B best', .003)):
            m = [r for r in rs if abs(r['lr'] - p) < 1e-9]
            if m:
                out.append(f'- {lab} prediction {p:g}: loss {m[0]["final"]:.4f}, gap to best sampled {m[0]["final"] - bL:+.4f}\n')
    plot_lr(rs, 'P1(c) 4.9B tokens', 'p1c_lr.png')


def p1d(runs, out):
    rs = with_tag(runs, 'a2-p1d')
    out.append('\n# P1(d): Hyperball at 1.2288B tokens\n\nPredicted optimum 0.00839 from the supplied Hyperball sweeps.\n\n' + lr_table(rs))
    if len(rs) >= 3:
        x, L, bx, bL = fit_lr_curve([(r['lr'], r['final']) for r in rs])
        out.append(f'Fitted optimum: lr* = {x:.4g}, loss* = {L:.4f}; best sampled {bx:g} ({bL:.4f}).\n')
    adamw = [r for r in provided() if r['parts'] == 'P1a' and r['tokens'] == 1228800000]
    if adamw:
        best = min(adamw, key=lambda r: r['final_val_loss'])
        out.append(f"AdamW supplied best at 1.2288B: lr {best['learning_rate']:g}, loss {best['final_val_loss']:.4f}.\n")
    plot_lr(rs, 'P1(d) Hyperball 1.23B tokens', 'p1d_lr.png')


def p2c(runs, out):
    rs = with_tag(runs, 'a2-p2c')
    out.append('\n# P2(c): LR-WD at 2.4576B tokens\n\nProduct-law prediction: lr*wd = 2.78e-4 so wd 0.0927 at lr 0.003; untuned baseline (153.6M best) lr 0.0015 wd 1.6.\n\n')
    rows = sorted(rs, key=lambda r: (r['lr'], r['wd']))
    out.append(md([{'lr': r['lr'], 'wd': r['wd'], 'product': r['lr'] * r['wd'], 'val_loss': r['final'], 'minutes': r['runtime_min']} for r in rows],
                  ['lr', 'wd', 'product', 'val_loss', 'minutes']))
    sweep = [(r['wd'], r['final']) for r in rs if abs(r['lr'] - .003) < 1e-9]
    if len(sweep) >= 3:
        x, L, a = quad_opt(sorted(sweep))
        out.append(f'Quadratic in log wd at lr 0.003: wd* = {x:.4g}, loss* = {L:.4f} (curvature {a:.3g}).\n')


def p32(runs, out):
    rs = with_tag(runs, 'a2-p32')
    prov = [r for r in provided() if r['tokens'] == 614400000 and r['batch_size'] == 64 and r['optimizer'] == 'adamw'
            and set(r['parts'].split('|')) & {'P1a', 'P2a'}]
    rows = [{'batch': r['batch'], 'lr': r['lr'], 'wd': r['wd'], 'beta1': r['beta1'], 'val_loss': r['final'], 'minutes': r['runtime_min'],
                 'src': 'ours'} for r in rs]
    rows += [{'batch': 64, 'lr': r['learning_rate'], 'wd': r['weight_decay'], 'beta1': r['beta1'], 'val_loss': r['final_val_loss'],
                  'minutes': None, 'src': 'supplied'} for r in prov]
    rows.sort(key=lambda r: (r['batch'], r['wd'], r['lr'], r['beta1']))
    out.append('\n# P3.2: batch-size sources and targets (614.4M tokens)\n\n' + md(rows, ['batch', 'lr', 'wd', 'beta1', 'val_loss', 'minutes', 'src']))
    # (a) LR optimum per batch at wd .1, beta1 .9
    res = lr_res = []
    for B in sorted({r['batch'] for r in rows}):
        pairs = [(r['lr'], r['val_loss']) for r in rows if r['batch'] == B and abs(r['wd'] - .1) < 1e-9 and abs(r['beta1'] - .9) < 1e-9]
        if len(pairs) >= 3:
            x, L, bx, bL = fit_lr_curve(pairs)
            res.append({'batch': B, 'n': len(pairs), 'lr_star': x, 'loss_star': L, 'best_lr': bx, 'best_loss': bL})
    if res:
        out.append('\n## (a) optimal LR vs batch size at wd 0.1\n\n' + md(res, ['batch', 'n', 'lr_star', 'loss_star', 'best_lr', 'best_loss']))
        Bs = np.array([r['batch'] for r in res]); ls = np.array([r['lr_star'] for r in res])
        src = Bs <= 64
        if src.sum() >= 2:
            p, lc = np.polyfit(np.log(Bs[src]), np.log(ls[src]), 1)
            out.append(f'Power law on sources (B<=64): lr* = {np.exp(lc):.4g} * B^{p:.3f}; predicts B=128: {np.exp(lc) * 128 ** p:.4g}, B=256: {np.exp(lc) * 256 ** p:.4g}\n')
    # (b) WD optimum per batch at lr .0015
    res = wd_res = []
    for B in sorted({r['batch'] for r in rows}):
        pairs = [(r['wd'], r['val_loss']) for r in rows if r['batch'] == B and abs(r['lr'] - .0015) < 1e-9 and abs(r['beta1'] - .9) < 1e-9]
        if len(pairs) >= 3:
            pairs = sorted(pairs)
            x, L, a = quad_opt(pairs)
            best_wd, best_L = min(pairs, key=lambda p: p[1])
            inside = a > 0 and pairs[0][0] <= x <= pairs[-1][0]
            res.append({'batch': B, 'n': len(pairs), 'wd_star': x if inside else best_wd, 'loss_star': L if inside else best_L,
                            'best_wd': best_wd, 'edge': '' if inside else 'grid edge (lower bound)'})
    if res:
        out.append('\n## (b) optimal WD vs batch size at lr 0.0015 (quadratic in log wd)\n\n'
                   + md(res, ['batch', 'n', 'wd_star', 'loss_star', 'best_wd', 'edge']))
        Bs = np.array([r['batch'] for r in res]); ws = np.array([r['wd_star'] for r in res])
        src = (Bs <= 64) & np.array([not r['edge'] for r in res])
        if src.sum() >= 2:
            p, lc = np.polyfit(np.log(Bs[src]), np.log(ws[src]), 1)
            out.append(f'Power law on interior sources (B in {sorted(Bs[src].tolist())}): wd* = {np.exp(lc):.4g} * B^{p:.3f}; '
                       f'predicts B=128: {np.exp(lc) * 128 ** p:.4g}, B=256: {np.exp(lc) * 256 ** p:.4g}. '
                       f'Linear hypothesis (wd* ∝ B, anchored at B=16): B=128: {ws[Bs == 16][0] * 8:.4g}, B=256: {ws[Bs == 16][0] * 16:.4g}\n')
    # (c) beta1
    mom = [r for r in rows if r['src'] == 'ours' and abs(r['beta1'] - .9) > 1e-9]
    if mom:
        out.append('\n## (c) beta1 ablation\n\n' + md(sorted(mom, key=lambda r: (r['batch'], r['beta1'])), ['batch', 'lr', 'wd', 'beta1', 'val_loss']))
    (TABLES / 'p32_fits.json').write_text(json.dumps({'rows': rows, 'lr_fits': lr_res, 'wd_fits': wd_res}, indent=1))


def p42(runs, out):
    rs = with_tag(runs, 'a2-p42a') + with_tag(runs, 'a2-p42c') + with_tag(runs, 'a2-p42d')
    ref = [r for r in provided() if r['parts'] == 'P1a' and r['tokens'] == 153600000]
    rows = [{'part': next(t for t in r['tags'] if t.startswith('a2-p42') and len(t) == 7)[-1], 'policy': r['policy'], 'width': r['width'], 'depth': r['depth'],
                 'lr': r['lr'], 'val_loss': r['final'], 'minutes': r['runtime_min']} for r in rs]
    rows += [{'part': 'src', 'policy': 'supplied', 'width': 512, 'depth': 8, 'lr': r['learning_rate'], 'val_loss': r['final_val_loss'], 'minutes': None} for r in ref]
    rows.sort(key=lambda r: (r['part'], r['policy'], r['width'], r['depth'], r['lr']))
    out.append('\n# P4.2: width/depth transfer at 153.6M tokens\n\n' + md(rows, ['part', 'policy', 'width', 'depth', 'lr', 'val_loss', 'minutes']))
    fits = []
    groups = defaultdict(list)
    for r in rows:
        groups[(r['policy'], r['width'], r['depth'])].append((r['lr'], r['val_loss']))
    for (pol, w, d), pairs in sorted(groups.items()):
        if len(pairs) >= 3:
            x, L, bx, bL = fit_lr_curve(pairs)
            at_src = dict(pairs).get(.003)
            fits.append({'policy': pol, 'width': w, 'depth': d, 'n': len(pairs), 'lr_star': x, 'loss_star': L, 'best_lr': bx, 'best_loss': bL,
                             'loss_at_source_lr': at_src, 'gap': (at_src - bL) if at_src is not None else None})
    if fits:
        out.append('\n## Fitted optima per configuration (source LR 0.003)\n\n' +
                   md(fits, ['policy', 'width', 'depth', 'n', 'lr_star', 'loss_star', 'best_lr', 'best_loss', 'loss_at_source_lr', 'gap']))
        for pol in ('baseline', 'mup'):
            pts = [(f['width'], f['lr_star']) for f in fits if f['policy'] in (pol, 'supplied') and f['depth'] == 8 and f['width'] <= 512]
            if len(pts) >= 3:
                ws, ls = np.array([p[0] for p in pts], float), np.array([p[1] for p in pts])
                p, lc = np.polyfit(np.log(ws), np.log(ls), 1)
                out.append(f'- {pol}: lr* = {np.exp(lc):.4g} * width^{p:.3f} on widths {sorted(ws.astype(int).tolist())}; predicted lr*(1024) = {np.exp(lc) * 1024 ** p:.4g}\n')
    json.dump(fits, (TABLES / 'p42_fits.json').open('w'), indent=1)


def plot_lr(rs, title, fname):
    if not rs:
        return
    rs = sorted(rs, key=lambda r: r['lr'])
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.semilogx([r['lr'] for r in rs], [r['final'] for r in rs], 'o-')
    ax.set_xlabel('learning rate'); ax.set_ylabel('final val loss'); ax.set_title(title)
    fig.tight_layout(); fig.savefig(FIG / fname, dpi=130); plt.close(fig)


def main():
    TABLES.mkdir(parents=True, exist_ok=True); FIG.mkdir(exist_ok=True)
    runs = load_runs()
    print(len(runs), 'runs loaded')
    out = []
    for fn in (p1c, p1d, p2c, p32, p42):
        fn(runs, out)
    (TABLES / 'a2_tables.md').write_text(''.join(out))
    print('wrote', TABLES / 'a2_tables.md')


if __name__ == '__main__':
    main()
