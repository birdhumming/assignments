"""Fit the supplied P1/P2 sweeps and print the predictions needed before launching targets."""
import numpy as np

from experiments.a2.provided_sweeps import load

B = 1e9


def by_budget(rows, key='learning_rate'):
    out = {}
    for r in rows:
        out.setdefault(r['tokens'], []).append((r[key], r['final_val_loss']))
    return {t: sorted(v) for t, v in out.items()}


def quad_opt(pairs):
    """Quadratic in log x through the points; return (x*, L*, curvature)."""
    x = np.log(np.array([p[0] for p in pairs])); y = np.array([p[1] for p in pairs])
    a, b, c = np.polyfit(x, y, 2)
    xs = -b / (2 * a)
    return float(np.exp(xs)), float(a * xs**2 + b * xs + c), float(a)


def powerlaw(xs, ys):
    """y = c x^p fit in log-log; return (c, p, R^2)."""
    lx, ly = np.log(xs), np.log(ys)
    p, lc = np.polyfit(lx, ly, 1)
    pred = p * lx + lc
    r2 = 1 - ((ly - pred)**2).sum() / ((ly - ly.mean())**2).sum()
    return float(np.exp(lc)), float(p), float(r2)


def p1(part_rows, label):
    opt = {t: quad_opt(v) for t, v in by_budget(part_rows).items()}
    print(f'\n== {label}: fitted optimum per budget')
    for t, (lr, L, a) in sorted(opt.items()):
        best = min(by_budget(part_rows)[t], key=lambda p: p[1])
        print(f'  {t/B:.4f}B  lr*={lr:.5f}  L*={L:.4f}  curv={a:.3f}  best sampled lr={best[0]:g} ({best[1]:.4f})')
    return opt


def fit_rule(opt, budgets, label):
    ts = np.array(sorted(b for b in budgets if b in opt)); lrs = np.array([opt[t][0] for t in ts])
    c, p, r2 = powerlaw(ts, lrs)
    print(f'  rule[{label}]: lr* = {c:.4g} * D^{p:+.3f}   R2={r2:.3f}')
    return c, p


def predict(c, p, D):
    return c * D**p


def p2_joint(rows):
    print('\n== P2(a): joint quadratic in (log lr, log wd) per budget')
    res = {}
    for t, rs in sorted({r['tokens']: [q for q in rows if q['tokens'] == r['tokens']] for r in rows}.items()):
        x = np.log([r['learning_rate'] for r in rs]); y = np.log([r['weight_decay'] for r in rs]); z = np.array([r['final_val_loss'] for r in rs])
        X = np.column_stack([np.ones_like(x), x, y, x*x, x*y, y*y])
        coef, *_ = np.linalg.lstsq(X, z, rcond=None)
        a, b, c_, d, e, f = coef
        H = np.array([[2*d, e], [e, 2*f]]); g = np.array([b, c_])
        xs, ys = np.linalg.solve(H, -g)
        Lmin = float(X[0] @ coef * 0 + a + b*xs + c_*ys + d*xs*xs + e*xs*ys + f*ys*ys)
        best = min(rs, key=lambda r: r['final_val_loss'])
        pd = 'min' if np.all(np.linalg.eigvals(H) > 0) else 'saddle/max'
        print(f'  {t/B:.4f}B  lr*={np.exp(xs):.5f} wd*={np.exp(ys):.4f} lr*wd*={np.exp(xs+ys):.3e} L*={Lmin:.4f} [{pd}] '
              f'| best sampled lr={best["learning_rate"]:g} wd={best["weight_decay"]:g} L={best["final_val_loss"]:.4f}  n={len(rs)}')
        res[t] = (float(np.exp(xs)), float(np.exp(ys)), Lmin, best)
    return res


if __name__ == '__main__':
    p1a = load('P1a'); p1b = load('P1b'); p1d = load('P1d'); p1e = load('P1e'); p2a = load('P2a')
    opt = p1(p1a, 'P1(a) AdamW linear, small budgets')
    c3, p3 = fit_rule(opt, opt, 'small 3')
    for D in (1.2288*B, 1.8432*B, 2.4576*B):
        print(f'  predicted lr* at {D/B:.4f}B from small-3 rule: {predict(c3, p3, D):.5f}')
    optb = p1(p1b, 'P1(b) AdamW linear, large budgets')
    allopt = {**opt, **optb}
    print('\n== P1(c): two rules -> 4.9152B')
    c6, p6 = fit_rule(allopt, allopt, 'all six')
    cL, pL = fit_rule(optb, optb, 'larger three')
    print(f'  predicted lr* at 4.9152B: all-six {predict(c6, p6, 4.9152*B):.5f}   larger-three {predict(cL, pL, 4.9152*B):.5f}')
    optd = p1(p1d, 'P1(d) Hyperball linear')
    cd, pd_ = fit_rule(optd, optd, 'hyperball 3')
    print(f'  predicted hyperball lr* at 1.2288B: {predict(cd, pd_, 1.2288*B):.5f}')
    print('  AdamW same three budgets:'); fit_rule(opt, opt, 'adamw 3')
    opte = p1(p1e, 'P1(e) AdamW cosine'); fit_rule(opte, opte, 'cosine 3')
    joint = p2_joint(p2a)
    ts = np.array(sorted(joint)); 
    print('\n== P2(b): power laws of lr*, wd*, lr*wd* vs tokens')
    for name, vals in (('lr*', [joint[t][0] for t in ts]), ('wd*', [joint[t][1] for t in ts]), ('lr*wd*', [joint[t][0]*joint[t][1] for t in ts])):
        c, p, r2 = powerlaw(ts, np.array(vals)); print(f'  {name}: {c:.4g} * D^{p:+.3f}  R2={r2:.3f}')
    c, p, _ = powerlaw(ts, np.array([joint[t][0]*joint[t][1] for t in ts]))
    prod = predict(c, p, 2.4576*B)
    print(f'  P2(c): predicted product at 2.4576B = {prod:.3e} -> WD at lr .003 = {prod/.003:.4f}')
    b153 = joint[int(.1536*B)][3]
    print(f'  P2(c) baseline (i): best 153.6M pair lr={b153["learning_rate"]:g} wd={b153["weight_decay"]:g}')
    print('\n== P1 vs P2 best measured per budget')
    for t in ts:
        p1best = min([r for r in p1a + p1b if r['tokens'] == t], key=lambda r: r['final_val_loss'])['final_val_loss']
        p2best = joint[t][3]['final_val_loss']
        print(f'  {t/B:.4f}B  P1 best {p1best:.4f}  P2 best {p2best:.4f}  gain {p1best-p2best:+.4f}')
