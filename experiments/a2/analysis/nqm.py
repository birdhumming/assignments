"""P3.1: noisy quadratic model on CPU. Writes results JSON + plots to $A2_DATA_DIR (default ~/a2)."""
import json
import os
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings('ignore', category=RuntimeWarning)

OUT = Path(os.environ.get('A2_DATA_DIR', Path.home() / 'a2'))
N = 8192
SAMPLES = 1024
H = np.array([1., 10.])
LR_GRID = np.logspace(-4, 0.5, 28)
rng = np.random.default_rng(0)

INIT_STD = (1., np.sqrt(.1))


def simulate(B, lr, *, opt='sgd', beta1=.9, beta2=.95, mu=0., sigma=1., H=H, init_std=INIT_STD,
             samples=SAMPLES, eps=1e-8):
    """Return mean final loss f(w_{N/B}) over independent initializations and noise draws."""
    steps = N // B
    d = len(H)
    w = rng.standard_normal((samples, d)) * np.asarray(init_std)
    m = np.zeros_like(w); v = np.zeros_like(w); s = np.zeros_like(w)
    noise_std = sigma / np.sqrt(B)
    for t in range(1, steps + 1):
        g = w * H + rng.standard_normal(w.shape) * noise_std
        if opt == 'sgd':
            s = mu * s + g
            w = w - lr * s
        else:  # adam (beta1 may be 0 -> RMSProp), bias-corrected
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * g * g
            mhat = m / (1 - beta1 ** t) if beta1 > 0 else g
            vhat = v / (1 - beta2 ** t)
            w = w - lr * mhat / (np.sqrt(vhat) + eps)
        if not np.all(np.isfinite(w)):
            return np.inf
    return float(np.mean(.5 * np.sum(H * w * w, axis=1)))


def sweep(B, lrs=LR_GRID, **kw):
    losses = np.array([simulate(B, lr, **kw) for lr in lrs])
    ok = np.isfinite(losses) & (losses < 1e3)
    return lrs[ok], losses[ok]


def fit_opt(lrs, losses):
    """Quadratic in log lr through the 5 points around the minimum; return (lr*, L*)."""
    i = int(np.argmin(losses)); lo, hi = max(0, i - 2), min(len(lrs), i + 3)
    x, y = np.log(lrs[lo:hi]), losses[lo:hi]
    if len(x) < 3:
        return float(lrs[i]), float(losses[i])
    a, b, _ = np.polyfit(x, y, 2)
    if a <= 0:
        return float(lrs[i]), float(losses[i])
    xs = float(np.clip(-b / (2 * a), x.min(), x.max()))
    # The fitted minimum can undershoot the sampled floor; report the best sampled loss instead.
    return float(np.exp(xs)), float(losses[i])


def powerlaw(xs, ys):
    p, lc = np.polyfit(np.log(xs), np.log(ys), 1)
    return float(np.exp(lc)), float(p)


def optimum_curve(batches, **kw):
    out = {}
    for B in batches:
        lrs, losses = sweep(B, **kw)
        out[B] = dict(lr_star=fit_opt(lrs, losses)[0], loss_star=fit_opt(lrs, losses)[1],
                      lrs=lrs.tolist(), losses=losses.tolist())
    return out


def source_target(label, **kw):
    src = optimum_curve([1, 2, 4, 8, 16, 32, 64], **kw)
    c, p = powerlaw(list(src), [src[B]['lr_star'] for B in src])
    pred = c * 256 ** p
    tgt = optimum_curve([128, 256, 512], **kw)
    loss_at_pred = simulate(256, pred, **kw)
    r = dict(label=label, source=src, exponent=p, coeff=c, pred_lr_256=pred, loss_at_pred_256=loss_at_pred,
             target=tgt, gap_256=loss_at_pred - tgt[256]['loss_star'])
    print(f"{label}: p={p:+.3f} pred lr*(256)={pred:.4g} tuned lr*(256)={tgt[256]['lr_star']:.4g} "
          f"loss pred={loss_at_pred:.4g} tuned={tgt[256]['loss_star']:.4g} gap={r['gap_256']:+.4g}")
    return r


def main():
    OUT.mkdir(exist_ok=True)
    res = {}
    res['sgd'] = source_target('SGD', opt='sgd')
    res['rmsprop'] = source_target('RMSProp', opt='adam', beta1=0.)
    res['adam'] = source_target('Adam', opt='adam', beta1=.9)
    res['sigma'] = {}
    for sigma in (1, 10, 100, 300):
        for opt, b1 in (('rmsprop', 0.), ('adam', .9)):
            res['sigma'][f'{opt}_sigma{sigma}'] = source_target(f'{opt} sigma={sigma}', opt='adam', beta1=b1, sigma=sigma)
    res['scalar'] = {}
    for sigma in (1, 10, 100, 300):
        for opt, b1 in (('rmsprop', 0.), ('adam', .9)):
            res['scalar'][f'{opt}_sigma{sigma}'] = source_target(
                f'scalar {opt} sigma={sigma}', opt='adam', beta1=b1, sigma=sigma, H=np.array([1.]), init_std=(np.sqrt(2.),))
    res['scalar']['sgd_sigma1'] = source_target('scalar SGD sigma=1', opt='sgd', H=np.array([1.]), init_std=(np.sqrt(2.),))
    # (d) momentum
    res['momentum'] = {}
    for B in (16, 256):
        for mu in (0., .9):
            lrs, losses = sweep(B, opt='sgd', mu=mu)
            lr_s, L_s = fit_opt(lrs, losses)
            res['momentum'][f'sgd_B{B}_mu{mu}'] = dict(lr_star=lr_s, loss_star=L_s)
            print(f'SGD B={B} mu={mu}: lr*={lr_s:.4g} loss*={L_s:.4g}')
        for b1 in (0., .5, .9, .95, .98, .99):
            lrs, losses = sweep(B, opt='adam', beta1=b1)
            lr_s, L_s = fit_opt(lrs, losses)
            res['momentum'][f'adam_B{B}_beta1_{b1}'] = dict(lr_star=lr_s, loss_star=L_s)
            print(f'Adam B={B} beta1={b1}: lr*={lr_s:.4g} loss*={L_s:.4g}')
    (OUT / 'nqm_results.json').write_text(json.dumps(res, indent=1, default=float))
    print('saved', OUT / 'nqm_results.json')


if __name__ == '__main__':
    main()
