"""Produce all P1-P7 tables (stdout) and plots ($A1_DATA_DIR/plots) from the pulled W&B runs."""
import json
import math
import os

import a1lib as L
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from a1lib import (
    curve,
    final_val,
    find,
    fmt,
    powerlaw_fit,
    series,
    smoothness,
)

OUT = os.path.join(L.DATA_DIR, "plots"); os.makedirs(OUT, exist_ok=True)
R = {}  # results dict for the writeup
def say(*a):
    print(*a); 
def fv(**spec):
    d = find(**spec); return final_val(d)
def name(**spec):
    d = find(**spec); return d["name"] if d else None

base = find(); BASE_VAL = final_val(base)
say("baseline", base["name"], fmt(BASE_VAL))

# ---------------- P1 ----------------
say("\n== P1a one-at-a-time sweeps (final val loss) ==")
sweeps = {
    "learning_rate": [3e-4, 1e-3, 3e-3, 9e-3, 2.7e-2],
    "batch_size": [16, 32, 64, 128, 256],
    "warmup_percent": [0.0, 0.003, 0.01, 0.03, 0.1],
    "weight_decay": [0.0, 0.01, 0.03, 0.1, 0.3, 1.0],
}
R["p1a"] = {}
fig, axes = plt.subplots(1, 4, figsize=(17, 4))
for ax, (k, vals) in zip(axes, sweeps.items()):
    ys = [fv(**{k: v}) for v in vals]
    R["p1a"][k] = dict(zip(map(str, vals), ys))
    say(k, [(v, fmt(y)) for v, y in zip(vals, ys)])
    xs = [v if v > 0 else (min(x for x in vals if x > 0) / 3) for v in vals]
    ax.plot(xs, ys, "o-"); ax.set_xscale("log"); ax.set_title(k); ax.set_ylabel("final val loss")
    ax.axhline(BASE_VAL, color="gray", ls="--", lw=0.8)
    for x, y, v in zip(xs, ys, vals):
        if not math.isnan(y): ax.annotate(f"{v:g}\n{y:.3f}", (x, y), fontsize=7, textcoords="offset points", xytext=(0, 6), ha="center")
fig.suptitle("P1(a): d8 recipe, one hyperparameter at a time (compute-matched 614M tokens)"); fig.tight_layout()
fig.savefig(f"{OUT}/p1a_sweeps.png", dpi=130); plt.close(fig)

say("\n== P1b pairs ==")
lrs = [1e-3, 3e-3, 9e-3]
pairs = {
    "batch_size": [16, 32, 64, 128, 256],
    "weight_decay": [0.01, 0.03, 0.1, 0.3, 1.0],
    "warmup_percent": [0.0, 0.01, 0.03, 0.1],
}
R["p1b"] = {}
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, (k, vals) in zip(axes, pairs.items()):
    R["p1b"][k] = {}
    for lr in lrs:
        ys = [fv(learning_rate=lr, **{k: v}) for v in vals]
        R["p1b"][k][str(lr)] = dict(zip(map(str, vals), ys))
        say(k, "lr", lr, [(v, fmt(y)) for v, y in zip(vals, ys)])
        xs = [v if v > 0 else 0.003 for v in vals]
        ax.plot(xs, ys, "o-", label=f"lr={lr:g}")
    ax.set_xscale("log"); ax.set_title(f"LR x {k}"); ax.legend(fontsize=8); ax.set_ylabel("final val loss")
    if k == "warmup_percent": ax.set_xlabel("(0 plotted at 0.003)")
# extra: lr 2.7e-2 with warmup 0.1 ; bs x wd1.0
extra = {"lr0.027 warmup0.1": fv(learning_rate=2.7e-2, warmup_percent=0.1), "lr0.027": fv(learning_rate=2.7e-2),
         "bs16 wd1.0": fv(batch_size=16, weight_decay=1.0), "bs256 wd1.0": fv(batch_size=256, weight_decay=1.0), "wd1.0": fv(weight_decay=1.0)}
R["p1b"]["extra"] = extra; say("extra", {k: fmt(v) for k, v in extra.items()})
fig.suptitle("P1(b): pairs that might co-vary"); fig.tight_layout(); fig.savefig(f"{OUT}/p1b_pairs.png", dpi=130); plt.close(fig)

say("\n== P1c schedules / betas / clipping ==")
scheds = ["linear", "cos", "constant", "wsd0.2", "wsd0.5"]
R["p1c"] = {"sched": {}}
for s in scheds:
    R["p1c"]["sched"][s] = {str(lr): fv(lr_schedule=s, learning_rate=lr) for lr in lrs}
    say(s, {lr: fmt(v) for lr, v in R["p1c"]["sched"][s].items()})
R["p1c"]["wsd_warmup"] = {str(w): fv(lr_schedule="wsd0.2", warmup_percent=w) for w in (0.0, 0.01, 0.1)}
R["p1c"]["example"] = {"wsd0.2 warm0": fv(learning_rate=9e-3, weight_decay=1.0, lr_schedule="wsd0.2", warmup_percent=0.0),
                       "wsd0.2 warm0.2": fv(learning_rate=9e-3, weight_decay=1.0, lr_schedule="wsd0.2", warmup_percent=0.2),
                       "linear warm0.1": fv(learning_rate=9e-3, weight_decay=1.0, lr_schedule="linear", warmup_percent=0.1),
                       "linear warm0.01 (lr9e-3 wd1.0)": fv(learning_rate=9e-3, weight_decay=1.0)}
R["p1c"]["beta1"] = {str(b): fv(beta1=b) for b in (0.8, 0.9, 0.95)}
R["p1c"]["beta2"] = {str(b): fv(beta2=b) for b in (0.9, 0.95, 0.99)}
R["p1c"]["gradclip"] = {"1.0": BASE_VAL, "none": fv(grad_norm=None)}
for k in ("wsd_warmup", "example", "beta1", "beta2", "gradclip"): say(k, {a: fmt(b) for a, b in R["p1c"][k].items()})
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
vals = [R["p1c"]["sched"][s]["0.003"] for s in scheds]
axes[0].bar(scheds, vals); axes[0].set_ylim(min(vals) - 0.02, max(vals) + 0.02); axes[0].set_ylabel("final val loss"); axes[0].set_title("schedule at peak LR 3e-3 (final val)")
for s_, v in zip(scheds, vals): axes[0].annotate(f"{v:.3f}", (s_, v), ha="center", va="bottom", fontsize=8)
for s in scheds:
    d = find(lr_schedule=s)
    if d: 
        xs, ys = curve(d, "val_loss"); axes[1].plot(xs, ys, label=s)
axes[1].set_ylim(2.85, 3.6); axes[1].set_xlabel("step"); axes[1].set_ylabel("val loss"); axes[1].legend(); axes[1].set_title("val curves by schedule (lr 3e-3)")
fig.tight_layout(); fig.savefig(f"{OUT}/p1c_schedules.png", dpi=130); plt.close(fig)

# ---------------- P2 ----------------
say("\n== P2 ==")
depths = [4, 5, 6, 7, 8, 9]
def nparams(depth):
    d = find(model_name=f"d{depth}")
    return d["config"]["parameter_count"] if d else None
N = {dp: nparams(dp) for dp in depths}
say("params", N)
recipes = {"baseline": {}, "constant": {"lr_schedule": "constant"}, "dropout0.2": {"dropout": 0.2}, "lr0.03": {"learning_rate": 0.03}}
R["p2a"] = {}
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for rname, ch in recipes.items():
    ys = [fv(model_name=f"d{dp}", **ch) for dp in depths]
    ok = [(N[dp], y) for dp, y in zip(depths, ys) if not math.isnan(y) and N[dp]]
    a, b = powerlaw_fit([n for n, _ in ok], [y for _, y in ok])
    a3, b3 = powerlaw_fit([N[dp] for dp in (4, 5, 6)], ys[:3])
    # residuals at d8/d9 from the d4-d6 fit
    pred89 = [a3 * N[dp] ** -b3 for dp in (8, 9)]
    # d20 param count: extrapolate from N(depth) ~ c*depth^3 fit
    ld = np.polyfit(np.log(depths), np.log([N[dp] for dp in depths]), 1)
    N20 = math.exp(ld[1]) * 20 ** ld[0]
    R["p2a"][rname] = {"losses": dict(zip(map(str, depths), ys)), "exp_all": b, "exp_d4d6": b3, "pred_d8d9_from_d4d6": pred89, "N20": N20, "pred_d20": a * N20 ** -b}
    say(rname, [fmt(y) for y in ys], f"exp(all)={b:.3f} exp(d4-6)={b3:.3f}  d4-6 fit predicts d8,d9 = {[fmt(p) for p in pred89]} vs actual {[fmt(ys[4]), fmt(ys[5])]}  d20(N={N20/1e6:.0f}M)->{a * N20 ** -b:.3f}")
    axes[0].plot([N[dp] for dp in depths], ys, "o-", label=f"{rname} (exp {b:.3f})")
    grid = np.logspace(np.log10(N[4]), np.log10(N20), 50)
    axes[1].plot([N[dp] for dp in depths], ys, "o", label=rname); axes[1].plot(grid, a * grid ** -b, "--", lw=0.8)
for ax in axes: ax.set_xscale("log"); ax.set_xlabel("non-embedding-inclusive params"); ax.set_ylabel("final val loss"); ax.legend(fontsize=8)
axes[1].set_yscale("log"); axes[1].set_title("power-law fits extrapolated to d20"); axes[0].set_title("P2(a): 4 recipes, d4-d9")
fig.tight_layout(); fig.savefig(f"{OUT}/p2a_scaling.png", dpi=130); plt.close(fig)

say("-- P2b slope benders --")
R["p2b"] = {"model_lr0.001": {}, "model_bs256": {}, "data": {}}
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for lab, ch in (("baseline", {}), ("lr0.001", {"learning_rate": 1e-3}), ("bs256", {"batch_size": 256})):
    ys = [fv(model_name=f"d{dp}", **ch) for dp in depths]
    ok = [(N[dp], y) for dp, y in zip(depths, ys) if not math.isnan(y)]
    a, b = powerlaw_fit([n for n, _ in ok], [y for _, y in ok]) if len(ok) >= 2 else (float("nan"), float("nan"))
    R["p2b"]["model_" + lab] = {"losses": dict(zip(map(str, depths), ys)), "exp": b}
    say("model-size", lab, [fmt(y) for y in ys], f"exp={b:.3f}")
    axes[0].plot([n for n, _ in ok], [y for _, y in ok], "o-", label=f"{lab} (exp {b:.3f})")
axes[0].set_xscale("log"); axes[0].set_title("model-size axis: recipe changes slope"); axes[0].legend(); axes[0].set_xlabel("params"); axes[0].set_ylabel("final val loss")
toks = [75000, 150000, 300000, 600000, 1200000]
for dp in (6, 8):
    for lab, ch in (("lr0.003", {}), ("lr0.001", {"learning_rate": 1e-3})):
        ys = [fv(model_name=f"d{dp}", num_train_sequences=n, **ch) for n in toks]
        ok = [(n * 1024, y) for n, y in zip(toks, ys) if not math.isnan(y)]
        a, b = powerlaw_fit([n for n, _ in ok], [y for _, y in ok]) if len(ok) >= 2 else (float("nan"), float("nan"))
        R["p2b"]["data"][f"d{dp}-{lab}"] = {"losses": dict(zip(map(str, toks), ys)), "exp": b}
        say("data", f"d{dp}", lab, [fmt(y) for y in ys], f"exp={b:.3f}")
        axes[1].plot([n for n, _ in ok], [y for _, y in ok], "o-", label=f"d{dp} {lab} (exp {b:.3f})")
axes[1].set_xscale("log"); axes[1].set_title("data axis: tokens (d6, d8; lr 3e-3 vs 1e-3)"); axes[1].legend(fontsize=8); axes[1].set_xlabel("train tokens")
fig.tight_layout(); fig.savefig(f"{OUT}/p2b_benders.png", dpi=130); plt.close(fig)

say("-- P2c breakers --")
breakers = {"sgd": {"optimizer_name": "sgd"}, "noQK lr0.03": {"qk_norm": False, "learning_rate": 0.03},
            "lr0.03 nowarm noclip": {"warmup_percent": 0.0, "learning_rate": 0.03, "grad_norm": None},
            "8 epochs x 75k": {"num_train_sequences": 75000, "num_epochs": 8.0}}
R["p2c"] = {}
fig, ax = plt.subplots(figsize=(7, 4.5))
ys = [fv(model_name=f"d{dp}") for dp in (4, 6, 8)]
ax.plot([N[dp] for dp in (4, 6, 8)], ys, "ko-", label="baseline")
for lab, ch in breakers.items():
    ys = [fv(model_name=f"d{dp}", **ch) for dp in (4, 6, 8)]
    R["p2c"][lab] = dict(zip(["d4", "d6", "d8"], ys)); say(lab, [fmt(y) for y in ys])
    ax.plot([N[dp] for dp in (4, 6, 8)], ys, "o-", label=lab)
ax.set_xscale("log"); ax.set_xlabel("params"); ax.set_ylabel("final val loss"); ax.legend(); ax.set_title("P2(c): interventions that break the power law")
fig.tight_layout(); fig.savefig(f"{OUT}/p2c_breakers.png", dpi=130); plt.close(fig)

# ---------------- P3 refresh ----------------
say("\n== P3 (refresh) ==")
def group(**spec):
    vals = []
    for s in (1, 2, 3, 42):
        d = find(model_seed=s, data_seed=s, **spec); vals.append(final_val(d))
    vals = [v for v in vals if not math.isnan(v)]
    return {"n": len(vals), "mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan"), "vals": vals}
R["p3c"] = {"d8 base": group(), "bs16": group(batch_size=16), "lr0.009": group(learning_rate=9e-3), "d4": group(model_name="d4"), "2x tokens": group(num_train_sequences=1200000)}
for k, v in R["p3c"].items(): say(k, v["n"], fmt(v["mean"]), fmt(v["std"]))

def vals_of(runs): return [final_val(d) for d in runs if d is not None]
p3_groups = {
    "paired seeds\n(ds=ms)": R["p3c"]["d8 base"]["vals"],
    "model seed\nonly": vals_of([find()] + [find(model_seed=s) for s in (1, 2, 3)]),
    "data seed\nonly": vals_of([find()] + [find(data_seed=s) for s in (1, 2, 3)]),
    "nondeterministic\nreps (same seeds)": vals_of([find()] + [find(run_name_suffix=f"nondeterministic-rep{i}") for i in (1, 2)]),
    "deterministic\nH100 x2 + A100": vals_of([find(deterministic=True, run_name_suffix=f"deterministic-reference-{k}") for k in ("1", "2", "a100")]),
    "d4 paired\nseeds": R["p3c"]["d4"]["vals"],
    "bs16 paired\nseeds": R["p3c"]["bs16"]["vals"],
    "2x tokens\nseeds": R["p3c"]["2x tokens"]["vals"],
    "lr 9e-3 paired\nseeds": R["p3c"]["lr0.009"]["vals"],
}
R["p3_groups"] = {k.replace("\n", " "): {"n": len(v), "mean": float(np.mean(v)), "std": float(np.std(v, ddof=1))} for k, v in p3_groups.items()}
for k, v in R["p3_groups"].items(): say(k, v["n"], fmt(v["mean"]), f'{v["std"]:.4f}')
fig, ax = plt.subplots(figsize=(13, 5.5))
for i, (k, v) in enumerate(p3_groups.items()):
    v = np.array(v); ax.scatter([i] * len(v), (v - v.mean()) * 1e3, s=60)
    ax.text(i, 12.5, f"mean {v.mean():.3f}\nstd {np.std(v, ddof=1)*1e3:.1f}e-3\nn={len(v)}", ha="center", va="top", fontsize=8)
ax.set_xticks(range(len(p3_groups))); ax.set_xticklabels(list(p3_groups), fontsize=8); ax.set_ylim(-12, 13); ax.axhline(0, color="k", lw=0.5)
ax.set_ylabel("final val loss - group mean (x1e-3)"); ax.set_title("P3: run-to-run variation of final val loss (each dot = one run)"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/p3_variation.png", dpi=130); plt.close(fig)

# ---------------- P5 ----------------
say("\n== P5 ==")
fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))
for lr in [3e-4, 1e-3, 3e-3, 9e-3, 2.7e-2]:
    d = find(learning_rate=lr)
    if d: xs, ys = curve(d, "train_loss"); k = np.ones(51)/51; axes[0].plot(xs[25:-25], np.convolve(ys, k, "valid"), label=f"lr {lr:g}", lw=1)
axes[0].set_ylim(2.8, 4.5); axes[0].set_title("macro: peak LR (51-step mean)"); axes[0].legend(fontsize=8)
for s in scheds:
    d = find(lr_schedule=s)
    if d: xs, ys = curve(d, "train_loss"); k = np.ones(51)/51; axes[1].plot(xs[25:-25], np.convolve(ys, k, "valid"), label=s, lw=1)
axes[1].set_ylim(2.8, 4.0); axes[1].set_title("macro: schedule"); axes[1].legend(fontsize=8)
for b in (0.8, 0.9, 0.95):
    d = find(beta1=b)
    if d: xs, ys = curve(d, "train_loss"); k = np.ones(51)/51; axes[2].plot(xs[25:-25], np.convolve(ys, k, "valid"), label=f"beta1 {b}", lw=1)
axes[2].set_ylim(2.8, 4.0); axes[2].set_title("macro: beta1"); axes[2].legend(fontsize=8)
for ax in axes: ax.set_xlabel("step"); ax.set_ylabel("train loss")
fig.tight_layout(); fig.savefig(f"{OUT}/p5a_macro.png", dpi=130); plt.close(fig)

micro = {"bs16": {"batch_size": 16}, "bs32": {"batch_size": 32}, "bs64 (base)": {}, "bs128": {"batch_size": 128}, "bs256": {"batch_size": 256},
         "lr3e-4": {"learning_rate": 3e-4}, "lr9e-3": {"learning_rate": 9e-3}, "lr2.7e-2": {"learning_rate": 2.7e-2},
         "beta2 0.9": {"beta2": 0.9}, "beta2 0.99": {"beta2": 0.99}, "beta1 0.8": {"beta1": 0.8}, "beta1 0.95": {"beta1": 0.95},
         "no clip": {"grad_norm": None}, "constant": {"lr_schedule": "constant"}, "wsd0.2": {"lr_schedule": "wsd0.2"}, "wd1.0": {"weight_decay": 1.0}}
R["p5b"] = {}
for lab, spec in micro.items():
    d = find(**spec)
    if d: R["p5b"][lab] = smoothness(d)
say("smoothness (std of train loss around 51-step mean, 2nd half):", {k: round(v, 4) for k, v in R["p5b"].items()})
fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
axes[0].barh(list(R["p5b"].keys()), list(R["p5b"].values())); axes[0].set_xlabel("loss jitter (std around 51-step mean)"); axes[0].set_title("micro: what changes smoothness")
for lab in ("bs16", "bs64 (base)", "bs256"):
    d = find(**micro[lab])
    if d is None: continue
    xs, ys = curve(d, "train_loss"); frac = xs / xs.max(); m = (frac > 0.60) & (frac < 0.63)
    axes[1].plot(frac[m], ys[m], lw=0.8, label=lab)
axes[1].set_title("zoom: raw train loss, 60-63% of run"); axes[1].legend(); axes[1].set_xlabel("fraction of run"); axes[1].set_ylabel("train loss")
fig.tight_layout(); fig.savefig(f"{OUT}/p5b_micro.png", dpi=130); plt.close(fig)

# ---------------- P6 ----------------
say("\n== P6 ==")
probe = find(run_name_suffix="actprobe")
if probe is None: probe = base
say("P6 run:", probe["name"])
def layer_stat(d, kind, mod, step):
    out = []
    for l in range(8):
        xs, ys = series(d, f"logging/rms/model.layers.{l}.{mod}/{kind}")
        if len(xs) == 0: out.append(float("nan")); continue
        i = int(np.argmin(np.abs(xs - step))); out.append(ys[i])
    return out
xs_g, _ = series(probe, "logging/rms/global/gradient"); T = xs_g.max() if len(xs_g) else 9374
snaps = {"start (step 100)": 100, "middle": T // 2, "end": T}
mods = ["self_attn.q_proj", "self_attn.o_proj", "mlp.up_proj", "mlp.down_proj"]
fig, axes = plt.subplots(3, 4, figsize=(18, 10))
R["p6a"] = {}
for i, kind in enumerate(("parameter", "gradient", "activation")):
    for j, mod in enumerate(mods):
        for lab, st in snaps.items():
            v = layer_stat(probe, kind, mod, st); axes[i, j].plot(range(1, 9), v, "o-", label=lab)
            R["p6a"][f"{kind}/{mod}/{lab}"] = v
        axes[i, j].set_title(f"{kind} RMS: {mod}"); axes[i, j].set_xlabel("layer"); axes[i, j].set_yscale("log")
        if i == 0 and j == 0: axes[i, j].legend(fontsize=8)
fig.suptitle(f"P6(a): per-layer RMS at start/middle/end of a standard d8 run ({probe['name']})"); fig.tight_layout(); fig.savefig(f"{OUT}/p6a_layers.png", dpi=130); plt.close(fig)

# global stats over training for interventions
inter = {"healthy d8": {"run_name_suffix": "actprobe"}, "lr0.03 noQK noclip": {"run_name_suffix": "actprobe", "learning_rate": 0.03, "qk_norm": False, "grad_norm": None},
         "wd1.0": {"run_name_suffix": "actprobe", "weight_decay": 1.0}, "wd0.0": {"run_name_suffix": "actprobe", "weight_decay": 0.0},
         "lr0.0003": {"learning_rate": 3e-4}, "lr0.009": {"learning_rate": 9e-3}, "bs256": {"batch_size": 256}, "no clip": {"grad_norm": None}, "constant": {"lr_schedule": "constant"}}
fig, axes = plt.subplots(1, 4, figsize=(20, 4.3))
R["p6c"] = {}
for lab, spec in inter.items():
    d = find(**spec)
    if d is None: say("missing", lab); continue
    R["p6c"][lab] = {}
    for ax, key in zip(axes[:3], ("logging/rms/global/parameter", "logging/rms/global/gradient", "logging/rms/global/activation")):
        xs, ys = series(d, key)
        if len(xs) == 0: continue
        sub = slice(None, None, max(1, len(xs) // 400)); ax.plot(xs[sub], ys[sub], lw=1, label=lab)
        ax.set_title(key.split("/")[-1] + " RMS (global)"); ax.set_yscale("log"); ax.set_xlabel("step")
        R["p6c"][lab][key.split("/")[-1]] = {"step0": float(ys[0]), "start": float(ys[min(1, len(ys)-1)]), "mid": float(ys[len(ys)//2]), "end": float(ys[-1])}
    xs, ys = series(d, "logging/layer_6_residual_rms")
    if len(xs): axes[3].plot(xs, ys, "o-", ms=3, lw=1, label=lab); R["p6c"][lab]["resid6"] = {"start": float(ys[0]), "mid": float(ys[len(ys)//2]), "end": float(ys[-1])}
axes[3].set_title("residual-stream RMS after layer 6 (probe, 8 val seqs)"); axes[3].set_yscale("log"); axes[3].set_xlabel("step")
axes[2].legend(fontsize=7, loc="center right")
fig.suptitle("P6(b,c): how interventions move parameter / gradient / activation statistics"); fig.tight_layout(); fig.savefig(f"{OUT}/p6c_interventions.png", dpi=130); plt.close(fig)
for lab, v in R["p6c"].items(): say(lab, {k: {a: f"{b:.3g}" for a, b in vv.items()} for k, vv in v.items()})

# ---------------- P7 ----------------
say("\n== P7 ==")
R["p7"] = {"qk": {}, "noqk": {}, "tied": {}}
fig, ax = plt.subplots(figsize=(7, 4.5))
lr7 = [1e-3, 3e-3, 9e-3, 3e-2]
for lab, spec in (("qk", {}), ("noqk", {"qk_norm": False})):
    ys = []
    for lr in lr7:
        v = fv(learning_rate=lr, **spec); R["p7"][lab][str(lr)] = v; ys.append(v)
    say(lab, [(lr, fmt(y)) for lr, y in zip(lr7, ys)])
    ax.plot(lr7, ys, "o-", label={"qk": "QK-norm on (default)", "noqk": "QK-norm off"}[lab])
for lr in (3e-3, 9e-3):
    v = fv(learning_rate=lr, tie_word_embeddings=True); R["p7"]["tied"][str(lr)] = v
say("tied", {k: fmt(v) for k, v in R["p7"]["tied"].items()})
ax.plot([3e-3, 9e-3], [R["p7"]["tied"]["0.003"], R["p7"]["tied"]["0.009"]], "s--", label="tied embeddings (QK on)")
ax.set_xscale("log"); ax.set_xlabel("peak LR"); ax.set_ylabel("final val loss"); ax.legend(); ax.set_title("P7: QK-norm off / tied embeddings vs LR (d8)")
fig.tight_layout(); fig.savefig(f"{OUT}/p7_qknorm.png", dpi=130); plt.close(fig)
# P7 curves for noqk
fig, ax = plt.subplots(figsize=(7, 4.2))
for lr in lr7:
    d = find(learning_rate=lr, qk_norm=False)
    if d: xs, ys = curve(d, "val_loss"); ax.plot(xs, ys, label=f"noQK lr {lr:g}")
d = find(learning_rate=0.03); 
if d: xs, ys = curve(d, "val_loss"); ax.plot(xs, ys, "k--", label="QK lr 0.03")
ax.set_ylim(2.8, 5.6); ax.legend(); ax.set_xlabel("step"); ax.set_ylabel("val loss"); ax.set_title("P7: val curves without QK-norm")
fig.tight_layout(); fig.savefig(f"{OUT}/p7_curves.png", dpi=130); plt.close(fig)

with open(os.path.join(L.DATA_DIR, "results.json"), "w") as fh:
    json.dump(R, fh, indent=1, default=str)
say("\nwrote results.json and plots")
