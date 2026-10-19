"""P2 figures -> experiments/a3_optimization/figures/p2_*.png"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.a3_optimization.analysis.p2_load import load_runs

OUT = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT, exist_ok=True)
runs = {r["name"].replace("-ema0.9-0.99-0.999-0.9999", ""): r for r in load_runs()}
ref = runs["model-d8-lr0.003-tok614M"]
LABEL = {"wsd0.0": "constant", "linear": "linear", "cos": "cosine", "wsd0.2": "plateau, 20% decay",
         "invsqrt": "inverse sqrt", "sgdr2000": "cosine restarts (2000)"}


def panel(names, fname, title, ema="0.999"):
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    for n in names:
        r = runs[n]
        lab = LABEL.get(r["schedule"], r["schedule"]) + (f", lr {r['peak_lr']:g}" if len({runs[m]['peak_lr'] for m in names}) > 1 else "")
        if r["dip"]:
            lab += f", dip {r['dip']}"
        steps = [t[0] for t in r["train"]]
        ax[0, 0].plot(steps, [t[1] for t in r["train"]], label=lab)
        ax[0, 1].plot(steps[::25], [t[2] for t in r["train"]][::25], lw=0.8)
        vs = [v["step"] for v in r["val"]]
        ax[1, 0].plot(vs, [v["val"] for v in r["val"]], label=lab)
        ax[1, 0].plot(vs, [v[f"ema{ema}"] for v in r["val"]], ls="--", color=ax[1, 0].lines[-1].get_color(), lw=0.8)
        ax[1, 1].plot(vs, [v["val"] - w["val"] for v, w in zip(r["val"], ref["val"])])
    ax[0, 0].set(ylabel="learning rate", title="learning rate")
    ax[0, 1].set(ylabel="training loss", ylim=(2.8, 3.8), title="training loss")
    ax[1, 0].set(ylabel="validation loss", ylim=(2.9, 3.6), xlabel="optimizer step",
                 title=f"validation loss (dashed: EMA {ema} of weights)")
    ax[1, 1].set(ylabel="val loss minus reference", xlabel="optimizer step", ylim=(-0.15, 0.3),
                 title="regret vs linear decay, lr 0.003")
    ax[1, 1].axhline(0, color="k", lw=0.5)
    ax[0, 0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=130)


panel([f"model-d8-lr0.003-tok614M{s}" for s in ("-wsd0.0", "", "-cos", "-wsd0.2")], "p2_schedules_lr3e-3.png",
      "P2(a) schedules at peak learning rate 0.003")
panel(["model-d8-lr0.003-tok614M-wsd0.2", "model-d8-lr0.003-tok614M-wsd0.2-dip0.4-0.45-0.5x0.1"], "p2_dip.png",
      "P2(c) temporary learning-rate dip (x0.1 over 40-50%) vs control", ema="0.99")
panel(["model-d8-lr0.003-tok614M", "model-d8-lr0.003-tok614M-cos", "model-d8-lr0.003-tok614M-invsqrt-warmupsteps94",
       "model-d8-lr0.003-tok614M-sgdr2000-warmupsteps94-minlr0.1"], "p2_horizon_free.png",
      "P2(d) horizon-free schedules vs horizon-aware baselines")
# EMA gap closure per schedule and peak LR
fig, ax = plt.subplots(figsize=(7, 4))
for sched in ("wsd0.0", "linear", "cos", "wsd0.2"):
    for lr in (0.0015, 0.003, 0.006):
        r = runs.get(f"model-d8-lr{lr:g}-tok614M" + ("" if sched == "linear" else f"-{sched}"))
        if r:
            f = r["final"]
            ax.plot([0, 0.9, 0.99, 0.999], [f["val"], f["ema0.9"], f["ema0.99"], f["ema0.999"]], marker="o",
                    label=f"{LABEL[sched]}, lr {lr:g}", alpha=0.5 + 0.5 * (lr == 0.003))
ax.set(xticks=[0, 0.9, 0.99, 0.999], xticklabels=["raw", "0.9", "0.99", "0.999"], xlabel="EMA decay",
       ylabel="final validation loss", title="P2(b) final loss of EMA weights (0.9999 off-scale, >4.0)")
ax.legend(fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "p2_ema.png"), dpi=130)
print("ok")
