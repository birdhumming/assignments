"""P1(a)(ii) attention figure from measured final checkpoints: per-head and layer-mean first-token mass, self mass and
mean lag versus query position, against the uniform-causal baseline. Writes figures/p1_attention.png and prints a table."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = Path("results/a4_stability")
FIG = ROOT / "figures"


def attention(run, stage="final"):
    d = json.loads((RES / run / f"{stage}.json").read_text())["attention"]
    layers = sorted(d, key=lambda k: int(k.split(".")[2]))
    q = np.array(d[layers[0]]["query_positions"])
    first = np.array([d[l]["first_mass"] for l in layers])      # L x H x Q
    self_ = np.array([d[l]["self_mass"] for l in layers])
    lag = np.array([d[l]["mean_lag"] for l in layers])
    uni = np.array(d[layers[0]]["uniform_first_self"])          # Q: uniform mass on first token = on self = 1/(q+1)
    return q, first, self_, lag, uni


def main():
    runs = {"d8": "a4-d8-lr0.003", "d8-no-prenorm": "a4-d8-no-prenorm-lr0.001", "d8-gated-attention": "a4-d8-gated-attention-lr0.006",
            "d8-postnorm": "a4-d8-postnorm-lr0.001"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    q, first, self_, lag, uni = attention(runs["d8"])
    for l in range(first.shape[0]):
        for h in range(first.shape[1]):
            axes[0].plot(q, first[l, h], color=plt.cm.viridis(l / 7), alpha=.35, lw=.8)
    axes[0].plot(q, first.mean((0, 1)), "k", lw=2.5, label="d8 mean over layers and heads")
    axes[0].plot(q, uni, "k--", lw=1.5, label="uniform causal baseline")
    axes[0].set_title("mass on the first token (thin: each of 64 heads, dark = deeper layer)")
    axes[0].set_ylabel("attention mass"); axes[0].legend(fontsize=8)
    for name, run in runs.items():
        q, f, s, lg, u = attention(run)
        axes[1].plot(q, s.mean((0, 1)), label=name)
        axes[2].plot(q, lg.mean((0, 1)), label=name)
    axes[1].plot(q, uni, "k--", label="uniform causal baseline")
    axes[1].set_title("mass on the current token (layer/head mean)"); axes[1].set_yscale("log"); axes[1].legend(fontsize=8)
    axes[2].plot(q, q / 2, "k--", label="uniform causal baseline")
    axes[2].set_title("mean lag to attended token (layer/head mean)"); axes[2].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel("query position")
    fig.tight_layout(); fig.savefig(FIG / "p1_attention.png", dpi=150)
    # table at query position ~512
    print("variant              first@q512  self@q512  lag@q512 | uniform first/self/lag")
    for run in sorted(RES.iterdir()):
        if "d8" not in run.name or not (run / "final.json").exists():
            continue
        q, f, s, lg, u = attention(run.name)
        i = int(np.argmin(np.abs(q - 512)))
        print(f"{run.name.removeprefix('a4-').rsplit('-lr',1)[0]:<20} {f[:, :, i].mean():9.3f} {s[:, :, i].mean():10.3f} {lg[:, :, i].mean():9.1f} | "
              f"{u[i]:.4f}/{u[i]:.4f}/{q[i]/2:.0f}")
    print("per-layer first-token mass at q512 for d8:", f"{[round(float(x), 2) for x in attention(runs['d8'])[1][:, :, i].mean(1)]}")
    print("per-head first-token mass at q512, d8 layer 7:", [round(float(x), 2) for x in attention(runs['d8'])[1][7, :, i]])


if __name__ == "__main__":
    main()
