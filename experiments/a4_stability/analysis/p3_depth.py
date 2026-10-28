"""P3 depth diagnostics figure for a set of runs (supplied additive baseline + our residual variants).

    uv run python -m experiments.a4_stability.analysis.p3_depth d8 a4-d8-lr0.003 a4-d8-attention-lr0.003 ...

Reads results/a4_stability/<run>/final.json (measure_supplied / measure_own output) and writes
figures/p3_<family>_depth.png: activation RMS, gradient RMS (full-loss and last-token loss),
branch/input RMS ratio and adjacent-depth cosine, all vs. depth-stream index (0 = embedding).
Also prints a compact table and writes figures/p3_<family>_routing.png when routing weights exist.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = Path("results/a4_stability")
FIG = ROOT / "figures"


def label(run):
    name = run[3:].rsplit("-lr", 1)[0]
    return {"d8": "additive (supplied)", "rnn": "additive (supplied)"}.get(name, name.split("-", 1)[-1])


def series(stats, key, objective="full"):
    return [row.get(key, np.nan) for row in stats[objective]["depth"]]


def main(family, runs, stage="final"):
    data = {r: json.loads((RES / r / f"{stage}.json").read_text()) for r in runs if (RES / r / f"{stage}.json").exists()}
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))
    panels = [("activation_rms", "full", "activation RMS", True), ("gradient_rms", "full", "gradient RMS (full loss, solid; last token, dashed)", True),
              ("raw_branch_input_ratio", "full", "branch output RMS / stream RMS", True), ("adjacent_cosine", "full", "cosine(stream$_{j-1}$, stream$_j$)", False)]
    print(f"{'run':34s} act RMS 0->L   grad bot/top  mean ratio  mean cos")
    for run, stats in data.items():
        act, grad, ratio, cos = (series(stats, k) for k in ("activation_rms", "gradient_rms", "raw_branch_input_ratio", "adjacent_cosine"))
        print(f"{run:34s} {act[0]:6.2f}->{act[-1]:6.2f}  {grad[0]/max(grad[-1],1e-30):8.2f}  {np.nanmean(ratio):8.3f}  {np.nanmean(cos):7.3f}")
        for ax, (key, obj, _, _) in zip(axes.flat, panels):
            y = series(stats, key, obj)
            line, = ax.plot(range(len(y)), y, marker=".", label=label(run))
            if key == "gradient_rms":
                ax.plot(range(len(y)), series(stats, key, "last"), ls="--", color=line.get_color())
    for ax, (_, _, title, log) in zip(axes.flat, panels):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("depth-stream index (0 = embedding)")
        if log:
            ax.set_yscale("log")
        ax.grid(alpha=.3)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"p3_{family}_depth.png", dpi=150)
    routed = {r: s["routing"] for r, s in data.items() if s.get("routing")}
    if routed:
        fig, axes = plt.subplots(1, len(routed), figsize=(4*len(routed), 4), squeeze=False)
        for ax, (run, routing) in zip(axes.flat, routed.items()):
            names = list(routing)
            n = max(len(np.ravel(routing[k])) for k in names)
            mat = np.full((len(names), n), np.nan)
            for i, k in enumerate(names):
                w = np.ravel(routing[k])
                mat[i, :len(w)] = w
            im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0)
            ax.set_title(label(run), fontsize=9)
            ax.set_xlabel("source (0 = embedding, then branch/block outputs)")
            ax.set_ylabel("router (branch input; last = output read)")
            fig.colorbar(im, ax=ax, fraction=.04)
        fig.tight_layout()
        fig.savefig(FIG / f"p3_{family}_routing.png", dpi=150)
    print("wrote", FIG / f"p3_{family}_depth.png", "(+routing)" if routed else "")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
