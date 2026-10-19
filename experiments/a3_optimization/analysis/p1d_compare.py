"""P1(d): compare the connectivity grid across training-recipe variants.

Prints one row per variant (cells passing B <= 0.01, failing cells, barrier at the
hardest cell, median averaging gain where averaging works) and draws
figures/p1d_variants.png: barrier against prefix length for y=250 and y=500.
"""

import argparse
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.a3_optimization.analysis.p1_tables import FIG_DIR, THRESHOLD, load
from experiments.a3_optimization.p1_mode_connectivity import PREFIX_STEPS, VARIANTS

ORDER = ["default", "lr1.5e-3", "lr6e-3", "bs256", "linear"]
LABEL = {"default": "default (lr 3e-3, bs 64, constant)", "lr1.5e-3": "lr 1.5e-3", "lr6e-3": "lr 6e-3",
         "bs256": "batch 256", "linear": "linear decay"}


def summary(cells):
    fails = sorted((x, y) for (x, y), c in cells.items() if c["barrier"] > THRESHOLD)
    gains = [c["endpoint_mean"] - c["average"] for (x, y), c in cells.items() if c["barrier"] <= THRESHOLD]
    worst = max(cells.values(), key=lambda c: c["barrier"])
    first_ok = min(x for x in PREFIX_STEPS if all(cells[(x, y)]["barrier"] <= THRESHOLD for (xx, y) in cells if xx == x))
    return {"pass": len(cells) - len(fails), "fails": fails, "worst": worst["barrier"],
            "gain": statistics.median(gains), "first_ok": first_ok,
            "end_1000_500": cells[(1000, 500)]["endpoint_mean"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="*", default=ORDER)
    args = parser.parse_args()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    print(f"{'variant':<12} {'pass':>5} {'first x all pass':>16} {'worst B':>8} {'median gain':>11} {'L(1000,500)':>11}  failing cells")
    for variant in args.variants:
        if variant not in VARIANTS:
            continue
        cells = load(variant)
        if cells is None:
            print(f"{variant:<12} (no results yet)")
            continue
        s = summary(cells)
        print(f"{variant:<12} {s['pass']:>3}/30 {s['first_ok']:>16} {s['worst']:>8.3f} {s['gain']:>11.3f} {s['end_1000_500']:>11.3f}  {s['fails']}")
        for ax, y in zip(axes, (250, 500)):
            ax.plot(PREFIX_STEPS, [cells[(x, y)]["barrier"] for x in PREFIX_STEPS], marker="o", label=LABEL[variant])
            ax.set_title(f"branch length y = {y}")
    for ax in axes:
        ax.axhline(THRESHOLD, color="k", ls="--", lw=0.8)
        ax.set_yscale("symlog", linthresh=0.05)
        ax.set_xscale("symlog", linthresh=50)
        ax.set_xlabel("prefix length x (updates)")
        ax.set_xticks(PREFIX_STEPS)
        ax.set_xticklabels([str(x) for x in PREFIX_STEPS])
    axes[0].set_ylabel("midpoint barrier B(x, y)")
    axes[1].legend(fontsize=8)
    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / "p1d_variants.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main()
