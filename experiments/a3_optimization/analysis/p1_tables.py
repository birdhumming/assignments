"""P1 tables and figures from results/a3-mode-connectivity*.json.

Barrier B(x, y) = L(0.5 average) - mean(L(branch 1), L(branch 2)).
Averaging "works" when B <= 0.01 (handout criterion).
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.a3_optimization.p1_mode_connectivity import (
    BRANCH_STEPS,
    PREFIX_STEPS,
    VARIANTS,
    results_path,
)

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
THRESHOLD = 0.01


def load(variant):
    path = results_path(variant)
    if not path.exists():
        return None
    rows = json.loads(path.read_text())
    cells = {}
    for row in rows:
        m = re.search(r"fork(\d+)", row["run_a"])
        x = int(m.group(1)) if m else 0
        y = row["step_a"] - x
        L = row["interpolation"]
        endpoints = (L[0], L[-1])
        mid = L[len(L) // 2]
        cells[(x, y)] = {
            "end_a": endpoints[0], "end_b": endpoints[1],
            "endpoint_mean": sum(endpoints) / 2,
            "average": mid,
            "ensemble": row.get("ensemble"),
            "barrier": mid - sum(endpoints) / 2,
            "curve": L, "alphas": row["alphas"],
        }
    return cells


def grid(cells, key, fmt="{:+.3f}"):
    head = "x \\ y | " + " | ".join(f"{y:>6}" for y in BRANCH_STEPS)
    lines = [head, "-" * len(head)]
    for x in PREFIX_STEPS:
        vals = []
        for y in BRANCH_STEPS:
            c = cells.get((x, y))
            vals.append(fmt.format(c[key]) if c else "   -  ")
        lines.append(f"{x:>5} | " + " | ".join(f"{v:>6}" for v in vals))
    return "\n".join(lines)


def report(variant, cells):
    print(f"\n=== {variant} ({len(cells)} cells) ===")
    print("barrier B(x,y) = L(average) - mean endpoint loss (<= 0.01 means averaging works)")
    print(grid(cells, "barrier"))
    print("\nendpoint mean loss")
    print(grid(cells, "endpoint_mean", "{:.3f}"))
    print("\nweight-average loss")
    print(grid(cells, "average", "{:.3f}"))
    print("\nensemble loss")
    print(grid(cells, "ensemble", "{:.3f}"))
    print("\nensemble gain = endpoint mean - ensemble")
    for c in cells.values():
        c["ens_gain"] = c["endpoint_mean"] - c["ensemble"]
        c["avg_gain"] = c["endpoint_mean"] - c["average"]
    print(grid(cells, "ens_gain"))
    print("\naverage gain = endpoint mean - average (positive: averaging beats either endpoint)")
    print(grid(cells, "avg_gain"))
    ok = sorted((x, y) for (x, y), c in cells.items() if c["barrier"] <= THRESHOLD)
    print(f"\ncells with B <= {THRESHOLD}: {ok}")


def figures(variant, cells):
    tag = "" if variant == "default" else f"-{variant}"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for y in BRANCH_STEPS:
        xs = [x for x in PREFIX_STEPS if (x, y) in cells]
        axes[0].plot(xs, [cells[(x, y)]["barrier"] for x in xs], marker="o", label=f"y={y}")
    axes[0].axhline(THRESHOLD, ls="--", c="k", lw=1)
    axes[0].set_xlabel("shared prefix steps x"); axes[0].set_ylabel("barrier B(x, y)")
    axes[0].set_yscale("symlog", linthresh=0.01); axes[0].legend(fontsize=8)
    axes[0].set_title(f"Midpoint barrier, {variant}")
    for x in PREFIX_STEPS:
        c = cells.get((x, max(BRANCH_STEPS)))
        if c:
            axes[1].plot(c["alphas"], c["curve"], marker=".", label=f"x={x}")
    axes[1].set_xlabel("alpha (0 = branch 1, 1 = branch 2)"); axes[1].set_ylabel("validation loss")
    axes[1].set_title(f"Interpolation at y={max(BRANCH_STEPS)}"); axes[1].legend(fontsize=8)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / f"p1_barrier{tag}.png"
    fig.savefig(out, dpi=130)
    print("wrote", out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="*", default=sorted(VARIANTS))
    args = parser.parse_args()
    for variant in args.variants:
        cells = load(variant)
        if cells is None:
            print(f"{variant}: no results yet ({results_path(variant).name})")
            continue
        report(variant, cells)
        figures(variant, cells)


if __name__ == "__main__":
    main()
