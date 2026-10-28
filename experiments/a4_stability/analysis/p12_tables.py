"""P1/P2 tables and LR-curve figures from the supplied manifest (results/manifest.json)."""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results" / "manifest.json"
FIG = ROOT / "figures"
BASELINE = {1: "d8", 2: "rnn"}
DIVERGED = 6.874  # uniform-over-4096 loss: the run never trained


def load():
    runs = json.loads(MANIFEST.read_text())["runs"]
    by = defaultdict(dict)
    for r in runs:
        loss = r.get("validation_loss")
        if r["status"] != "complete" or loss is None or loss > DIVERGED:
            loss = None
        by[(r["problem"], r["variant"])][r["lr"]] = (loss, r["status"])
    return by


def best(cells):
    finite = [(l, lr) for lr, (l, _) in cells.items() if l is not None]
    return min(finite) if finite else (None, None)


def main():
    by = load()
    print(f"{'variant':<22} {'best':>7} {'lr':>8} {'vs base':>8}  diverged/nonfinite at")
    base = {p: best(by[(p, v)])[0] for p, v in BASELINE.items()}
    for (p, v) in sorted(by):
        loss, lr = best(by[(p, v)])
        bad = [f"{lr:g}" for lr, (l, s) in sorted(by[(p, v)].items()) if l is None]
        delta = f"{loss - base[p]:+.3f}" if loss is not None else "  --"
        shown = ("--", "--") if loss is None else (f"{loss:.4f}", f"{lr:g}")
        print(f"P{p} {v:<19} {shown[0]:>7} {shown[1]:>8} {delta:>8}  {' '.join(bad)}")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, p in zip(axes, (1, 2)):
        for (pp, v), cells in sorted(by.items()):
            if pp != p:
                continue
            lrs = sorted(cells)
            ys = [cells[lr][0] if cells[lr][0] is not None else float("nan") for lr in lrs]
            ax.plot(lrs, ys, marker="o", lw=2 if v == BASELINE[p] else 1, label=v)
        ax.set_xscale("log")
        ax.set_xlabel("peak learning rate")
        ax.set_ylabel("final validation loss")
        ax.set_title("P1 transformer variants" if p == 1 else "P2 RNN variants")
        ax.set_ylim(2.85, 3.5 if p == 1 else 4.5)
        ax.legend(fontsize=7, ncol=2)
    FIG.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / "p12_lr_curves.png", dpi=150)
    print("wrote", FIG / "p12_lr_curves.png")


if __name__ == "__main__":
    main()
