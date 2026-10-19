"""P4(c): summarise the projected-update continuation runs (c-*.jsonl) and plot them."""

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "a3-inside-the-hessian"
FIG = ROOT / "figures" / "p4c_subspace.png"

ORDER = ["c-full", "c-top", "c-removed", "c-top-k50", "c-removed-k50", "c-full-lr0.3",
         "c-removed-lr0.3", "c-removed-refresh5", "c-full-step305", "c-removed-step305"]


def load(name):
    rows = [json.loads(line) for line in (RESULTS / f"{name}.jsonl").open()]
    start = rows[0]
    steps = [r for r in rows if r["kind"] == "step"]
    refresh = [r for r in rows if r["kind"] == "refresh"]
    vals = [r["val_loss"] for r in rows if r.get("val_loss") is not None]
    fractions = [r["top_fraction"] for r in steps if r.get("top_fraction") is not None]
    return {
        "name": name, "start": start, "steps": steps, "refresh": refresh,
        "val_start": start["val_loss"], "val_end": vals[-1],
        "train_last20": statistics.mean(r["loss"] for r in steps[-20:]),
        "frac_mean": statistics.mean(fractions), "frac_last": fractions[-1],
        "lmax_start": refresh[0]["lambda_max"], "lmax_end": refresh[-1]["lambda_max"],
    }


def main():
    runs = {n: load(n) for n in ORDER}
    print(f"{'job':20s} {'arm':8s} start  k  refresh lr  | val start -> end  dval  | train last20 | top frac mean/last | lmax start->end")
    for r in runs.values():
        s = r["start"]
        print(f"{r['name']:20s} {r['name'].split('-')[1]:8s} {s['step']:5d} {s['k']:3d} {s['refresh']:3d} {s['lr_scale']:.1f} | "
              f"{r['val_start']:.3f} -> {r['val_end']:.3f} {r['val_end'] - r['val_start']:+.3f} | {r['train_last20']:.3f} | "
              f"{r['frac_mean']:.3f}/{r['frac_last']:.3f} | {r['lmax_start']:.2f}->{r['lmax_end']:.2f}")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for name, color in (("c-full", "k"), ("c-removed", "C0"), ("c-top", "C3"), ("c-top-k50", "C1"), ("c-removed-k50", "C2")):
        r = runs[name]
        x = [s["step"] for s in r["steps"]]
        axes[0].plot(x, [s["loss"] for s in r["steps"]], color=color, lw=1, label=name[2:])
        axes[1].plot(x, [s["top_fraction"] for s in r["steps"]], color=color, lw=1, label=name[2:])
        axes[2].plot([q["step"] for q in r["refresh"]], [q["lambda_max"] for q in r["refresh"]], color=color, marker=".", lw=1, label=name[2:])
    axes[0].set_ylabel("training loss")
    axes[0].set_ylim(3.4, 4.2)
    axes[1].set_ylabel("fraction of gradient norm$^2$ in top-k subspace")
    axes[1].set_yscale("log")
    axes[2].set_ylabel(r"$\lambda_{\max}$ at refresh")
    axes[2].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("optimizer step (continued from 1526)")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[0].set_title("SGD+momentum continuation, 300 updates")
    fig.tight_layout()
    fig.savefig(FIG, dpi=130)
    print("wrote", FIG)


if __name__ == "__main__":
    main()
