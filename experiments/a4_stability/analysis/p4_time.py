"""P4 temporal diagnostics figure: tanh RNN (supplied) vs GRU / ablations / NormGate (ours).

    uv run python -m experiments.a4_stability.analysis.p4_time a4-rnn-lr0.002 a4-gru-lr0.003 a4-normgate-lr0.003 ...

Reads results/a4_stability/<run>/final.json, writes figures/p4_time.png (top layer: last-token hidden-gradient RMS,
hidden-state RMS, saturation vs position) and prints per-layer last-token gradient reach and mean saturation.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .p12_diagnostics import temporal_summary

ROOT = Path(__file__).resolve().parents[1]
RES = Path("results/a4_stability")
FIG = ROOT / "figures"


def label(run):
    name = run[3:].rsplit("-lr", 1)[0]
    return "tanh RNN (supplied)" if name == "rnn" else name


def main(runs, out="p4_time.png"):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    print(f"{'run':30s} reach(1e-3) per layer        mean sat per layer")
    for run in runs:
        path = RES / run / "final.json"
        if not path.exists():
            print(run, "missing")
            continue
        s = json.loads(path.read_text())
        t = temporal_summary(s)
        print(f"{run:30s} {t['reach_1e3']!s:26s} {t['sat_mean']}")
        lw = 2 if run.startswith("a4-rnn-") else 1
        axes[0].plot(np.array(s["last"]["hidden_gradient_rms"])[-1], label=label(run), lw=lw)
        axes[1].plot(np.array(s["full"]["hidden_rms"])[-1], label=label(run), lw=lw)
        axes[2].plot(np.array(s["full"]["fraction_abs_gt_095"])[-1], label=label(run), lw=lw)
    axes[0].set_yscale("log"); axes[0].set_title("top-layer hidden-gradient RMS, last-token loss")
    axes[1].set_title("top-layer hidden-state RMS"); axes[2].set_title("top-layer saturation Pr(|h|>0.95)")
    for ax in axes:
        ax.set_xlabel("position t"); ax.grid(alpha=.3)
    axes[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / out, dpi=150)
    print("wrote", FIG / out)


if __name__ == "__main__":
    main(sys.argv[1:])
