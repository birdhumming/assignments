"""P1/P2 diagnostics tables + figures from results/a4_stability/<run>/<stage>.json (measure_supplied output).

Prints per-variant depth summaries (activation RMS growth, gradient ratio top/bottom, last-token gradient reach),
attention-sink summaries (first/self mass vs uniform baseline), RNN temporal summaries (hidden grad reach,
saturation) and context-removal losses; writes figures/p1_depth.png, p1_attention.png, p2_time.png, p12_context.png.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = Path("results/a4_stability")
FIG = ROOT / "figures"
STAGES = ("init", "early", "final")


def load_all():
    out = {}
    for run in sorted(RES.iterdir()):
        stages = {s: json.loads((run / f"{s}.json").read_text()) for s in STAGES if (run / f"{s}.json").exists()}
        if stages:
            out[run.name] = stages
    return out


def variant(run):
    return run.rsplit("-lr", 1)[0].removeprefix("a4-")


def depth_summary(stats):
    d = stats["full"]["depth"]
    act = [r["activation_rms"] for r in d]
    grad = [r["gradient_rms"] for r in d]
    last = [r["gradient_rms"] for r in stats["last"]["depth"]]
    cos = [r.get("adjacent_cosine") for r in d if r.get("adjacent_cosine") is not None]
    return dict(act_in=act[1], act_out=act[-1], act_growth=act[-1] / max(act[1], 1e-12),
                grad_ratio=grad[1] / max(grad[-1], 1e-30), last_grad_ratio=last[1] / max(last[-1], 1e-30),
                cos_mean=float(np.mean(cos)) if cos else float("nan"))


def attention_summary(stats):
    rows = []
    for name, a in stats.get("attention", {}).items():
        first, self_m, uni = np.array(a["first_mass"]), np.array(a["self_mass"]), np.array(a["uniform_first_self"])
        rows.append((name, float(first.mean()), float(self_m.mean()), float(uni[..., 0].mean()) if uni.ndim else float(uni.mean()), float(np.mean(a["mean_lag"]))))
    return rows


def temporal_summary(stats):
    hg = np.array(stats["last"]["hidden_gradient_rms"])  # layer x t
    sat = np.array(stats["full"]["fraction_abs_gt_095"])
    hr = np.array(stats["full"]["hidden_rms"])
    T = hg.shape[1]
    def steps_back(g):
        below = np.where(g[::-1] < g[-1] * 1e-3)[0]
        return int(below[0]) if len(below) else int(T)
    reach = [steps_back(hg[l]) if hg[l][-1] > 0 else 0 for l in range(hg.shape[0])]
    return dict(reach_1e3=reach, sat_mean=sat.mean(1).round(3).tolist(), hidden_rms_end=hr[:, -1].round(2).tolist(),
                hidden_rms_start=hr[:, 1].round(2).tolist())


def main():
    data = load_all()
    print("== depth (final): act_in -> act_out (growth), grad ratio bottom/top (L), (L_T), adjacent cosine")
    for run, stages in data.items():
        if "final" not in stages:
            continue
        s = depth_summary(stages["final"])
        print(f"{variant(run):<20} {s['act_in']:6.2f} -> {s['act_out']:7.2f} (x{s['act_growth']:.1f})  {s['grad_ratio']:8.1f} {s['last_grad_ratio']:8.1f}  cos {s['cos_mean']:.3f}  val {stages['final']['full']['loss']:.3f}")
    print("\n== attention (final, mean over layers): first-token mass, self mass, uniform-first baseline, mean lag")
    for run, stages in data.items():
        if "final" in stages and stages["final"].get("attention"):
            rows = attention_summary(stages["final"])
            m = np.array([r[1:] for r in rows]).mean(0)
            print(f"{variant(run):<20} first {m[0]:.3f} self {m[1]:.3f} uniform-first {m[2]:.3f} lag {m[3]:.1f}  per-layer first {[round(r[1],2) for r in rows]}")
    print("\n== RNN time (final): last-token grad reach (steps back to 1e-3 of final), saturation, hidden rms start->end")
    for run, stages in data.items():
        if "final" in stages and stages["final"]["full"]["hidden_rms"]:
            t = temporal_summary(stages["final"])
            print(f"{variant(run):<20} reach {t['reach_1e3']} sat {t['sat_mean']} rms {t['hidden_rms_start']}->{t['hidden_rms_end']}")
    print("\n== context removal (final): loss increase when keeping K recent tokens (BOS dropped / kept)")
    for run, stages in data.items():
        if "final" in stages:
            c = stages["final"]["context_loss_increase"]
            print(f"{variant(run):<20} " + " ".join(f"K{k}:{c[f'{k}_first_False']:+.2f}/{c[f'{k}_first_True']:+.2f}" for k in (1, 4, 16, 64, 256, 512)))
    FIG.mkdir(exist_ok=True)
    # figures
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for run, stages in data.items():
        v = variant(run)
        if not v.startswith("d8") or "final" not in stages:
            continue
        d = stages["final"]["full"]["depth"]
        axes[0].plot([r["activation_rms"] for r in d], label=v, lw=2 if v == "d8" else 1)
        axes[1].plot([r["gradient_rms"] for r in d], label=v, lw=2 if v == "d8" else 1)
        axes[2].plot([r["gradient_rms"] for r in stages["final"]["last"]["depth"]], label=v, lw=2 if v == "d8" else 1)
    for ax, t in zip(axes, ("activation RMS", "gradient RMS, full loss", "gradient RMS, last-token loss")):
        ax.set_title(t); ax.set_xlabel("depth step (0 = embedding, odd = after attention)"); ax.set_yscale("log")
    axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "p1_depth.png", dpi=150)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for run, stages in data.items():
        v = variant(run)
        if not v.startswith("rnn") or "final" not in stages:
            continue
        s = stages["final"]
        lw = 2 if v == "rnn" else 1
        axes[0].plot(np.array(s["full"]["hidden_gradient_rms"])[-1], label=v, lw=lw)
        axes[1].plot(np.array(s["last"]["hidden_gradient_rms"])[-1], label=v, lw=lw)
        axes[2].plot(np.array(s["full"]["hidden_rms"])[-1], label=v, lw=lw)
        axes[3].plot(np.array(s["full"]["fraction_abs_gt_095"])[-1], label=v, lw=lw)
    axes[0].set_yscale("log"); axes[0].set_title("top-layer hidden-gradient RMS, full loss L")
    axes[1].set_yscale("log"); axes[1].set_title("top-layer hidden-gradient RMS, last-token loss L_T")
    axes[2].set_title("top-layer hidden-state RMS"); axes[3].set_title("top-layer tanh saturation Pr(|h|>0.95)")
    for ax in axes:
        ax.set_xlabel("position t")
    axes[2].legend(fontsize=7, ncol=2, loc="lower center")
    fig.tight_layout(); fig.savefig(FIG / "p2_time.png", dpi=150)
    fig, ax = plt.subplots(figsize=(6, 4))
    for run, stages in data.items():
        v = variant(run)
        if v in ("d8", "rnn", "rnn1", "rnn4", "d8-gated-residual", "rnn-gated-residual") and "final" in stages:
            c = stages["final"]["context_loss_increase"]
            ks = (1, 4, 16, 64, 256, 512)
            ax.plot(ks, [c[f"{k}_first_True"] for k in ks], marker="o", label=f"{v} (BOS kept)")
    ax.set_xscale("log"); ax.set_xlabel("recent tokens kept"); ax.set_ylabel("last-token loss increase vs full context"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "p12_context.png", dpi=150)
    print("wrote figures")


if __name__ == "__main__":
    main()
