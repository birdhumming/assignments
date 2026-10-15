"""P3 tables and figures: eta * lambda_max over training for every finished run.

Full-batch gradient descent: threshold 2 (raw Hessian). Minibatch AdamW: threshold 38
(preconditioned Hessian, beta_1 = 0.9). Uses the course's sharpness.history/summarize.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.a3_optimization.p3_edge_of_stability import (
    EXPLORATION_RUNS,
    FULL_BATCH_RUNS,
    MINIBATCH_RUNS,
)
from experiments.a3_optimization.sharpness import (
    THRESHOLDS,
    history,
    summarize,
    window_median,
)
from train import training_run_name

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
CACHE = Path(__file__).resolve().parents[1] / "results" / "p3_histories.json"


def label(config):
    name = training_run_name(config)
    return name.replace("model-d8-", "").replace("-epochs1500.0-tok32.8K-wd0.0-emb16-nogradclip-b10.0-sgd", "")


def is_full_batch(config):
    return config.optimizer_name == "sgd"


def load_all(configs, refresh=False):
    cache = json.loads(CACHE.read_text()) if CACHE.exists() and not refresh else {}
    out = {}
    for config in configs:
        name = training_run_name(config)
        if name not in cache:
            try:
                rows = history(config)
            except Exception as e:  # noqa: BLE001 - unfinished or missing run
                print(f"skip {label(config)}: {type(e).__name__}: {e}")
                continue
            cache[name] = {"rows": list(rows), "losses": list(getattr(rows, "losses", []))}
        out[name] = cache[name]
    CACHE.parent.mkdir(exist_ok=True)
    CACHE.write_text(json.dumps(cache))
    return out


def as_history(entry):
    from experiments.a3_optimization.sharpness import History

    rows = History(entry["rows"])
    rows.losses = [tuple(x) for x in entry["losses"]]
    return rows


def fmt(v, nd=2):
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def full_batch_table(configs, data):
    print("\n=== full-batch gradient descent (threshold eta*lambda = 2) ===")
    print(f"{'run':<34} {'eta*lam@0':>9} {'cross':>6} {'plateau':>8} {'exit':>6} {'up<':>5} {'up>':>5} {'final loss':>10}")
    for config in configs:
        name = training_run_name(config)
        if name not in data:
            continue
        rows = as_history(data[name])
        s = summarize(rows)
        final = rows.losses[-1][1] if rows.losses else None
        print(f"{label(config):<34} {fmt(s['initial']):>9} {fmt(s['crossing']):>6} {fmt(s['plateau']):>8} "
              f"{fmt(s.get('exit')):>6} {fmt(s.get('loss_up_before')):>5} {fmt(s.get('loss_up_after')):>5} {fmt(final, 3):>10}")


def minibatch_table(configs, data):
    print("\n=== minibatch AdamW (threshold eta*lambda_pre = 38) ===")
    print(f"{'run':<58} {'median 100-180M':>15} {'final':>8} {'raw eta*lam final':>17} {'final loss':>10}")
    for config in configs:
        name = training_run_name(config)
        if name not in data:
            continue
        rows = as_history(data[name])
        pre = [r for r in rows if r.get("eta_lambda_pre") is not None]
        raw = [r for r in rows if r.get("eta_lambda") is not None]
        med = window_median(rows, config, 100e6, 180e6)
        final = rows.losses[-1][1] if rows.losses else None
        print(f"{label(config):<58} {fmt(med):>15} {fmt(pre[-1]['eta_lambda_pre'] if pre else None):>8} "
              f"{fmt(raw[-1]['eta_lambda'] if raw else None):>17} {fmt(final, 3):>10}")


def plot(configs, data, key, threshold, title, out, xkey="step"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for config in configs:
        name = training_run_name(config)
        if name not in data:
            continue
        rows = as_history(data[name])
        pts = [(r[xkey], r[key]) for r in rows if r.get(key) is not None and r[key] > 0]
        if not pts:
            continue
        axes[0].plot(*zip(*pts), marker=".", ms=3, lw=1, label=label(config))
        if rows.losses:
            axes[1].plot(*zip(*rows.losses), lw=1, label=label(config))
    axes[0].axhline(threshold, ls="--", c="k", lw=1, label=f"threshold {threshold:g}")
    axes[0].set_yscale("log"); axes[0].set_xlabel("update"); axes[0].set_ylabel(key.replace("_", " "))
    axes[0].set_title(title); axes[0].legend(fontsize=7)
    axes[1].set_xlabel("update"); axes[1].set_ylabel("training loss"); axes[1].set_yscale("log")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / out, dpi=130)
    print("wrote", FIG_DIR / out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="re-pull every history from W&B")
    args = parser.parse_args()
    all_runs = FULL_BATCH_RUNS + MINIBATCH_RUNS + EXPLORATION_RUNS
    data = load_all(all_runs, refresh=args.refresh)
    print(f"{len(data)}/{len(all_runs)} runs with histories")
    fb = [c for c in all_runs if is_full_batch(c)]
    mb = [c for c in all_runs if not is_full_batch(c)]
    full_batch_table(fb, data)
    minibatch_table(mb, data)
    plot(FULL_BATCH_RUNS, data, "eta_lambda", THRESHOLDS["raw"], "Full-batch GD: eta * lambda_max", "p3_fullbatch_core.png")
    plot(fb, data, "eta_lambda", THRESHOLDS["raw"], "All full-batch runs", "p3_fullbatch_all.png")
    plot(mb, data, "eta_lambda_pre", THRESHOLDS["pre"], "Minibatch AdamW: eta * lambda_max (preconditioned)",
         "p3_minibatch.png")


if __name__ == "__main__":
    main()
