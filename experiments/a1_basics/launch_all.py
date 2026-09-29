"""Launch every a1_basics run in one Modal app (priority order, 2 GPUs).

Identical configs shared between problems are deduplicated by run name, so the
d8 baseline etc. train once. `--dry-run` prints the manifest without launching.
"""

import argparse
import csv
import sys
from pathlib import Path

from experiments.a1_basics import (
    p1_hyperparameters,
    p2_scaling_law_reliability,
    p3_measuring_variation,
    p4_amplification,
    p5_loss_curve_augury,
    p6_activation_gradient_norms,
    p7_own_prediction_problem,
)
from experiments.a1_basics.common import dedupe
from modal_train import launch_training_jobs
from train import checked_train_config, training_run_name


# Priority order: earlier problems keep their runs when --budget-hours bites.
# P3/P4 first (noise floor is needed to read every other result), then the
# scaling ladder, then the P1 gap-fill (21 P1 runs already exist), then P6/P7.
PROBLEMS = [
    ("p3", p3_measuring_variation.RUNS),
    ("p4", p4_amplification.RUNS),
    ("p2", p2_scaling_law_reliability.RUNS),
    ("p1", p1_hyperparameters.RUNS),
    ("p5", p5_loss_curve_augury.RUNS),
    ("p6", p6_activation_gradient_norms.RUNS),
    ("p7", p7_own_prediction_problem.RUNS),
]
A100_RUNS = p3_measuring_variation.A100_RUNS

MANIFEST_PATH = Path(__file__).resolve().parent / "run_manifest.csv"


def depth_of(config):
    if config.model_config is not None:
        return int(config.model_config.name.lstrip("d"))
    return int(config.model_name.lstrip("d"))


# Measured from the finished a1 d8 runs in W&B: 10.3-12.6 min wall clock at the
# 614M-token default on H100 (the older lr_tuning runs took ~25 min, but those
# were logged under heavier eval settings). Depth costs are relative to this.
D8_MINUTES = 11.0
DEPTH_COST = {4: 0.30, 5: 0.40, 6: 0.50, 7: 0.75, 8: 1.00, 9: 1.40}
A100_SLOWDOWN = 2.0


def estimated_minutes(config, gpu="H100"):
    tokens_scale = (config.num_train_sequences * float(config.num_epochs)) / 600_000
    minutes = D8_MINUTES * DEPTH_COST[depth_of(config)] * tokens_scale
    return minutes * (A100_SLOWDOWN if gpu == "A100" else 1.0)


def build_manifest():
    rows, all_configs, problems_by_name = [], [], {}
    for problem, runs in PROBLEMS:
        for config in runs:
            checked_train_config(config)
            name = training_run_name(config)
            problems_by_name.setdefault(name, []).append(problem)
            all_configs.append(config)
    unique = dedupe(all_configs)
    for config in unique:
        name = training_run_name(config)
        rows.append(
            {
                "run_name": name,
                "problems": "+".join(problems_by_name[name]),
                "tags": ",".join(config.wandb_tags),
                "gpu": "H100",
                "est_minutes": f"{estimated_minutes(config):.1f}",
            }
        )
    for config in A100_RUNS:
        rows.append(
            {
                "run_name": training_run_name(config),
                "problems": "p3",
                "tags": ",".join(config.wandb_tags),
                "gpu": "A100",
                "est_minutes": f"{estimated_minutes(config, gpu='A100'):.1f}",
            }
        )
    return unique, rows


def within_budget(rows, budget_hours):
    """Keep runs in priority order until the estimated budget is exhausted.

    Rows arrive in PROBLEMS order, so earlier problems keep their runs and the
    tail of the cheapest-to-lose problems is what gets dropped.
    """
    if budget_hours is None:
        return rows, []
    kept, dropped, spent = [], [], 0.0
    for row in rows:
        minutes = float(row["est_minutes"])
        if spent + minutes > budget_hours * 60:
            dropped.append(row)
            continue
        kept.append(row)
        spent += minutes
    return kept, dropped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-parallel-runs", type=int, default=2)
    parser.add_argument(
        "--budget-hours",
        type=float,
        default=None,
        help="Drop the lowest-priority runs so the estimate stays under this many GPU-hours.",
    )
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        default=None,
        help="Skip run names listed in this earlier manifest CSV (already submitted).",
    )
    args = parser.parse_args()

    unique, rows = build_manifest()
    if args.exclude_manifest is not None:
        with args.exclude_manifest.open(newline="") as f:
            already = {row["run_name"] for row in csv.DictReader(f)}
        rows = [row for row in rows if row["run_name"] not in already]
    rows, dropped = within_budget(rows, args.budget_hours)
    keep_names = {row["run_name"] for row in rows}
    unique = [c for c in unique if training_run_name(c) in keep_names]
    a100_runs = [c for c in A100_RUNS if training_run_name(c) in keep_names]
    with MANIFEST_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    total_minutes = sum(float(r["est_minutes"]) for r in rows)
    print(f"{len(rows)} unique runs, ~{total_minutes / 60:.1f} GPU-hours estimated.")
    if dropped:
        dropped_hours = sum(float(r["est_minutes"]) for r in dropped) / 60
        print(f"{len(dropped)} runs dropped to fit the budget (~{dropped_hours:.1f} GPU-hours).")
    print(f"Manifest written to {MANIFEST_PATH}")
    if args.dry_run:
        for row in rows:
            print(f"  [{row['problems']:>8}] {row['gpu']} {row['est_minutes']:>5}m {row['run_name']}")
        return 0

    launch_training_jobs(unique, max_parallel_runs=args.max_parallel_runs)
    if a100_runs:
        launch_training_jobs(a100_runs, gpu="A100", max_parallel_runs=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
