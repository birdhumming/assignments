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


PROBLEMS = [
    ("p1", p1_hyperparameters.RUNS),
    ("p3", p3_measuring_variation.RUNS),
    ("p4", p4_amplification.RUNS),
    ("p2", p2_scaling_law_reliability.RUNS),
    ("p6", p6_activation_gradient_norms.RUNS),
    ("p7", p7_own_prediction_problem.RUNS),
    ("p5", p5_loss_curve_augury.RUNS),
]
A100_RUNS = p3_measuring_variation.A100_RUNS

MANIFEST_PATH = Path(__file__).resolve().parent / "run_manifest.csv"


def depth_of(config):
    if config.model_config is not None:
        return int(config.model_config.name.lstrip("d"))
    return int(config.model_name.lstrip("d"))


# Rough H100 minutes per run relative to a ~12.5 min d8 default (fixed 614M tokens).
DEPTH_COST = {4: 0.30, 5: 0.40, 6: 0.50, 7: 0.75, 8: 1.00, 9: 1.40}


def estimated_minutes(config):
    tokens_scale = (config.num_train_sequences * float(config.num_epochs)) / 600_000
    return 12.5 * DEPTH_COST[depth_of(config)] * tokens_scale


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
                "est_minutes": f"{estimated_minutes(config) * 2:.1f}",
            }
        )
    return unique, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-parallel-runs", type=int, default=2)
    args = parser.parse_args()

    unique, rows = build_manifest()
    with MANIFEST_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    total_minutes = sum(float(r["est_minutes"]) for r in rows)
    print(f"{len(rows)} unique runs, ~{total_minutes / 60:.1f} GPU-hours estimated.")
    print(f"Manifest written to {MANIFEST_PATH}")
    if args.dry_run:
        for row in rows:
            print(f"  [{row['problems']:>8}] {row['gpu']} {row['est_minutes']:>5}m {row['run_name']}")
        return 0

    launch_training_jobs(unique, max_parallel_runs=args.max_parallel_runs)
    launch_training_jobs(A100_RUNS, gpu="A100", max_parallel_runs=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
