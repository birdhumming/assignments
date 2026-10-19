"""Budgeted A3 launch waves (max 2 H100s). Preview by default; `--execute` submits.

    uv run python -m experiments.a3_optimization.waves p2_wave1 [--execute]
"""
import argparse
import csv
from dataclasses import replace
from pathlib import Path

from experiments.a3_optimization.p2_river_valley import (
    EXPERIMENT_KEY as P2_KEY,
)
from experiments.a3_optimization.p2_river_valley import (
    HORIZON_FREE_RUNS,
    REWARM_RUNS,
)
from experiments.a3_optimization.p3_edge_of_stability import (
    EXPLORATION_RUNS as P3_EXPLORE,
)
from experiments.a3_optimization.p3_edge_of_stability import (
    RUNS as P3_CORE,
)
from train import TrainConfig, training_run_name

MANIFEST_DIR = Path(__file__).resolve().parent / "manifests"
D8_MINUTES = 11.5  # one 614M-token d8 run on an H100, measured in A1/A2

EMA_DECAYS = (0.9, 0.99, 0.999, 0.9999)  # evaluation-only; adds averaging timescales for P2(b)
SCHEDULES = ("wsd0.0", "linear", "cos", "wsd0.2")
PEAK_LRS = (3e-3, 1.5e-3, 6e-3)  # handout P2(a): tune the peak over these alternatives


def p2_schedule(lr_schedule, learning_rate):
    return TrainConfig(lr_schedule=lr_schedule, learning_rate=learning_rate,
                       ema_decays=EMA_DECAYS, wandb_tags=(P2_KEY,))


P2_WAVE1 = ([p2_schedule(s, lr) for lr in PEAK_LRS for s in SCHEDULES]
            + [replace(c, ema_decays=EMA_DECAYS) for c in REWARM_RUNS + HORIZON_FREE_RUNS])

def p2_dip(dip, learning_rate=3e-3):
    return TrainConfig(lr_schedule="wsd0.2", learning_rate=learning_rate, lr_dip=dip,
                       ema_decays=EMA_DECAYS, wandb_tags=(P2_KEY,))


def p2_budget(num_epochs, lr_schedule="cos", learning_rate=3e-3):
    return TrainConfig(lr_schedule=lr_schedule, learning_rate=learning_rate, num_epochs=num_epochs,
                       ema_decays=EMA_DECAYS, wandb_tags=(P2_KEY,))


# Dip depth / width / timing on wsd0.2 at the best peak LR, plus horizon-matched baselines
# (cos tuned to 25% and 50% of the tokens) to score the horizon-free schedules.
P2_WAVE2 = [
    p2_dip((0.4, 0.45, 0.5, 0.3)),
    p2_dip((0.4, 0.45, 0.5, 0.01)),
    p2_dip((0.4, 0.5, 0.6, 0.1)),
    p2_dip((0.2, 0.25, 0.3, 0.1)),
    p2_dip((0.6, 0.65, 0.7, 0.1)),
    p2_budget(0.25),
    p2_budget(0.5),
]

WAVES = {"p2_wave1": P2_WAVE1, "p2_wave2": P2_WAVE2, "p3_core": P3_CORE,
         "p3_explore_trimmed": P3_EXPLORE[:10], "p3_explore_rest": P3_EXPLORE[10:]}


def estimated_minutes(config):
    tokens = config.num_epochs * config.num_train_sequences * 1024
    steps_scale = 64 / config.batch_size if config.num_train_sequences >= 600_000 else 1.0
    if config.stop_at_step is not None:
        tokens = min(tokens, config.stop_at_step * config.batch_size * 1024)
    return D8_MINUTES * tokens / (600_000 * 1024) * max(1.0, steps_scale ** 0.5)


def preview(configs, name):
    MANIFEST_DIR.mkdir(exist_ok=True)
    path = MANIFEST_DIR / f"{name}.csv"
    rows = [{"run_name": training_run_name(c), "lr_schedule": c.lr_schedule,
             "learning_rate": c.learning_rate, "est_minutes": round(estimated_minutes(c), 1)}
            for c in configs]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"  {r['est_minutes']:>6}m  {r['run_name']}")
    print(f"{len(rows)} runs, ~{sum(r['est_minutes'] for r in rows) / 60:.1f} GPU-hours -> {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("wave", choices=sorted(WAVES))
    p.add_argument("--execute", action="store_true")
    p.add_argument("--max-parallel-runs", type=int, default=2)
    a = p.parse_args()
    configs = WAVES[a.wave]
    names = [training_run_name(c) for c in configs]
    assert len(set(names)) == len(names), "duplicate run names"
    preview(configs, a.wave)
    if a.execute:
        from modal_train import launch_training_jobs
        launch_training_jobs(configs, max_parallel_runs=a.max_parallel_runs)


if __name__ == "__main__":
    main()
