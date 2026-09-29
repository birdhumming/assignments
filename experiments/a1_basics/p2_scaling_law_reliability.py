"""Problem 2: scaling-law reliability across the d4-d9 depth ladder.

(a) four recipes x d4..d9 (handout grid; d9 is the pre-registered
    extrapolation test for every recipe)
(b) slope-bending interventions that stay in the power-law regime
    (model-size axis d4..d8 and a data axis at d6/d8)
(c) interventions intended to break the power law, at d4/d6/d8 so we can
    see whether "breaking" depends on scale
"""

from experiments.a1_basics.common import dedupe, run
from modal_train import launch_training_jobs
from model_config import depth_model_config


def depth_run(depth, *tags, **changes):
    return run(*tags, model_config=depth_model_config(depth), **changes)


RECIPES = {
    "baseline": dict(learning_rate=0.003, lr_schedule="linear", dropout=0.0),
    "constant": dict(learning_rate=0.003, lr_schedule="constant", dropout=0.0),
    "dropout": dict(learning_rate=0.003, lr_schedule="linear", dropout=0.2),
    "lr0.03": dict(learning_rate=0.03, lr_schedule="linear", dropout=0.0),
}

# (a) Handout grid. Pre-register d8/d9 predictions from d4-d7 before reading them.
PART_A = [
    depth_run(depth, "a1-p2", "a1-p2a", f"a1-p2a-{name}", **recipe)
    for name, recipe in RECIPES.items()
    for depth in range(4, 10)
]

# (b) Slope benders. Model-size axis: mild optimizer changes at d4..d7 (the d8
# points are the P1 runs lr=0.001 and bs=256).
PART_B_MODEL = [
    depth_run(depth, "a1-p2", "a1-p2b-model", **changes)
    for depth in (4, 5, 6, 7)
    for changes in ({"learning_rate": 1e-3}, {"batch_size": 256})
]
# Data axis: shorter horizons at d6 and d8, baseline vs lower LR.
DATA_HORIZONS = (75_000, 150_000, 300_000)
PART_B_DATA = [
    depth_run(depth, "a1-p2", "a1-p2b-data", num_train_sequences=n, **changes)
    for depth in (6, 8)
    for n in DATA_HORIZONS
    for changes in ({}, {"learning_rate": 1e-3})
]

# (c) Scaling-law breakers at d4, d6 and d8.
PART_C = [
    depth_run(depth, "a1-p2", "a1-p2c", **changes)
    for depth in (4, 6, 8)
    for changes in (
        {"optimizer_name": "sgd"},
        {"qk_norm": False, "learning_rate": 0.03},
        {"warmup_percent": 0.0, "learning_rate": 0.03, "grad_norm": None},
        # Same token budget, but 8 epochs over a 75k-sequence subset (repetition).
        {"num_train_sequences": 75_000, "num_epochs": 8.0},
    )
]

RUNS = dedupe([*PART_A, *PART_B_MODEL, *PART_B_DATA, *PART_C])


def main():
    launch_training_jobs(RUNS, max_parallel_runs=2)


if __name__ == "__main__":
    main()
