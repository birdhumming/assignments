"""Problem 3: measuring run-to-run variation.

(a) all sources at once: paired model/data seeds
(b) isolated sources: model seed only, data seed only, hardware only
    (deterministic references, repeated non-deterministic runs, and the same
    deterministic config on an A100 via A100_RUNS)
(c) does variation change with hyperparameters / scale / training length?
"""

from experiments.a1_basics.common import dedupe, run
from modal_train import launch_training_jobs
from model_config import depth_model_config


SEEDS = (1, 2, 3)

DETERMINISTIC_REFERENCES = [
    run("a1-p3", "a1-p3b-hardware", deterministic=True, run_name_suffix="deterministic-reference-1"),
    run("a1-p3", "a1-p3b-hardware", deterministic=True, run_name_suffix="deterministic-reference-2"),
]

# (a) Vary everything: paired seeds, GPU nondeterminism left on.
PART_A = [run("a1-p3", "a1-p3a-all", model_seed=s, data_seed=s) for s in SEEDS]

# (b) One source at a time.
PART_B = (
    [run("a1-p3", "a1-p3b-model-seed", model_seed=s) for s in SEEDS[:2]]
    + [run("a1-p3", "a1-p3b-data-seed", data_seed=s) for s in SEEDS[:2]]
    # Hardware-only: identical seeds, non-deterministic kernels, repeated.
    + [run("a1-p3", "a1-p3b-hardware", run_name_suffix=f"nondeterministic-rep{i}") for i in (1, 2)]
    + DETERMINISTIC_REFERENCES
)

# Launched separately on a different GPU type to expose hardware nondeterminism.
A100_RUNS = [
    run("a1-p3", "a1-p3b-hardware", deterministic=True, run_name_suffix="deterministic-reference-a100"),
]

# (c) Variation under other hyperparameters / scales (paired seeds as in (a)).
# (c) Seeds 1,2 here pair with the seed-42 runs of the same recipe from P1
# (lr 0.009, bs 16) and P2 (d4), giving 3 samples per recipe. The "longer
# training" axis is deferred: 2x-token d8 runs cost 50 min each.
PART_C = (
    [run("a1-p3", "a1-p3c-lr009", learning_rate=9e-3, model_seed=s, data_seed=s) for s in SEEDS[:2]]
    + [run("a1-p3", "a1-p3c-bs16", batch_size=16, model_seed=s, data_seed=s) for s in SEEDS[:2]]
    + [
        run("a1-p3", "a1-p3c-d4", model_config=depth_model_config(4), model_seed=s, data_seed=s)
        for s in SEEDS
    ]
)

RUNS = dedupe([*PART_A, *PART_B, *PART_C])


def main():
    launch_training_jobs(RUNS, max_parallel_runs=2)
    launch_training_jobs(A100_RUNS, gpu="A100", max_parallel_runs=1)


if __name__ == "__main__":
    main()
