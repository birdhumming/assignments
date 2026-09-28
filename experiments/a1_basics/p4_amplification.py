"""Problem 4: amplification of a micro-perturbation under fully deterministic training.

train.py's perturb_one_token hook is extended with `perturb_row` (which shuffled
training row is edited, i.e. *when* the perturbation is seen: step = row // 64)
and `perturb_num_tokens` (how many tokens starting at position 100 become id 17).

(a) size of the perturbation at step 0
(b) same perturbations injected at 25% / 50% / 90% of training
"""

from experiments.a1_basics.common import dedupe, run
from modal_train import launch_training_jobs


BATCH_SIZE = 64
TOTAL_STEPS = 600_000 // BATCH_SIZE  # 9375

REFERENCE = run("a1-p4", "a1-p4a", deterministic=True)

PART_A = [REFERENCE] + [
    run("a1-p4", "a1-p4a", deterministic=True, perturb_one_token=True, perturb_num_tokens=n)
    for n in (1, 10, 100, 1024)
]

PERTURB_STEPS = {
    "25pct": int(0.25 * TOTAL_STEPS),
    "50pct": int(0.50 * TOTAL_STEPS),
    "90pct": int(0.90 * TOTAL_STEPS),
}

PART_B = [
    run(
        "a1-p4",
        "a1-p4b",
        f"a1-p4b-{label}",
        deterministic=True,
        perturb_one_token=True,
        perturb_row=step * BATCH_SIZE,
        perturb_num_tokens=n,
    )
    for label, step in PERTURB_STEPS.items()
    for n in (1, 1024)
]

RUNS = dedupe([*PART_A, *PART_B])


def main():
    launch_training_jobs(RUNS, max_parallel_runs=2)


if __name__ == "__main__":
    main()
