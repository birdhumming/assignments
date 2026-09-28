"""Problem 1: hyperparameter sweeps on the standard d8 recipe (compute-matched).

(a) one-at-a-time log-spaced sweeps (base 3) of LR, batch size, warmup, WD
(b) paired sweeps for the pairs we expect to co-vary
(c) LR-schedule interactions plus the remaining optimizer knobs
"""

from experiments.a1_basics.common import dedupe, run
from modal_train import launch_training_jobs


BASELINE = run("a1-p1", "a1-baseline")

# (a) individual sweeps. Multiplier 3 around the default, 5 points each.
SWEEP_A = (
    [run("a1-p1", "a1-p1a-lr", learning_rate=lr) for lr in (3e-4, 1e-3, 9e-3, 2.7e-2)]
    + [run("a1-p1", "a1-p1a-bs", batch_size=bs) for bs in (16, 32, 128, 256)]
    + [run("a1-p1", "a1-p1a-warmup", warmup_percent=w) for w in (0.0, 0.003, 0.03, 0.1)]
    + [run("a1-p1", "a1-p1a-wd", weight_decay=wd) for wd in (0.0, 0.01, 0.03, 0.3, 1.0)]
)

# (b) paired sweeps.
SWEEP_B = (
    # LR x batch size: does the optimal LR move with batch size?
    [
        run("a1-p1", "a1-p1b-lr-bs", learning_rate=lr, batch_size=bs)
        for lr in (1e-3, 9e-3)
        for bs in (16, 32, 128, 256)
    ]
    # LR x weight decay: AdamW decay is lr*wd, so these should be linked.
    + [
        run("a1-p1", "a1-p1b-lr-wd", learning_rate=lr, weight_decay=wd)
        for lr in (1e-3, 9e-3)
        for wd in (0.01, 1.0)
    ]
    # LR x warmup: does a longer warmup rescue high LRs?
    + [
        run("a1-p1", "a1-p1b-lr-warmup", learning_rate=lr, warmup_percent=w)
        for lr, w in ((9e-3, 0.1), (2.7e-2, 0.1), (1e-3, 0.0))
    ]
    # batch size x weight decay: fewer/more steps change the total decay applied.
    + [
        run("a1-p1", "a1-p1b-bs-wd", batch_size=bs, weight_decay=1.0)
        for bs in (16, 256)
    ]
)

# (c) schedules and the other optimizer hyperparameters.
SWEEP_C = (
    [run("a1-p1", "a1-p1c-schedule", lr_schedule=s) for s in ("cos", "constant", "wsd0.2", "wsd0.5")]
    + [
        run("a1-p1", "a1-p1c-schedule-lr", lr_schedule=s, learning_rate=lr)
        for s in ("cos", "wsd0.2")
        for lr in (1e-3, 9e-3)
    ]
    + [run("a1-p1", "a1-p1c-schedule-lr", lr_schedule="constant", learning_rate=1e-3)]
    # Handout medium example: lr 0.009, wd 1.0, three schedule/warmup combos.
    + [
        run("a1-p1", "a1-p1c-example", learning_rate=9e-3, weight_decay=1.0, lr_schedule=s, warmup_percent=w)
        for s, w in (("wsd0.2", 0.0), ("wsd0.2", 0.2), ("linear", 0.1))
    ]
    + [run("a1-p1", "a1-p1c-schedule-warmup", lr_schedule="wsd0.2", warmup_percent=w) for w in (0.0, 0.1)]
    + [run("a1-p1", "a1-p1c-beta1", beta1=b) for b in (0.8, 0.95)]
    + [run("a1-p1", "a1-p1c-beta2", beta2=b) for b in (0.9, 0.99)]
    + [run("a1-p1", "a1-p1c-gradclip", grad_norm=None)]
)

RUNS = dedupe([BASELINE, *SWEEP_A, *SWEEP_B, *SWEEP_C])


def main():
    launch_training_jobs(RUNS, max_parallel_runs=2)


if __name__ == "__main__":
    main()
