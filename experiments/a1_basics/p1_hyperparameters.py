"""Problem 1: hyperparameter sweeps on the standard d8 recipe (compute-matched).

(a) one-at-a-time log-spaced sweeps (base 3) of LR, batch size, warmup, WD
(b) paired sweeps for the pairs we expect to co-vary
(c) LR-schedule interactions plus the remaining optimizer knobs

21 runs already exist in W&B (tag a1-basics-p1b-v3): lr in {1e-3, 3e-3, 9e-3}
crossed with {default, bs32, bs128, wd0.03, wd0.3, warmup0.03, warmup0.1}.
Those grid points are deliberately *not* repeated here; this file fills the
gaps (sweep endpoints, extreme pairs, schedules, betas, clipping).
"""

from experiments.a1_basics.common import dedupe, run
from modal_train import launch_training_jobs


BASELINE = run("a1-p1", "a1-baseline")

# (a) individual sweeps: extend the existing x3 points one more decade each way.
SWEEP_A = (
    [run("a1-p1", "a1-p1a-lr", learning_rate=lr) for lr in (3e-4, 2.7e-2)]
    + [run("a1-p1", "a1-p1a-bs", batch_size=bs) for bs in (16, 256)]
    + [run("a1-p1", "a1-p1a-wd", weight_decay=1.0)]
)

# (b) paired sweeps at the corners the existing 3x3 grids do not reach.
SWEEP_B = (
    # LR x batch size: does the optimal LR move with batch size?
    [
        run("a1-p1", "a1-p1b-lr-bs", learning_rate=lr, batch_size=bs)
        for lr, bs in ((1e-3, 16), (9e-3, 256))
    ]
    # LR x weight decay: AdamW decay per step is lr*wd, so these should be linked.
    + [run("a1-p1", "a1-p1b-lr-wd", learning_rate=9e-3, weight_decay=1.0)]
    # LR x warmup: does a longer warmup rescue a too-high LR, or none hurt a high LR?
    + [
        run("a1-p1", "a1-p1b-lr-warmup", learning_rate=lr, warmup_percent=w)
        for lr, w in ((2.7e-2, 0.1), (9e-3, 0.0))
    ]
)

# (c) schedules and the other optimizer hyperparameters.
SWEEP_C = (
    [run("a1-p1", "a1-p1c-schedule", lr_schedule=s) for s in ("constant", "wsd0.2")]
    + [
        run("a1-p1", "a1-p1c-schedule-lr", lr_schedule=s, learning_rate=lr)
        for s, lr in (("wsd0.2", 9e-3),)
    ]
    # Handout medium example: lr 0.009, wd 1.0, three schedule/warmup combos.
    + [
        run("a1-p1", "a1-p1c-example", learning_rate=9e-3, weight_decay=1.0, lr_schedule=s, warmup_percent=w)
        for s, w in (("wsd0.2", 0.0), ("wsd0.2", 0.2), ("linear", 0.1))
    ]
    + [run("a1-p1", "a1-p1c-beta1", beta1=0.8)]
    + [run("a1-p1", "a1-p1c-beta2", beta2=0.99)]
    + [run("a1-p1", "a1-p1c-gradclip", grad_norm=None)]
)

RUNS = dedupe([BASELINE, *SWEEP_A, *SWEEP_B, *SWEEP_C])


def main():
    launch_training_jobs(RUNS, max_parallel_runs=2)


if __name__ == "__main__":
    main()
