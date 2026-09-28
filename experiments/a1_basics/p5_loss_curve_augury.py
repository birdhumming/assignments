"""Problem 5: loss-curve shapes.

No new training: the runs below are the P1 configs whose curves we study
(macro shape: LR / schedule / beta1; micro smoothness: batch size, beta2,
grad clipping). They are listed so launch_all.py tags them and so the plotting
notebook can find them by name.
"""

from experiments.a1_basics.common import dedupe, run
from modal_train import launch_training_jobs


RUNS = dedupe(
    [
        run("a1-p5"),
        run("a1-p5", batch_size=256),
        run("a1-p5", batch_size=16),
        run("a1-p5", learning_rate=3e-4),
        run("a1-p5", learning_rate=9e-3),
        run("a1-p5", learning_rate=2.7e-2),
        run("a1-p5", lr_schedule="constant"),
        run("a1-p5", lr_schedule="wsd0.2"),
        run("a1-p5", beta1=0.8),
        run("a1-p5", beta2=0.99),
        run("a1-p5", grad_norm=None),
    ]
)


def main():
    launch_training_jobs(RUNS, max_parallel_runs=2)


if __name__ == "__main__":
    main()
