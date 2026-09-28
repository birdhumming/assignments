"""Problem 7: our own prediction problem.

Question (multiple choice): QK-normalization is on by default. Turn it off and
sweep the peak LR over {0.001, 0.003, 0.009, 0.03}. Relative to the standard
d8 run (2.926):
  A. every no-QK-norm run is within +0.02 of its QK-norm counterpart
  B. no-QK-norm only hurts (>+0.05) at LR >= 0.009
  C. no-QK-norm diverges (non-finite loss) at 0.03 but is within +0.02 elsewhere
  D. no-QK-norm is *better* than QK-norm at the lowest LR and worse at the highest
The QK-norm counterparts at these LRs come from P1. As a second axis we also
tie the input/output embeddings at two LRs.
"""

from experiments.a1_basics.common import dedupe, run
from modal_train import launch_training_jobs


RUNS = dedupe(
    [run("a1-p7", "a1-p7-noqknorm", qk_norm=False, learning_rate=lr) for lr in (1e-3, 3e-3, 9e-3, 3e-2)]
    + [run("a1-p7", "a1-p7-tiedemb", tie_word_embeddings=True)]
)


def main():
    launch_training_jobs(RUNS, max_parallel_runs=2)


if __name__ == "__main__":
    main()
