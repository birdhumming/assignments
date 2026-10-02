"""Wave 1: the long target runs whose configs are fixed by the supplied-sweep fits.

Predictions (from experiments/a2/analysis/sources.py, recorded before launch):
  P1(c)  lr* at 4.9152B: all-six-budget law 0.00373, large-three law 0.00223
  P1(d)  Hyperball lr* at 1.2288B: 0.00839 (power law over the three source budgets)
  P2(c)  lr*wd* product law -> 2.78e-4 at 2.4576B -> wd 0.0927 at peak lr .003
"""
import argparse

from experiments.a2.common import dedupe, preview, run

P1C_PREDICTED = {'all_six': .00373, 'larger_three': .00223}
P1D_PREDICTED = .00839
P2C_PREDICTED_WD = .0927

P1C = [run('a2-p1c', tokens=4_915_200_000, suffix='a2-p1-D4915m', learning_rate=lr)
       for lr in (.003, .0015, .006, *P1C_PREDICTED.values())]
P1D = [run('a2-p1d', tokens=1_228_800_000, suffix='a2-p1d-hyperball', optimizer_name='adamh', learning_rate=lr)
       for lr in (P1D_PREDICTED, .006, .01, .015)]
P2C = ([run('a2-p2c', tokens=2_457_600_000, suffix='a2-p2c', learning_rate=.003, weight_decay=P2C_PREDICTED_WD),
        run('a2-p2c', tokens=2_457_600_000, suffix='a2-p2c', learning_rate=.0015, weight_decay=1.6)]
       + [run('a2-p2c', tokens=2_457_600_000, suffix='a2-p2c', learning_rate=.003, weight_decay=wd)
          for wd in (.05, .1, .2)])

RUNS = dedupe(P1C + P1D + P2C)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--max-parallel-runs', type=int, default=2)
    a = p.parse_args()
    preview(RUNS, 'experiments/a2/wave1_manifest.csv')
    if a.execute:
        from experiments.a2.modal_launcher import launch_training_jobs
        launch_training_jobs(RUNS, max_parallel_runs=a.max_parallel_runs)


if __name__ == '__main__':
    main()
