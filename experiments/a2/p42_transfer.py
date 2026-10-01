"""P4.2: width and depth transfer at 153.6M tokens (baseline vs muP; muP vs Depth-muP vs CompleteP).

Source LR = Problem 1's best sampled LR at 153.6M tokens (0.003). Width 512 / depth 8 comes from the
supplied P1(a) sweep, so only widths 128/256 (a), width 1024 (c), and depths 4/16 (d) are new.
"""
import argparse

from experiments.a2.common import dedupe, preview, run
from model_config import LMConfig

TOKENS = 153_600_000
SOURCE_LR = .003
WIDTH_LRS = (.00075, .0015, .003, .006, .012)
DEPTH_LRS = (.0015, .003, .006)
BUILDER = 'experiments.a2.mup:build_model'
OPTIMIZER = 'experiments.a2.mup:build_optimizer'


def lm_config(width, depth):
    return LMConfig(f'a2-w{width}-d{depth}', 4096, 1024, width, int(3.5 * width), depth, width // 64, width // 64)


def transfer_run(part, policy, width, depth, lr, **overrides):
    return run(f'a2-p42{part}', f'a2-p42-{policy}', tokens=TOKENS, suffix=f'a2-p42{part}-{policy}',
               model_config=lm_config(width, depth), learning_rate=lr,
               model_builder=BUILDER, model_builder_kwargs=dict(policy=policy, reference_width=512, reference_depth=8),
               optimizer_builder=OPTIMIZER, **overrides)


def width_runs(widths=(128, 256), lrs=WIDTH_LRS):
    return [transfer_run('a', pol, w, 8, lr) for pol in ('baseline', 'mup') for w in widths for lr in lrs]


def depth_runs(depths=(4, 16), lrs=DEPTH_LRS):
    return [transfer_run('d', pol, 512, d, lr) for pol in ('mup', 'depth_mup', 'completep') for d in depths for lr in lrs]


def heldout_runs(predicted, lrs=(.00075, .0015, .003, .006)):
    """predicted: {policy: fitted lr* at width 1024}; also runs the transferred source LR and a local sweep."""
    out = []
    for pol, lr_star in predicted.items():
        for lr in dict.fromkeys((lr_star, SOURCE_LR, *lrs)):
            out.append(transfer_run('c', pol, 1024, 8, lr))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--part', choices=('a', 'd', 'ad', 'c'), default='ad')
    p.add_argument('--predicted', nargs=2, type=float, metavar=('BASELINE_LR', 'MUP_LR'))
    a = p.parse_args()
    runs = []
    if 'a' in a.part:
        runs += width_runs()
    if 'd' in a.part:
        runs += depth_runs()
    if a.part == 'c':
        if a.predicted is None:
            p.error('--part c needs --predicted BASELINE_LR MUP_LR')
        runs += heldout_runs(dict(zip(('baseline', 'mup'), a.predicted)))
    runs = dedupe(runs)
    preview(runs, f'experiments/a2/p42_{a.part}_manifest.csv')
    if a.execute:
        from experiments.a2.modal_launcher import launch_training_jobs
        launch_training_jobs(runs, max_parallel_runs=2)


if __name__ == '__main__':
    main()
