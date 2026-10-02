"""P3.2: what should scale with batch size (d8, 614.4M tokens, linear decay, 1% warmup, microbatch min(B,64)).

Supplied at batch 64: LR sweep {.0015,.003,.006} at WD .1 (P1a) and WD {.1,.2,.4} at LR .0015 (P2a).
(a) LR sweep per batch at WD .1, grids centred on the linear-scaling guess lr = .003 * B / 64.
(b) WD sweep per batch at LR .0015 (hypothesis ii), then targets at batches 128/256 from both laws.
(c) beta1 ablation at batches 8 and 256 holding each batch's best measured (LR, WD).
"""
import argparse

from experiments.a2.common import dedupe, preview, run

LR_GRIDS = {8: (.0004, .00075, .0015), 16: (.00075, .0015, .003), 32: (.0015, .003, .006)}
WD_GRID = (.05, .1, .2, .4)
WD_LR = .0015
# From p32_predictions.md (fits on B=8..64 sources, recorded before any target run).
PRED_LR = {128: .00556, 256: .00819}
PRED_WD = {128: .63, 256: 1.10}
LR_SWEEP = {128: (.003, .006, .012), 256: (.006, .012, .024)}
WD_SWEEP = {128: (.4, .8, 1.6), 256: (.8, 1.6, 3.2)}
BETA1_TRIMMED = (0., .5, .98)


def p32_run(part, batch, lr, wd, **overrides):
    return run('a2-p32', f'a2-p32{part}', batch=batch, suffix='a2-p32', learning_rate=lr, weight_decay=wd, **overrides)


def source_runs(wd_grid=WD_GRID):
    runs = [p32_run('a', B, lr, .1) for B, lrs in LR_GRIDS.items() for lr in lrs]
    runs += [p32_run('b', B, WD_LR, wd) for B in LR_GRIDS for wd in wd_grid]
    return runs


def target_runs(pred_lr, pred_wd, lr_sweep, wd_sweep):
    """pred_lr/pred_wd: {batch: value}; sweeps: {batch: tuple}. Hypothesis i at WD .1, ii at LR .0015."""
    runs = []
    for B in (128, 256):
        runs += [p32_run('b', B, lr, .1) for lr in dict.fromkeys((pred_lr[B], *lr_sweep[B]))]
        runs += [p32_run('b', B, WD_LR, wd) for wd in dict.fromkeys((pred_wd[B], *wd_sweep[B]))]
    return runs


def momentum_runs(best, betas=(0., .5, .95, .98)):
    """best: {batch: (lr, wd)}; beta1=.9 is the already-run best pair."""
    return [p32_run('c', B, lr, wd, beta1=b1) for B, (lr, wd) in best.items() for b1 in betas]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--wd-grid', type=float, nargs='+', default=WD_GRID)
    p.add_argument('--targets', action='store_true', help='B=128/256 runs from the fits in p32_predictions.md (trimmed: 4 points each)')
    p.add_argument('--beta1', nargs=3, type=float, metavar=('BATCH', 'LR', 'WD'), action='append',
                   help='momentum ablation at this batch and its best (lr, wd); repeatable')
    a = p.parse_args()
    if a.targets or a.beta1:
        runs = []
        if a.targets:
            runs += target_runs(PRED_LR, PRED_WD, LR_SWEEP, WD_SWEEP)
        for B, lr, wd in a.beta1 or ():
            runs += momentum_runs({int(B): (lr, wd)}, betas=BETA1_TRIMMED)
        runs = dedupe(runs)
        preview(runs, 'experiments/a2/p32_targets_manifest.csv')
    else:
        runs = dedupe(source_runs(tuple(a.wd_grid)))
        preview(runs, 'experiments/a2/p32_sources_manifest.csv')
    if a.execute:
        from experiments.a2.modal_launcher import launch_training_jobs
        launch_training_jobs(runs, max_parallel_runs=2)


if __name__ == '__main__':
    main()
