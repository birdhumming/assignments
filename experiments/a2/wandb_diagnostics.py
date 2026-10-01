"""W&B curves and plots for saved A2 alignment measurements."""
from collections import defaultdict
import math


def configure(wandb):
    for group in ('alignment', 'readout_alignment'):
        axis = f'logging/{group}/step'
        wandb.define_metric(axis, hidden=True)
        wandb.define_metric(f'logging/{group}/*', step_metric=axis, step_sync=False)


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def update_metrics(rows, step):
    metrics = {'alignment/step': step}
    for row in rows:
        if finite(row.get('alpha')):
            metrics['alignment/alpha/' + row['parameter']] = row['alpha']
    return metrics


def readout_metrics(row):
    metrics = {'readout_alignment/step': row['step']}
    for kind, values in row.get('readout_alignment', {}).items():
        for key in ('omega', 'ratio'):
            if finite(values.get(key)):
                metrics[f'readout_alignment/{kind}_{key}'] = values[key]
    return metrics


def log_records(wandb, alignments, features):
    """Append saved curves using their own training-step axes, never W&B _step."""
    configure(wandb)
    steps = defaultdict(dict)
    grouped = defaultdict(list)
    for row in alignments:
        grouped[row['step']].append(row)
    for step, rows in grouped.items():
        steps[step].update(update_metrics(rows, step))
    for row in features:
        steps[row['step']].update(readout_metrics(row))
    for step in sorted(steps):
        wandb.log({'logging/' + k: v for k, v in steps[step].items()})
    log_plots(wandb, alignments=alignments, features=features)


def log_plots(wandb, *, alignments=(), features=()):
    curves = defaultdict(dict)
    for row in alignments:
        if finite(row.get('alpha')):
            curves[row['parameter']][row['step']] = row['alpha']
    plots = {}
    if curves:
        keys = sorted(curves)
        plots['alignment_charts/update_alpha'] = wandb.plot.line_series(
            xs=[sorted(curves[k]) for k in keys],
            ys=[[curves[k][s] for s in sorted(curves[k])] for k in keys],
            keys=keys, title='Update alignment by matrix', xname='Completed optimizer updates')
    curves = defaultdict(dict)
    for row in features:
        for kind, values in row.get('readout_alignment', {}).items():
            if finite(values.get('omega')):
                curves[kind][row['step']] = values['omega']
    if curves:
        keys = sorted(curves)
        plots['alignment_charts/readout_omega'] = wandb.plot.line_series(
            xs=[sorted(curves[k]) for k in keys],
            ys=[[curves[k][s] for s in sorted(curves[k])] for k in keys],
            keys=keys, title='Readout–feature alignment', xname='Completed optimizer updates')
    if plots:
        wandb.log(plots)
