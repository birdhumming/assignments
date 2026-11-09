"""Shared loaders for the A5 analysis scripts (W&B pull + Modal measurements)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUNS = Path('results/a5_capacity/runs.json')
MEASURES = Path('results/a5_capacity/measurements.json')
PLOTS = Path('plots/a5_capacity')
PLOTS.mkdir(parents=True, exist_ok=True)


def load():
    runs = json.loads(RUNS.read_text()) if RUNS.exists() else {}
    for r in runs.values():  # a cancelled Modal call exits cleanly, so W&B marks it finished without a val_loss
        if r.get('state') == 'finished' and r.get('val_loss') is None:
            r['state'] = 'cancelled'
    meas = json.loads(MEASURES.read_text()) if MEASURES.exists() else {}
    return runs, meas


def run(runs, variant, lr=.003, depth=8, width=512):
    return runs.get(f'a5-{variant}-d{depth}-w{width}-lr{lr:g}')


def final_loss(r):
    """Finished runs only: W&B summary val_loss is the last logged validation loss."""
    if r is None or r.get('state') != 'finished':
        return None
    return r.get('val_loss')


def curve(r):
    h = [x for x in (r or {}).get('history', []) if x.get('val_loss') is not None and x.get('optimizer_step') is not None]
    h.sort(key=lambda x: x['optimizer_step'])
    return [x['optimizer_step'] for x in h], [x['val_loss'] for x in h]


def fmt(x, nd=4):
    return '--' if x is None else f'{x:.{nd}f}'


def save(fig, name):
    fig.tight_layout()
    fig.savefig(PLOTS/name, dpi=160)
    plt.close(fig)
    print('wrote', PLOTS/name)
