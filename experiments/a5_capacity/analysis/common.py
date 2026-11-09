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


# W&B records overwritten when a later run resumed the same run id (summary values copied from the earlier pull).
LOST = {'a5-moe-hash-d8-w512-lr0.003': dict(state='finished', val_loss=2.9312479496002197, step=9374, history=[]),  # record later resumed by moe-all
        'a5-unique-reset-d8-w512-lr0.003': dict(state='finished', val_loss=3.0017778873443604, step=9374, history=[])}


def load():
    runs = json.loads(RUNS.read_text()) if RUNS.exists() else {}
    runs = {**LOST, **runs}
    meas = json.loads(MEASURES.read_text()) if MEASURES.exists() else {}
    for name, m in meas.items():  # W&B record lost but the final checkpoint exists: use its held-out NLL (marked in tables)
        if m.get('nll') is not None and runs.get(name, {}).get('val_loss') is None:
            runs[name] = dict(runs.get(name, {}), state='finished', val_loss=m['nll'], from_ckpt=True, history=[])
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


def tag(label, r):
    return label + r'$^\dagger$' if (r or {}).get('from_ckpt') else label


def fmt(x, nd=4):
    return '--' if x is None else f'{x:.{nd}f}'


def save(fig, name):
    fig.tight_layout()
    fig.savefig(PLOTS/name, dpi=160)
    plt.close(fig)
    print('wrote', PLOTS/name)
