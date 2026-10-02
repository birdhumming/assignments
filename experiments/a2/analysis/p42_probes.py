"""P4.2(a)/(e): alignment and probe diagnostics through training for the width/depth transfer runs."""
import json
import sys
from pathlib import Path

import numpy as np

DATA = Path('/home/ubuntu/a2')
HIDDEN = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')


def load(tag_prefix='a2-p42'):
    out = []
    for f in sorted((DATA / 'runs').glob('*.json')):
        r = json.loads(f.read_text())
        if any(t.startswith(tag_prefix) for t in r['tags']) and r['state'] == 'finished':
            out.append(r)
    return out


def series(hist, key):
    return [(h['_step'], h[key]) for h in hist if h.get(key) is not None]


def alpha_summary(hist, step):
    """Median alpha over hidden matrices (and lm_head alone) at the alignment log nearest `step`."""
    rows = [h for h in hist if h.get('logging/alignment/step') is not None]
    if not rows:
        return None, None
    h = min(rows, key=lambda h: abs(h['logging/alignment/step'] - step))
    hid = [v for k, v in h.items() if k.startswith('logging/alignment/alpha/model.layers.') and any(s in k for s in HIDDEN) and v is not None]
    head = h.get('logging/alignment/alpha/lm_head.weight')
    return (float(np.median(hid)) if hid else None), head


def omega_at(hist, step):
    rows = [h for h in hist if h.get('logging/readout_alignment/movement_omega') is not None]
    if not rows:
        return None
    h = min(rows, key=lambda h: abs(h['logging/readout_alignment/step'] - step))
    return h['logging/readout_alignment/movement_omega']


def probe_at(hist, key, step):
    s = series(hist, key)
    if not s:
        return None
    return min(s, key=lambda p: abs(p[0] - step))[1]


def fmt(x):
    return '' if x is None else f'{x:.3g}'


def main():
    runs = load()
    last = max(h['_step'] for r in runs for h in r['history'])
    checkpoints = {'step 1': 1, 'step 5': 5, 'step 100': 100, 'mid': last // 2, 'end': last}
    lines = ['# P4.2 probes through training (median alpha_upd over hidden matrices; alpha of readout; omega_move; logit RMS; final-norm movement)\n']
    for r in sorted(runs, key=lambda r: (r['tags'], (r['config'].get('model_config') or {}).get('hidden_size'), r['config']['learning_rate'])):
        c = r['config']; mc = c.get('model_config') or {}
        pol = (c.get('model_builder_kwargs') or {}).get('policy', 'supplied')
        hist = sorted(r['history'], key=lambda h: h['_step'])
        hdr = f"\n## {pol} width {mc.get('hidden_size')} depth {mc.get('num_hidden_layers')} lr {c['learning_rate']:g} (final val {r['summary'].get('val_loss'):.4f})\n"
        lines.append(hdr)
        lines.append('| checkpoint | alpha hidden (median) | alpha readout | omega_move | logit RMS | final-norm movement |\n|---|---|---|---|---|---|')
        for name, st in checkpoints.items():
            ah, ar = alpha_summary(hist, st)
            lines.append(f"| {name} (step {st}) | {fmt(ah)} | {fmt(ar)} | {fmt(omega_at(hist, st))} | {fmt(probe_at(hist, 'logging/probe/logit_rms', st))} | {fmt(probe_at(hist, 'logging/probe/final_movement', st))} |")
    out = DATA / 'tables' / 'p42_probes.md'
    out.write_text('\n'.join(lines) + '\n')
    print('wrote', out, len(runs), 'runs')


if __name__ == '__main__':
    sys.exit(main())
