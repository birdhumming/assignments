"""Measure supplied P1/P2 checkpoints (init rebuilt with seed 42, step-100 early, final) on Modal.

    uv run python -m experiments.a4_stability.measure_supplied [variant ...]

Writes results/a4_stability/<run>/{init,early,final}.json with diagnostics.profile output plus the
paired context-removal losses used in P2(f). Default: the best-LR run of every supplied variant.
"""
import json
import os
import sys
from pathlib import Path

RESULTS = Path('results/a4_stability')
REFERENCE_ROOT = Path(os.environ.get('A4_REFERENCE_ROOT', '/root/shared_data/a4_stability'))
KEEPS = (1, 4, 16, 64, 256, 512)


def best_runs(variants=None):
    runs = json.loads(Path(__file__).with_name('results').joinpath('manifest.json').read_text())['runs']
    best = {}
    for r in runs:
        if r['status'] != 'complete' or r.get('validation_loss') is None or r['validation_loss'] > 6.8:
            continue
        if variants and r['variant'] not in variants:
            continue
        if r['variant'] not in best or r['validation_loss'] < best[r['variant']]['validation_loss']:
            best[r['variant']] = r
    return best


def measure_stage(model, tokens):
    import torch

    from experiments.a4_stability.diagnostics import context_loss, profile
    stats = profile(model, tokens)
    with torch.no_grad():
        full = context_loss(model, tokens, keep=tokens.shape[1]-1)
        stats['context_loss_increase'] = {
            f'{keep}_first_{first}': (context_loss(model, tokens, keep, first)-full).mean().item()
            for keep in KEEPS for first in (False, True)}
        stats['full_last_loss'] = full.mean().item()
    return stats


def measure_runs(jobs, root=REFERENCE_ROOT, device='cuda', batch=8, early='early'):
    """jobs: list of (run_name, variant, lr). Returns {run_name: {stage: stats}}.
    `early` is the step-100 checkpoint relative to the run dir: `early/` (supplied) or `step_100.pt` (ours)."""
    import torch

    from experiments.a4_stability import _runtime
    _runtime.DISABLE_COMPILE = True
    import torch._dynamo
    torch._dynamo.config.disable = True  # probes run eagerly; compiled GRU cells hit the recompile limit under hooks
    from data import dclm_val_dataset, load_token_dataset
    from experiments.a4_stability.sweeps import config
    from model_io import load_model
    from train import build_model
    validation = load_token_dataset(dclm_val_dataset(), role='val', data_seed=42)
    tokens = torch.tensor(validation[:batch]['input_ids'].astype('int64'), device=device)
    out = {}
    for run, variant, lr in jobs:
        out[run] = {}
        stages = {'init': None, 'early': Path(root)/run/early, 'final': Path(root)/run}
        for stage, path in stages.items():
            try:
                if path is None:
                    model = build_model(config(variant, lr, precision='fp32'), device)
                elif path.exists():
                    model = load_model(path, device=device, precision='fp32')
                else:
                    out[run][stage] = {'missing': str(path)}
                    continue
                model.eval()
                out[run][stage] = measure_stage(model, tokens)
                out[run][stage]['num_params'] = sum(p.numel() for p in model.parameters())
            except Exception as e:  # keep going; record the failure
                out[run][stage] = {'error': f'{type(e).__name__}: {e}'[:500]}
            finally:
                model = None
                torch.cuda.empty_cache()
            print(run, stage, {k: v for k, v in out[run][stage].items() if k in ('error', 'missing', 'full_last_loss')}, flush=True)
    return out


if __name__ == '__main__':
    import modal

    from modal_utils import MODAL_ENVIRONMENT, VOLUME_MOUNTS, app, build_image

    from .diagnostics import save
    variants = sys.argv[1:] or None
    jobs = [(r['run_name'], v, r['lr']) for v, r in sorted(best_runs(variants).items())]
    print(len(jobs), 'runs:', [j[0] for j in jobs])
    probe = app.function(image=build_image(), volumes=VOLUME_MOUNTS, gpu=os.environ.get('A4_MEASURE_GPU', 'H100'), timeout=3*3600)(measure_runs)
    with modal.enable_output(), app.run(environment_name=MODAL_ENVIRONMENT):
        results = probe.remote(jobs)
    for run, stages in results.items():
        for stage, stats in stages.items():
            save(RESULTS/run/f'{stage}.json', stats)
    print('wrote', RESULTS)
