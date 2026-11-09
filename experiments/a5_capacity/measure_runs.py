"""Post-training diagnostics for our saved a5-* checkpoints, on matched held-out tokens.

    uv run python -m experiments.a5_capacity.measure_runs [run_name ...]   # default: every a5-* dir

Writes results/a5_capacity/measurements.json (merged with earlier entries). For loop models the
held-out NLL is also measured at other inference loop counts.
"""
import json
import sys
from pathlib import Path

OUTPUT = Path('results/a5_capacity/measurements.json')
GPU = ['H100', 'H200', 'A100-80GB']


def held_out(run_name, device, sequences=8, context=128):
    import numpy as np
    import torch
    if run_name.startswith('a5-spam-'):
        _, _, v, w, _ = run_name.split('-')
        from .p2_readout import training_config
        cfg = training_config(int(v[1:]), int(w[1:]), root='/tmp/a5_capacity/spam')
        ids = np.fromfile(Path(cfg.val_dataset.path)/'tokens.bin', dtype=np.uint16).reshape(-1, 64)
        return torch.as_tensor(ids[:sequences].astype('int64'), device=device)
    from data import load_token_dataset, dclm_val_dataset
    validation = load_token_dataset(dclm_val_dataset(), role='val', data_seed=42)
    tokens = torch.as_tensor(validation[:max(sequences, 64)]['input_ids'].astype('int64'), device=device)
    return tokens[:sequences, :context]


def loop_count_sweep(model, tokens, passes):
    """Held-out NLL when the trained loop model runs a different number of core passes."""
    import torch
    from train import causal_lm_loss
    from .p4_looping import LoopLM
    if not isinstance(model, LoopLM) or not getattr(model, 'tied', True):
        return None
    out = {}
    for p in passes:
        model.eval_passes = p
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            out[str(p)] = float(causal_lm_loss(model(tokens), tokens))
    model.eval_passes = None
    return out


def measure_one(run_name, device='cuda'):
    import torch
    from model_io import load_model
    from modal_utils import MODAL_MODEL_DIR
    from .diagnostics import measure_model, signal_profile
    path = MODAL_MODEL_DIR/run_name
    model = load_model(path, device=device, precision='fp32')
    tokens = held_out(run_name, device)
    result = measure_model(model, tokens)
    builder_kwargs = json.loads((path/'metadata.json').read_text()).get('model_builder_kwargs', {})
    result['model_builder_kwargs'] = builder_kwargs
    from .p4_looping import LoopLM
    if isinstance(model, LoopLM):
        trained = model.passes
        candidates = sorted(set([1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, trained, trained-1, trained+1]))
        big = held_out(run_name, device, sequences=64, context=512)
        result['loop_sweep'] = loop_count_sweep(model, big, [p for p in candidates if p >= 1])
        result['trained_passes'] = trained
        result['signals'] = signal_profile(model, tokens[:2])
    elif not run_name.startswith('a5-spam-') and 'moe' not in run_name:
        result['signals'] = signal_profile(model, tokens[:2])
    del model
    torch.cuda.empty_cache()
    return result


def measure_many(run_names=None):
    import torch
    from modal_utils import MODAL_MODEL_DIR
    if not run_names:
        run_names = sorted(p.name for p in MODAL_MODEL_DIR.glob('a5-*') if (p/'metadata.json').exists())
    out = {}
    for name in run_names:
        try:
            out[name] = measure_one(name)
            print('measured', name, {k: out[name].get(k) for k in ('nll', 'routers', 'gradient_energy', 'loop_sweep')}, flush=True)
        except Exception as e:  # keep going; report the failure
            out[name] = dict(error=f'{type(e).__name__}: {e}')
            print('FAILED', name, out[name]['error'], flush=True)
    return out


if __name__ == '__main__':
    import modal
    from modal_utils import app, build_image, VOLUME_MOUNTS, MODAL_ENVIRONMENT
    remote = app.function(image=build_image(), volumes=VOLUME_MOUNTS, gpu=GPU,
                          cpu=8, memory=32768, timeout=3600)(measure_many)
    with modal.enable_output(), app.run(environment_name=MODAL_ENVIRONMENT):
        result = remote.remote(sys.argv[1:] or None)
    previous = json.loads(OUTPUT.read_text()) if OUTPUT.exists() else {}
    previous.update(result)
    OUTPUT.write_text(json.dumps(previous, indent=1) + '\n')
    print(OUTPUT, len(result), 'measured')
