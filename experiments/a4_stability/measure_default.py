"""Measure a trained default d8 from the course's read-only shared volume.

    uv run python -m experiments.a4_stability.measure_default

Edit RUN to inspect another supplied run. This launches one measurement job,
then writes the returned measurements locally; it does not train a model.
"""
import os
from pathlib import Path
import torch

RUN = 'a4-d8-lr0.003'
REFERENCE_ROOT = Path(os.environ.get('A4_REFERENCE_ROOT', '/root/shared_data/a4_stability'))
OUTPUT = Path('results/a4_stability') / f'{RUN}.json'


def measure(run=RUN, root=REFERENCE_ROOT, device='cuda'):
    from data import dclm_val_dataset, load_token_dataset
    from model_io import load_model
    from experiments.a4_stability.diagnostics import profile, context_loss
    model = load_model(Path(root)/run, device=device, precision='fp32')
    validation = load_token_dataset(dclm_val_dataset(), role='val', data_seed=42)
    tokens = torch.tensor(validation[:1]['input_ids'].astype('int64'), device=device)
    stats = profile(model, tokens)
    full = context_loss(model, tokens, keep=tokens.shape[1]-1)
    stats['context_loss_increase'] = {
        f'{keep}_first_{first}': (context_loss(model, tokens, keep, first)-full).tolist()
        for keep in (16, 64, 256) for first in (False, True)}
    return stats


if __name__ == '__main__':
    import modal
    from modal_utils import app, build_image, VOLUME_MOUNTS, MODAL_ENVIRONMENT
    from .diagnostics import save
    probe = app.function(image=build_image(), volumes=VOLUME_MOUNTS, gpu='H100')(measure)
    with modal.enable_output(), app.run(environment_name=MODAL_ENVIRONMENT):
        save(OUTPUT, probe.remote())
    print(OUTPUT)
