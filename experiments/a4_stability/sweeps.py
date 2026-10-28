"""P1/P2 reference grid and configs for students' local GPU or Modal runs.

Edit the call at the bottom, or import config(), learning_rates(), and launch().
Students use their own W&B project from utils.py, as usual.
"""
from dataclasses import dataclass

from . import p1_transformer_depth as p1, p2_rnn_depth as p2
from . import p3_attention_residuals as p3, p4_recurrent_updates as p4
from .common import training_config

LR_FACTORS = (.25, .5, 1., 2., 4.)
PUBLIC_PROJECT = 'hashimoto-group/public-alchemy-a4'
SHARED_ROOT = '/root/shared_data/a4_stability'


@dataclass(frozen=True)
class Variant:
    problem: int
    model: type
    center: float


# Centers are informed by the instructor trial, not selected from these runs.
# P3 implementations and the base GRU support student experiments; no runs are supplied.
VARIANTS = {}
for key, model in p1.MODELS.items():
    name = 'd8' if key == 'baseline' else 'd8-' + key.replace('_', '-')
    center = .009 if key == 'gated_norm' else .001 if key in ('no_prenorm', 'postnorm') else .003
    VARIANTS[name] = Variant(1, model, center)
for key, model in p2.MODELS.items():
    name = {'plain1': 'rnn1', 'plain4': 'rnn4', 'modern': 'rnn'}.get(key, 'rnn-' + key.replace('_', '-'))
    VARIANTS[name] = Variant(2, model, .003 if key == 'plain1' else .001)
for key, model in p3.MODELS.items():
    family, variant = key.split('_', 1)
    if variant == 'baseline':
        continue  # Reuse P1/P2's five baseline runs.
    name = ('d8' if family == 'transformer' else 'rnn') + '-' + variant
    center = .001 if family == 'rnn' else .009 if variant == 'block2' else .003
    VARIANTS[name] = Variant(3, model, center)
for key, model in p4.MODELS.items():
    if key == 'tanh':
        continue  # Reuse the modern tanh RNN.
    name = 'gru' if key == 'gru' else 'gru-' + key.replace('_', '-')
    VARIANTS[name] = Variant(4, model, .003)

PROVIDED_VARIANTS = tuple(name for name, variant in VARIANTS.items() if variant.problem in (1, 2))


def learning_rates(name):
    return tuple(float(f'{VARIANTS[name].center*factor:.8g}') for factor in LR_FACTORS)


def run_name(name, lr):
    return f'a4-{name}-lr{lr:g}'


def config(name, lr=None, **overrides):
    """One normal TrainConfig. This controls local names, Modal names, and W&B."""
    variant = VARIANTS[name]
    lr = variant.center if lr is None else lr
    options = dict(run_name=run_name(name, lr), run_name_suffix=None,
                   keep_checkpoint_steps=(100,),
                   wandb_tags=('a4', 'a4-reference', f'p{variant.problem}', name))
    options.update(overrides)
    return training_config(variant.model, lr, problem=variant.problem, **options)


def grid(names=None):
    return [config(name, lr) for name in (PROVIDED_VARIANTS if names is None else names)
            for lr in learning_rates(name)]


def launch(names=('d8',), *, rates=None, local=False, dry_run=False, **overrides):
    """Launch only the named models. Omit rates for their five-rate grids."""
    configs = [config(name, lr, **overrides) for name in names
               for lr in (learning_rates(name) if rates is None else rates)]
    if dry_run:
        for cfg in configs:
            print(cfg.run_name, cfg.model_builder, cfg.learning_rate)
    elif local:
        from train import train
        for cfg in configs:
            train(cfg)
    else:
        from modal_train import launch_training_jobs
        return launch_training_jobs(configs, max_parallel_runs=min(2, len(configs)))


def launch_one(name='d8', lr=None, **kwargs):
    return launch((name,), rates=(VARIANTS[name].center if lr is None else lr,), **kwargs)


if __name__ == '__main__':
    # Reference sweeps are supplied. Edit this only for a new experiment.
    launch(('d8',), dry_run=True)
