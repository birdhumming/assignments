"""Use the native training recipe; measure detailed diagnostics afterwards."""
import sys
from dataclasses import replace

from model_config import MODEL_CONFIGS
from train import TrainConfig


def training_config(model, lr=.003, *, problem, **overrides):
    module = model.__module__
    if module == '__main__':
        module = sys.modules[module].__spec__.name
    architecture = replace(MODEL_CONFIGS['d8'], name=model.model_name,
                           num_hidden_layers=model.num_layers)
    config = TrainConfig(
        model_config=architecture, model_builder=f'{module}:{model.__qualname__}',
        learning_rate=lr, torch_compile_mode='default',
        qk_norm=getattr(model, 'use_qk_norm', True),
        run_name_suffix=f'a4-{model.__name__}',
        wandb_tags=('a4', f'p{problem}', model.__name__))
    return replace(config, **overrides)
