"""A2 optimizer factories for the shared trainer's optimizer_builder hook."""
import math
import torch
from optimizers import should_apply_weight_decay
from experiments.a2.hyperball import AdamH, build_adamh_parameter_groups
from experiments.a2.muon import SingleDeviceMuonWithAuxAdam


def build_muon_parameter_groups(model, learning_rate, weight_decay, beta1, beta2,
                                adam_learning_rate=3e-4, momentum=.95, eps=1e-8):
    """Muon for hidden Linear weights; masked AdamW for every other parameter.

    Models must expose their output layer as lm_head or head. For a different
    architecture, construct groups explicitly and use SingleDeviceMuonWithAuxAdam.
    """
    heads = [getattr(model, name, None) for name in ('lm_head', 'head')]
    heads = [head for head in heads if isinstance(head, torch.nn.Module)]
    if not heads:
        raise ValueError('Muon routing requires model.lm_head or model.head')
    excluded = {id(p) for head in heads for p in head.parameters()}
    embedding_ids = {id(p) for module in model.modules() if isinstance(module, torch.nn.Embedding)
                     for p in module.parameters()}
    excluded.update(embedding_ids)
    hidden, decayed, unscaled = [], [], []
    seen = set()
    for module in model.modules():
        for name, p in module.named_parameters(recurse=False):
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            if isinstance(module, torch.nn.Linear) and name == 'weight' and p.ndim == 2 and id(p) not in excluded:
                hidden.append(p)
            elif should_apply_weight_decay(module, name) and id(p) not in embedding_ids:
                decayed.append(p)
            else:
                unscaled.append(p)
    if not hidden:
        raise ValueError('Muon requires at least one hidden Linear weight')
    groups = [dict(params=hidden, use_muon=True, lr=learning_rate,
                   momentum=momentum, weight_decay=weight_decay)]
    for params, wd in ((decayed, weight_decay), (unscaled, 0.)):
        if params:
            groups.append(dict(params=params, use_muon=False, lr=adam_learning_rate,
                               betas=(beta1, beta2), eps=eps, weight_decay=wd))
    return groups


def build_optimizer(model, optimizer_name, learning_rate, weight_decay, beta1, beta2,
                    *, adam_learning_rate=None, momentum=.95, eps=1e-8):
    """Build Hyperball or Muon; no implicit width/depth parameterization changes."""
    for name, value in [('learning_rate', learning_rate), ('weight_decay', weight_decay),
                        ('eps', eps), ('adam_learning_rate', adam_learning_rate)]:
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError(f'{name} must be finite and nonnegative')
    if not all(0 <= b < 1 for b in (beta1, beta2, momentum)):
        raise ValueError('betas and momentum must be in [0, 1)')
    if optimizer_name in {'adamh', 'hyperball'}:
        if weight_decay != 0:
            raise ValueError('Hyperball requires weight_decay=0')
        groups, _ = build_adamh_parameter_groups(model, learning_rate, adam_learning_rate)
        return AdamH(groups, betas=(beta1, beta2), eps=eps)
    if optimizer_name == 'muon':
        groups = build_muon_parameter_groups(model, learning_rate, weight_decay, beta1, beta2,
            adam_learning_rate=3e-4 if adam_learning_rate is None else adam_learning_rate,
            momentum=momentum, eps=eps)
        return SingleDeviceMuonWithAuxAdam(groups)
    raise ValueError(f'Expected adamh, hyperball, or muon; got {optimizer_name!r}')
