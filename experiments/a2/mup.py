"""P4.2 parameterizations for full training runs via train.py's model/optimizer builder hooks.

policy='baseline' : the course default (initialize_model + one LR) at any width/depth.
policy='mup'      : width muP anchored so that width n0 (512) reproduces the baseline exactly:
                    embedding std 1/n0, readout std 1/sqrt(n0) with output multiplier 1/m,
                    hidden std 1/sqrt(fan_in), hidden-matrix LR eta/m; m = n/n0.
policy='depth_mup', 'completep' : muP width rules plus the Problem 4.1 depth table with r = L/L0
                    (residual multiplier c_L, block LR multiplier, block Adam eps multiplier).
Weight decay stays masked like the baseline (no decay on embeddings/norms); AdamW's decoupled decay
multiplies by each group's LR, so hidden matrices under muP also decay more slowly (see writeup).
"""
import math

import torch
from torch import nn

from modeling import AutoregressiveLM, LlamaRMSNorm, _truncated_normal_
from optimizers import ADAMW_EPSILON

POLICIES = ('baseline', 'mup', 'depth_mup', 'completep')
HIDDEN_SUFFIXES = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')


def depth_multipliers(policy, depth, reference_depth):
    r = depth / reference_depth
    if policy in ('baseline', 'mup'):
        return dict(residual=1., block_lr=1., block_eps=1.)
    if policy == 'depth_mup':
        return dict(residual=r ** -.5, block_lr=r ** -.5, block_eps=r ** -.5)
    if policy == 'completep':
        return dict(residual=1. / r, block_lr=1., block_eps=1. / r)
    raise ValueError(policy)


def classify(name):
    """embed | head | hidden | block_norm | final_norm for a named parameter of AutoregressiveLM."""
    if name == 'model.embed_tokens.weight':
        return 'embed'
    if name == 'lm_head.weight':
        return 'head'
    if name == 'model.norm.weight':
        return 'final_norm'
    if name.startswith('model.layers.'):
        if any(name.endswith(f'{s}.weight') for s in HIDDEN_SUFFIXES):
            return 'hidden'
        return 'block_norm'
    raise ValueError(f'Unclassified parameter {name}')


class ScaledLM(AutoregressiveLM):
    def __init__(self, config, dtype=torch.float32, qk_norm=True, tie_word_embeddings=False, dropout=0.,
                 *, policy='mup', reference_width=512, reference_depth=8):
        if policy not in POLICIES:
            raise ValueError(policy)
        if tie_word_embeddings:
            raise ValueError('ScaledLM assumes untied embeddings')
        super().__init__(config, dtype=dtype, qk_norm=qk_norm, tie_word_embeddings=False, dropout=dropout)
        self.policy = policy
        self.reference_width = reference_width
        self.reference_depth = reference_depth
        self.width_multiplier = config.hidden_size / reference_width
        self.output_multiplier = 1. if policy == 'baseline' else 1. / self.width_multiplier
        self.residual_multiplier = depth_multipliers(policy, config.num_hidden_layers, reference_depth)['residual']
        if self.residual_multiplier != 1.:
            scale = self.residual_multiplier
            for layer in self.model.layers:
                layer.self_attn.register_forward_hook(lambda m, a, out: out * scale)
                layer.mlp.register_forward_hook(lambda m, a, out: out * scale)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None):
        logits = super().forward(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
        return logits if self.output_multiplier == 1. else logits * self.output_multiplier

    @torch.no_grad()
    def initialize_parameters(self):
        n = self.config.hidden_size
        n0 = self.reference_width if self.policy != 'baseline' else n
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.lm_head:
                _truncated_normal_(module.weight, 1. / math.sqrt(module.weight.shape[1]))
            elif isinstance(module, LlamaRMSNorm):
                module.weight.fill_(1.)
        base = torch.empty(self.config.vocab_size, n, dtype=torch.float32)
        _truncated_normal_(base, 1.)
        base = base.to(self.model.embed_tokens.weight.dtype)
        self.model.embed_tokens.weight.copy_(base / n0)
        self.lm_head.weight.copy_(base / math.sqrt(n0))


def build_model(model_config, dtype=torch.float32, qk_norm=True, tie_word_embeddings=False, dropout=0.,
                **kwargs):
    return ScaledLM(model_config, dtype=dtype, qk_norm=qk_norm, tie_word_embeddings=tie_word_embeddings,
                    dropout=dropout, **kwargs)


def parameter_groups(model, learning_rate, weight_decay, eps=ADAMW_EPSILON):
    """Groups ordered so param_groups[0] carries the base LR (train.py logs that one)."""
    policy = getattr(model, 'policy', 'baseline')
    m = getattr(model, 'width_multiplier', 1.)
    mult = depth_multipliers(policy, model.config.num_hidden_layers, getattr(model, 'reference_depth', 8))
    hidden_lr = learning_rate if policy == 'baseline' else learning_rate / m
    spec = {
        'embed': dict(lr=learning_rate, eps=eps, weight_decay=0.),
        'head': dict(lr=learning_rate, eps=eps, weight_decay=weight_decay),
        'final_norm': dict(lr=learning_rate, eps=eps, weight_decay=0.),
        'hidden': dict(lr=hidden_lr * mult['block_lr'], eps=eps * mult['block_eps'], weight_decay=weight_decay),
        'block_norm': dict(lr=learning_rate * mult['block_lr'], eps=eps * mult['block_eps'], weight_decay=0.),
    }
    buckets = {k: [] for k in spec}
    for name, p in model.named_parameters():
        if p.requires_grad:
            buckets[classify(name)].append(p)
    return [dict(params=ps, **spec[k]) for k, ps in buckets.items() if ps]


def build_optimizer(model, optimizer_name, learning_rate, weight_decay, beta1, beta2):
    if optimizer_name != 'adamw':
        raise ValueError('experiments.a2.mup only supports adamw')
    groups = parameter_groups(model, learning_rate, weight_decay)
    fused = any(p.is_cuda for g in groups for p in g['params'])
    return torch.optim.AdamW(groups, lr=learning_rate, betas=(beta1, beta2), weight_decay=0.,
                             eps=ADAMW_EPSILON, fused=fused)


if __name__ == '__main__':
    from model_config import LMConfig
    for policy, n, L in (('baseline', 128, 8), ('mup', 128, 8), ('mup', 512, 8), ('depth_mup', 512, 4), ('completep', 512, 16)):
        cfg = LMConfig(f'a2-w{n}-d{L}', 4096, 64, n, int(3.5 * n), L, n // 64, n // 64)
        model = build_model(cfg, policy=policy)
        model.initialize_parameters()
        opt = build_optimizer(model, 'adamw', .003, .1, .9, .95)
        x = torch.randint(0, 4096, (2, 64))
        logits = model(x)
        print(policy, n, L, 'out_mult', model.output_multiplier, 'res_mult', round(model.residual_multiplier, 4),
              'logit rms', round(logits.float().pow(2).mean().sqrt().item(), 3),
              'embed std', round(model.model.embed_tokens.weight.std().item(), 5),
              'groups', [(round(g['lr'], 6), g['eps'], g['weight_decay'], len(g['params'])) for g in opt.param_groups])
