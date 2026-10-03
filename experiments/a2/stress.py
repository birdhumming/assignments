"""P4.1 architecture and measurements; students supply initialization/LR rules.

No parameterization recipe, fitted optimum, or sweep result is included.
"""
import hashlib
import json
import math
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint

from data import PreprocessedTokenDataset
from experiments.a2.probe_math import measure_delta
from experiments.a2.readout import readout_alignment
from model_config import LMConfig
from modeling import LlamaRMSNorm, LlamaRotaryEmbedding, apply_rotary_pos_emb


@dataclass(frozen=True)
class StressConfig:
    width: int = 640
    depth: int = 2
    head_dim: int = 64
    vocab: int = 4096
    context: int = 1024
    batch: int = 64
    steps: int = 5
    microbatch: int = 8
    probe_sequences: int = 8
    checkpoint_blocks: int = 16
    seed: int = 42
    precision: str = 'fp32'

    def validate(self):
        for key, value in asdict(self).items():
            if key == 'precision':
                if value not in ('fp32', 'mp'):
                    raise ValueError('precision must be fp32 (width) or mp (depth)')
                continue
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f'{key} must be an integer')
            if key != 'seed' and value <= 0:
                raise ValueError(f'{key} must be positive')
        if self.width % self.head_dim or self.head_dim % 2 or self.context < 2:
            raise ValueError('Require width divisible by even head_dim, context >= 2')


class GatedSiLUBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.head_dim = c.head_dim
        self.residual_multiplier = 1.0  # Student policy may change this.
        self.norm1 = LlamaRMSNorm(c.width, 1e-5)
        self.norm2 = LlamaRMSNorm(c.width, 1e-5)
        self.qnorm = LlamaRMSNorm(c.head_dim, 1e-5)
        self.knorm = LlamaRMSNorm(c.head_dim, 1e-5)
        self.q, self.k, self.v, self.o = (
            nn.Linear(c.width, c.width, bias=False) for _ in range(4))
        inner = int(3.5 * c.width)
        self.gate = nn.Linear(c.width, inner, bias=False)
        self.up = nn.Linear(c.width, inner, bias=False)
        self.down = nn.Linear(inner, c.width, bias=False)

    def forward(self, x, cos, sin):
        h = self.norm1(x)
        shape = (*x.shape[:2], x.shape[-1] // self.head_dim, self.head_dim)
        q, k, v = [p(h).view(shape).transpose(1, 2) for p in (self.q, self.k, self.v)]
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        q, k = self.qnorm(q), self.knorm(k)
        h = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        h = h.transpose(1, 2).contiguous().reshape(x.shape)
        x = x + self.residual_multiplier * self.o(h).float()
        h = self.norm2(x)
        h = self.down(F.silu(self.gate(h)) * self.up(h))
        return x + self.residual_multiplier * h.float()


class StressTransformer(nn.Module):
    def __init__(self, config, device='cpu'):
        super().__init__()
        config.validate()
        self.config = c = config
        self.output_multiplier = 1.0  # Student policy may change this.
        # No implicit PyTorch initialization serves as an assignment answer.
        with torch.device('meta'):
            self.embed = nn.Embedding(c.vocab, c.width)
            self.blocks = nn.ModuleList([GatedSiLUBlock(c) for _ in range(c.depth)])
            self.norm = LlamaRMSNorm(c.width, 1e-5)
            self.head = nn.Linear(c.width, c.vocab, bias=False)
        self.to(dtype=torch.float32)
        self.to_empty(device=device)
        rc = LMConfig('p31-course', c.vocab, c.context, c.width, int(3.5*c.width), c.depth,
                      c.width // c.head_dim, c.width // c.head_dim, head_dim=c.head_dim)
        self.rope = LlamaRotaryEmbedding(rc).to(device)
        with torch.no_grad():
            for p in self.parameters():
                p.fill_(float('nan'))

    def forward(self, tokens):
        x = self.embed(tokens)
        cos, sin = self.rope(x, torch.arange(tokens.shape[1], device=x.device)[None, :])
        for start in range(0, len(self.blocks), self.config.checkpoint_blocks):
            stop = min(start + self.config.checkpoint_blocks, len(self.blocks))
            def segment(h, start=start, stop=stop):
                for i in range(start, stop):
                    h = self.blocks[i](h, cos, sin)
                return h
            if self.training and torch.is_grad_enabled():
                x = checkpoint(segment, x, use_reentrant=False, preserve_rng_state=False)
            else:
                x = segment(x)
        return self.head(self.norm(x) * self.output_multiplier)


def load_tokens(path, sequences, context=1024):
    """Read a fixed contiguous token prefix, rechunked without dropping tokens.

    Call once per train/validation split, then reuse these tensors across runs.
    This only reads the requested prefix, not the full corpus into RAM.
    """
    if sequences <= 0 or context < 2:
        raise ValueError('Invalid requested token shape')
    data = PreprocessedTokenDataset(path)
    needed = sequences * context
    if needed > data.tokens.size:
        raise ValueError('Dataset is too short')
    return torch.from_numpy(np.array(data.tokens.reshape(-1)[:needed], dtype=np.int64)).reshape(sequences, context)


def _rms(x):
    return x.double().square().mean().sqrt().item()


def _loss(logits, tokens):
    return F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1))


@contextmanager
def _fp32_math():
    matmul = torch.backends.cuda.matmul.allow_tf32
    cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with sdpa_kernel(SDPBackend.MATH):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn


class StressProbe:
    """Fixed-input diagnostics with CPU feature references and matrix snapshots.

    Selected feature references persist; alignment also retains each matrix's
    pre-update inputs and weights on CPU when enabled.
    """
    def __init__(self, model, tokens, alignment=True):
        self.model = model
        self.tokens = tokens
        self.initial = {}
        self.initial_readout = model.head.weight.detach().float().cpu().clone()
        self.maps = {name: module for name, module in model.named_modules()
                     if isinstance(module, nn.Linear)} if alignment else {}
        n = len(model.blocks)
        self.positions = sorted({0, n // 2, n - 1})

    @torch.no_grad()
    def sample(self):
        features, residual, branches, inputs, handles = {}, {}, {}, {}, []
        readout = {}
        def feature(name):
            def hook(module, args, output):
                h = output.detach().cpu().clone()
                if name not in self.initial:
                    self.initial[name] = h.clone()
                if name == 'final_norm':
                    readout.update(readout_alignment(self.model.head.weight, output,
                                   self.initial_readout, self.initial[name]))
                initial = _rms(self.initial[name])
                movement = _rms(h - self.initial[name])
                features[name] = {'rms': _rms(h), 'movement': movement, 'initial_rms': initial,
                    'relative_movement': movement / initial if initial else None}
                residual[name] = _rms(args[0])
            return hook
        for i in self.positions:
            b = self.model.blocks[i]
            handles.append(b.norm1.register_forward_hook(feature(f'blocks.{i}.norm1')))
            for part in ['o', 'down']:
                name = f'blocks.{i}.{part}'
                def branch(module, args, output, name=name):
                    branches[name] = _rms(output)
                handles.append(getattr(b, part).register_forward_hook(branch))
        handles.append(self.model.norm.register_forward_hook(feature('final_norm')))
        for name, module in self.maps.items():
            def capture(module, args, name=name):
                inputs[name] = args[0].detach().cpu().clone()
            handles.append(module.register_forward_pre_hook(capture))
        was_training = self.model.training
        try:
            self.model.eval()
            logits = self.model(self.tokens)
            row = {'logit_rms': _rms(logits), 'features': features,
                       'residual_rms': residual, 'unscaled_branch_rms': branches,
                       'readout_alignment': readout}
        finally:
            for h in handles:
                h.remove()
            self.model.train(was_training)
        return row, inputs

    @torch.no_grad()
    def snapshot(self):
        return {name: module.weight.detach().cpu().clone() for name, module in self.maps.items()}

    @torch.no_grad()
    def updates(self, before, inputs, step):
        rows = []
        for name, module in self.maps.items():
            old = before[name].to(module.weight.device)
            rows.append(dict(step=step, input_step=step - 1, parameter=name + '.weight',
                             **measure_delta(old, module.weight, inputs[name])))
        return rows


def _validate_tokens(tokens, count, c, label):
    if tokens.dtype != torch.long or tokens.ndim != 2 or tokens.shape != (count, c.context):
        raise ValueError(f'{label} must be int64 [{count}, {c.context}]')
    if count == 0:
        raise ValueError(f'{label} cannot be empty')
    if tokens.min() < 0 or tokens.max() >= c.vocab:
        raise ValueError(f'{label} contains out-of-vocabulary IDs')


def run(config, train_tokens, val_tokens, *, base_lr, initialize_fn=None,
        groups_fn=None, device='cuda', alignment=True, output=None):
    """Run config.steps constant-LR Adam updates (five by default), without a sweep.

    Callbacks are mandatory student work. Parameters, gradients, Adam states,
    and residual additions stay FP32. Depth runs use BF16 autocast; width runs
    keep all computation FP32 with the math attention backend.
    Microbatches accumulate to the specified total batch before each update.
    """
    config.validate()
    if initialize_fn is None or groups_fn is None:
        raise NotImplementedError('Supply both student callbacks before allocating the stress model.')
    if not math.isfinite(base_lr) or base_lr <= 0:
        raise ValueError('base_lr must be positive')
    if output is not None and Path(output).exists():
        raise FileExistsError(output)
    _validate_tokens(train_tokens, config.steps * config.batch, config, 'train_tokens')
    _validate_tokens(val_tokens, len(val_tokens), config, 'val_tokens')
    if not len(val_tokens):
        raise ValueError('Validation data cannot be empty')
    device = torch.device(device)
    attention = _fp32_math() if config.precision == 'fp32' else nullcontext()
    # Evaluation/probes and optimizer updates share this scope. Cached casts
    # would retain no-grad weights or weights from before an optimizer update.
    with torch.random.fork_rng(devices=[device.index or 0] if device.type == 'cuda' else []), attention, torch.autocast(device.type, dtype=torch.bfloat16, enabled=config.precision == 'mp', cache_enabled=False):
        model = StressTransformer(config, device)
        torch.manual_seed(config.seed)
        with torch.no_grad():
            initialize_fn(model)
        for name, p in model.named_parameters():
            if p.dtype != torch.float32 or not torch.isfinite(p).all():
                raise ValueError(f'{name} must be fully initialized and FP32')
        if not all(math.isfinite(v) for v in [model.output_multiplier, *[b.residual_multiplier for b in model.blocks]]):
            raise ValueError('Forward multipliers must be finite')
        groups = list(groups_fn(model, base_lr))
        for g in groups:
            if set(g) != {'params', 'lr', 'eps'} or not all(math.isfinite(g[k]) and g[k] > 0 for k in ['lr', 'eps']):
                raise ValueError('Each parameter group needs positive lr and eps')
            g['params'] = list(g['params'])
        params = [p for g in groups for p in g['params']]
        if len(params) != len(set(map(id, params))) or set(map(id, params)) != set(map(id, model.parameters())):
            raise ValueError('Groups must cover every model parameter exactly once')
        optimizer = torch.optim.Adam(groups, betas=(.9, .95), weight_decay=0., foreach=False)
        probe = StressProbe(model, val_tokens[:config.probe_sequences].to(device), alignment)
        history, alignments = [], []
        def evaluate():
            model.eval()
            total = 0.
            with torch.no_grad():
                for b in val_tokens.split(config.microbatch):
                    b = b.to(device)
                    total += _loss(model(b), b).item() * len(b)
            model.train()
            return total / len(val_tokens)
        row, inputs = probe.sample()
        history.append(dict(step=0, val_loss=evaluate(), **row))
        for step in range(1, config.steps + 1):
            before = probe.snapshot()
            optimizer.zero_grad(set_to_none=True)
            total = 0.
            batch = train_tokens[(step - 1) * config.batch:step * config.batch]
            for b in batch.split(config.microbatch):
                b = b.to(device)
                loss = _loss(model(b), b) * (len(b) / config.batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite training loss at step {step}')
                loss.backward()
                total += loss.detach().item()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError(f'Missing or nonfinite gradient at step {step}')
            optimizer.step()
            alignments.extend(probe.updates(before, inputs, step))
            row, inputs = probe.sample()
            history.append(dict(step=step, train_loss=total, val_loss=evaluate(), **row))
        def digest(t):
            return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        result = {'config': asdict(config), 'base_lr': base_lr,
            'precision': 'float32' if config.precision == 'fp32' else 'bfloat16_autocast_fp32_residual',
            'history': history, 'alignment': alignments,
            'data_sha256': {'train': digest(train_tokens), 'validation': digest(val_tokens)},
            'parameter_groups': [{'names': [n for n, p in model.named_parameters()
                if any(p is q for q in g['params'])], 'lr': g['lr'], 'eps': g['eps']} for g in groups],
            'output_multiplier': model.output_multiplier,
            'residual_multipliers': [b.residual_multiplier for b in model.blocks],
            'probe_block_indices': probe.positions}
        encoded = json.dumps(result, indent=2, allow_nan=False) + '\n'
        if output is not None:
            path = Path(output)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('x') as f:
                f.write(encoded)
        return result
