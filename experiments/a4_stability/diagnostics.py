"""Fixed-input depth/time probes for separate checkpoint analysis.

profile(model, tokens) returns full/final-token gradients on the same inputs.
Depth entries are actual branch inputs, followed by the final stream. Branch
ratios use raw F(N(x)) before depth gates/mixing. Temporal entries retain the
real h_t nodes before stacking: gradients of a plotting-only stack are different.
"""
import json
import math

import torch
from torch import nn
from torch.nn import functional as F
from .p1_transformer_depth import Attention
from .p2_rnn_depth import Recurrent


def rms(x):
    return 0. if x is None else x.detach().double().square().mean().sqrt().item()


def attention_summary(q, k):
    k = k.repeat_interleave(q.shape[1]//k.shape[1], 1)
    length = q.shape[-2]
    query = torch.linspace(min(1, length-1), length-1, min(32, max(1, length-1)), device=q.device).long().unique()
    keys = torch.arange(length, device=q.device)
    scores = q[:, :, query].float() @ k.float().transpose(-1, -2) / math.sqrt(q.shape[-1])
    p = scores.masked_fill(keys[None] > query[:, None], -torch.inf).softmax(-1)
    self_mass = p.gather(-1, query[None, None, :, None].expand(*p.shape[:-1], 1)).squeeze(-1)
    return dict(query_positions=query.tolist(), first_mass=p[..., 0].mean(0).tolist(),
                self_mass=self_mass.mean(0).tolist(), uniform_first_self=(1/(query.float()+1)).tolist(),
                mean_lag=(p*(query[:, None]-keys[None])).sum(-1).mean(0).tolist())


def profile(model, tokens):
    """FP32 weights (the default precision='mp'), unpadded sequences, full BPTT."""
    if next(model.parameters()).dtype != torch.float32:
        raise ValueError("Use precision='mp' or 'fp32' for FP32 diagnostic probes.")
    training = model.training
    sequences = [m for m in model.modules() if isinstance(m, Recurrent)]
    flags = [m.record_states for m in sequences]
    model.eval()
    for m in sequences:
        m.record_states = True
    result, handles = {}, []
    inputs, updates, final, qk, routes = {}, {}, [], {}, {}

    def remember(tensor):
        if tensor.requires_grad:
            tensor.retain_grad()
        return tensor

    for j, branch in enumerate(model.branches()):
        def before(module, args, j=j):
            inputs[j] = remember(args[0])
        def after(module, args, output, j=j):
            updates[j] = output
        handles.append(branch.register_forward_pre_hook(before))
        handles.append(getattr(branch, 'operation', branch).register_forward_hook(after))
    handles.append(model.output_norm.register_forward_pre_hook(lambda m, a: final.append(remember(a[0]))))
    for name, module in model.named_modules():
        if isinstance(module, Attention):
            qk[name] = {}
            for label, norm in [('q', module.q_norm), ('k', module.k_norm)]:
                def capture(module, args, output, name=name, label=label):
                    qk[name][label] = output.detach()
                handles.append(norm.register_forward_hook(capture))
        if isinstance(module, nn.Softmax):
            def route(module, args, output, name=name):
                routes[name] = output.detach().mean((1, 2)).tolist()
            handles.append(module.register_forward_hook(route))
    try:
        with torch.enable_grad(), torch.autocast(tokens.device.type, enabled=False):
            for objective in ('full', 'last'):
                inputs.clear(); updates.clear(); final.clear()
                model.zero_grad(set_to_none=True)
                logits = model(tokens[:, :-1])
                loss = (F.cross_entropy(logits[:, -1].float(), tokens[:, -1]) if objective == 'last' else
                        F.cross_entropy(logits.flatten(0, 1).float(), tokens[:, 1:].flatten()))
                loss.backward()
                stream = [inputs[j] for j in sorted(inputs)] + [final[-1]]
                depth = []
                for j, x in enumerate(stream):
                    row = dict(activation_rms=rms(x), gradient_rms=rms(x.grad))
                    if j in updates:
                        row['raw_branch_input_ratio'] = rms(updates[j])/max(rms(x), 1e-30)
                    if j:
                        row['adjacent_cosine'] = F.cosine_similarity(stream[j-1].detach().float(), x.detach().float(), dim=-1).mean().item()
                    depth.append(row)
                result[objective] = dict(loss=loss.item(), depth=depth,
                    hidden_rms=[[rms(h) for h in m.states] for m in sequences],
                    hidden_gradient_rms=[[rms(h.grad) for h in m.states] for m in sequences],
                    fraction_abs_gt_095=[[h.detach().abs().gt(.95).float().mean().item() for h in m.states] for m in sequences])
                if objective == 'last':
                    result['attention'] = {name:attention_summary(v['q'],v['k']) for name,v in qk.items()}
                    result['routing'] = dict(routes)
                    result['temporal_cosine'] = []
                    for m in sequences:
                        h = torch.stack([state.detach() for state in m.states], 1).float()
                        result['temporal_cosine'].append({str(lag):F.cosine_similarity(h[:, lag:], h[:, :-lag], dim=-1).mean().item()
                                                         for lag in (1, 4, 16, 64) if lag < h.shape[1]})
    finally:
        for handle in handles:
            handle.remove()
        for m, flag in zip(sequences, flags):
            m.record_states, m.states = flag, []
        model.zero_grad(set_to_none=True)
        model.train(training)
    return result


@torch.no_grad()
def context_loss(model, tokens, keep, preserve_first=True):
    """Per-target losses for paired context tests; reset state, preserve positions.

    keep counts recent input tokens. preserve_first separately restores the
    original first/BOS token. Compare each result with keep=tokens.shape[1]-1.
    """
    length = tokens.shape[1]-1
    if not 1 <= keep <= length:
        raise ValueError(f'keep must be between 1 and {length}')
    positions = torch.arange(length-keep, length, device=tokens.device)
    if preserve_first and keep < length:
        positions = torch.cat((positions.new_zeros(1), positions))
    training = model.training
    model.eval()
    try:
        logits = model(tokens[:, positions], position_ids=positions[None])
        return F.cross_entropy(logits[:, -1].float(), tokens[:, -1], reduction='none')
    finally:
        model.train(training)


def save(path, value):
    def clean(x):
        if isinstance(x, dict):
            return {k:clean(v) for k,v in x.items()}
        if isinstance(x, list):
            return [clean(v) for v in x]
        return None if isinstance(x, float) and not math.isfinite(x) else x
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False)+'\n')
