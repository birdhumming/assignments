"""Problem 3(c): our router variants, as small edits of RouterMLP.

hash:    route by token id modulo 4 (no router parameters consulted; input-independent of hidden state).
st:      straight-through combine: hard unit weight in the forward pass, softmax gradient to the router.
zloss:   learned top-1 plus a router z-loss, lambda * mean(logsumexp(logits)^2), to bound logit scale.
"""
import torch
from torch import nn
from .p3_moe import MoELM, RouterMLP


class OwnRouterMLP(RouterMLP):
    def __init__(self, config, *, own=None, zloss=0., **kwargs):
        super().__init__(config, **kwargs)
        self.own, self.zloss = own, zloss
        self.token_ids = None

    def forward(self, hidden):
        if self.own == 'hash' and self.token_ids is not None:
            # Fixed token-to-expert assignment; keep the router unused and deterministic.
            x = hidden.flatten(0, 1)
            selected = (self.token_ids.flatten() % 4)[:, None]
            return self._dispatch(hidden, x, selected, torch.ones_like(selected, dtype=x.dtype),
                                  self.router(self.norm(x)).float().softmax(-1))
        return super().forward(hidden)

    def _dispatch(self, hidden, x, selected, weights, p):
        counts = torch.bincount(selected.flatten(), minlength=4)
        self.aux_loss = torch.zeros((), device=x.device)
        order = selected.flatten().argsort(stable=True)
        tokens = torch.arange(len(x), device=x.device)[order]
        chunks = x[tokens].split(counts.tolist())
        values = torch.cat([self.experts.compute(chunk, e) for e, chunk in enumerate(chunks) if len(chunk)])
        out = torch.zeros_like(x).index_add(0, tokens, values*weights.flatten()[order, None])
        if self.record:
            from .diagnostics import router_metrics
            self.stats = router_metrics(p, selected)
        return out.reshape_as(hidden)


def _patched_forward(self, hidden):
    """Straight-through and z-loss hooks around the parent forward (same dispatch path)."""
    x = hidden.flatten(0, 1)
    logits = self.router(self.norm(x)).float()
    if self.training and self.noise:
        logits = logits + self.noise*torch.randn_like(logits)
    p = logits.softmax(-1)
    selected = (logits+self.bias).topk(1, -1).indices
    pe = p.gather(-1, selected)
    if self.own == 'st':
        weights = (1 + pe - pe.detach()).to(x.dtype)        # value 1, gradient of p_e
    else:
        weights = (4*pe).to(x.dtype)
    counts = torch.bincount(selected.flatten(), minlength=4)
    shares = counts.float()/selected.numel()
    aux = self.balance*4*(shares.detach()*p.mean(0)).sum()
    if self.zloss:
        aux = aux + self.zloss*logits.logsumexp(-1).square().mean()
    out = OwnRouterMLP._dispatch(self, hidden, x, selected, weights, p)
    self.aux_loss = aux
    return out


class OwnMoELM(MoELM):
    def __init__(self, config, *, router=None, **kwargs):
        super(MoELM, self).__init__(config, **kwargs)
        router = dict(router or {})
        for block in self.model.layers:
            block.mlp = OwnRouterMLP(config, **router)
            if router.get('own') in ('st', 'zloss'):
                block.mlp.forward = _patched_forward.__get__(block.mlp, OwnRouterMLP)

    def forward(self, input_ids=None, **kwargs):
        for block in self.model.layers:
            block.mlp.token_ids = input_ids
        return super().forward(input_ids, **kwargs)
