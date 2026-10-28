"""Problem 3: replace additive depth skips; leave the branches themselves fixed.

    uv run python -m experiments.a4_stability.p3_attention_residuals
"""
import torch
from torch import nn
from modeling import LlamaRMSNorm
from .p1_transformer_depth import Transformer, NoResidual
from .p2_rnn_depth import ModernRNN


class DepthEMA:
    """x' = beta*x + F(N(x)), once per recurrent/attention or MLP branch."""
    residual_type = NoResidual  # branch(x) now returns the raw F(N(x)).
    beta = .5

    def run_depth(self, x, **context):
        for branch in self.branches():
            x = self.beta*x + branch(x, **context)
        return x


class DepthHalf:
    """Add F(N(x)) to the mean of the previous two depth-stream states.

    The first branch uses the embedding alone as its residual stream.
    """
    residual_type = NoResidual

    def run_depth(self, x, **context):
        previous = None
        for branch in self.branches():
            residual = x if previous is None else .5*(x+previous)
            previous, x = x, residual + branch(x, **context)
        return x


class AttentionMix(nn.Module):
    """Score normalized keys, but mix the original, unnormalized values."""
    def __init__(self, width):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(width))
        self.key_norm = LlamaRMSNorm(width, eps=1e-6)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, sources):
        values = torch.stack(sources)
        scores = (self.key_norm(values.float()) * self.query).sum(-1)
        weights = self.softmax(scores)
        return (values * weights[..., None].to(values.dtype)).sum(0)


class UniformMix(nn.Module):
    def __init__(self, width):
        super().__init__()

    def forward(self, sources):
        return torch.stack(sources).mean(0)


class AttentionDepth:
    """Route over the embedding, completed block sums, and a partial block sum.

    block_size=1 retains every raw branch output (full AttnRes).
    The final read routes once more before the model's usual output norm/head.
    """
    residual_type = NoResidual
    mix_type, block_size = AttentionMix, 1

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.routers = nn.ModuleList(self.mix_type(config.hidden_size)
                                     for _ in range(len(self.branches())+1))
        self.routers.to(dtype=next(self.parameters()).dtype)

    def initialize_parameters(self):
        super().initialize_parameters()
        for router in self.routers:
            if isinstance(router, AttentionMix):
                nn.init.zeros_(router.query)
                nn.init.ones_(router.key_norm.weight)

    def run_depth(self, x, **context):
        sources, partial = [x], None
        for j, branch in enumerate(self.branches()):
            available = sources + ([] if partial is None else [partial])
            update = branch(self.routers[j](available), **context)
            partial = update if partial is None else partial + update
            if (j+1) % self.block_size == 0:
                sources.append(partial)
                partial = None
        return self.routers[-1](sources + ([] if partial is None else [partial]))


class UniformDepth(AttentionDepth):
    mix_type = UniformMix


class HalfTransformer(DepthHalf, Transformer):
    pass


class EMA01Transformer(DepthEMA, Transformer):
    beta = .1


class EMA09Transformer(DepthEMA, Transformer):
    beta = .9


class UniformTransformer(UniformDepth, Transformer):
    pass


class AttentionTransformer(AttentionDepth, Transformer):
    pass


class Block2Transformer(AttentionTransformer):
    block_size = 2


class Block4Transformer(AttentionTransformer):
    block_size = 4


class HalfRNN(DepthHalf, ModernRNN):
    pass


class EMA01RNN(DepthEMA, ModernRNN):
    beta = .1


class EMA09RNN(DepthEMA, ModernRNN):
    beta = .9


class UniformRNN(UniformDepth, ModernRNN):
    pass


class AttentionRNN(AttentionDepth, ModernRNN):
    pass


class Block2RNN(AttentionRNN):
    block_size = 2


class Block4RNN(AttentionRNN):
    block_size = 4


# TODO: subclass DepthEMA, DepthHalf, or AttentionDepth to try your own depth scheme.
# Add the resulting model class to MODELS, with a new name for its checkpoints.
MODELS = dict(transformer_baseline=Transformer, rnn_baseline=ModernRNN,
              transformer_half=HalfTransformer, transformer_ema01=EMA01Transformer,
              transformer_ema09=EMA09Transformer, transformer_uniform=UniformTransformer,
              transformer_attention=AttentionTransformer, transformer_block2=Block2Transformer,
              transformer_block4=Block4Transformer, rnn_half=HalfRNN, rnn_ema01=EMA01RNN,
              rnn_ema09=EMA09RNN, rnn_uniform=UniformRNN, rnn_attention=AttentionRNN,
              rnn_block2=Block2RNN, rnn_block4=Block4RNN)


if __name__ == "__main__":
    from .sweeps import launch_one
    launch_one('rnn-block2')  # Edit the variant and LR here.
