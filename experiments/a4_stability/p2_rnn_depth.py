"""Problem 2: plain RNNs, then a four-layer RNN with transformer depth tools.

    uv run python -m experiments.a4_stability.p2_rnn_depth
"""
import math
import torch
from torch import nn
from modeling import LlamaRMSNorm, rotate_half

from . import _runtime
from .p1_transformer_depth import (Residual, NoResidual, NoPreNorm, PostNorm,
                                   GatedResidual, GatedNorm, MLP)


class TanhCell(nn.Module):
    """The only temporal memory is h: h' = tanh(Wx + Uh + b)."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input = nn.Linear(config.hidden_size, config.hidden_size)
        self.hidden = nn.Linear(config.hidden_size, config.hidden_size)

    def project(self, x):
        return (self.input(x),)

    def step(self, projected, h):
        return torch.tanh(projected[0] + self.hidden(h))


class Recurrent(nn.Module):
    """Apply one cell across time. The backend does not change cell.step()."""
    def __init__(self, config, cell_type=TanhCell):
        super().__init__()
        self.config, self.cell = config, cell_type(config)
        self.record_states, self.states = False, []

    def inputs(self, x, position_ids=None):
        return x

    def cell_steps(self, projected, h):
        return _runtime.steps(self.cell, projected, h)

    def forward(self, x, position_ids=None, **context):
        x = self.inputs(x, position_ids)
        if type(self.cell) is TanhCell and not self.record_states:
            return _runtime.fused_tanh(self, x)
        return _runtime.recurrence(self, x)


class RotaryRecurrent(Recurrent):
    """Rotate the normalized recurrent input, not the temporal hidden state."""
    def inputs(self, x, position_ids=None):
        if position_ids is None:
            position_ids = torch.arange(x.shape[1], device=x.device)[None]
        freq = 500_000. ** (-torch.arange(0, x.shape[-1], 2, device=x.device).float() / x.shape[-1])
        angles = position_ids[..., None].float() * freq
        angles = torch.cat((angles, angles), -1)
        return (x * angles.cos() + rotate_half(x) * angles.sin()).to(x.dtype)


class GatedRecurrent(Recurrent):
    """Gate eight output groups; carry the UNGATED h to the next token."""
    def __init__(self, config, cell_type=TanhCell):
        super().__init__(config, cell_type)
        self.gate = nn.Linear(config.hidden_size, 8)

    def forward(self, x, **context):
        h = super().forward(x, **context)
        groups = h.unflatten(-1, (8, -1))
        return (groups * self.gate(x).sigmoid().unsqueeze(-1)).flatten(-2)


class RNNBlock(nn.Module):
    def __init__(self, config, residual_type, recurrent_type, cell_type):
        super().__init__()
        self.recurrent = residual_type(recurrent_type(config, cell_type), nn.LayerNorm(config.hidden_size))
        self.mlp = residual_type(MLP(config), nn.LayerNorm(config.hidden_size))

    def forward(self, x, **context):
        return self.mlp(self.recurrent(x, **context), **context)


class Stack(nn.ModuleList):
    def forward(self, x, **context):
        for layer in self:
            x = layer(x, **context)
        return x


class PlainRNN(nn.Module):
    num_layers, model_name = 4, "rnn4"
    cell_type = TanhCell

    def __init__(self, config, dtype=torch.float32, **kwargs):
        super().__init__()
        if kwargs.get('dropout') or kwargs.get('tie_word_embeddings'):
            raise ValueError("These comparisons use dropout=0 and untied embeddings.")
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.stack = Stack(Recurrent(config, self.cell_type) for _ in range(config.num_hidden_layers))
        self.output_norm = nn.Identity()
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.to(dtype=dtype)

    def initialize_parameters(self):
        """Orthogonal temporal matrices; Xavier input/MLP maps; zero biases."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LayerNorm, LlamaRMSNorm)):
                nn.init.ones_(module.weight)
                if getattr(module, 'bias', None) is not None:
                    nn.init.zeros_(module.bias)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) and name.endswith('.hidden'):
                nn.init.orthogonal_(module.weight)
        nn.init.normal_(self.embedding.weight, std=1/math.sqrt(self.config.hidden_size))
        nn.init.normal_(self.lm_head.weight, std=1/math.sqrt(self.config.hidden_size))

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.lm_head

    def branches(self):
        return list(self.stack)

    def run_depth(self, x, **context):
        return self.stack(x, **context)

    def forward(self, input_ids, position_ids=None, attention_mask=None):
        x = self.embedding(input_ids)
        x = self.run_depth(x, position_ids=position_ids)
        return self.lm_head(self.output_norm(x))


class OneLayerRNN(PlainRNN):
    num_layers, model_name = 1, "rnn1"


class ModernRNN(PlainRNN):
    """Four [LayerNorm -> RNN -> depth skip -> LayerNorm -> SwiGLU -> skip] blocks."""
    residual_type, recurrent_type = Residual, Recurrent

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.stack = Stack(RNNBlock(config, self.residual_type, self.recurrent_type, self.cell_type)
                           for _ in range(config.num_hidden_layers))
        self.to(dtype=next(self.embedding.parameters()).dtype)

    def branches(self):
        return [branch for block in self.stack for branch in (block.recurrent, block.mlp)]

    def run_depth(self, x, **context):
        if self.cell_type is not TanhCell and context.get('position_ids') is None:
            return _runtime.run_stack(self.stack, x)
        return self.stack(x, **context)


class NoResidualRNN(ModernRNN):
    residual_type = NoResidual


class NoPreNormRNN(ModernRNN):
    residual_type = NoPreNorm


class RoPERNN(ModernRNN):
    recurrent_type = RotaryRecurrent


class FinalNormRNN(ModernRNN):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.output_norm = LlamaRMSNorm(config.hidden_size, eps=1e-5).to(self.embedding.weight.dtype)


class PostNormRNN(ModernRNN):
    residual_type = PostNorm


class GatedResidualRNN(ModernRNN):
    residual_type = GatedResidual


class GatedOutputRNN(ModernRNN):
    recurrent_type = GatedRecurrent


class GatedNormRNN(ModernRNN):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        for branch in self.branches():
            branch.norm = GatedNorm(branch.norm, config.hidden_size).to(self.embedding.weight.dtype)


MODELS = dict(plain1=OneLayerRNN, plain4=PlainRNN, modern=ModernRNN,
              no_residual=NoResidualRNN, no_prenorm=NoPreNormRNN, rope=RoPERNN,
              final_norm=FinalNormRNN, postnorm=PostNormRNN,
              gated_residual=GatedResidualRNN, gated_output=GatedOutputRNN, gated_norm=GatedNormRNN)


if __name__ == "__main__":
    from .sweeps import launch_one
    launch_one('rnn')  # Edit the variant and LR here.
