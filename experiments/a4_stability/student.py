"""Student variants for Problems 3(d) and 4: registered in sweeps.VARIANTS via MODELS.

P3(d) hybrid residuals: keep the additive depth stream (identity path) and feed each
branch an attention mixture over the embedding and the raw outputs of earlier branches.
    sources = [u0]; x = u0
    for branch j: update = F_j(N_j(route_j(sources))); x = x + update; sources.append(update)
    return x
P4 cells (carry only h across time):
    NoResetCell    GRU with r = 1.
    ResetOnlyCell  GRU with z = 1 (tanh RNN whose recurrent input is gated by r).
    InputGateCell  GRU whose update gate reads only x_t (no U_z h).
    NormGateCell   new update: no reset, RMSNorm on h inside the candidate, update-gate bias -1
                   (start by keeping ~73% of the old state): h' = (1-z) h + z tanh(W x + U RMSNorm(h) + b).
"""
import torch
from torch import nn
from modeling import LlamaRMSNorm
from .p1_transformer_depth import Transformer
from .p2_rnn_depth import ModernRNN
from .p3_attention_residuals import AttentionDepth
from .p4_recurrent_updates import GRUCell


class HybridDepth(AttentionDepth):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.routers = self.routers[:-1]

    def run_depth(self, x, **context):
        sources = [x]
        for j, branch in enumerate(self.branches()):
            update = branch(self.routers[j](sources), **context)
            x = x + update
            sources.append(update)
        return x


class HybridTransformer(HybridDepth, Transformer):
    pass


class HybridRNN(HybridDepth, ModernRNN):
    pass


class NoResetCell(GRUCell):
    def make_reset(self, config):
        return None

    def project(self, x):
        return self.input(x), self.update.project(x)

    def step(self, projected, h):
        candidate_input, update_input = projected
        z = self.update(update_input, h)
        return self.combine(h, torch.tanh(candidate_input + self.hidden(h)), z)


class ResetOnlyCell(GRUCell):
    def make_update(self, config):
        return None

    def project(self, x):
        return self.input(x), self.reset.project(x)

    def step(self, projected, h):
        candidate_input, reset_input = projected
        r = self.reset(reset_input, h)
        return torch.tanh(candidate_input + self.hidden(r*h))


class InputGateCell(GRUCell):
    def make_update(self, config):
        return nn.Linear(config.hidden_size, config.hidden_size)

    def project(self, x):
        return self.input(x), self.reset.project(x), self.update(x)

    def step(self, projected, h):
        candidate_input, reset_input, update_input = projected
        r, z = self.reset(reset_input, h), update_input.sigmoid()
        return self.combine(h, torch.tanh(candidate_input + self.hidden(r*h)), z)


class NormGateCell(NoResetCell):
    def __init__(self, config):
        super().__init__(config)
        self.norm = LlamaRMSNorm(config.hidden_size, eps=1e-6)

    def step(self, projected, h):
        candidate_input, update_input = projected
        z = self.update(update_input, h)
        return self.combine(h, torch.tanh(candidate_input + self.hidden(self.norm(h))), z)


class NoResetGRU(ModernRNN):
    cell_type = NoResetCell


class ResetOnlyGRU(ModernRNN):
    cell_type = ResetOnlyCell


class InputGateGRU(ModernRNN):
    cell_type = InputGateCell


class NormGateGRU(ModernRNN):
    cell_type = NormGateCell
    update_bias = -1.

    def initialize_parameters(self):
        super().initialize_parameters()
        for module in self.modules():
            if isinstance(module, NormGateCell):
                nn.init.constant_(module.update.input.bias, self.update_bias)


class NormGateGRU0(NormGateGRU):
    update_bias = 0.


# name -> (problem, model, lr center)
MODELS = {'d8-hybrid': (3, HybridTransformer, .003), 'rnn-hybrid': (3, HybridRNN, .001),
          'gru-no-reset': (4, NoResetGRU, .003), 'gru-reset-only': (4, ResetOnlyGRU, .003),
          'gru-input-gate': (4, InputGateGRU, .003), 'normgate': (4, NormGateGRU, .003),
          'normgate-b0': (4, NormGateGRU0, .003)}
