"""Problem 4: edit the temporal update, keeping the modern RNN's depth stack.

    uv run python -m experiments.a4_stability.p4_recurrent_updates

z is the fraction of NEW information, following the handout. Reset is applied
BEFORE the candidate's recurrent projection; torch.nn.GRU uses a different form.
"""
import torch
from torch import nn

from .p2_rnn_depth import ModernRNN, TanhCell


class Gate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input = nn.Linear(config.hidden_size, config.hidden_size)
        self.hidden = nn.Linear(config.hidden_size, config.hidden_size)

    def project(self, x):
        return self.input(x)

    def forward(self, projected, h):
        return (projected + self.hidden(h)).sigmoid()


class GRUCell(TanhCell):
    def __init__(self, config):
        super().__init__(config)
        self.reset = self.make_reset(config)
        self.update = self.make_update(config)

    def make_reset(self, config):
        return Gate(config)

    def make_update(self, config):
        return Gate(config)

    def project(self, x):
        return self.input(x), self.reset.project(x), self.update.project(x)

    def step(self, projected, h):
        candidate_input, reset_input, update_input = projected
        r, z = self.reset(reset_input, h), self.update(update_input, h)
        candidate = torch.tanh(candidate_input + self.hidden(r*h))
        return self.combine(h, candidate, z)

    def combine(self, h, candidate, z):
        return (1-z)*h + z*candidate


class GRU(ModernRNN):
    cell_type = GRUCell


# TODO: subclass TanhCell or GRUCell, implement step(projected, h), and select
# it through a ModernRNN subclass's cell_type. Carry only h across time.
MODELS = dict(gru=GRU, tanh=ModernRNN)


if __name__ == "__main__":
    from .sweeps import launch_one
    launch_one('gru')  # Edit the variant and LR here.
