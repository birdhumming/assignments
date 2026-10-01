"""Offline fixtures for the public model-builder and W&B interfaces."""
from types import SimpleNamespace
from unittest.mock import Mock
import torch._dynamo  # Load lazy optimizer dependencies before sys.modules patches.
from modeling import AutoregressiveLM


def mock_wandb():
    wandb = Mock()
    wandb.init.return_value = SimpleNamespace(id='test-run', url='test-run')
    wandb.util.generate_id.return_value = 'test-run'
    return wandb


class ScaledTestLM(AutoregressiveLM):
    """Arbitrary constant scale to check custom model reconstruction."""
    def __init__(self, config, *, scale=1., **kwargs):
        super().__init__(config, **kwargs)
        self.scale = scale

    def forward(self, input_ids=None, attention_mask=None, position_ids=None):
        hidden = self.model(input_ids=input_ids, attention_mask=attention_mask,
                            position_ids=position_ids)
        return self.lm_head(hidden * self.scale)
