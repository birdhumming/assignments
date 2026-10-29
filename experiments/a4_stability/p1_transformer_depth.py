"""Problem 1: each subclass changes one component of the class transformer.

    uv run python -m experiments.a4_stability.p1_transformer_depth
"""
import torch
from torch import nn
from torch.nn import functional as F

from modeling import (
    AutoregressiveLM,
    LlamaAttention,
    LlamaMLP,
    LlamaRMSNorm,
    apply_rotary_pos_emb,
    causal_attention_mask,
    initialize_model,
    repeat_kv,
)


class Residual(nn.Module):
    """x' = x + F(N(x)). Both attention and MLP use this wrapper."""
    def __init__(self, operation, norm):
        super().__init__()
        self.operation, self.norm = operation, norm

    def forward(self, x, **context):
        return x + self.operation(self.norm(x), **context)


class NoResidual(Residual):
    def forward(self, x, **context):
        return self.operation(self.norm(x), **context)


class NoPreNorm(Residual):
    def __init__(self, operation, norm):
        super().__init__(operation, nn.Identity())


class PostNorm(Residual):
    def forward(self, x, **context):
        return self.norm(x + self.operation(x, **context))


class GatedResidual(Residual):
    """x' = x + sigmoid(W N(x) + b) * F(N(x))."""
    def __init__(self, operation, norm):
        super().__init__(operation, norm)
        width = operation.config.hidden_size
        self.gate = nn.Linear(width, width)

    def forward(self, x, **context):
        y = self.norm(x)
        return x + self.gate(y).sigmoid() * self.operation(y, **context)


class GatedNorm(nn.Module):
    def __init__(self, norm, width):
        super().__init__()
        self.norm, self.gate = norm, nn.Linear(width, width)

    def forward(self, x):
        y = self.norm(x)
        return self.gate(y).sigmoid() * y


class Attention(LlamaAttention):
    """Native SDPA, with small override points for RoPE and head output gates."""
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        if not self.qk_norm:
            self.q_norm, self.k_norm = nn.Identity(), nn.Identity()

    def rotate(self, q, k, position_embeddings):
        return apply_rotary_pos_emb(q, k, *position_embeddings)

    def gate_heads(self, heads, x):
        return heads

    def forward(self, x, *, position_embeddings, attention_mask=None, **context):
        batch, length, _ = x.shape
        q = self.q_proj(x).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        q, k = self.rotate(q, k, position_embeddings)
        q, k = self.q_norm(q), self.k_norm(k)
        k, v = repeat_kv(k, self.num_key_value_groups), repeat_kv(v, self.num_key_value_groups)
        mask = None if attention_mask is None else causal_attention_mask(length, attention_mask, x.device, q.dtype)
        heads = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=mask is None)
        heads = self.gate_heads(heads.transpose(1, 2), x)
        return self.o_proj(heads.reshape(batch, length, -1))


class NoRoPEAttention(Attention):
    def rotate(self, q, k, position_embeddings):
        return q, k


class GatedAttention(Attention):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.gate = nn.Linear(config.hidden_size, self.num_heads)

    def gate_heads(self, heads, x):
        return heads * self.gate(x).sigmoid().unsqueeze(-1)


class MLP(LlamaMLP):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

    def forward(self, x, **context):
        return super().forward(x)


class TransformerBlock(nn.Module):
    def __init__(self, config, residual_type, attention_type, qk_norm):
        super().__init__()
        self.attention = residual_type(attention_type(config, qk_norm=qk_norm),
                                       LlamaRMSNorm(config.hidden_size, eps=1e-5))
        self.mlp = residual_type(MLP(config), LlamaRMSNorm(config.hidden_size, eps=1e-5))

    def forward(self, x, **context):
        return self.mlp(self.attention(x, **context), **context)


class Transformer(AutoregressiveLM):
    residual_type, attention_type = Residual, Attention
    model_name, num_layers = "d8", 8
    use_qk_norm = True

    def __init__(self, config, dtype=torch.float32, qk_norm=True, tie_word_embeddings=False, dropout=0.):
        if dropout or tie_word_embeddings:
            raise ValueError("These comparisons use dropout=0 and untied embeddings.")
        super().__init__(config, qk_norm=qk_norm)
        self.model.layers = nn.ModuleList([
            TransformerBlock(config, self.residual_type, self.attention_type, qk_norm)
            for _ in range(config.num_hidden_layers)])
        self.to(dtype=dtype)

    def branches(self):
        return [branch for block in self.model.layers for branch in (block.attention, block.mlp)]

    def initialize_parameters(self):
        initialize_model(self)

    @property
    def output_norm(self):
        return self.model.norm

    def run_depth(self, x, **context):
        for block in self.model.layers:
            x = block(x, **context)
        return x

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        x = self.model.embed_tokens(input_ids)
        if position_ids is None:
            position_ids = torch.arange(x.shape[1], device=x.device)[None]
        context = dict(position_ids=position_ids, attention_mask=attention_mask,
                       position_embeddings=self.model.rotary_emb(x, position_ids))
        return self.lm_head(self.output_norm(self.run_depth(x, **context)))


class NoResidualTransformer(Transformer):
    """Replace x + F(N(x)) by F(N(x)) in both sublayers."""
    residual_type = NoResidual


class NoPreNormTransformer(Transformer):
    residual_type = NoPreNorm


class NoQKNormTransformer(Transformer):
    use_qk_norm = False

    def __init__(self, config, **kwargs):
        kwargs['qk_norm'] = False
        super().__init__(config, **kwargs)


class NoRoPETransformer(Transformer):
    attention_type = NoRoPEAttention


class NoFinalNormTransformer(Transformer):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.model.norm = nn.Identity()


class PostNormTransformer(Transformer):
    residual_type = PostNorm


class GatedResidualTransformer(Transformer):
    residual_type = GatedResidual


class GatedAttentionTransformer(Transformer):
    attention_type = GatedAttention


class GatedNormTransformer(Transformer):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        for branch in self.branches():
            branch.norm = GatedNorm(branch.norm, config.hidden_size).to(next(self.parameters()).dtype)


MODELS = dict(baseline=Transformer, no_residual=NoResidualTransformer,
              no_prenorm=NoPreNormTransformer, no_qknorm=NoQKNormTransformer,
              no_rope=NoRoPETransformer, no_final_norm=NoFinalNormTransformer,
              postnorm=PostNormTransformer, gated_residual=GatedResidualTransformer,
              gated_attention=GatedAttentionTransformer, gated_norm=GatedNormTransformer)


if __name__ == "__main__":
    from .sweeps import launch_one
    launch_one('d8')  # Edit the variant and LR, e.g. ('d8-no-residual', .001).
