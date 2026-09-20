"""Observation projections + transformer trunk.

Spec §4.2: small transformer, 6-8 layers, ``d_model`` 384-512, 8 heads, pre-norm,
learned positional embeddings over the observation token set. Outputs a fixed set of
context tokens consumed by the decoder via cross-attention.

Attention and MLP submodules are written out as explicit ``nn.Linear`` layers rather
than using ``nn.MultiheadAttention`` / ``nn.TransformerEncoderLayer``, because §4.5
requires a registry of every projectable linear layer (trunk attention Q/K/V/O, trunk
MLPs, ...) and §7.1 hooks each layer's *input*. Fused QKV weights or C++-fused
attention would make per-layer Q/K/V bases impossible to separate.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """Multi-head self-attention with separate Q, K, V, O projections."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model {d_model} must be divisible by n_heads {n_heads}"
            )
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        q = self.q_proj(x).view(batch, seq, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, self.n_heads, self.d_head).transpose(1, 2)
        attended = F.scaled_dot_product_attention(q, k, v)
        merged = attended.transpose(1, 2).reshape(batch, seq, self.d_model)
        return self.out_proj(merged)


class CrossAttention(nn.Module):
    """Multi-head cross-attention from queries to a context sequence."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model {d_model} must be divisible by n_heads {n_heads}"
            )
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        ctx_len = context.shape[1]
        q = self.q_proj(x).view(batch, seq, self.n_heads, self.d_head).transpose(1, 2)
        k = (
            self.k_proj(context)
            .view(batch, ctx_len, self.n_heads, self.d_head)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(context)
            .view(batch, ctx_len, self.n_heads, self.d_head)
            .transpose(1, 2)
        )
        attended = F.scaled_dot_product_attention(q, k, v)
        merged = attended.transpose(1, 2).reshape(batch, seq, self.d_model)
        return self.out_proj(merged)


class FeedForward(nn.Module):
    """Two-layer MLP with GELU."""

    def __init__(self, d_model: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class TrunkBlock(nn.Module):
    """Pre-norm self-attention + MLP block (§4.2)."""

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = FeedForward(d_model, mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ObservationTrunk(nn.Module):
    """Fuse vision, state and language tokens into a fixed set of context tokens.

    Token layout, in this fixed order (the order the registry and every analysis
    assume):

    1. ``n_context`` learned context queries, whose outputs are returned
    2. one state token, from a linear projection of proprioception
    3. language tokens
    4. vision patch tokens, cameras concatenated in ``EmbodimentSpec.cameras`` order

    Returning only the learned context queries keeps the decoder's cross-attention
    cost independent of image resolution and camera count.
    """

    def __init__(
        self,
        d_state: int,
        d_model: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        n_context_tokens: int = 32,
        max_tokens: int = 1024,
    ) -> None:
        super().__init__()
        if not 6 <= n_layers <= 8:
            raise ValueError(f"§4.2 specifies 6-8 trunk layers, got {n_layers}")
        if not 384 <= d_model <= 512:
            raise ValueError(f"§4.2 specifies d_model 384-512, got {d_model}")

        self.d_model = d_model
        self.n_context_tokens = n_context_tokens
        self.max_tokens = max_tokens

        self.state_projection = nn.Linear(d_state, d_model)
        self.context_queries = nn.Parameter(torch.zeros(n_context_tokens, d_model))
        nn.init.normal_(self.context_queries, std=0.02)

        # Learned positional embeddings over the whole observation token set (§4.2).
        self.position_embedding = nn.Parameter(torch.zeros(max_tokens, d_model))
        nn.init.normal_(self.position_embedding, std=0.02)

        self.blocks = nn.ModuleList(
            TrunkBlock(d_model, n_heads, mlp_ratio) for _ in range(n_layers)
        )
        self.norm_out = nn.LayerNorm(d_model)

    def forward(
        self,
        state: torch.Tensor,
        language_tokens: torch.Tensor,
        vision_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            state: ``(B, d_state)``.
            language_tokens: ``(B, L, d_model)``.
            vision_tokens: ``(B, V, d_model)`` with cameras already concatenated.

        Returns:
            ``(B, n_context_tokens, d_model)`` context tokens.
        """
        batch = state.shape[0]
        state_token = self.state_projection(state).unsqueeze(1)
        queries = self.context_queries.unsqueeze(0).expand(batch, -1, -1)

        tokens = torch.cat([queries, state_token, language_tokens, vision_tokens], dim=1)
        seq = tokens.shape[1]
        if seq > self.max_tokens:
            raise ValueError(
                f"observation token set has length {seq} but the trunk was built for "
                f"max_tokens={self.max_tokens}; raise max_tokens in the policy config"
            )
        tokens = tokens + self.position_embedding[:seq].unsqueeze(0)

        for block in self.blocks:
            tokens = block(tokens)

        return self.norm_out(tokens[:, : self.n_context_tokens])
