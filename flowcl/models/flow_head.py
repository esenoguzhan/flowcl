"""Conditional flow-matching action decoder.

Spec §4.3, which this module implements literally::

    A_1 = target action chunk, A_0 ~ N(0, I)
    A_s = (1 - s) A_0 + s A_1,   s ~ p(s) on [0, 1]
    network predicts v_theta(A_s, o, s)
    loss = masked MSE( v_theta(A_s, o, s), A_1 - A_0 )

The regression target is ``A_1 - A_0`` because for the straight-line interpolant above
``dA_s/ds = A_1 - A_0`` exactly, independent of ``s``.

``s`` is conditioned in via a sinusoidal embedding plus AdaLN modulation, and the
sampler for ``p(s)`` is swappable (uniform, logit-normal) because §7.3/§9 need
``rho_l(s)`` and ``c_l(s)`` measured under a known ``s`` distribution, and a
hard-coded sampler would make that a code change rather than a config change.

Inference uses an Euler ODE solver with ``N = 10`` steps (§4.4).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from flowcl.models.trunk import CrossAttention, FeedForward, SelfAttention

# §9 / §7.3 flow-time bins. Right-open except the last, which includes s = 1.
S_BINS: tuple[tuple[float, float], ...] = (
    (0.00, 0.25),
    (0.25, 0.50),
    (0.50, 0.75),
    (0.75, 1.00),
)


def bin_index(s: torch.Tensor) -> torch.Tensor:
    """Map ``s`` values to :data:`S_BINS` indices.

    ``s = 1.0`` belongs to the last bin, matching the half-open convention in §9.
    """
    idx = torch.floor(s * len(S_BINS)).long()
    return idx.clamp_(0, len(S_BINS) - 1)


# ---- p(s) samplers -------------------------------------------------------------


class SSampler(ABC):
    """Distribution over flow time ``s`` in ``[0, 1]``."""

    name: str

    @abstractmethod
    def sample(
        self,
        n: int,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Return ``(n,)`` samples in ``[0, 1]``."""


class UniformSSampler(SSampler):
    """``s ~ U(0, 1)``."""

    name = "uniform"

    def sample(self, n, device, generator=None):
        return torch.rand(n, device=device, generator=generator)


class LogitNormalSSampler(SSampler):
    """``s = sigmoid(z)``, ``z ~ N(mu, sigma^2)``.

    Concentrates samples near ``s = 0.5``, the standard choice in the flow-matching
    literature for putting capacity where the velocity field is hardest.
    """

    name = "logit_normal"

    def __init__(self, mean: float = 0.0, std: float = 1.0) -> None:
        self.mean = mean
        self.std = std

    def sample(self, n, device, generator=None):
        z = torch.randn(n, device=device, generator=generator) * self.std + self.mean
        return torch.sigmoid(z)


class BinnedSSampler(SSampler):
    """Uniform *within* one :data:`S_BINS` bin.

    Used by §7.5's s-binned method, where each microbatch draws its ``s`` from a
    single bin.
    """

    name = "binned"

    def __init__(self, bin_idx: int) -> None:
        if not 0 <= bin_idx < len(S_BINS):
            raise ValueError(
                f"bin_idx {bin_idx} out of range for {len(S_BINS)} bins"
            )
        self.bin_idx = bin_idx
        self.low, self.high = S_BINS[bin_idx]

    def sample(self, n, device, generator=None):
        u = torch.rand(n, device=device, generator=generator)
        return self.low + u * (self.high - self.low)


S_SAMPLERS: dict[str, type[SSampler]] = {
    "uniform": UniformSSampler,
    "logit_normal": LogitNormalSSampler,
}


def build_s_sampler(name: str, **kwargs) -> SSampler:
    if name not in S_SAMPLERS:
        raise ValueError(
            f"Unknown s sampler {name!r}; choose from {sorted(S_SAMPLERS)}"
        )
    return S_SAMPLERS[name](**kwargs)


# ---- s conditioning ------------------------------------------------------------


class SinusoidalSEmbedding(nn.Module):
    """Sinusoidal embedding of flow time, followed by a small MLP.

    ``s`` lives in ``[0, 1]`` while the usual sinusoidal frequency ladder assumes
    integer-scale inputs, so ``s`` is scaled by ``max_period_scale`` first.
    """

    def __init__(self, dim: int, max_period_scale: float = 1000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"embedding dim must be even, got {dim}")
        self.dim = dim
        self.max_period_scale = max_period_scale
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """``(B,)`` flow times -> ``(B, dim)`` embeddings."""
        if s.ndim != 1:
            raise ValueError(f"expected (B,) flow times, got {tuple(s.shape)}")
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=s.device).float() / half
        )
        angles = (s * self.max_period_scale).unsqueeze(1) * freqs.unsqueeze(0)
        embedding = torch.cat([torch.cos(angles), torch.sin(angles)], dim=1)
        return self.mlp(embedding)


class AdaLNModulation(nn.Module):
    """Produce per-block AdaLN scale/shift/gate triples from the ``s`` embedding.

    These are exactly the "AdaLN/FiLM modulation parameters" that §7.4 freezes in
    projection experiments: they carry flow-time information, and projecting their
    gradients onto a subspace estimated across all ``s`` would confound the very
    signal §7.5 studies.

    Zero-initialised output layer so each block starts as the identity, which is the
    standard DiT-style initialisation and makes early training stable.
    """

    def __init__(self, d_model: int, s_dim: int, n_groups: int = 3) -> None:
        super().__init__()
        self.n_groups = n_groups
        self.d_model = d_model
        self.proj = nn.Linear(s_dim, n_groups * 3 * d_model)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, s_embedding: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return ``3 * n_groups`` tensors of shape ``(B, 1, d_model)``."""
        out = F.silu(s_embedding)
        out = self.proj(out)
        chunks = out.chunk(self.n_groups * 3, dim=-1)
        return tuple(chunk.unsqueeze(1) for chunk in chunks)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN: ``x * (1 + scale) + shift``."""
    return x * (1 + scale) + shift


class FlowDecoderBlock(nn.Module):
    """AdaLN-modulated block: self-attention, cross-attention to context, MLP."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        s_dim: int,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        # elementwise_affine=False: AdaLN supplies scale and shift, so a learned
        # affine here would be redundant and would double-count in the §7.4 allowlist.
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.self_attn = SelfAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.cross_attn = CrossAttention(d_model, n_heads)
        self.norm3 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.mlp = FeedForward(d_model, mlp_ratio)
        self.modulation = AdaLNModulation(d_model, s_dim, n_groups=3)

    def forward(
        self, x: torch.Tensor, context: torch.Tensor, s_embedding: torch.Tensor
    ) -> torch.Tensor:
        (
            shift_sa,
            scale_sa,
            gate_sa,
            shift_ca,
            scale_ca,
            gate_ca,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.modulation(s_embedding)

        x = x + gate_sa * self.self_attn(modulate(self.norm1(x), shift_sa, scale_sa))
        x = x + gate_ca * self.cross_attn(
            modulate(self.norm2(x), shift_ca, scale_ca), context
        )
        x = x + gate_mlp * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
        return x


class FlowHead(nn.Module):
    """Predicts the flow-matching velocity field ``v_theta(A_s, o, s)``.

    Args:
        d_action: Action width for this embodiment (never padded, §0).
        horizon: ``H``, chunk length.
        d_model: Decoder width.
        n_layers: Decoder depth.
        n_heads: Attention heads.
        s_dim: Flow-time embedding width.
    """

    def __init__(
        self,
        d_action: int,
        horizon: int,
        d_model: int = 512,
        n_layers: int = 4,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        s_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.d_action = d_action
        self.horizon = horizon
        self.d_model = d_model
        s_dim = s_dim or d_model

        self.action_in = nn.Linear(d_action, d_model)
        self.chunk_position = nn.Parameter(torch.zeros(horizon, d_model))
        nn.init.normal_(self.chunk_position, std=0.02)

        self.s_embedding = SinusoidalSEmbedding(s_dim)
        self.blocks = nn.ModuleList(
            FlowDecoderBlock(d_model, n_heads, s_dim, mlp_ratio)
            for _ in range(n_layers)
        )
        self.norm_out = nn.LayerNorm(d_model, elementwise_affine=False)
        self.final_modulation = AdaLNModulation(d_model, s_dim, n_groups=1)
        self.action_out = nn.Linear(d_model, d_action)
        # Start by predicting zero velocity; the loss then shapes it from scratch
        # rather than from an arbitrary random field.
        nn.init.zeros_(self.action_out.weight)
        nn.init.zeros_(self.action_out.bias)

    def forward(
        self, noisy_actions: torch.Tensor, context: torch.Tensor, s: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            noisy_actions: ``A_s``, shape ``(B, H, d_action)``.
            context: Trunk context tokens, ``(B, n_context, d_model)``.
            s: Flow times, ``(B,)``.

        Returns:
            Predicted velocity, ``(B, H, d_action)``.
        """
        if noisy_actions.shape[1:] != (self.horizon, self.d_action):
            raise ValueError(
                f"expected (B, {self.horizon}, {self.d_action}) actions, got "
                f"{tuple(noisy_actions.shape)}"
            )
        if s.shape != (noisy_actions.shape[0],):
            raise ValueError(
                f"expected ({noisy_actions.shape[0]},) flow times, got "
                f"{tuple(s.shape)}"
            )

        s_embedding = self.s_embedding(s)
        x = self.action_in(noisy_actions) + self.chunk_position.unsqueeze(0)

        for block in self.blocks:
            x = block(x, context, s_embedding)

        shift, scale, _gate = self.final_modulation(s_embedding)
        x = modulate(self.norm_out(x), shift, scale)
        return self.action_out(x)

    # ---- inference ------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        context: torch.Tensor,
        n_steps: int = 10,
        generator: torch.Generator | None = None,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Integrate the velocity field from ``s = 0`` to ``s = 1`` (§4.4).

        Fixed-step forward Euler with ``n_steps`` increments of ``1 / n_steps``.

        Args:
            context: ``(B, n_context, d_model)``.
            n_steps: ``N`` in §4.4; 10 by default.
            generator: RNG for the initial noise, so a rollout is reproducible.
            noise: Supply ``A_0`` directly instead of sampling it. Used by tests that
                need bitwise determinism without depending on RNG internals.

        Returns:
            ``A_1``, shape ``(B, H, d_action)``.
        """
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")

        batch = context.shape[0]
        device = context.device
        if noise is None:
            actions = torch.randn(
                batch,
                self.horizon,
                self.d_action,
                device=device,
                dtype=context.dtype,
                generator=generator,
            )
        else:
            if noise.shape != (batch, self.horizon, self.d_action):
                raise ValueError(
                    f"noise must be ({batch}, {self.horizon}, {self.d_action}), got "
                    f"{tuple(noise.shape)}"
                )
            actions = noise.to(device=device, dtype=context.dtype).clone()

        ds = 1.0 / n_steps
        for step in range(n_steps):
            s = torch.full((batch,), step * ds, device=device, dtype=actions.dtype)
            actions = actions + ds * self.forward(actions, context, s)
        return actions
