"""Activation capture on registry layers, and the gradient-reachability probe (spec §7.1).

Spec §7.1::

    Forward hooks on every registry layer capture the layer input x in R^{d_l}.
    - Tag every captured sample with the s value used in that forward pass, the task,
      and the token position type.
    - Subsample tokens (e.g. random 10-20 per sample); fix the subsampling seed.
    - Build R_l with N >~ 10 · d_l; log the actual N/d_l ratio per layer.
    - Hooks must be removable and must not be active during evaluation rollouts.

What is accumulated is the uncentered float64 Gram ``Σ x x^T`` per layer, on CPU — the
sufficient statistic for :mod:`flowcl.analysis.subspace`. The raw ``R_l`` is never
held. Tags survive as sufficient statistics too: token types as per-layer *counts*, and
(only when asked for, see ``s_bin_edges``) per-flow-time-bin Grams for the layers whose
input actually depends on ``s``. The task tag is the capture itself: one capture is run
per ``(checkpoint, task)``.

**Which positions count.** The action mask acts only in the loss;
:class:`flowcl.models.trunk.SelfAttention` applies no attention mask. So a padded
action position can still produce a non-zero output gradient ``δ`` wherever a later
operation mixes tokens, and then its input *does* enter ``∇W = δ x^T``. Layers with
action-position tokens therefore accumulate two Grams, ``valid`` (masked positions
dropped) and ``all``, and :func:`gradient_reachability` decides per layer which one
reflects the real update: it inspects ``∂L/∂z`` on the module **output** ``z``. The input
gradient ``W^T δ`` is the wrong quantity — it vanishes whenever ``W`` does (e.g. the
zero-initialised ``action_out``) even though ``δ ≠ 0``.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Sequence

import torch
import torch.nn as nn

# Token position types (§7.1 tags). Order fixes the index used in the count vectors.
TOKEN_TYPES: tuple[str, ...] = (
    "context_query",
    "state",
    "language",
    "vision",
    "action",
    "context",
)
_TYPE_INDEX = {name: i for i, name in enumerate(TOKEN_TYPES)}

# Layer kinds, by what the layer's input tokens are.
KIND_STATE = "state_vector"  # trunk.state_projection: one (B, d_state) vector
KIND_TRUNK = "trunk_tokens"  # trunk blocks: the full observation token sequence
KIND_CONTEXT = "context_tokens"  # decoder cross-attn K/V: the trunk's context tokens
KIND_ACTION = "action_tokens"  # every other decoder layer: the H action positions

# Only action-position inputs depend on the flow time s: the trunk never sees s, and
# cross-attention K/V read the (s-independent) trunk context.
S_DEPENDENT_KINDS = frozenset({KIND_ACTION})

VIEW_VALID = "valid"
VIEW_ALL = "all"

# Set on the owning module while a capture is active; checked before evaluation.
_ACTIVE_ATTR = "_flowcl_active_capture"


def layer_kind(name: str) -> str:
    """Classify a §4.5 registry name by what its input tokens are."""
    if name == "trunk.state_projection":
        return KIND_STATE
    if name.startswith("trunk."):
        return KIND_TRUNK
    if name.startswith("flow_head."):
        if ".cross_attn." in name and name.endswith(("k_proj", "v_proj")):
            return KIND_CONTEXT
        return KIND_ACTION
    raise ValueError(f"cannot classify registry layer {name!r}")


def expected_reachability(
    names: Sequence[str], n_decoder_layers: int
) -> dict[str, bool]:
    """Hand-derived answer to "do padded action positions get ``δ ≠ 0``?".

    For action-position layers only. With unmasked self-attention:

    * decoder blocks ``0 .. L-2`` — reachable: every output feeds a later
      self-attention's K/V at the same position, and those feed valid queries;
    * ``action_in`` — reachable, for the same reason;
    * last block self-attn K/V — reachable: a padded key/value feeds valid queries;
    * last block self-attn Q and out_proj, cross-attn Q/O, MLP, and ``action_out`` —
      unreachable: position ``p``'s output reaches only position ``p``'s loss term.

    Not used to *select* anything — :func:`gradient_reachability` measures the rule on
    the real model. This is the cross-check that the measurement is recorded against.
    """
    last = f"flow_head.blocks.{n_decoder_layers - 1}."
    out = {}
    for name in names:
        if layer_kind(name) != KIND_ACTION:
            continue
        if name == "flow_head.action_out":
            out[name] = False
        elif name.startswith(last):
            out[name] = name.endswith(("self_attn.k_proj", "self_attn.v_proj"))
        else:
            out[name] = True
    return out


@dataclass(frozen=True)
class CaptureLayer:
    """One hooked layer."""

    name: str
    module: nn.Linear
    kind: str

    @property
    def d_in(self) -> int:
        return self.module.in_features


@dataclass(frozen=True)
class TrunkLayout:
    """Token layout of the trunk sequence (see :class:`flowcl.models.trunk.ObservationTrunk`).

    ``[context queries | state | language | vision]``, in that fixed order.
    """

    n_context: int
    n_language: int

    def type_ids(self, seq_len: int) -> torch.Tensor:
        n_vision = seq_len - self.n_context - 1 - self.n_language
        if n_vision <= 0:
            raise ValueError(
                f"trunk sequence of length {seq_len} is too short for layout "
                f"{self.n_context} context + 1 state + {self.n_language} language + "
                "vision tokens"
            )
        return torch.cat(
            [
                torch.full((self.n_context,), _TYPE_INDEX["context_query"]),
                torch.full((1,), _TYPE_INDEX["state"]),
                torch.full((self.n_language,), _TYPE_INDEX["language"]),
                torch.full((n_vision,), _TYPE_INDEX["vision"]),
            ]
        )


@dataclass
class LayerAccumulator:
    """Streaming statistics for one layer."""

    name: str
    kind: str
    d_in: int
    views: tuple[str, ...]
    gram: dict[str, torch.Tensor] = field(default_factory=dict)
    n: dict[str, int] = field(default_factory=dict)
    type_counts: dict[str, torch.Tensor] = field(default_factory=dict)
    binned_view: str | None = None
    binned_gram: list[torch.Tensor] | None = None
    binned_n: list[int] | None = None

    def type_count_dict(self, view: str) -> dict[str, int]:
        counts = self.type_counts[view]
        return {t: int(counts[i]) for i, t in enumerate(TOKEN_TYPES) if counts[i]}


@dataclass
class _BatchContext:
    s: torch.Tensor  # (B,), CPU
    s_bins: torch.Tensor | None  # (B,), CPU
    action_mask: torch.Tensor | None  # (B, H), CPU bool


class ActivationCapture:
    """Accumulate uncentered input Grams on registry layers while active.

    Use as a context manager; every forward pass inside it must run inside
    :meth:`batch_context` (or :meth:`forward_policy`), which supplies the ``s`` tags and
    the action mask. A hook firing outside that context raises rather than recording an
    untagged sample.

    Args:
        owner: Module the hooks belong to; marked while active so
            :func:`assert_no_active_capture` can refuse to evaluate it.
        layers: Layers to hook, in registry order.
        tokens_per_sample: Maximum positions kept per sample, per layer, per view.
        subsample_seed: Seed of the CPU generator that picks positions.
        trunk_layout: Required if any layer is :data:`KIND_TRUNK`.
        s_bin_edges: Flow-time bin edges. ``None`` disables per-bin Grams.
        binned_views: ``layer -> view`` whose Gram is also split per ``s`` bin. Required
            for every s-dependent layer when ``s_bin_edges`` is given.
    """

    def __init__(
        self,
        owner: nn.Module,
        layers: Sequence[CaptureLayer],
        tokens_per_sample: int,
        subsample_seed: int,
        trunk_layout: TrunkLayout | None = None,
        s_bin_edges: Sequence[float] | None = None,
        binned_views: dict[str, str] | None = None,
    ) -> None:
        if tokens_per_sample < 1:
            raise ValueError(f"tokens_per_sample must be >= 1, got {tokens_per_sample}")
        if not layers:
            raise ValueError("ActivationCapture received no layers")
        if trunk_layout is None and any(l.kind == KIND_TRUNK for l in layers):
            raise ValueError("trunk layers need a TrunkLayout for token-type tags")

        self.owner = owner
        self.layers = tuple(layers)
        self.tokens_per_sample = int(tokens_per_sample)
        self.subsample_seed = int(subsample_seed)
        self.trunk_layout = trunk_layout
        self._generator = torch.Generator(device="cpu").manual_seed(self.subsample_seed)
        self._handles: list = []
        self._context: _BatchContext | None = None

        self.s_bin_edges: tuple[float, ...] | None = None
        if s_bin_edges is not None:
            edges = tuple(float(e) for e in s_bin_edges)
            if len(edges) < 2 or any(b <= a for a, b in zip(edges, edges[1:])):
                raise ValueError(f"s_bin_edges must be strictly increasing, got {edges}")
            self.s_bin_edges = edges

        self.accumulators: dict[str, LayerAccumulator] = {}
        for layer in self.layers:
            views = (
                (VIEW_VALID, VIEW_ALL) if layer.kind == KIND_ACTION else (VIEW_ALL,)
            )
            acc = LayerAccumulator(
                name=layer.name, kind=layer.kind, d_in=layer.d_in, views=views
            )
            for view in views:
                acc.gram[view] = torch.zeros(
                    layer.d_in, layer.d_in, dtype=torch.float64
                )
                acc.n[view] = 0
                acc.type_counts[view] = torch.zeros(len(TOKEN_TYPES), dtype=torch.long)

            if self.s_bin_edges is not None and layer.kind in S_DEPENDENT_KINDS:
                view = (binned_views or {}).get(layer.name)
                if view not in views:
                    raise ValueError(
                        f"{layer.name}: s-binning needs binned_views[{layer.name!r}] "
                        f"in {views}, got {view!r}"
                    )
                n_bins = len(self.s_bin_edges) - 1
                acc.binned_view = view
                acc.binned_gram = [
                    torch.zeros(layer.d_in, layer.d_in, dtype=torch.float64)
                    for _ in range(n_bins)
                ]
                acc.binned_n = [0] * n_bins
            self.accumulators[layer.name] = acc

    @classmethod
    def for_policy(cls, policy, **kwargs) -> "ActivationCapture":
        """Hook every §4.5 registry layer of a :class:`~flowcl.models.policy.FlowPolicy`."""
        layers = [
            CaptureLayer(entry.name, entry.module, layer_kind(entry.name))
            for entry in policy.projectable_layers()
        ]
        layout = TrunkLayout(
            n_context=policy.trunk.n_context_tokens,
            n_language=policy.text_encoder.max_length,
        )
        return cls(policy, layers, trunk_layout=layout, **kwargs)

    # ---- lifecycle -------------------------------------------------------------

    def __enter__(self) -> "ActivationCapture":
        if getattr(self.owner, _ACTIVE_ATTR, None) is not None:
            raise RuntimeError("an ActivationCapture is already active on this module")
        for layer in self.layers:
            acc = self.accumulators[layer.name]
            self._handles.append(
                layer.module.register_forward_pre_hook(self._make_hook(layer, acc))
            )
        setattr(self.owner, _ACTIVE_ATTR, self)
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._context = None
        if getattr(self.owner, _ACTIVE_ATTR, None) is self:
            delattr(self.owner, _ACTIVE_ATTR)

    @property
    def active(self) -> bool:
        return bool(self._handles)

    @contextmanager
    def batch_context(self, s: torch.Tensor, action_mask: torch.Tensor | None = None):
        """Supply the per-sample tags for the forward passes run inside."""
        if not self.active:
            raise RuntimeError("batch_context() used outside the capture's `with` block")
        s_cpu = s.detach().to("cpu", torch.float64)
        if s_cpu.ndim != 1:
            raise ValueError(f"s must be (B,), got {tuple(s_cpu.shape)}")
        s_bins = None
        if self.s_bin_edges is not None:
            low, high = self.s_bin_edges[0], self.s_bin_edges[-1]
            if bool((s_cpu < low).any() or (s_cpu > high).any()):
                raise ValueError(
                    f"s outside the bin range [{low}, {high}]: "
                    f"min {float(s_cpu.min())}, max {float(s_cpu.max())}"
                )
            inner = torch.tensor(self.s_bin_edges[1:-1], dtype=torch.float64)
            # right=True: a value on an inner edge belongs to the upper bin, and
            # s == high falls in the last bin, matching flow_head.S_BINS.
            s_bins = torch.bucketize(s_cpu, inner, right=True)
        mask = None
        if action_mask is not None:
            mask = action_mask.detach().to("cpu") > 0
            if mask.shape[0] != s_cpu.shape[0]:
                raise ValueError(
                    f"action_mask batch {mask.shape[0]} != s batch {s_cpu.shape[0]}"
                )
        self._context = _BatchContext(s=s_cpu, s_bins=s_bins, action_mask=mask)
        try:
            yield
        finally:
            self._context = None

    @torch.no_grad()
    def forward_policy(self, policy, batch: dict, s: torch.Tensor, noise: torch.Tensor):
        """One tagged, gradient-free fp32 forward of the real training computation."""
        with self.batch_context(s, batch["action_mask"]):
            return policy(batch, s=s, noise=noise)

    # ---- accumulation ----------------------------------------------------------

    def _make_hook(self, layer: CaptureLayer, acc: LayerAccumulator):
        def hook(module, args):
            x = args[0]
            if x.shape[-1] != module.in_features:
                raise RuntimeError(
                    f"{layer.name}: captured input has last dim {x.shape[-1]} but the "
                    f"layer's in_features is {module.in_features}. The basis must live "
                    "in the input space (§7.3 orientation)."
                )
            if self._context is None:
                raise RuntimeError(
                    f"{layer.name} ran outside ActivationCapture.batch_context(); a "
                    "captured sample must carry its s tag (§7.1)"
                )
            self._accumulate(layer, acc, x.detach())

        return hook

    def _select(self, eligible: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Up to ``tokens_per_sample`` random eligible positions per sample (CPU)."""
        batch, seq = eligible.shape
        m = min(self.tokens_per_sample, seq)
        scores = torch.rand((batch, seq), generator=self._generator)
        # Ineligible positions sort last (rand is in [0, 1)).
        scores = scores.masked_fill(~eligible, 2.0)
        order = scores.argsort(dim=1)[:, :m]
        keep = eligible.gather(1, order)
        rows = torch.arange(batch).unsqueeze(1).expand(batch, m)
        return rows[keep], order[keep]

    def _accumulate(
        self, layer: CaptureLayer, acc: LayerAccumulator, x: torch.Tensor
    ) -> None:
        ctx = self._context
        if layer.kind == KIND_STATE:
            if x.ndim != 2:
                raise RuntimeError(f"{layer.name}: expected (B, d), got {tuple(x.shape)}")
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise RuntimeError(f"{layer.name}: expected (B, T, d), got {tuple(x.shape)}")
        batch, seq, _ = x.shape
        if batch != ctx.s.shape[0]:
            raise RuntimeError(
                f"{layer.name}: activation batch {batch} != tagged batch {ctx.s.shape[0]}"
            )

        if layer.kind == KIND_TRUNK:
            type_ids = self.trunk_layout.type_ids(seq)
        else:
            token_type = {
                KIND_STATE: "state",
                KIND_CONTEXT: "context",
                KIND_ACTION: "action",
            }[layer.kind]
            type_ids = torch.full((seq,), _TYPE_INDEX[token_type])

        all_positions = torch.ones((batch, seq), dtype=torch.bool)
        eligible = {VIEW_ALL: all_positions}
        if layer.kind == KIND_ACTION:
            if ctx.action_mask is None:
                raise RuntimeError(f"{layer.name}: action layer captured without a mask")
            if tuple(ctx.action_mask.shape) != (batch, seq):
                raise RuntimeError(
                    f"{layer.name}: action_mask {tuple(ctx.action_mask.shape)} does not "
                    f"match activation positions {(batch, seq)}"
                )
            eligible[VIEW_VALID] = ctx.action_mask

        for view in acc.views:
            b, t = self._select(eligible[view])
            if b.numel() == 0:
                continue
            rows = x[b.to(x.device), t.to(x.device)].to("cpu", torch.float64)
            acc.gram[view].addmm_(rows.T, rows)
            acc.n[view] += rows.shape[0]
            acc.type_counts[view] += torch.bincount(
                type_ids[t], minlength=len(TOKEN_TYPES)
            )
            if acc.binned_view == view:
                bins = ctx.s_bins[b]
                for k in range(len(acc.binned_gram)):
                    sel = bins == k
                    if bool(sel.any()):
                        part = rows[sel]
                        acc.binned_gram[k].addmm_(part.T, part)
                        acc.binned_n[k] += part.shape[0]


def assert_no_active_capture(module: nn.Module) -> None:
    """Raise if an :class:`ActivationCapture` is active on ``module`` (§7.1)."""
    capture = getattr(module, _ACTIVE_ATTR, None)
    if capture is not None:
        raise RuntimeError(
            "An ActivationCapture is active on this policy. §7.1: analysis hooks must "
            "not be active during evaluation rollouts. Exit the capture first."
        )


# ---- gradient reachability ---------------------------------------------------


def gradient_reachability(
    owner: nn.Module,
    layers: Sequence[CaptureLayer],
    compute_loss: Callable[[], torch.Tensor],
    padded: torch.Tensor,
) -> dict[str, bool]:
    """Do padded positions receive a non-zero output gradient ``δ = ∂L/∂z``?

    One forward/backward of ``compute_loss`` with ``retain_grad()`` on each layer's
    output. No optimiser step; every parameter's ``.grad`` is restored afterwards, so
    the model is left exactly as it was.

    Unreachable rows come out *exactly* zero (they are products with a zero upstream
    gradient), so ``!= 0`` is a sound test.

    Args:
        owner: Module whose parameters' ``.grad`` must be preserved.
        layers: Layers whose output positions align with ``padded``.
        compute_loss: Runs the forward and returns the scalar training loss.
        padded: ``(B, T)`` bool, True where a position is loss-masked.

    Returns:
        ``layer -> reachable``.

    Raises:
        RuntimeError: If a layer's ``δ`` is zero on the *valid* rows as well. The probe
            then carries no information (e.g. an untrained model with zero-initialised
            AdaLN gates), and guessing would be a silent fallback.
    """
    padded = padded.bool()
    if padded.ndim != 2:
        raise ValueError(f"padded must be (B, T), got {tuple(padded.shape)}")
    if not bool(padded.any()) or bool(padded.all()):
        raise ValueError(
            "reachability probe needs a batch with both padded and valid positions; "
            f"got {int(padded.sum())} padded of {padded.numel()}"
        )

    saved = {
        name: (p.grad.detach().clone() if p.grad is not None else None)
        for name, p in owner.named_parameters()
    }
    outputs: dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(name: str):
        def hook(module, args, output):
            if name in outputs:
                raise RuntimeError(f"{name} ran twice in one probe forward")
            if output.requires_grad:
                output.retain_grad()
            outputs[name] = output

        return hook

    for layer in layers:
        handles.append(layer.module.register_forward_hook(make_hook(layer.name)))

    try:
        with torch.enable_grad():
            loss = compute_loss()
            if not loss.requires_grad:
                raise RuntimeError("probe loss does not require grad")
            loss.backward()

        result = {}
        for layer in layers:
            z = outputs.get(layer.name)
            if z is None:
                raise RuntimeError(f"{layer.name} did not run in the probe forward")
            if z.grad is None:
                raise RuntimeError(f"{layer.name}: output received no gradient")
            grad = z.grad.detach()
            if tuple(grad.shape[:2]) != tuple(padded.shape):
                raise RuntimeError(
                    f"{layer.name}: output positions {tuple(grad.shape[:2])} do not "
                    f"match the padding mask {tuple(padded.shape)}"
                )
            mask = padded.to(grad.device)
            if not bool((grad[~mask] != 0).any()):
                raise RuntimeError(
                    f"{layer.name}: δ is zero on valid positions too, so the probe is "
                    "uninformative (untrained or zero-gated model?)"
                )
            result[layer.name] = bool((grad[mask] != 0).any())
        return result
    finally:
        for handle in handles:
            handle.remove()
        for name, p in owner.named_parameters():
            p.grad = saved[name]


def probe_policy_reachability(
    policy, batch: dict, generator: torch.Generator | None = None
) -> dict[str, bool]:
    """:func:`gradient_reachability` on a FlowPolicy's action-position layers.

    Uses the normal masked flow-matching loss in fp32.
    """
    layers = [
        CaptureLayer(entry.name, entry.module, layer_kind(entry.name))
        for entry in policy.projectable_layers()
        if layer_kind(entry.name) == KIND_ACTION
    ]
    padded = batch["action_mask"] == 0
    device_type = batch["action_mask"].device.type

    def compute_loss() -> torch.Tensor:
        with torch.autocast(device_type=device_type, enabled=False):
            return policy(batch, generator=generator)["loss"]

    return gradient_reachability(policy, layers, compute_loss, padded)
