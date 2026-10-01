"""Output-side sensitivity of registry layers, for two-sided protection (thesis plan E1).

GPM protects each layer's input subspace ``M``: every update ``D`` keeps ``D M = 0``. Two-sided
protection keeps only ``U^T D M = 0``, where ``U`` spans the directions of the layer's *output*
the policy's velocity is sensitive to. For a layer ``y_t = W x_t`` (position ``t``), a weight change
``ΔW`` moves the velocity by ``Σ_t J_t ΔW x_t`` to first order, with ``J_t = ∂v/∂y_t``. With
``x_t`` in ``span M`` the part through ``U U^T ΔW M M^T`` vanishes, and the rest is bounded by the
energy of ``Σ_t J_t^T J_t`` outside ``U``. So ``U`` is the energy-thresholded basis of that output
Gram (the output factor of the Gauss-Newton matrix), built with
:func:`flowcl.analysis.subspace.extend_basis` exactly as ``M`` is built from the input Gram.

**Hutchinson.** For ``u ~ N(0, I)`` over the (masked) velocity, the gradient of ``<u, v>`` with
respect to ``y_t`` is ``g_t = J_t^T u``, and ``E_u[g_t g_t^T] = J_t^T J_t``.
:class:`OutputGradientCapture` retains every hooked layer's output during one forward;
:meth:`OutputGradientCapture.backward` differentiates ``R`` independent probes through it (only
the outputs, never a parameter's ``.grad``) and accumulates ``Σ g g^T`` per probe, on subsampled
positions, in float64 on the CPU: one ``(d_out, d_out)`` Gram per layer and probe.

Positions whose output gradient is exactly zero carry no sensitivity (e.g. the last trunk block's
non-context positions) and are not eligible; this changes no Gram, only how many informative rows
the per-sample subsample keeps.
"""

from __future__ import annotations

import statistics
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Sequence

import torch
from torch import nn

from flowcl.analysis.hooks import (
    KIND_ACTION,
    KIND_STATE,
    VIEW_ALL,
    VIEW_VALID,
    CaptureLayer,
    layer_kind,
)


@dataclass
class OutputAccumulator:
    """One layer's output Grams, one per probe."""

    name: str
    kind: str
    d_out: int
    view: str
    gram: list[torch.Tensor] = field(default_factory=list)
    n: list[int] = field(default_factory=list)


class OutputGradientCapture:
    """Retain registry outputs during a forward; accumulate Hutchinson output Grams on demand.

    Use as a context manager. Every forward inside it must run inside :meth:`recording` (which
    supplies the action mask); :meth:`backward` then consumes the retained outputs.

    Args:
        owner: Module the hooks belong to.
        layers: Layers to hook, in registry order.
        views: ``layer -> view`` (``valid`` or ``all``); ``valid`` restricts action-position
            layers to unpadded positions. Non-action layers must use ``all``.
        n_probes: Independent Hutchinson probes per forward (``R``), one Gram each.
        tokens_per_sample: Maximum positions kept per sample, per layer and probe.
        subsample_seed: Seed of the CPU generator that picks positions.
    """

    def __init__(
        self,
        owner: nn.Module,
        layers: Sequence[CaptureLayer],
        views: dict[str, str],
        n_probes: int,
        tokens_per_sample: int,
        subsample_seed: int,
    ) -> None:
        if n_probes < 1:
            raise ValueError(f"n_probes must be >= 1, got {n_probes}")
        if tokens_per_sample < 1:
            raise ValueError(f"tokens_per_sample must be >= 1, got {tokens_per_sample}")
        if not layers:
            raise ValueError("OutputGradientCapture received no layers")
        self.owner = owner
        self.layers = tuple(layers)
        self.n_probes = int(n_probes)
        self.tokens_per_sample = int(tokens_per_sample)
        self._generator = torch.Generator(device="cpu").manual_seed(int(subsample_seed))
        self._handles: list = []
        self._recording = False
        self._mask: torch.Tensor | None = None
        self._outputs: dict[str, torch.Tensor] = {}
        self.accumulators: dict[str, OutputAccumulator] = {}
        for layer in self.layers:
            view = views[layer.name]
            if view not in (VIEW_VALID, VIEW_ALL):
                raise ValueError(f"{layer.name}: unknown view {view!r}")
            if view == VIEW_VALID and layer.kind != KIND_ACTION:
                raise ValueError(f"{layer.name}: only action-position layers have a valid view")
            d_out = layer.module.out_features
            self.accumulators[layer.name] = OutputAccumulator(
                name=layer.name, kind=layer.kind, d_out=d_out, view=view,
                gram=[torch.zeros(d_out, d_out, dtype=torch.float64) for _ in range(self.n_probes)],
                n=[0] * self.n_probes,
            )

    @classmethod
    def for_policy(cls, policy, **kwargs) -> "OutputGradientCapture":
        """Hook every §4.5 registry layer of a :class:`~flowcl.models.policy.FlowPolicy`."""
        layers = [
            CaptureLayer(entry.name, entry.module, layer_kind(entry.name))
            for entry in policy.projectable_layers()
        ]
        return cls(policy, layers, **kwargs)

    # ---- lifecycle -------------------------------------------------------------

    def __enter__(self) -> "OutputGradientCapture":
        for layer in self.layers:
            self._handles.append(layer.module.register_forward_hook(self._make_hook(layer)))
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._outputs.clear()
        self._recording = False

    @contextmanager
    def recording(self, action_mask: torch.Tensor | None = None):
        """Retain the outputs of the forward passes run inside (with the batch's action mask)."""
        if not self._handles:
            raise RuntimeError("recording() used outside the capture's `with` block")
        self._outputs.clear()
        self._mask = None if action_mask is None else action_mask.detach().to("cpu") > 0
        self._recording = True
        try:
            yield
        finally:
            self._recording = False

    def _make_hook(self, layer: CaptureLayer):
        def hook(module, args, output):
            if output.shape[-1] != module.out_features:
                raise RuntimeError(
                    f"{layer.name}: captured output has last dim {output.shape[-1]} but the "
                    f"layer's out_features is {module.out_features}; U must live in the output "
                    "space"
                )
            if self._recording:
                if layer.name in self._outputs:
                    raise RuntimeError(f"{layer.name} ran twice in one recorded forward")
                self._outputs[layer.name] = output

        return hook

    # ---- the probes --------------------------------------------------------------

    def backward(
        self,
        velocity: torch.Tensor,
        weight: torch.Tensor | None,
        generator: torch.Generator,
    ) -> None:
        """``R`` Hutchinson probes through the retained graph; accumulate each probe's Grams.

        ``weight`` broadcasts against ``velocity`` (the action mask, ``(B, H, 1)``) or is
        ``None``. ``u`` is drawn on the CPU from ``generator``, so the draws do not depend on the
        device and do not touch any other stream.
        """
        missing = [layer.name for layer in self.layers if layer.name not in self._outputs]
        if missing:
            raise RuntimeError(f"no retained output for {len(missing)} layer(s), e.g. {missing[:3]}")
        names = [layer.name for layer in self.layers]
        outputs = [self._outputs[n] for n in names]
        not_grad = [n for n, o in zip(names, outputs) if not o.requires_grad]
        if not_grad:
            raise RuntimeError(f"outputs without a graph (forward ran under no_grad?): {not_grad[:3]}")
        for r in range(self.n_probes):
            u = torch.randn(tuple(velocity.shape), generator=generator, dtype=torch.float32)
            u = u.to(device=velocity.device, dtype=velocity.dtype)
            if weight is not None:
                u = u * weight.to(velocity.dtype)
            grads = torch.autograd.grad(
                (velocity * u).sum(), outputs, retain_graph=r < self.n_probes - 1, allow_unused=True
            )
            for name, out, g in zip(names, outputs, grads):
                if g is None:
                    g = torch.zeros_like(out)
                self._accumulate(self.accumulators[name], r, g.detach())
        self._outputs.clear()

    def _accumulate(self, acc: OutputAccumulator, r: int, g: torch.Tensor) -> None:
        if acc.kind == KIND_STATE:
            if g.ndim != 2:
                raise RuntimeError(f"{acc.name}: expected (B, d), got {tuple(g.shape)}")
            g = g.unsqueeze(1)
        if g.ndim != 3 or g.shape[-1] != acc.d_out:
            raise RuntimeError(f"{acc.name}: expected (B, T, {acc.d_out}), got {tuple(g.shape)}")
        batch, seq, _ = g.shape
        eligible = (g != 0).any(dim=-1).to("cpu")
        if acc.view == VIEW_VALID:
            if self._mask is None or tuple(self._mask.shape) != (batch, seq):
                raise RuntimeError(f"{acc.name}: the valid view needs a (B, H) action mask")
            eligible = eligible & self._mask
        b, t = _select(eligible, self.tokens_per_sample, self._generator)
        if b.numel() == 0:
            return
        rows = g[b.to(g.device), t.to(g.device)].to("cpu", torch.float64)
        acc.gram[r].addmm_(rows.T, rows)
        acc.n[r] += rows.shape[0]


def _select(eligible: torch.Tensor, m: int, generator: torch.Generator):
    """Up to ``m`` random eligible positions per sample (as ``ActivationCapture._select``)."""
    batch, seq = eligible.shape
    m = min(m, seq)
    scores = torch.rand((batch, seq), generator=generator)
    scores = scores.masked_fill(~eligible, 2.0)
    order = scores.argsort(dim=1)[:, :m]
    keep = eligible.gather(1, order)
    rows = torch.arange(batch).unsqueeze(1).expand(batch, m)
    return rows[keep], order[keep]


# ---- capacity accounting (pure) ------------------------------------------------------------------


def _half(name: str) -> str:
    return "trunk" if name.startswith("trunk.") else "decoder"


def protected_fraction(k_out: Sequence[float], k_in: Sequence[float], d_out: Sequence[int],
                       d_in: Sequence[int]) -> dict:
    """Parameter-weighted protected fractions over a set of weights ``(d_out, d_in)``.

    ``two_sided = Σ k_out k_in / Σ d_out d_in`` (protecting ``U ⊗ M``); ``gpm = Σ d_out k_in /
    Σ d_out d_in`` (protecting ``I ⊗ M``); ``rho_out = Σ k_out d_in / Σ d_out d_in``. With
    ``k_out = d_out`` the two-sided fraction is GPM's.
    """
    if not (len(k_out) == len(k_in) == len(d_out) == len(d_in)) or not k_out:
        raise ValueError("protected_fraction needs equally long, non-empty sequences")
    for ko, ki, do, di in zip(k_out, k_in, d_out, d_in):
        if not (0 <= ko <= do and 0 <= ki <= di):
            raise ValueError(f"rank out of range: k_out {ko}/{do}, k_in {ki}/{di}")
    total = float(sum(do * di for do, di in zip(d_out, d_in)))
    return {
        "two_sided": sum(ko * ki for ko, ki in zip(k_out, k_in)) / total,
        "gpm": sum(do * ki for do, ki in zip(d_out, k_in)) / total,
        "rho_out": sum(ko * di for ko, di in zip(k_out, d_in)) / total,
        "rho_out_median": statistics.median(ko / do for ko, do in zip(k_out, d_out)),
        "rho_in_median": statistics.median(ki / di for ki, di in zip(k_in, d_in)),
        "n_layers": len(k_out),
        "n_params": int(total),
    }


def fraction_table(k_out: dict, k_in: dict, dims: dict, groups: dict) -> dict:
    """:func:`protected_fraction` for every half, every group and ``all``.

    ``dims``: ``layer -> (d_out, d_in)``; ``groups``: ``layer -> group``.
    """
    names = list(dims)
    if set(k_out) != set(names) or set(k_in) != set(names):
        raise ValueError("k_out, k_in and dims must cover the same layers")
    sets = {"all": names}
    for n in names:
        sets.setdefault(_half(n), []).append(n)
        sets.setdefault(groups[n], []).append(n)
    return {
        key: protected_fraction([k_out[n] for n in sel], [k_in[n] for n in sel],
                                [dims[n][0] for n in sel], [dims[n][1] for n in sel])
        for key, sel in sets.items()
    }


def forecast_k(k_early: float, k_late: float, d: int, tasks_late: int, steps_between: int,
               targets: Sequence[int]) -> dict[int, float]:
    """A linear FORECAST of a union's rank: the mean per-task increment between two observed
    tasks, extrapolated from the later one and capped at ``d``. Not a measurement."""
    if steps_between < 1:
        raise ValueError("steps_between must be >= 1")
    if k_late < k_early:
        raise ValueError(f"a union cannot shrink ({k_early} -> {k_late})")
    slope = (k_late - k_early) / steps_between
    return {int(t): float(min(d, k_late + (t - tasks_late) * slope)) for t in targets}


def subspace_overlap(A: torch.Tensor, B: torch.Tensor) -> float:
    """``||A^T B||_F^2 / max(k_A, k_B)`` for orthonormal ``A``, ``B``: 1 iff the spans agree."""
    k = max(A.shape[1], B.shape[1])
    if k == 0:
        return 1.0
    return float((A.T.to(torch.float64) @ B.to(torch.float64)).pow(2).sum()) / k
