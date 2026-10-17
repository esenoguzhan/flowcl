"""Output-side sensitivity of the integrated, executed action prefix (E1-B; D2's angle B in
``docs/thesis_plan.md``; ``configs/analysis/twosided_angleb.yaml``).

E1 (:mod:`flowcl.analysis.sensitivity`) measured each registry layer's output sensitivity for the
whole velocity at a random flow time. E1-B measures it for what the robot executes: the policy
samples a chunk by integrating its velocity field with forward Euler from noise
(:meth:`flowcl.models.flow_head.FlowHead.sample`) and executes the first ``K`` actions.
:func:`integrate` is the same arithmetic with autograd on: the trunk encodes once, the decoder runs
``n_steps`` times.

**Coherent targets, exact over output directions.** For each action dimension d,
``T_d = sum_{k<K} a_{k,d} / sqrt(K)``: a constant offset across the executed prefix, the temporal
direction the recovery study perturbed. One vector-Jacobian product (VJP) per d gives
``g_d = dT_d / dy`` at every hooked output, so ``sum_d r_d g_d g_d^T`` is exact for any diagonal
metric ``r`` over the 7 coherent outputs, conditional on the captured samples and rows (no
Hutchinson probe). Rows are accumulated into one Gram per (phase bin, group of dimensions), each
row scaled by ``c_d``, so any weighting ``w[bin, group]`` is a linear combination of the stored
Grams.

**The independent-step comparison.** Hutchinson probes ``u`` over the ``K x D`` prefix (supplied by
the caller, already weighted) give ``E[g g^T] = J^T diag(r) J``.

**Rows.** A hooked decoder layer runs once per Euler step; every run's positions are rows. Per sample
and layer, up to ``tokens_per_sample`` rows are drawn among the (call, position) pairs whose output
gradient is nonzero for a generic random combination of the executed prefix's ``K x D`` outputs.
That support is almost surely the union of every prefix output's support, so it contains every
coherent and every Hutchinson VJP's support. The row set is drawn once per batch and serves every
VJP, and only one VJP's gradients are held at a time.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Sequence

import torch
from torch import nn

from flowcl.analysis.hooks import KIND_STATE, CaptureLayer
from flowcl.analysis.sensitivity import _select


def integrate(policy, batch: dict, noise: torch.Tensor, n_steps: int) -> torch.Tensor:
    """``A_1`` by forward Euler from ``noise``, autograd on: the arithmetic of ``policy.sample``."""
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    context = policy.encode_observation(batch)
    head = policy.flow_head
    actions = noise.to(device=context.device, dtype=context.dtype).clone()
    size = context.shape[0]
    ds = 1.0 / n_steps
    for step in range(n_steps):
        s = torch.full((size,), step * ds, device=context.device, dtype=actions.dtype)
        actions = actions + ds * head.forward(actions, context, s)
    return actions


@dataclass
class IntegratedAccumulator:
    """One layer's Grams: ``coherent[bin, group]`` and one per Hutchinson probe."""

    name: str
    kind: str
    d_out: int
    coherent: torch.Tensor                      # (n_bins, n_groups, d_out, d_out), float64
    hutch: list[torch.Tensor] = field(default_factory=list)
    n: int = 0                                  # rows (shared by every VJP)
    n_bin: list[int] = field(default_factory=list)
    calls: int = 0                              # hooked runs in the last recorded forward


class IntegratedOutputCapture:
    """Retain every run of every hooked output during an integration; accumulate the exact coherent
    Grams per (phase bin, group) and the Hutchinson Grams on demand.

    Args:
        owner: Module the hooks belong to.
        layers: Layers to hook, in registry order.
        groups: The dimension groups, as ``(lo, hi)`` index ranges of the action (position,
            rotation, gripper).
        n_bins: Phase bins.
        n_probes: Hutchinson Grams (the independent-step comparison).
        tokens_per_sample: Maximum rows per sample, per layer.
        subsample_seed: Seed of the CPU generator that draws the generic combination and the rows.
    """

    def __init__(self, owner: nn.Module, layers: Sequence[CaptureLayer], groups: Sequence[tuple[int, int]],
                 n_bins: int, n_probes: int, tokens_per_sample: int, subsample_seed: int) -> None:
        if tokens_per_sample < 1:
            raise ValueError(f"tokens_per_sample must be >= 1, got {tokens_per_sample}")
        if not layers:
            raise ValueError("IntegratedOutputCapture received no layers")
        self.owner = owner
        self.layers = tuple(layers)
        self.groups = [tuple(int(x) for x in g) for g in groups]
        self.n_bins = int(n_bins)
        self.n_probes = int(n_probes)
        self.tokens_per_sample = int(tokens_per_sample)
        self._generator = torch.Generator(device="cpu").manual_seed(int(subsample_seed))
        self._handles: list = []
        self._recording = False
        self._outputs: dict[str, list[torch.Tensor]] = {}
        self.accumulators: dict[str, IntegratedAccumulator] = {}
        for layer in self.layers:
            d = layer.module.out_features
            self.accumulators[layer.name] = IntegratedAccumulator(
                name=layer.name, kind=layer.kind, d_out=d,
                coherent=torch.zeros(self.n_bins, len(self.groups), d, d, dtype=torch.float64),
                hutch=[torch.zeros(d, d, dtype=torch.float64) for _ in range(self.n_probes)],
                n_bin=[0] * self.n_bins)

    @classmethod
    def for_policy(cls, policy, **kwargs) -> "IntegratedOutputCapture":
        """Hook every registry layer of a :class:`~flowcl.models.policy.FlowPolicy`."""
        from flowcl.analysis.hooks import layer_kind

        layers = [CaptureLayer(e.name, e.module, layer_kind(e.name)) for e in policy.projectable_layers()]
        return cls(policy, layers, **kwargs)

    # ---- lifecycle -------------------------------------------------------------

    def __enter__(self) -> "IntegratedOutputCapture":
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
    def recording(self):
        """Retain every run of every hooked output inside (an integration has several per layer)."""
        if not self._handles:
            raise RuntimeError("recording() used outside the capture's `with` block")
        self._outputs = {layer.name: [] for layer in self.layers}
        self._recording = True
        try:
            yield
        finally:
            self._recording = False

    def _make_hook(self, layer: CaptureLayer):
        def hook(module, args, output):
            if output.shape[-1] != module.out_features:
                raise RuntimeError(f"{layer.name}: captured output has last dim {output.shape[-1]} but the "
                                   f"layer's out_features is {module.out_features}; U must live in the output space")
            if self._recording:
                self._outputs[layer.name].append(output)

        return hook

    # ---- the VJPs ----------------------------------------------------------------

    def backward(self, prefix: torch.Tensor, scale: Sequence[float], bins: torch.Tensor,
                 probes: Sequence[torch.Tensor] = ()) -> None:
        """All VJPs through the retained integration; accumulate every Gram.

        Args:
            prefix: The executed prefix ``(B, K, D)`` of the integrated chunk, with its graph.
            scale: ``c_d`` per action dimension (a row of ``g_d`` is scaled by ``c_d``).
            bins: ``(B,)`` phase bin of each sample.
            probes: ``n_probes`` Hutchinson tensors ``(B, K, D)``, already weighted.
        """
        if len(probes) != self.n_probes:
            raise ValueError(f"expected {self.n_probes} probes, got {len(probes)}")
        if prefix.ndim != 3 or not prefix.requires_grad:
            raise RuntimeError("the prefix needs shape (B, K, D) and a graph (integration under no_grad?)")
        size, k, d = prefix.shape
        if len(scale) != d or any(hi > d for _, hi in self.groups):
            raise ValueError("scale and groups must match the action dimension")
        bins = bins.detach().to("cpu", torch.long)
        if tuple(bins.shape) != (size,) or int(bins.min()) < 0 or int(bins.max()) >= self.n_bins:
            raise ValueError(f"bins must be ({size},) in [0, {self.n_bins})")
        names = [layer.name for layer in self.layers]
        missing = [n for n in names if not self._outputs.get(n)]
        if missing:
            raise RuntimeError(f"no retained output for {len(missing)} layer(s), e.g. {missing[:3]}")
        calls = {n: len(self._outputs[n]) for n in names}
        flat = [o for n in names for o in self._outputs[n]]
        if any(not o.requires_grad for o in flat):
            raise RuntimeError("outputs without a graph (forward ran under no_grad?)")
        group_of = {}
        for g, (lo, hi) in enumerate(self.groups):
            for dim in range(lo, hi):
                group_of[dim] = g

        generic = torch.randn((size, k, d), generator=self._generator, dtype=torch.float32)
        targets = [(prefix * generic.to(prefix.device, prefix.dtype)).sum()]
        targets += [prefix[..., dim].sum() / math.sqrt(k) for dim in range(d)]
        targets += [(prefix * u.to(prefix.device, prefix.dtype)).sum() for u in probes]
        rows: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for j, target in enumerate(targets):
            grads = torch.autograd.grad(target, flat, retain_graph=j < len(targets) - 1, allow_unused=True)
            at = 0
            for name in names:
                acc = self.accumulators[name]
                parts = []
                for out, g in zip(self._outputs[name], grads[at: at + calls[name]]):
                    g = torch.zeros_like(out) if g is None else g.detach()
                    parts.append(g.unsqueeze(1) if acc.kind == KIND_STATE else g)
                at += calls[name]
                if any(p.ndim != 3 or p.shape[0] != size or p.shape[-1] != acc.d_out for p in parts):
                    raise RuntimeError(f"{name}: expected (B, T, {acc.d_out}) gradients")
                g = torch.cat(parts, dim=1)
                if j == 0:           # the generic combination: the eligible rows, drawn once
                    eligible = (g != 0).any(dim=-1).to("cpu")
                    rows[name] = _select(eligible, self.tokens_per_sample, self._generator)
                    b_idx = rows[name][0]
                    acc.n += int(b_idx.numel())
                    acc.calls = calls[name]
                    for b in range(self.n_bins):
                        acc.n_bin[b] += int((bins[b_idx] == b).sum())
                    continue
                b_idx, t_idx = rows[name]
                if b_idx.numel() == 0:
                    continue
                sel = g[b_idx.to(g.device), t_idx.to(g.device)].to("cpu", torch.float64)
                if j <= d:           # coherent dimension j - 1
                    dim = j - 1
                    sel = sel * float(scale[dim])
                    row_bins = bins[b_idx]
                    for b in torch.unique(row_bins).tolist():
                        part = sel[row_bins == b]
                        acc.coherent[b, group_of[dim]].addmm_(part.T, part)
                else:                # Hutchinson probe j - 1 - d
                    acc.hutch[j - 1 - d].addmm_(sel.T, sel)
            del grads
        self._outputs = {n: [] for n in names}


def combine_grams(coherent: torch.Tensor, weights) -> torch.Tensor:
    """``sum_{b,g} w[b, g] G[b, g]`` for a stored ``(n_bins, n_groups, d, d)`` tensor."""
    w = torch.as_tensor(weights, dtype=torch.float64)
    if tuple(w.shape) != tuple(coherent.shape[:2]):
        raise ValueError(f"weights must be {tuple(coherent.shape[:2])}, got {tuple(w.shape)}")
    return torch.einsum("bg,bgij->ij", w, coherent.to(torch.float64))
