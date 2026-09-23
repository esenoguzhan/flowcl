"""Gradient Projection Memory with AdamW-safe update projection (§6 item 5, §7.3, §7.4).

Source: Saha, Garg & Roy, "Gradient Projection Memory for Continual Learning", ICLR 2021
(arXiv:2103.09762). For a fully connected layer the new task's gradient is projected onto
the orthogonal complement of the stored input basis (their Eq. 6)::

    ∇W L ← ∇W L − (∇W L) M M^T

**Deviation, and the reported name.** The paper trains with plain SGD, where projecting
the gradient *is* projecting the update. This codebase trains with AdamW (the Gate 1
recipe). Adam's per-coordinate scaling does not preserve subspaces and AdamW's decoupled
weight decay shrinks weights in every direction, so a projected gradient does not give an
orthogonal *applied* update (Gate 3 §6.2). This method therefore projects twice:

1. ``modify_gradients`` — ``G ← G P`` with ``P = I − M M^T``, so Adam's moments only see the
   free-subspace gradient;
2. ``after_step`` — the realised displacement ``ΔW = W − W_ref`` is replaced by ``ΔW P``, so
   the weights never move in protected directions whatever Adam or weight decay did.

Because that is stricter than canonical GPM, the method reports itself as
``gpm_projected_adam`` (``projection: gradient_and_update``), never as plain "GPM". A
gradient-only ablation would be ``gpm_grad_only``; it is not implemented yet.

Details inferred rather than taken from the paper (§11):

* the memory is the §7.2 basis from :mod:`flowcl.analysis.subspace`, primary
  (gradient-reachable) view, at one ``eps`` for every layer (the paper tunes per layer);
* projection acts on :meth:`~flowcl.models.policy.FlowPolicy.projectable_parameters` only,
  and *every* other parameter is frozen (:func:`freeze_to_allowlist`): §7.4's list plus the
  encoder projections, context queries and positional embedding that §7.4 does not name;
* the incremental memory update (paper Eq. 8–9, §7.2) is not implemented yet
  (``update_memory`` must be false); the pilot uses a fixed Task-1 memory.
"""

from __future__ import annotations

import math

import torch

from flowcl.analysis.subspace import SubspaceBasis
from flowcl.methods.base import BaseMethod, register_method

PROJECTION_NAMES = {"gradient_and_update": "gpm_projected_adam"}


# ---- §7.4 allowlist -----------------------------------------------------------


def allowlist(policy) -> tuple[str, ...]:
    """Parameters allowed to train in projection experiments: registry weights only."""
    return tuple(policy.projectable_parameters())


def freeze_to_allowlist(policy) -> dict:
    """Freeze every parameter outside :func:`allowlist`; return what was done.

    Must run before the optimiser is built, so frozen tensors never enter AdamW (and so
    never receive weight decay).
    """
    allowed = set(allowlist(policy))
    params = dict(policy.named_parameters())
    missing = sorted(allowed - set(params))
    if missing:
        raise RuntimeError(f"allowlisted parameters do not exist: {missing}")
    newly_frozen = []
    for name, param in params.items():
        if name in allowed:
            param.requires_grad_(True)
        elif param.requires_grad:
            param.requires_grad_(False)
            newly_frozen.append(name)
    assert_allowlist(policy)
    return {
        "trainable_tensors": len(allowed),
        "trainable_params": sum(params[n].numel() for n in allowed),
        "newly_frozen": newly_frozen,
        "newly_frozen_params": sum(params[n].numel() for n in newly_frozen),
    }


def assert_allowlist(policy) -> None:
    """Raise unless the trainable set is exactly :func:`allowlist` (§7.4)."""
    allowed = set(allowlist(policy))
    trainable = {n for n, p in policy.named_parameters() if p.requires_grad}
    extra, absent = sorted(trainable - allowed), sorted(allowed - trainable)
    if extra or absent:
        raise RuntimeError(
            "§7.4 projection protocol violated.\n"
            f"  trainable but not allowlisted (forgetting would leak through these): {extra}\n"
            f"  allowlisted but frozen: {absent}"
        )


# ---- the method ---------------------------------------------------------------


class _ProjectedLayer:
    """Per-layer runtime state: projector, basis, reference weights, tolerances."""

    def __init__(self, name: str, weight: torch.nn.Parameter, basis: SubspaceBasis, eps: float, weight_atol: float):
        d_in = weight.shape[1]
        if basis.d_in != d_in:
            raise ValueError(f"{name}: basis d_in {basis.d_in} != weight d_in {d_in}")
        k = basis.ranks[eps]
        M = basis.vectors[:, :k].to(torch.float64)
        device = weight.device
        self.name = name
        self.weight = weight
        self.k = k
        self.full_rank = k == d_in
        if self.full_rank:
            # Exactly zero, not a numerically built near-zero matrix: nothing may move.
            self.P = torch.zeros(d_in, d_in, dtype=torch.float32, device=device)
        else:
            eye = torch.eye(d_in, dtype=torch.float64)
            self.P = (eye - M @ M.T).to(device=device, dtype=torch.float32)
        self.M = M.to(device=device, dtype=torch.float32)
        self.W_ref = weight.detach().clone()
        self.atol = weight_atol * float(self.W_ref.norm())
        self.max_residual = 0.0
        self.max_residual_ratio = 0.0  # ||D M|| / (atol + rtol ||D||); must stay <= 1
        self.displacement = torch.zeros_like(self.W_ref)  # cumulative, for reporting


@register_method
class GPM(BaseMethod):
    """Hard projection against a fixed input-subspace memory; see the module docstring."""

    name = "gpm"

    def __init__(
        self,
        eps: float = 0.95,
        projection: str = "gradient_and_update",
        residual_rtol: float = 1e-3,
        residual_weight_atol: float = 1e-6,
        log_interval: int = 100,
        update_memory: bool = False,
    ) -> None:
        super().__init__()
        if projection not in PROJECTION_NAMES:
            raise ValueError(
                f"projection {projection!r} not implemented; available "
                f"{sorted(PROJECTION_NAMES)}"
            )
        if not 0.0 < eps <= 1.0:
            raise ValueError(f"eps must lie in (0, 1], got {eps}")
        if log_interval < 1:
            raise ValueError(f"log_interval must be >= 1, got {log_interval}")
        self.eps = float(eps)
        self.projection = projection
        self.residual_rtol = float(residual_rtol)
        self.residual_weight_atol = float(residual_weight_atol)
        self.log_interval = int(log_interval)
        self.update_memory = bool(update_memory)

        self._memory: dict[str, SubspaceBasis] = {}
        self._layers: list[_ProjectedLayer] = []
        self._active = False
        self.memory_extended: bool | None = None
        # Logged every log_interval steps: layer -> c_l of the raw gradient / of Adam's step.
        self.gradient_c: dict[int, dict[str, float]] = {}
        self.update_c: dict[int, dict[str, float]] = {}
        self.residuals: dict[str, dict] = {}
        self.cumulative_displacement_norm: dict[str, float] = {}

    @property
    def display_name(self) -> str:
        return PROJECTION_NAMES[self.projection]

    def config(self) -> dict:
        return {
            "eps": self.eps,
            "projection": self.projection,
            "residual_rtol": self.residual_rtol,
            "residual_weight_atol": self.residual_weight_atol,
            "log_interval": self.log_interval,
            "update_memory": self.update_memory,
        }

    def set_memory(self, bases: dict[str, SubspaceBasis]) -> None:
        """Install a fixed memory (e.g. the Task-1 bases Gate 2 persisted)."""
        lacking = sorted(n for n, b in bases.items() if self.eps not in b.ranks)
        if lacking:
            raise ValueError(f"bases lack eps={self.eps} for {lacking}")
        self._memory = dict(bases)

    # ---- §6 hooks --------------------------------------------------------------

    def on_task_start(self, task_idx, policy, dataset) -> None:
        if not self._memory:
            if task_idx == 0:
                self._active = False  # Task 1 trains unconstrained (paper Alg. 1)
                return
            raise RuntimeError(
                f"GPM at task {task_idx} has no memory; call set_memory() or train task 0 first"
            )
        assert_allowlist(policy)
        registry = policy.projectable_layers()
        names = [e.name for e in registry]
        if sorted(self._memory) != sorted(names):
            raise ValueError(
                "memory layers do not match the registry: missing "
                f"{sorted(set(names) - set(self._memory))}, extra "
                f"{sorted(set(self._memory) - set(names))}"
            )
        self._layers = [
            _ProjectedLayer(
                e.name, e.module.weight, self._memory[e.name], self.eps, self.residual_weight_atol
            )
            for e in registry
        ]
        self._active = True

    def modify_gradients(self, policy, batch_meta) -> None:
        if not self._active:
            return
        step = batch_meta["step"]
        record = step % self.log_interval == 0
        logged = {}
        with torch.no_grad():
            for layer in self._layers:
                G = layer.weight.grad
                if G is None:
                    raise RuntimeError(f"{layer.name}: no gradient at step {step}")
                if record:
                    logged[layer.name] = _ratio(G, layer.M)
                G.copy_(G @ layer.P)
        if record:
            self.gradient_c[step] = logged

    def after_step(self, policy, step_meta) -> None:
        if not self._active:
            return
        step = step_meta["step"]
        record = step % self.log_interval == 0
        logged = {}
        residuals, norms, finite = [], [], []
        with torch.no_grad():
            for layer in self._layers:
                W = layer.weight
                delta = W - layer.W_ref
                if record:
                    logged[layer.name] = _ratio(delta, layer.M)
                W.copy_(torch.addmm(layer.W_ref, delta, layer.P))
                applied = W - layer.W_ref
                residuals.append((applied @ layer.M).norm())
                norms.append(applied.norm())
                finite.append(torch.isfinite(W).all())
                layer.displacement.add_(applied)
                layer.W_ref.copy_(W)
            # One device->host transfer per step instead of three per layer.
            residuals = torch.stack(residuals).cpu().tolist()
            norms = torch.stack(norms).cpu().tolist()
            finite = torch.stack(finite).cpu().tolist()
        for layer, residual, norm, ok in zip(self._layers, residuals, norms, finite):
            if not ok:
                raise RuntimeError(f"{layer.name}: non-finite weights at step {step}")
            bound = layer.atol + self.residual_rtol * norm
            if residual > bound:
                raise RuntimeError(
                    f"{layer.name}: applied update not orthogonal to the memory at step "
                    f"{step}: ||D M|| = {residual:.3e} > {bound:.3e} (atol {layer.atol:.3e} "
                    f"+ rtol {self.residual_rtol} * ||D|| {norm:.3e})"
                )
            layer.max_residual = max(layer.max_residual, residual)
            if bound > 0:
                layer.max_residual_ratio = max(layer.max_residual_ratio, residual / bound)
        if record:
            self.update_c[step] = logged

    def on_task_end(self, task_idx, policy, dataset) -> None:
        if self.update_memory:
            raise NotImplementedError(
                "incremental GPM memory update (paper Eq. 8-9, §7.2) arrives in build step "
                "8; run with update_memory=false for the fixed-memory pilot"
            )
        self.memory_extended = False
        for layer in self._layers:
            self.residuals[layer.name] = {
                "max_residual": layer.max_residual,
                "max_residual_over_bound": layer.max_residual_ratio,
                "atol": layer.atol,
                "k": layer.k,
                "full_rank": layer.full_rank,
            }
            self.cumulative_displacement_norm[layer.name] = float(layer.displacement.norm())
        # Release ~0.5 GB of projectors and reference weights; the logs stay.
        self._layers = []
        self._active = False

    # ---- reporting -------------------------------------------------------------

    def stored_bytes(self) -> int:
        """§8.2: the memory is the bases, k x d_in floats per layer (float32 in use)."""
        return sum(b.ranks[self.eps] * b.d_in * 4 for b in self._memory.values())

    def state_dict(self) -> dict:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "config": self.config(),
            "memory_layers": len(self._memory),
            "stored_mb": self.stored_bytes() / 1e6,
            "memory_extended": self.memory_extended,
            "gradient_c": {str(k): v for k, v in self.gradient_c.items()},
            "update_c": {str(k): v for k, v in self.update_c.items()},
            "residuals": self.residuals,
            "cumulative_displacement_norm": self.cumulative_displacement_norm,
        }

    def describe(self) -> str:
        return f"{self.display_name} (eps {self.eps}, stores {self.stored_bytes() / 1e6:.2f} MB)"


def _ratio(A: torch.Tensor, M: torch.Tensor) -> float:
    """``||A M|| / ||A||`` (NaN for a zero ``A``, e.g. a skipped AMP step)."""
    total = float(A.norm())
    if total == 0.0:
        return math.nan
    return float((A @ M).norm()) / total
