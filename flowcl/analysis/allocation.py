"""Allocation rules at matched capacity (the allocation-rules pilot, ``configs/analysis/allocation_rules.yaml``).

GPM extends its memory after each task layer by layer (:func:`flowcl.analysis.subspace.extend_basis`).
How *much* the extension protects and *where* it protects are tied together in the threshold rules:
a flat ``eps`` and the new-energy rule both decide each layer's rank from that layer alone. Here the
total is fixed from outside. A **capacity target** is an integer protected count
``C = Σ_l d_out,l k_l`` (the numerator of the parameter-weighted protected fraction,
:func:`flowcl.analysis.sensitivity.protected_fraction`), and a rule with one knob distributes it:

* ``flat``     τ = x                                         (x ∈ [0.95, 1]; the e99 family)
* ``novelty``  τ = max(0.95, min(1, p + x (1 − p)))           (x = f ∈ [0, 0.99999]; ne90's rule)
* ``size``     k = max(k(0.95), min(#{λ_i ≥ x · tr K}, cap))  (x = δ ∈ [1e-12, 1]; larger δ protects less)
* ``room``     novelty with f_l = min(0.99999, x (1 − ρ_l)),  ρ_l = k_before,l / d_in,l   (x = κ ∈ [0, 1e3])

Every ``k`` is a count of **added** directions; τ-rules add the smallest ``k`` reaching τ exactly as
Eq. 9 does in :func:`extend_basis` (:func:`k_for_target` is that arithmetic). The 0.95 floor holds
for the arm's own memory and activations.

:func:`match_capacity` bisects the knob to the two attainable counts on either side of the target
(the **bracket**) and takes the end nearest the target (an exact tie goes to the upper end). The
allocation is taken from the bracket end itself. When the floor alone already exceeds the target
(``floor_bound``) or the knob's maximum stays below it (``ceiling_bound``), the endpoint is taken and
its error recorded; the caller's gate decides. A target below the current count raises (memory never
shrinks), as does one above the registry's full capacity.

Everything here is pure and CPU-only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from flowcl.analysis.subspace import adaptive_target

FLOOR_EPS = 0.95
F_MAX = 0.99999


@dataclass
class LayerStats:
    """One layer's extension inputs (:func:`flowcl.analysis.subspace.residual_stats` plus ``d_out``)."""

    name: str
    d_in: int
    d_out: int
    k_before: int
    proj: float
    total: float
    eigenvalues: torch.Tensor
    cap: int
    cumulative: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0 <= self.k_before <= self.d_in:
            raise ValueError(f"{self.name}: k_before {self.k_before} outside [0, {self.d_in}]")
        if not 0 <= self.cap <= self.d_in - self.k_before:
            raise ValueError(f"{self.name}: cap {self.cap} outside [0, {self.d_in - self.k_before}]")
        self.eigenvalues = self.eigenvalues.to(torch.float64)
        # extend_basis's expression, operation for operation.
        self.cumulative = self.proj * self.total + torch.cumsum(self.eigenvalues, dim=0)

    @classmethod
    def from_residual(cls, name: str, d_out: int, stats: dict) -> "LayerStats":
        return cls(name=name, d_in=stats["d_in"], d_out=int(d_out), k_before=stats["k_before"],
                   proj=stats["proj"], total=stats["total"], eigenvalues=stats["eigenvalues"],
                   cap=stats["cap"])


def k_for_target(s: LayerStats, tau: float) -> int:
    """Directions Eq. 9 adds for target ``tau`` (:func:`extend_basis`'s arithmetic): 0 if met."""
    if s.cap == 0 or s.proj >= tau:
        return 0
    target = torch.tensor([tau * s.total], dtype=torch.float64)
    k = int(torch.searchsorted(s.cumulative, target, side="left")) + 1
    return max(1, min(k, s.cap))


def _flat(s: LayerStats, x: float, eps: float) -> int:
    return k_for_target(s, x)


def _novelty(s: LayerStats, x: float, eps: float) -> int:
    return k_for_target(s, adaptive_target(s.proj, eps, x))


def _size(s: LayerStats, x: float, eps: float) -> int:
    if s.cap == 0:
        return 0
    large = int((s.eigenvalues >= x * s.total).sum())
    return max(k_for_target(s, eps), min(large, s.cap))


def _room(s: LayerStats, x: float, eps: float) -> int:
    f = min(F_MAX, x * (1.0 - s.k_before / s.d_in))
    return k_for_target(s, adaptive_target(s.proj, eps, f))


@dataclass(frozen=True)
class Rule:
    """``k_added(stats, x, eps)``; ``least``/``most`` are the knob values that protect least/most."""

    name: str
    knob: str
    k_added: object
    least: float
    most: float
    geometric: bool = False

    def mid(self, a: float, b: float) -> float:
        return math.sqrt(a * b) if self.geometric else 0.5 * (a + b)


RULES: dict[str, Rule] = {
    "flat": Rule("flat", "eps", _flat, least=FLOOR_EPS, most=1.0),
    "novelty": Rule("novelty", "f", _novelty, least=0.0, most=F_MAX),
    "size": Rule("size", "delta", _size, least=1.0, most=1e-12, geometric=True),
    "room": Rule("room", "kappa", _room, least=0.0, most=1e3),
}


def allocate(rule: Rule | str, stats: list[LayerStats], x: float, eps: float = FLOOR_EPS) -> dict[str, int]:
    """``layer -> k_added`` for knob ``x``."""
    rule = RULES[rule] if isinstance(rule, str) else rule
    return {s.name: int(rule.k_added(s, x, eps)) for s in stats}


def protected_count(stats: list[LayerStats], k_added: dict[str, int] | None = None) -> int:
    """``Σ_l d_out,l (k_before,l + k_added,l)``."""
    k_added = k_added or {}
    return sum(s.d_out * (s.k_before + k_added.get(s.name, 0)) for s in stats)


def full_count(stats: list[LayerStats]) -> int:
    """``Σ_l d_out,l d_in,l``: the registry's weight count."""
    return sum(s.d_out * s.d_in for s in stats)


def match_capacity(rule: Rule | str, stats: list[LayerStats], target: int, eps: float = FLOOR_EPS,
                   iterations: int = 200) -> dict:
    """The rule's allocation nearest to protected count ``target`` (module docstring).

    Returns ``status`` (``matched`` / ``floor_bound`` / ``ceiling_bound``), the chosen ``knob``,
    ``k_added`` per layer, ``count``, ``target``, ``error`` (count − target) and ``error_pp`` (in
    percentage points of the registry), the bracket (``lower`` < target <= ``upper`` counts, ``None``
    outside a match), ``floor_count`` and ``ceiling_count``, the ``jump_layers`` whose k differs
    between the bracket's ends, and ``current`` / ``full``.
    """
    rule = RULES[rule] if isinstance(rule, str) else rule
    target = int(target)
    current, full = protected_count(stats), full_count(stats)
    if target < current:
        raise ValueError(f"target {target} is below the current protected count {current}; memory never shrinks")
    if target > full:
        raise ValueError(f"target {target} exceeds the registry's {full} weights")

    def count(alloc):
        return protected_count(stats, alloc)

    least, most = allocate(rule, stats, rule.least, eps), allocate(rule, stats, rule.most, eps)
    floor_count, ceiling_count = count(least), count(most)
    out = {"rule": rule.name, "knob_name": rule.knob, "target": target, "current": current, "full": full,
           "floor_count": floor_count, "ceiling_count": ceiling_count, "lower": None, "upper": None,
           "jump_layers": []}
    if floor_count >= target:
        status, x, chosen = ("matched" if floor_count == target else "floor_bound"), rule.least, least
    elif ceiling_count < target:
        status, x, chosen = "ceiling_bound", rule.most, most
    else:
        a, b = rule.least, rule.most          # invariant: count(a) < target <= count(b)
        for _ in range(iterations):
            m = rule.mid(a, b)
            if m in (a, b):
                break
            if count(allocate(rule, stats, m, eps)) >= target:
                b = m
            else:
                a = m
        lower, upper = allocate(rule, stats, a, eps), allocate(rule, stats, b, eps)
        lo_c, up_c = count(lower), count(upper)
        if not lo_c < target <= up_c:
            raise RuntimeError(f"bisection lost its bracket: {lo_c} < {target} <= {up_c} fails")
        out.update(lower=lo_c, upper=up_c, knob_lower=a, knob_upper=b,
                   jump_layers=sorted(n for n in upper if upper[n] != lower[n]))
        status = "matched"
        if up_c - target <= target - lo_c:   # an exact tie goes to the upper end
            x, chosen = b, upper
        else:
            x, chosen = a, lower
    c = count(chosen)
    out.update(status=status, knob=x, k_added=chosen, count=c, error=c - target,
               error_pp=100.0 * (c - target) / full)
    return out


# ---- measures ------------------------------------------------------------------------------------


def reallocated(k_a: dict[str, int], k_b: dict[str, int], dims: dict[str, tuple[int, int]]) -> float:
    """Capacity reallocated across layers, ``½ Σ d_out |k_a − k_b| / Σ d_out d_in`` (a fraction).

    ``dims`` is ``layer -> (d_out, d_in)``; ``k`` are total ranks. At equal totals, moving one
    direction's capacity from one layer to another counts once.
    """
    if sorted(k_a) != sorted(k_b) or sorted(k_a) != sorted(dims):
        raise ValueError("reallocated needs the same layers in both allocations and the dims")
    total = sum(do * di for do, di in dims.values())
    return 0.5 * sum(dims[n][0] * abs(k_a[n] - k_b[n]) for n in dims) / total


def subspace_overlap(M_a: dict[str, torch.Tensor], M_b: dict[str, torch.Tensor],
                     d_out: dict[str, int]) -> float:
    """``Σ d_out ‖M_aᵀ M_b‖²_F / Σ d_out min(k_a, k_b)``: 1 when the smaller memory lies in the larger."""
    if sorted(M_a) != sorted(M_b) or sorted(M_a) != sorted(d_out):
        raise ValueError("subspace_overlap needs the same layers in both memories and d_out")
    num = den = 0.0
    for n in M_a:
        a, b = M_a[n].to(torch.float64), M_b[n].to(torch.float64)
        num += d_out[n] * float((a.T @ b).pow(2).sum())
        den += d_out[n] * min(a.shape[1], b.shape[1])
    return num / den if den else 1.0
