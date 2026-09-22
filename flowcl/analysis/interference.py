"""Gradient interference ``c_l``: per-layer decomposition against a protected basis (§7.3).

Spec §7.3::

    For layer l with gradient G_l:
        G_∥ = G_l M_l M_l^T
        G_⊥ = G_l - G_∥
        c_l = ||G_∥||_F / ||G_l||_F

    This normalizes by the full gradient norm, so c_l in [0, 1]. [...] nn.Linear.weight is
    (d_out, d_in); the projection acts on the **input** dimension — get this wrong and
    the whole method is silently broken; unit-test it.

Orientation is asserted by shape (``G.shape[1] == M.shape[0] == d_in``). For square
layers — most of the registry — a shape check cannot catch a transpose, which is why
``tests/test_interference.py`` has a square-matrix orientation test with a known answer.

Everything here computes in float64. ``c_l`` is a ratio of two energies that are equal
in exact arithmetic when the basis spans the whole input space (e.g. ``action_in`` at
``rho = 1``); in lower precision the parallel part can exceed the total by rounding.
A ratio above ``1 + RATIO_ATOL`` raises; anything below that is clamped to 1.

``ProjectedOptimizer`` (build order §10 step 8) reuses :func:`decompose`.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch

# Rounding slack for ||G_∥||² <= ||G||² in float64. Real violations (a non-orthonormal
# basis, a transposed projection) are orders of magnitude larger.
RATIO_ATOL = 1e-9

# Tolerance for V^T V = I when a basis is handed to this module.
_ORTHONORMALITY_ATOL = 1e-6


def _check_gradient(G: torch.Tensor, d_in: int, layer: str) -> None:
    if G.ndim != 2:
        raise ValueError(
            f"{layer}: expected a (d_out, d_in) weight gradient, got shape "
            f"{tuple(G.shape)}. Biases are excluded from projection (§7.4) and have "
            "no input subspace."
        )
    if G.shape[1] != d_in:
        raise ValueError(
            f"{layer}: gradient is {tuple(G.shape)} but the basis lives in d_in={d_in}. "
            "nn.Linear.weight is (d_out, d_in); the projection acts on dim 1 (§7.3)."
        )
    nonfinite = int((~torch.isfinite(G)).sum())
    if nonfinite:
        raise ValueError(f"{layer}: gradient has {nonfinite} non-finite entries")


def assert_orthonormal(V: torch.Tensor, layer: str = "") -> None:
    """Raise unless ``V`` has orthonormal columns (``V^T V = I``)."""
    if V.ndim != 2:
        raise ValueError(f"{layer}: basis must be (d_in, k), got {tuple(V.shape)}")
    gram = V.T.to(torch.float64) @ V.to(torch.float64)
    error = float((gram - torch.eye(V.shape[1], dtype=torch.float64, device=V.device)).abs().max())
    if error > _ORTHONORMALITY_ATOL:
        raise ValueError(f"{layer}: basis is not orthonormal (max |V^T V - I| = {error:.3e})")


def decompose(
    G: torch.Tensor, M: torch.Tensor, layer: str = ""
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(G_∥, G_⊥)`` with ``G_∥ = G M M^T`` (§7.3). ``M`` is ``(d_in, k)``, orthonormal."""
    if M.ndim != 2:
        raise ValueError(f"{layer}: basis must be (d_in, k), got {tuple(M.shape)}")
    _check_gradient(G, M.shape[0], layer)
    G64 = G.to(torch.float64)
    M64 = M.to(device=G.device, dtype=torch.float64)
    parallel = (G64 @ M64) @ M64.T
    return parallel, G64 - parallel


def c_from_energies(parallel_energy: float, total_energy: float, layer: str = "") -> float:
    """``c_l = sqrt(||G_∥||² / ||G||²)`` with the rounding rule in the module docstring."""
    return math.sqrt(_ratio(parallel_energy, total_energy, layer))


def _ratio(parallel_energy: float, total_energy: float, layer: str) -> float:
    if total_energy <= 0.0:
        raise ValueError(
            f"{layer}: zero gradient, so c_l = ||G_∥||/||G|| is undefined; refusing to "
            "report a default"
        )
    ratio = parallel_energy / total_energy
    if ratio > 1.0 + RATIO_ATOL:
        raise ValueError(
            f"{layer}: ||G_∥||² exceeds ||G||² by {ratio - 1:.3e}; the basis is not "
            "orthonormal or the projection is on the wrong side"
        )
    return min(ratio, 1.0)


def interference_ratio(G: torch.Tensor, M: torch.Tensor, layer: str = "") -> float:
    """``c_l = ||G M M^T||_F / ||G||_F`` exactly as §7.3 writes it."""
    parallel, _ = decompose(G, M, layer)
    total = float((G.to(torch.float64) ** 2).sum())
    return c_from_energies(float((parallel**2).sum()), total, layer)


def projected_energies(
    G: torch.Tensor, V: torch.Tensor, ranks: Sequence[int], layer: str = ""
) -> tuple[float, list[float]]:
    """``(||G||², [||G V[:, :k]||² for k in ranks])`` from one matmul.

    ``V`` must have orthonormal columns and hold the basis at the largest rank; smaller
    bases are its prefixes (:class:`flowcl.analysis.subspace.SubspaceBasis`). For an
    orthonormal ``M``, ``||G M M^T||_F = ||G M||_F``, so column energies of ``G V``
    summed up to ``k`` give the parallel energy at every rank at once.
    """
    if V.ndim != 2:
        raise ValueError(f"{layer}: basis must be (d_in, k), got {tuple(V.shape)}")
    _check_gradient(G, V.shape[0], layer)
    bad = [k for k in ranks if not 1 <= k <= V.shape[1]]
    if bad:
        raise ValueError(f"{layer}: ranks {bad} outside [1, {V.shape[1]}]")

    G64 = G.to(torch.float64)
    total = float((G64**2).sum())
    column_energy = ((G64 @ V.to(device=G.device, dtype=torch.float64)) ** 2).sum(dim=0)
    cumulative = torch.cumsum(column_energy, dim=0)
    return total, [float(cumulative[k - 1]) for k in ranks]


def prefix_interference(
    G: torch.Tensor, V: torch.Tensor, ranks: dict, layer: str = ""
) -> dict:
    """``{key: c_l}`` for each basis rank in ``ranks`` (e.g. ``{eps: k_eps}``)."""
    keys = list(ranks)
    total, parallel = projected_energies(G, V, [ranks[k] for k in keys], layer)
    return {key: c_from_energies(p, total, layer) for key, p in zip(keys, parallel)}


def mean_ratio(parallel: Sequence[float], total: Sequence[float], layer: str = "") -> float:
    """Arithmetic mean of per-batch ``c_l`` — each batch is one optimiser step."""
    if len(parallel) != len(total) or not parallel:
        raise ValueError(f"{layer}: need matching, non-empty per-batch energies")
    return sum(c_from_energies(p, t, layer) for p, t in zip(parallel, total)) / len(
        parallel
    )


def energy_weighted_ratio(
    parallel: Sequence[float], total: Sequence[float], layer: str = ""
) -> float:
    """``sqrt(Σ_b ||G_∥,b||² / Σ_b ||G_b||²)`` — large-gradient batches weigh more."""
    if len(parallel) != len(total) or not parallel:
        raise ValueError(f"{layer}: need matching, non-empty per-batch energies")
    return c_from_energies(sum(parallel), sum(total), layer)
