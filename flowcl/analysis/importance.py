"""Basis importance for Scaled Gradient Projection (Saha & Roy, AAAI 2023, Eq. 2, 6-10).

Source: Saha & Roy, "Continual Learning with Scaled Gradient Projection", AAAI 2023
(arXiv:2302.01386). SGP keeps GPM's memory ``M`` (Eq. 8-9,
:func:`flowcl.analysis.subspace.extend_basis`, unchanged) and adds one importance
``λ_i ∈ (0, 1]`` per basis vector; the projection becomes ``G (I − M Λ Mᵀ)``, so a
direction with ``λ = 1`` is blocked as in GPM and the others are scaled by ``1 − λ``.

Everything here works on Gram matrices ``K = R Rᵀ`` of the task's layer inputs, never on
``R`` itself:

* **Eq. 2** — ``λ_i = (α+1) σ_i / (α σ_i + max_j σ_j)``, evaluated as
  ``(α+1) r_i / (α r_i + 1)`` with ``r = σ / max σ``, so the largest gets exactly ``1.0``
  (``r = 1.0`` exactly) and float rounding is clamped to ``<= 1``.
* **Eq. 6-8** — the surrogate singular value of an *old* basis vector under the current
  task, ``σ'_i² = Σ_j (m_iᵀ u_j)² σ_j² = m_iᵀ K m_i`` (the paper's ``(C⊙C) σ²`` over the SVD
  of the projected representation; no explicit ``R`` needed).
* **Eq. 10** — ``λ_i ← min(1, λ_i + λ_{i,τ})`` for old bases; new bases take ``λ_{i,τ}``.

The new bases' ``σ̂`` are the square roots of the residual Gram's leading eigenvalues,
exactly the spectrum :func:`~flowcl.analysis.subspace.extend_basis` returns. One detail is
inferred (§11): when a task adds no basis, Eq. 2's max runs over the old bases' ``σ'`` only.
"""

from __future__ import annotations

import math

import torch

# diag(Mᵀ K M) of a PSD Gram is >= 0; anything below -NEG_REL * tr K is not round-off.
NEG_REL = 1e-10


def sgp_importance(sigma: torch.Tensor, alpha: float) -> torch.Tensor:
    """Eq. 2: ``λ = (α+1) r / (α r + 1)``, ``r = σ / max σ`` (float64, exactly 1 at the max)."""
    if not (math.isfinite(alpha) and alpha > 0):
        raise ValueError(f"alpha must be a positive finite number, got {alpha}")
    s = sigma.to(torch.float64)
    if s.ndim != 1 or s.numel() == 0:
        raise ValueError(f"sigma must be a non-empty 1-D tensor, got shape {tuple(s.shape)}")
    if not torch.isfinite(s).all():
        raise ValueError("sigma has non-finite entries")
    if (s < 0).any():
        raise ValueError(f"sigma has negative entries (min {float(s.min()):.3e})")
    top = s.max()
    if not float(top) > 0.0:
        raise ValueError("max sigma is 0: the task has no energy on these bases")
    r = s / top
    return torch.clamp((alpha + 1.0) * r / (alpha * r + 1.0), max=1.0)


def _symmetric(K: torch.Tensor) -> torch.Tensor:
    K = K.to(torch.float64)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"Gram must be square, got {tuple(K.shape)}")
    if not torch.isfinite(K).all():
        raise ValueError("Gram has non-finite entries")
    return 0.5 * (K + K.T)


def basis_energy(M: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """``diag(Mᵀ K M)``: each basis vector's energy under ``K`` (float64, round-off >= 0)."""
    K = _symmetric(K)
    M = M.to(torch.float64)
    if M.shape[0] != K.shape[0]:
        raise ValueError(f"basis is ({M.shape[0]}, k) but the Gram is {K.shape[0]}x{K.shape[0]}")
    if M.shape[1] == 0:
        return torch.zeros(0, dtype=torch.float64)
    energy = ((K @ M) * M).sum(dim=0)
    floor = -NEG_REL * abs(float(torch.trace(K)))
    if float(energy.min()) < floor:
        raise ValueError(
            f"basis energy {float(energy.min()):.3e} is negative beyond round-off "
            f"({floor:.3e}); the Gram is not PSD"
        )
    return energy.clamp(min=0.0)


def surrogate_sigma(M: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Eq. 6-8 on a Gram: ``σ'_i = sqrt(m_iᵀ K m_i)`` for every column of ``M``."""
    return basis_energy(M, K).sqrt()


def accumulate_importance(
    prev: torch.Tensor, task_old: torch.Tensor, task_new: torch.Tensor
) -> torch.Tensor:
    """Eq. 10: ``[min(1, prev + task_old); task_new]`` (float64)."""
    prev, task_old = prev.to(torch.float64), task_old.to(torch.float64)
    if prev.shape != task_old.shape:
        raise ValueError(
            f"old importance has {tuple(prev.shape)} entries, this task's {tuple(task_old.shape)}"
        )
    return torch.cat([torch.clamp(prev + task_old, max=1.0), task_new.to(torch.float64)])


def importance_update(
    M_old: torch.Tensor | None,
    K: torch.Tensor,
    residual_spectrum: torch.Tensor,
    k_added: int,
    prev: torch.Tensor,
    alpha: float,
) -> dict[str, torch.Tensor]:
    """One layer's importance after a memory extension (Eq. 2, 6-10).

    Args:
        M_old: The memory *before* the extension (``None`` or empty at the first task).
        K: The task's input Gram (the same one the extension used).
        residual_spectrum: The residual Gram's eigenvalues, descending, as returned by
            :func:`~flowcl.analysis.subspace.extend_basis` (empty if nothing was added).
        k_added: How many bases the extension appended.
        prev: The importance of ``M_old``'s columns (empty at the first task).

    Returns:
        ``sigma_old`` (σ'), ``sigma_new`` (σ̂), ``lambda_task`` (λ_τ over both),
        ``lambda_prev`` and ``lambda`` (accumulated, one per column of the new memory).
    """
    k_old = 0 if M_old is None else M_old.shape[1]
    prev = prev.to(torch.float64)
    if prev.numel() != k_old:
        raise ValueError(f"importance has {prev.numel()} entries for a {k_old}-column memory")
    if k_added < 0 or k_added > residual_spectrum.numel():
        raise ValueError(
            f"k_added {k_added} but only {residual_spectrum.numel()} residual eigenvalues"
        )
    sigma_old = (
        torch.zeros(0, dtype=torch.float64) if k_old == 0 else surrogate_sigma(M_old, K)
    )
    sigma_new = residual_spectrum[:k_added].to(torch.float64).clamp(min=0).sqrt()
    lam_task = sgp_importance(torch.cat([sigma_old, sigma_new]), alpha)
    lam = accumulate_importance(prev, lam_task[:k_old], lam_task[k_old:])
    return {
        "sigma_old": sigma_old,
        "sigma_new": sigma_new,
        "lambda_task": lam_task,
        "lambda_prev": prev,
        "lambda": lam,
    }


def protected_energy_fraction(M: torch.Tensor, lam: torch.Tensor, K: torch.Tensor) -> float:
    """``Σ λ_i m_iᵀ K m_i / tr K``: the importance-weighted share of the task's energy."""
    energy = basis_energy(M, K)
    if lam.shape != energy.shape:
        raise ValueError(f"{lam.numel()} importances for {energy.numel()} basis vectors")
    total = float(torch.trace(_symmetric(K)))
    if total <= 0.0:
        raise ValueError(f"Gram has non-positive trace {total:.3e}")
    return float((lam.to(torch.float64) * energy).sum()) / total
