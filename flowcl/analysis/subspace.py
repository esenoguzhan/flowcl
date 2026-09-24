"""Input-subspace bases and occupancy ``rho_l`` (spec §7.2).

Spec §7.2::

    R_l = U Σ V^T; keep k_l = smallest k with Σ_{i<=k} σ_i² >= ε · Σ_i σ_i²,
    default ε = 0.95 (sweep ε in {0.90, 0.95, 0.99}).
    M_l = U[:, :k_l],  rho_l = k_l / d_l.
    Persist bases to disk per (run_id, task_idx, layer) with the energy threshold, k_l,
    d_l, N, and singular-value spectrum.

**Gram route.** ``R_l`` is never materialised. :mod:`flowcl.analysis.hooks` streams the
uncentered Gram ``K_l = R_l R_l^T = Σ_n x_n x_n^T`` in float64, and :func:`gram_eigh`
eigendecomposes it. Because ``K_l = U Σ² U^T``, its eigenvectors are ``R_l``'s left
singular vectors and its eigenvalues are ``σ²`` — the same ``M_l`` and the same energy
criterion as the SVD, without holding sixteen 2048 × 20k matrices in memory. A test pins
the equivalence against ``torch.linalg.svd``.

**Uncentered** on purpose: a linear layer's weight gradient is ``δ x^T``, which lives in
the span of the *raw* inputs. Centering would drop the mean direction, which is exactly
the direction every update touches. This is GPM's convention (Saha et al., ICLR 2021).

**Numerical behaviour** (a Gram matrix is PSD in exact arithmetic, not in floating point):

1. non-finite entries raise;
2. ``K`` is symmetrised and eigendecomposed in float64;
3. an eigenvalue below ``-neg_tol · λ_max`` raises — that is not rounding, it is an
   accumulation bug; smaller negatives are clamped to zero;
4. eigenpairs are sorted descending;
5. numerical rank ``r = #{λ_i > rank_tol · λ_max}``. ``rank_tol`` defaults to
   ``d · eps(float64)``. This is a relative tolerance **on Gram eigenvalues**; it is *not*
   the square of ``matrix_rank``'s singular-value criterion (``σ_i > τ σ_max`` would be
   ``λ_i > τ² λ_max``);
6. ``energy_rank`` is capped at ``r``, and ``ε = 1`` returns ``r`` explicitly so
   cumulative rounding cannot report fewer directions than the matrix has.

GPM's incremental multi-task accumulation (the other half of §7.2, paper Eq. 8-9) is
:func:`extend_basis`, also on the Gram route. Two quantities are kept apart throughout:
**capacity occupancy** ``rho_l = k_l / d_l`` (dimensions protected) and
**``proj_energy_fraction``** ``= tr(M^T K M) / tr K`` (share of a new task's input energy
already inside the memory).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import torch

# Format version for persisted bases. Bump when the payload's meaning changes.
BASES_FORMAT_VERSION = 1

# Sanity bound on ``V^T V - I`` for eigenvectors from float64 ``eigh``. Actual error
# is ~1e-13; this only has to catch a transposed or truncated basis.
_ORTHONORMALITY_ATOL = 1e-6


@dataclass
class SubspaceBasis:
    """One layer's input-subspace basis at every swept energy threshold.

    Attributes:
        layer: Registry name (§4.5).
        d_in: Input width — the space the basis lives in.
        n_samples: Number of activation vectors (columns of ``R_l``) behind the Gram.
        numerical_rank: ``r`` as defined in the module docstring.
        singular_values: ``σ``, descending, length ``d_in`` (clamped to ``>= 0``).
        thresholds: Swept energy thresholds ``ε``, ascending.
        ranks: ``ε -> k_l``.
        rhos: ``ε -> k_l / d_l``.
        vectors: ``(d_in, max_k)`` orthonormal columns. The basis at a smaller ``ε`` is a
            prefix of this one, so one tensor serves every threshold.
    """

    layer: str
    d_in: int
    n_samples: int
    numerical_rank: int
    singular_values: torch.Tensor
    thresholds: tuple[float, ...]
    ranks: dict[float, int]
    rhos: dict[float, float]
    vectors: torch.Tensor
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.vectors.ndim != 2 or self.vectors.shape[0] != self.d_in:
            raise ValueError(
                f"{self.layer}: basis vectors must be (d_in={self.d_in}, k), got "
                f"{tuple(self.vectors.shape)}. A (k, d_in) basis would project onto "
                "the output dimension, which §7.3 warns silently breaks the method."
            )
        max_k = max(self.ranks.values())
        if self.vectors.shape[1] != max_k:
            raise ValueError(
                f"{self.layer}: stored {self.vectors.shape[1]} vectors but the "
                f"largest rank is {max_k}"
            )
        if set(self.ranks) != set(self.thresholds) or set(self.rhos) != set(
            self.thresholds
        ):
            raise ValueError(
                f"{self.layer}: ranks/rhos keys {sorted(self.ranks)} / "
                f"{sorted(self.rhos)} do not match thresholds {self.thresholds}"
            )

    def basis(self, eps: float) -> torch.Tensor:
        """``M_l`` at energy threshold ``eps``: ``(d_in, k_l)``."""
        if eps not in self.ranks:
            raise KeyError(
                f"{self.layer}: eps={eps} was not swept; available {self.thresholds}"
            )
        return self.vectors[:, : self.ranks[eps]]

    @property
    def samples_per_dim(self) -> float:
        return self.n_samples / self.d_in

    def summary(self) -> dict:
        """JSON-friendly numbers, without the tensors."""
        return {
            "layer": self.layer,
            "d_in": self.d_in,
            "n_samples": self.n_samples,
            "samples_per_dim": self.samples_per_dim,
            "numerical_rank": self.numerical_rank,
            "ranks": {str(k): v for k, v in self.ranks.items()},
            "rhos": {str(k): v for k, v in self.rhos.items()},
        }

    def to_payload(self) -> dict:
        return {
            "layer": self.layer,
            "d_in": self.d_in,
            "n_samples": self.n_samples,
            "numerical_rank": self.numerical_rank,
            "singular_values": self.singular_values,
            "thresholds": list(self.thresholds),
            "ranks": dict(self.ranks),
            "rhos": dict(self.rhos),
            "vectors": self.vectors,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "SubspaceBasis":
        return cls(
            layer=payload["layer"],
            d_in=int(payload["d_in"]),
            n_samples=int(payload["n_samples"]),
            numerical_rank=int(payload["numerical_rank"]),
            singular_values=payload["singular_values"],
            thresholds=tuple(float(t) for t in payload["thresholds"]),
            ranks={float(k): int(v) for k, v in payload["ranks"].items()},
            rhos={float(k): float(v) for k, v in payload["rhos"].items()},
            vectors=payload["vectors"],
            meta=dict(payload.get("meta", {})),
        )


def validate_thresholds(thresholds) -> tuple[float, ...]:
    """Energy thresholds must be distinct values in ``(0, 1]``; returned ascending."""
    values = tuple(sorted(float(t) for t in thresholds))
    if not values:
        raise ValueError("at least one energy threshold is required")
    if len(set(values)) != len(values):
        raise ValueError(f"duplicate energy thresholds: {thresholds}")
    bad = [t for t in values if not 0.0 < t <= 1.0]
    if bad:
        raise ValueError(f"energy thresholds must lie in (0, 1], got {bad}")
    return values


def gram_eigh(
    gram: torch.Tensor,
    layer: str,
    neg_tol: float,
    rank_tol: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Eigendecompose an uncentered Gram matrix with the checks in the module docstring.

    Args:
        gram: ``(d, d)`` ``Σ x x^T``.
        layer: For error messages.
        neg_tol: Relative tolerance for negative eigenvalues (see module docstring).
        rank_tol: Relative Gram-eigenvalue tolerance for numerical rank; ``None``
            means ``d · eps(float64)``.

    Returns:
        ``(eigenvalues, eigenvectors, numerical_rank)``: eigenvalues ``(d,)``
        descending and clamped to ``>= 0``, eigenvectors ``(d, d)`` with matching
        column order, both float64.
    """
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError(f"{layer}: Gram must be square, got {tuple(gram.shape)}")
    d = gram.shape[0]

    gram = gram.to(dtype=torch.float64)
    nonfinite = int((~torch.isfinite(gram)).sum())
    if nonfinite:
        raise ValueError(
            f"{layer}: Gram matrix has {nonfinite} non-finite entries (of {d * d}); "
            "an activation contained NaN/inf"
        )

    gram = 0.5 * (gram + gram.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    # eigh is ascending; §7.2's k_l needs the leading directions first.
    eigenvalues = eigenvalues.flip(0)
    eigenvectors = eigenvectors.flip(1)

    lam_max = float(eigenvalues[0])
    if lam_max <= 0.0:
        raise ValueError(
            f"{layer}: Gram matrix has no positive eigenvalue (λ_max = {lam_max:.3e}); "
            "the layer's inputs were all zero, so it has no subspace to measure"
        )
    lam_min = float(eigenvalues[-1])
    if lam_min < -neg_tol * lam_max:
        raise ValueError(
            f"{layer}: Gram matrix is not PSD: λ_min = {lam_min:.3e} < "
            f"-{neg_tol:.1e} · λ_max ({lam_max:.3e}). This is beyond rounding error "
            "and indicates an accumulation bug."
        )
    eigenvalues = eigenvalues.clamp(min=0.0)

    tol = rank_tol if rank_tol is not None else d * torch.finfo(torch.float64).eps
    numerical_rank = int((eigenvalues > tol * lam_max).sum())
    return eigenvalues, eigenvectors, numerical_rank


def energy_rank(eigenvalues: torch.Tensor, eps: float, numerical_rank: int) -> int:
    """Smallest ``k`` with ``Σ_{i<=k} λ_i >= eps · Σ_i λ_i``, capped at the numerical rank.

    Args:
        eigenvalues: ``σ²``, descending, non-negative.
        eps: Energy threshold in ``(0, 1]``.
        numerical_rank: From :func:`gram_eigh`.
    """
    if not 0.0 < eps <= 1.0:
        raise ValueError(f"eps must lie in (0, 1], got {eps}")
    if numerical_rank < 1:
        raise ValueError(f"numerical_rank must be >= 1, got {numerical_rank}")
    if eps == 1.0:
        # Explicit: cumulative rounding could otherwise stop short of r.
        return numerical_rank

    cumulative = torch.cumsum(eigenvalues.to(torch.float64), dim=0)
    target = eps * cumulative[-1]
    k = int(torch.searchsorted(cumulative, target.unsqueeze(0), side="left")) + 1
    return min(k, numerical_rank)


def basis_from_gram(
    gram: torch.Tensor,
    layer: str,
    n_samples: int,
    thresholds,
    neg_tol: float,
    rank_tol: float | None = None,
    meta: dict | None = None,
) -> SubspaceBasis:
    """Build a :class:`SubspaceBasis` from an accumulated Gram matrix."""
    thresholds = validate_thresholds(thresholds)
    if n_samples < 1:
        raise ValueError(f"{layer}: n_samples must be >= 1, got {n_samples}")

    eigenvalues, eigenvectors, numerical_rank = gram_eigh(
        gram, layer, neg_tol=neg_tol, rank_tol=rank_tol
    )
    d_in = gram.shape[0]
    ranks = {eps: energy_rank(eigenvalues, eps, numerical_rank) for eps in thresholds}
    max_k = max(ranks.values())
    vectors = eigenvectors[:, :max_k].contiguous()

    assert_orthonormal_columns(vectors, layer)

    return SubspaceBasis(
        layer=layer,
        d_in=d_in,
        n_samples=int(n_samples),
        numerical_rank=numerical_rank,
        singular_values=eigenvalues.sqrt(),
        thresholds=thresholds,
        ranks=ranks,
        rhos={eps: k / d_in for eps, k in ranks.items()},
        vectors=vectors,
        meta=dict(meta or {}),
    )


def assert_orthonormal_columns(vectors: torch.Tensor, layer: str) -> None:
    """Raise unless ``vectors^T vectors = I`` (an empty basis is trivially orthonormal)."""
    k = vectors.shape[1]
    if k == 0:
        return
    gram_check = vectors.T.to(torch.float64) @ vectors.to(torch.float64)
    error = float((gram_check - torch.eye(k, dtype=torch.float64)).abs().max())
    if error > _ORTHONORMALITY_ATOL:
        raise RuntimeError(
            f"{layer}: basis is not orthonormal (max |V^T V - I| = {error:.3e})"
        )


# ---- GPM incremental memory (paper Eq. 8-9, spec §7.2) ------------------------

# The captured-energy post-condition may undershoot eps by at most this much. It absorbs
# only float64 rounding; numerical-rank truncation drops eigenvalues <= rank_tol*λmax,
# whose total energy is orders of magnitude smaller.
ENERGY_TOL = 1e-6


def captured_energy_fraction(M: torch.Tensor, K: torch.Tensor) -> float:
    """``tr(M^T K M) / tr K`` — the share of a Gram's energy inside ``span(M)``.

    Clamped to ``[0, 1]``: in exact arithmetic it lies there for orthonormal ``M`` and PSD
    ``K``; the clamp only removes rounding outside that range.
    """
    total = float(torch.trace(K))
    if total <= 0.0:
        raise ValueError(f"Gram has non-positive trace {total:.3e}")
    if M.shape[1] == 0:
        return 0.0
    inside = float(torch.trace(M.T @ K @ M))
    return min(max(inside, 0.0), total) / total


def check_captured_energy(
    M: torch.Tensor, K: torch.Tensor, eps: float, layer: str, tol: float = ENERGY_TOL
) -> float:
    """The unconditional post-condition of :func:`extend_basis`: captured >= eps - tol."""
    fraction = captured_energy_fraction(M, K)
    if fraction < eps - tol:
        raise RuntimeError(
            f"{layer}: memory captures {fraction:.9f} of the task's input energy, below "
            f"eps - tol = {eps - tol:.9f} (k = {M.shape[1]} of d = {M.shape[0]}). A full-rank "
            "memory captures ~100%, so this is a numerical failure or an over-aggressive "
            "rank tolerance, not capacity exhaustion."
        )
    return fraction


# Below this much energy outside memory (as a share of tr K), a task has no new energy: the
# adaptive target would otherwise ask for 90% of rounding noise.
NEW_ENERGY_ATOL = 1e-12


def adaptive_target(proj: float, eps: float, new_energy_fraction: float | None) -> float:
    """The captured-energy target: ``eps``, or ``min(1, max(eps, proj + f (1 - proj)))``.

    Returns ``eps`` itself (not a recomputed equal float) whenever it is the larger term, so
    the default and the adaptive variant take bit-identical paths on a first task.
    """
    if new_energy_fraction is None:
        return eps
    adaptive = proj + new_energy_fraction * (1.0 - proj)
    return min(1.0, adaptive) if adaptive > eps else eps


def extend_basis(
    M: torch.Tensor | None,
    gram: torch.Tensor,
    eps: float,
    layer: str,
    neg_tol: float,
    rank_tol: float | None = None,
    energy_tol: float = ENERGY_TOL,
    new_energy_fraction: float | None = None,
) -> tuple[torch.Tensor, dict]:
    """GPM's incremental memory update on a Gram matrix (Saha et al. 2021, Eq. 8-9).

    With ``R`` the new task's activations and ``K = R R^T``::

        ||R||^2      = tr K
        ||R_proj||^2 = tr(M^T K M)                      (energy already in memory)
        R_hat R_hat^T = (I - M M^T) K (I - M M^T)        (Eq. 8, residual)
        add the smallest k with tr(M^T K M) + sum_{i<=k} λ_hat_i >= eps tr K   (Eq. 9)

    ``M = None`` or empty reduces to Eq. 5 (a fresh basis). The new directions are
    re-orthogonalised against ``M`` (projection, then QR), ``[M, U]`` is asserted
    orthonormal, and the captured energy ``tr(M_new^T K M_new)/tr K >= eps - energy_tol`` is
    asserted *unconditionally*. Capacity exhaustion (``k_after == d``) is reported
    separately in ``info``; a full memory captures ~100% and cannot violate the check.

    ``new_energy_fraction = f`` (adaptive variant) raises the target to
    ``min(1, max(eps, proj + f (1 - proj)))``: at least ``f`` of the task's energy *not
    already in memory* is protected, never less than ``eps`` of the total. At ``proj = 0``
    (a first task) the target is exactly ``eps`` and the path is identical to ``None``.
    Because ``proj + f (1 - proj) > proj`` whenever ``proj < 1``, every layer with any new
    energy left is extended.

    Returns:
        ``(M_new, info)`` with ``M_new`` float64 ``(d, k_after)``.
    """
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError(f"{layer}: Gram must be square, got {tuple(gram.shape)}")
    if not 0.0 < eps <= 1.0:
        raise ValueError(f"eps must lie in (0, 1], got {eps}")
    if new_energy_fraction is not None and not 0.0 < new_energy_fraction < 1.0:
        raise ValueError(f"new_energy_fraction must lie in (0, 1), got {new_energy_fraction}")
    d = gram.shape[0]
    K = gram.to(torch.float64)
    nonfinite = int((~torch.isfinite(K)).sum())
    if nonfinite:
        raise ValueError(f"{layer}: Gram has {nonfinite} non-finite entries")
    K = 0.5 * (K + K.T)

    M = torch.zeros(d, 0, dtype=torch.float64) if M is None else M.to(torch.float64)
    if M.shape[0] != d:
        raise ValueError(f"{layer}: memory is ({M.shape[0]}, k) but the Gram is {d}x{d}")
    assert_orthonormal_columns(M, layer)
    k_before = M.shape[1]

    total = float(torch.trace(K))
    if total <= 0.0:
        raise ValueError(
            f"{layer}: Gram has non-positive trace {total:.3e}; the task produced no "
            "input energy at this layer"
        )
    proj_fraction = captured_energy_fraction(M, K)
    target_fraction = adaptive_target(proj_fraction, eps, new_energy_fraction)

    residual_spectrum = torch.zeros(0, dtype=torch.float64)
    no_new_energy = new_energy_fraction is not None and 1.0 - proj_fraction <= NEW_ENERGY_ATOL
    if proj_fraction >= target_fraction or k_before == d or no_new_energy:
        added = torch.zeros(d, 0, dtype=torch.float64)
    else:
        P = torch.eye(d, dtype=torch.float64) - M @ M.T
        K_res = P @ K @ P
        eigenvalues, eigenvectors, rank = gram_eigh(K_res, layer, neg_tol=neg_tol, rank_tol=rank_tol)
        residual_spectrum = eigenvalues
        cap = min(rank, d - k_before)
        cumulative = proj_fraction * total + torch.cumsum(eigenvalues, dim=0)
        target = torch.tensor([target_fraction * total], dtype=torch.float64)
        k = int(torch.searchsorted(cumulative, target, side="left")) + 1
        k = max(1, min(k, cap))
        U = eigenvectors[:, :k]
        U = U - M @ (M.T @ U)  # re-orthogonalise against the memory
        Q, Rq = torch.linalg.qr(U)
        if float(Rq.diagonal().abs().min()) < 1e-8:
            raise RuntimeError(
                f"{layer}: new directions are numerically dependent on the memory "
                f"(min |diag R| = {float(Rq.diagonal().abs().min()):.3e})"
            )
        added = Q

    M_new = torch.cat([M, added], dim=1)
    assert_orthonormal_columns(M_new, layer)
    captured = check_captured_energy(M_new, K, target_fraction, layer, tol=energy_tol)
    k_after = M_new.shape[1]
    return M_new, {
        "k_before": k_before,
        "k_added": k_after - k_before,
        "k_after": k_after,
        "d_in": d,
        "rho_after": k_after / d,
        "proj_energy_fraction": proj_fraction,
        "target_fraction": target_fraction,
        "captured_energy_fraction": captured,
        "capacity_exhausted": k_after == d,
        "residual_spectrum": residual_spectrum,
    }


def save_bases(path: str | Path, bases: dict[str, SubspaceBasis], meta: dict) -> Path:
    """Persist one ``(run_id, task_idx)``'s bases, every layer at every threshold.

    Written atomically (temporary file, then ``os.replace``), so an interrupted write
    never leaves a partial file that a checkpoint could point to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(
        {
            "version": BASES_FORMAT_VERSION,
            "meta": dict(meta),
            "layers": list(bases),
            "bases": {name: basis.to_payload() for name, basis in bases.items()},
        },
        tmp,
    )
    os.replace(tmp, path)
    return path


def load_bases(path: str | Path) -> tuple[dict[str, SubspaceBasis], dict]:
    """Inverse of :func:`save_bases`; layer order is preserved."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No bases file at {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    version = payload.get("version")
    if version != BASES_FORMAT_VERSION:
        raise ValueError(
            f"{path}: bases format version {version} != {BASES_FORMAT_VERSION}; "
            "refusing to reinterpret it"
        )
    bases = {
        name: SubspaceBasis.from_payload(payload["bases"][name])
        for name in payload["layers"]
    }
    return bases, payload["meta"]
