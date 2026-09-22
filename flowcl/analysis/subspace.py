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

GPM's incremental multi-task accumulation (the other half of §7.2) is deliberately not
here yet; it arrives with the ``gpm`` method (build order §10 step 8).
"""

from __future__ import annotations

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

    gram_check = vectors.T @ vectors
    error = float((gram_check - torch.eye(max_k, dtype=vectors.dtype)).abs().max())
    if error > _ORTHONORMALITY_ATOL:
        raise RuntimeError(
            f"{layer}: basis is not orthonormal (max |V^T V - I| = {error:.3e})"
        )

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


def save_bases(path: str | Path, bases: dict[str, SubspaceBasis], meta: dict) -> Path:
    """Persist one ``(run_id, task_idx)``'s bases, every layer at every threshold."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": BASES_FORMAT_VERSION,
            "meta": dict(meta),
            "layers": list(bases),
            "bases": {name: basis.to_payload() for name, basis in bases.items()},
        },
        path,
    )
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
