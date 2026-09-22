"""§7.2 bases and rho_l on synthetic matrices with known answers."""

from __future__ import annotations

import pytest
import torch

from flowcl.analysis.subspace import (
    basis_from_gram,
    energy_rank,
    gram_eigh,
    load_bases,
    save_bases,
)

NEG_TOL = 1e-8


def synthetic_activations(sigmas, d: int, n: int, seed: int = 0) -> torch.Tensor:
    """``R = U diag(σ) V^T`` with random orthonormal ``U`` (d×r) and ``V`` (n×r)."""
    g = torch.Generator().manual_seed(seed)
    r = len(sigmas)
    u, _ = torch.linalg.qr(torch.randn(d, r, generator=g, dtype=torch.float64))
    v, _ = torch.linalg.qr(torch.randn(n, r, generator=g, dtype=torch.float64))
    return u @ torch.diag(torch.tensor(sigmas, dtype=torch.float64)) @ v.T


def test_ranks_and_rhos_match_the_energy_criterion():
    # σ² = 50, 30, 14, 4.5, 1.5 -> total 100; cumulative 50, 80, 94, 98.5, 100.
    # Thresholds sit strictly between cumulative values so rounding cannot flip them.
    sigmas = [50**0.5, 30**0.5, 14**0.5, 4.5**0.5, 1.5**0.5]
    d = 16
    r = synthetic_activations(sigmas, d=d, n=200)
    basis = basis_from_gram(
        r @ r.T,
        "toy",
        n_samples=200,
        thresholds=[0.4, 0.9, 0.95, 0.99, 1.0],
        neg_tol=NEG_TOL,
    )
    assert basis.ranks == {0.4: 1, 0.9: 3, 0.95: 4, 0.99: 5, 1.0: 5}
    assert basis.rhos[0.95] == pytest.approx(4 / d)
    assert basis.numerical_rank == 5


def test_eps_one_returns_numerical_rank_not_d():
    """Rank-r data in d > r dimensions: eps=1 must give k = r."""
    r = synthetic_activations([3.0, 2.0, 1.0], d=32, n=100)
    basis = basis_from_gram(r @ r.T, "toy", 100, [1.0], neg_tol=NEG_TOL)
    assert basis.ranks[1.0] == 3
    assert basis.rhos[1.0] == pytest.approx(3 / 32)


def test_energy_rank_is_capped_at_numerical_rank():
    eigenvalues = torch.tensor([4.0, 1.0, 1e-30, 0.0], dtype=torch.float64)
    assert energy_rank(eigenvalues, 1.0, numerical_rank=2) == 2
    assert energy_rank(eigenvalues, 0.999999999, numerical_rank=2) == 2
    with pytest.raises(ValueError):
        energy_rank(eigenvalues, 0.0, numerical_rank=2)


def test_gram_route_matches_svd():
    g = torch.Generator().manual_seed(1)
    r = torch.randn(12, 300, generator=g, dtype=torch.float64) * torch.linspace(
        3, 0.1, 12, dtype=torch.float64
    ).unsqueeze(1)
    eigenvalues, eigenvectors, rank = gram_eigh(r @ r.T, "toy", neg_tol=NEG_TOL)
    u, s, _ = torch.linalg.svd(r, full_matrices=False)

    assert rank == 12
    torch.testing.assert_close(eigenvalues, s**2)
    # Same spans (columns agree up to sign).
    for k in (1, 4, 12):
        p_gram = eigenvectors[:, :k] @ eigenvectors[:, :k].T
        p_svd = u[:, :k] @ u[:, :k].T
        torch.testing.assert_close(p_gram, p_svd, atol=1e-8, rtol=0)


def test_basis_is_orthonormal_descending_and_input_oriented():
    r = synthetic_activations([5.0, 4.0, 3.0, 2.0], d=10, n=80)
    basis = basis_from_gram(r @ r.T, "toy", 80, [0.9, 0.99], neg_tol=NEG_TOL)
    assert basis.vectors.shape == (10, basis.ranks[0.99])
    torch.testing.assert_close(
        basis.vectors.T @ basis.vectors,
        torch.eye(basis.vectors.shape[1], dtype=torch.float64),
    )
    assert torch.all(basis.singular_values[:-1] >= basis.singular_values[1:])
    assert basis.singular_values.shape == (10,)


def test_smaller_eps_basis_is_a_prefix():
    r = synthetic_activations([5.0, 4.0, 3.0, 2.0, 1.0], d=10, n=80)
    basis = basis_from_gram(r @ r.T, "toy", 80, [0.5, 0.9, 0.99], neg_tol=NEG_TOL)
    small = basis.basis(0.5)
    assert small.shape[1] == basis.ranks[0.5]
    torch.testing.assert_close(small, basis.vectors[:, : basis.ranks[0.5]])
    with pytest.raises(KeyError):
        basis.basis(0.95)


def test_tiny_negative_eigenvalues_are_clamped():
    gram = torch.diag(torch.tensor([2.0, 1.0, -1e-14], dtype=torch.float64))
    eigenvalues, _, rank = gram_eigh(gram, "toy", neg_tol=NEG_TOL)
    assert float(eigenvalues.min()) == 0.0
    assert rank == 2


def test_meaningfully_negative_eigenvalue_raises():
    gram = torch.diag(torch.tensor([2.0, 1.0, -0.1], dtype=torch.float64))
    with pytest.raises(ValueError, match="not PSD"):
        gram_eigh(gram, "toy", neg_tol=NEG_TOL)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_gram_raises(bad):
    gram = torch.eye(3, dtype=torch.float64)
    gram[0, 1] = bad
    with pytest.raises(ValueError, match="non-finite"):
        gram_eigh(gram, "toy", neg_tol=NEG_TOL)


def test_zero_gram_raises():
    with pytest.raises(ValueError, match="no positive eigenvalue"):
        gram_eigh(torch.zeros(4, 4, dtype=torch.float64), "toy", neg_tol=NEG_TOL)


def test_save_load_round_trip(tmp_path):
    r = synthetic_activations([3.0, 1.0], d=6, n=40)
    basis = basis_from_gram(r @ r.T, "layer.a", 40, [0.9, 0.99], neg_tol=NEG_TOL)
    path = save_bases(tmp_path / "task0.pt", {"layer.a": basis}, {"run_id": "x"})
    loaded, meta = load_bases(path)
    assert meta == {"run_id": "x"}
    assert list(loaded) == ["layer.a"]
    torch.testing.assert_close(loaded["layer.a"].vectors, basis.vectors)
    assert loaded["layer.a"].ranks == basis.ranks
    assert loaded["layer.a"].rhos == basis.rhos
