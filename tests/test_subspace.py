"""§7.2 bases and rho_l on synthetic matrices with known answers."""

from __future__ import annotations

import pytest
import torch

from flowcl.analysis.subspace import (
    assert_orthonormal_columns,
    basis_from_gram,
    captured_energy_fraction,
    check_captured_energy,
    energy_rank,
    extend_basis,
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


# ---- GPM incremental memory (extend_basis) --------------------------------------

def activations_in(indices, d, n, seed, scales=None):
    """R (d x n) whose columns lie in span(e_i for i in indices)."""
    g = torch.Generator().manual_seed(seed)
    R = torch.zeros(d, n, dtype=torch.float64)
    for j, i in enumerate(indices):
        s = 1.0 if scales is None else scales[j]
        R[i] = s * torch.randn(n, generator=g, dtype=torch.float64)
    return R


def projector(M):
    return M @ M.T


def test_nested_subspaces_add_only_the_new_directions():
    d = 10
    RA = activations_in([0, 1, 2], d, 400, 0)
    M1, info1 = extend_basis(None, RA @ RA.T, 0.999, "toy", neg_tol=NEG_TOL)
    assert info1["k_after"] == 3
    RB = activations_in([1, 2, 3, 4], d, 400, 1)
    M2, info2 = extend_basis(M1, RB @ RB.T, 0.999, "toy", neg_tol=NEG_TOL)
    assert info2["k_added"] == 2
    added = M2[:, 3:]
    expected = torch.zeros(d, d, dtype=torch.float64)
    expected[3, 3] = expected[4, 4] = 1.0
    torch.testing.assert_close(projector(added), expected, atol=1e-8, rtol=0)
    assert 0.3 < info2["proj_energy_fraction"] < 0.7  # e1, e2 were already protected
    assert info2["rho_after"] == 5 / d


def test_data_already_in_memory_adds_nothing():
    d = 8
    R = activations_in([0, 1, 2], d, 200, 0)
    M, _ = extend_basis(None, R @ R.T, 0.95, "toy", neg_tol=NEG_TOL)
    R2 = activations_in([0, 1, 2], d, 200, 5)
    M2, info = extend_basis(M, R2 @ R2.T, 0.95, "toy", neg_tol=NEG_TOL)
    assert info["k_added"] == 0 and M2.shape == M.shape
    assert info["proj_energy_fraction"] == pytest.approx(1.0)


def test_empty_memory_matches_basis_from_gram():
    r = synthetic_activations([50**0.5, 30**0.5, 14**0.5, 4.5**0.5, 1.5**0.5], d=16, n=200)
    K = r @ r.T
    M, info = extend_basis(None, K, 0.95, "toy", neg_tol=NEG_TOL)
    reference = basis_from_gram(K, "toy", 200, [0.95], neg_tol=NEG_TOL)
    assert info["k_after"] == reference.ranks[0.95] == 4
    torch.testing.assert_close(projector(M), projector(reference.basis(0.95)), atol=1e-10, rtol=0)


def paper_reference(M, R, eps):
    """Eq. 8-9 on explicit R via SVD, as in the reference implementation."""
    total = float((R**2).sum())
    R_proj = M @ (M.T @ R) if M.shape[1] else torch.zeros_like(R)
    R_hat = R - R_proj
    U, S, _ = torch.linalg.svd(R_hat, full_matrices=False)
    acc = float((R_proj**2).sum())
    if acc >= eps * total:
        return M
    k = 0
    for s in S:
        k += 1
        acc += float(s**2)
        if acc >= eps * total:
            break
    return torch.cat([M, U[:, :k]], dim=1)


def test_gram_route_matches_the_paper_on_explicit_activations():
    g = torch.Generator().manual_seed(3)
    d = 12
    mix = torch.randn(d, d, generator=g, dtype=torch.float64)
    RA = mix @ torch.diag(torch.linspace(3, 0.05, d, dtype=torch.float64)) @ torch.randn(d, 300, generator=g, dtype=torch.float64)
    RB = torch.randn(d, 300, generator=g, dtype=torch.float64) * torch.linspace(0.1, 2, d, dtype=torch.float64).unsqueeze(1)
    M = None
    ref = torch.zeros(d, 0, dtype=torch.float64)
    for R in (RA, RB):
        M, _ = extend_basis(M, R @ R.T, 0.9, "toy", neg_tol=NEG_TOL)
        ref = paper_reference(ref, R, 0.9)
        assert M.shape == ref.shape
        torch.testing.assert_close(projector(M), projector(ref), atol=1e-8, rtol=0)


def test_memory_is_capped_at_full_rank_and_reports_exhaustion():
    d = 6
    M = None
    for seed in range(4):
        R = torch.randn(d, 100, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
        M, info = extend_basis(M, R @ R.T, 0.99, "toy", neg_tol=NEG_TOL)
    assert M.shape[1] == d and info["capacity_exhausted"]
    assert info["captured_energy_fraction"] == pytest.approx(1.0)
    R = torch.randn(d, 100, generator=torch.Generator().manual_seed(9), dtype=torch.float64)
    M2, info = extend_basis(M, R @ R.T, 0.99, "toy", neg_tol=NEG_TOL)
    assert info["k_added"] == 0 and M2.shape[1] == d


def test_orthonormal_after_many_extensions():
    d = 32
    M = None
    for seed in range(5):
        g = torch.Generator().manual_seed(seed)
        R = torch.randn(d, 4, generator=g, dtype=torch.float64) @ torch.randn(4, 200, generator=g, dtype=torch.float64)
        M, info = extend_basis(M, R @ R.T, 0.95, "toy", neg_tol=NEG_TOL)
        assert info["captured_energy_fraction"] >= 0.95 - 1e-6
    torch.testing.assert_close(M.T @ M, torch.eye(M.shape[1], dtype=torch.float64), atol=1e-10, rtol=0)


def test_post_condition_and_orthonormality_catch_broken_extensions():
    d = 8
    R = activations_in([0, 1, 2, 3], d, 200, 0)
    K = R @ R.T
    M, _ = extend_basis(None, K, 0.95, "toy", neg_tol=NEG_TOL)
    with pytest.raises(RuntimeError, match="numerical failure"):
        check_captured_energy(M[:, :1], K, 0.95, "toy")  # truncated below the criterion
    broken = torch.cat([M, M[:, :1] + 1e-3], dim=1)  # not re-orthogonalised
    with pytest.raises(RuntimeError, match="not orthonormal"):
        assert_orthonormal_columns(broken, "toy")


def test_projected_energy_is_clamped_and_tiny_negatives_do_not_raise():
    d = 5
    M = torch.eye(d, dtype=torch.float64)[:, :2]
    K = torch.diag(torch.tensor([1.0, 1.0, 1.0, 1.0, -1e-14], dtype=torch.float64))
    assert 0.0 <= captured_energy_fraction(M, K) <= 1.0
    M2, info = extend_basis(M, K, 0.95, "toy", neg_tol=NEG_TOL)
    assert info["captured_energy_fraction"] >= 0.95


@pytest.mark.parametrize("bad", [float("nan"), 0.0])
def test_non_finite_or_zero_grams_raise(bad):
    K = torch.full((4, 4), bad, dtype=torch.float64)
    with pytest.raises(ValueError):
        extend_basis(None, K, 0.95, "toy", neg_tol=NEG_TOL)
