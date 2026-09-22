"""§7.3 gradient decomposition and c_l on synthetic cases with known answers."""

from __future__ import annotations

import math

import pytest
import torch

from flowcl.analysis.interference import (
    assert_orthonormal,
    decompose,
    energy_weighted_ratio,
    interference_ratio,
    mean_ratio,
    prefix_interference,
    projected_energies,
)


def orthonormal(d: int, k: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
    return q[:, :k]


def split_basis(d: int = 8, k: int = 3, seed: int = 0):
    """``(M, M_perp)``: an orthonormal basis and an orthonormal complement direction set."""
    q = orthonormal(d, d, seed)
    return q[:, :k], q[:, k:]


def test_gradient_in_span_gives_one_and_orthogonal_gives_zero():
    M, perp = split_basis()
    u = torch.randn(5, dtype=torch.float64)
    assert interference_ratio(torch.outer(u, M[:, 0]), M) == pytest.approx(1.0)
    assert interference_ratio(torch.outer(u, perp[:, 0]), M) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("theta", [0.1, 0.7, 1.2])
def test_mixed_direction_gives_cos_theta(theta):
    M, perp = split_basis()
    v = math.cos(theta) * M[:, 1] + math.sin(theta) * perp[:, 2]
    G = torch.outer(torch.randn(4, dtype=torch.float64), v)
    assert interference_ratio(G, M) == pytest.approx(math.cos(theta))


def test_projection_acts_on_the_input_dimension_of_a_square_weight():
    """§7.3 orientation test. For a square layer a shape check cannot catch a transpose.

    G = u v^T with u in span(M) (the *output* side) and v orthogonal to M (the input
    side). Projecting on the input dimension gives c = 0; a transposed implementation
    (M M^T G) would give c = 1.
    """
    M, perp = split_basis(d=8, k=3)
    G = torch.outer(M[:, 0], perp[:, 0])  # (8, 8): d_out == d_in
    assert interference_ratio(G, M) == pytest.approx(0.0, abs=1e-12)
    assert float(torch.linalg.norm(M @ M.T @ G) / torch.linalg.norm(G)) == pytest.approx(1.0)


def test_decomposition_is_orthogonal_and_complete():
    M, _ = split_basis(d=10, k=4)
    G = torch.randn(6, 10, dtype=torch.float64)
    par, perp = decompose(G, M)
    torch.testing.assert_close(par + perp, G)
    assert float((par * perp).sum()) == pytest.approx(0.0, abs=1e-10)
    assert float((par**2).sum() + (perp**2).sum()) == pytest.approx(float((G**2).sum()))


def test_prefix_interference_matches_direct_computation_at_every_rank():
    V = orthonormal(12, 9, seed=3)
    G = torch.randn(5, 12, dtype=torch.float64)
    ranks = {k: k for k in range(1, 10)}
    prefix = prefix_interference(G, V, ranks)
    for k in ranks:
        assert prefix[k] == pytest.approx(interference_ratio(G, V[:, :k]))


def test_full_rank_basis_gives_exactly_one():
    V = orthonormal(7, 7)
    G = torch.randn(512, 7)
    assert prefix_interference(G, V, {"all": 7})["all"] == pytest.approx(1.0, abs=1e-12)


def test_aggregates_weigh_batches_as_documented():
    """Batch 1: norm 1, fully parallel (c=1). Batch 2: norm 100, orthogonal (c=0)."""
    parallel, total = [1.0, 0.0], [1.0, 1e4]
    assert mean_ratio(parallel, total) == pytest.approx(0.5)
    assert energy_weighted_ratio(parallel, total) == pytest.approx(math.sqrt(1 / 10001))


@pytest.mark.parametrize(
    "G, match",
    [
        (torch.zeros(3, 8, dtype=torch.float64), "zero gradient"),
        (torch.full((3, 8), float("nan"), dtype=torch.float64), "non-finite"),
        (torch.randn(3, 7, dtype=torch.float64), "d_in=8"),
        (torch.randn(8, dtype=torch.float64), "Biases are excluded"),
    ],
)
def test_bad_gradients_raise(G, match):
    M, _ = split_basis(d=8, k=3)
    with pytest.raises(ValueError, match=match):
        interference_ratio(G, M)


def test_non_orthonormal_basis_is_caught():
    with pytest.raises(ValueError, match="orthonormal"):
        assert_orthonormal(2.0 * orthonormal(6, 2))
    # A parallel energy larger than the total means the basis was not orthonormal.
    G = torch.randn(3, 6, dtype=torch.float64)
    with pytest.raises(ValueError, match="exceeds"):
        prefix_interference(G, 2.0 * orthonormal(6, 6), {"all": 6})


def test_projected_energies_rejects_ranks_beyond_the_basis():
    V = orthonormal(6, 3)
    with pytest.raises(ValueError, match="outside"):
        projected_energies(torch.randn(2, 6, dtype=torch.float64), V, [4])
