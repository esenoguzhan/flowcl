"""Flow-time (``s``) characterization: principal angles and the Gate 4 rules (§7.5, §9).

Gate 4 asks whether ``s``-conditioning is justified: reproducible variation in
``rho_l(s)``, ``c_l(s)`` and non-trivial principal angles across the four ``S_BINS``,
over >= 3 seeds. The rules are pre-registered in ``configs/analysis/flowtime.yaml``;
this module holds them as pure functions so every branch is unit-tested:

* :func:`rho_rule` — the replicate-mean occupancy range against ``max(floor, k * noise)``,
  noise being the largest within-bin replicate difference;
* :func:`angle_rule` — the mean overlap of the extreme bins over all four replicate
  pairs against the within-bin (sampling-noise) overlap minus a margin;
* :func:`c_rule` — the paired bin-3 minus bin-0 difference of per-batch ``c_l`` (common
  random numbers), a bootstrap CI that must exclude 0, required for both basis
  replicates with the same sign;
* :func:`reproducible` — a criterion passes on the *same* layers in every seed.

Failure does not establish that no layer contains flow-time-dependent geometry; it means
that broad, layerwise-reproducible evidence sufficient to justify a general ``s``-binned
method was not obtained.
"""

from __future__ import annotations

import itertools
import math

import torch

from flowcl.analysis.metrics import bootstrap_ci

# Float slack at inclusive boundaries (the inputs are float64 ratios).
_SLACK = 1e-12


def principal_cosines(Ma: torch.Tensor, Mb: torch.Tensor) -> torch.Tensor:
    """Cosines of the principal angles between ``span(Ma)`` and ``span(Mb)``, descending.

    Both must have orthonormal columns; there are ``min(k_a, k_b)`` of them.
    """
    if Ma.shape[0] != Mb.shape[0]:
        raise ValueError(f"bases live in different spaces: {tuple(Ma.shape)} vs {tuple(Mb.shape)}")
    if Ma.shape[1] == 0 or Mb.shape[1] == 0:
        raise ValueError("principal angles need non-empty bases")
    cos = torch.linalg.svdvals(Ma.to(torch.float64).T @ Mb.to(torch.float64))
    return cos.clamp(0.0, 1.0)


def overlap(Ma: torch.Tensor, Mb: torch.Tensor) -> float:
    """``||Ma^T Mb||_F^2 / min(k_a, k_b)``: the mean cos^2 of the principal angles.

    1 when the smaller subspace lies inside the larger; 0 when they are orthogonal.
    """
    cos = principal_cosines(Ma, Mb)
    return float((cos**2).sum()) / min(Ma.shape[1], Mb.shape[1])


def cosine_summary(cos: torch.Tensor) -> dict:
    """Quantiles of a principal-cosine spectrum (the full spectra are too large to store)."""
    q = torch.quantile(cos, torch.tensor([0.0, 0.1, 0.5], dtype=cos.dtype))
    return {"n": int(cos.numel()), "min": float(q[0]), "p10": float(q[1]),
            "median": float(q[2]), "mean_cos2": float((cos**2).mean())}


# ---- the per-layer rules ------------------------------------------------------------


def rho_rule(rho_a: list[float], rho_b: list[float], min_delta: float, noise_multiple: float) -> dict:
    """(i) ``rho_l(s)`` varies: the replicate-mean range beats ``max(min_delta, k * eta)``.

    ``rho_a[b]`` / ``rho_b[b]``: occupancy of bin ``b`` in replicates A and B.
    """
    if len(rho_a) != len(rho_b) or len(rho_a) < 2:
        raise ValueError("need matching per-bin occupancies for two replicates")
    mean = [(a + b) / 2 for a, b in zip(rho_a, rho_b)]
    delta = max(mean) - min(mean)
    eta = max(abs(a - b) for a, b in zip(rho_a, rho_b))
    threshold = max(min_delta, noise_multiple * eta)
    return {"rho_mean": mean, "delta": delta, "eta": eta, "threshold": threshold,
            "passed": delta >= threshold - _SLACK}


def angle_rule(across: list[float], within: list[float], margin: float) -> dict:
    """(ii) Non-trivial angles: mean cross-bin overlap <= mean within-bin overlap - margin.

    ``across``: overlaps of the extreme bins over all four replicate pairs;
    ``within``: the within-bin A/B overlaps of those two bins.
    """
    if len(across) != 4 or len(within) != 2:
        raise ValueError(f"expected 4 cross-replicate and 2 within-bin overlaps, got "
                         f"{len(across)} and {len(within)}")
    ov_across = sum(across) / 4
    ov_within = sum(within) / 2
    return {"ov_across": ov_across, "ov_within": ov_within, "margin": margin,
            "passed": ov_across <= ov_within - margin + _SLACK}


def paired_difference(c_low: list[float], c_high: list[float], n_bootstrap: int,
                      confidence: float, seed: int) -> dict:
    """Mean and bootstrap CI of the per-batch paired difference ``c_high - c_low``."""
    if len(c_low) != len(c_high) or not c_low:
        raise ValueError("paired per-batch values must be non-empty and of equal length")
    diffs = [h - l for l, h in zip(c_low, c_high)]
    est = bootstrap_ci(diffs, n_bootstrap=n_bootstrap, confidence=confidence, seed=seed)
    return {"delta": float(est.value), "low": float(est.low), "high": float(est.high),
            "n_batches": len(diffs)}


def c_rule(per_replicate: dict[str, tuple[list[float], list[float]]], min_delta: float,
           n_bootstrap: int, confidence: float, seed: int) -> dict:
    """(iii) ``c_l(s)`` varies: for every basis replicate, the paired difference
    ``c(bin_high) - c(bin_low)`` has ``|delta| >= min_delta`` and a CI excluding 0 — with
    the same sign across replicates.

    ``per_replicate``: replicate -> (per-batch c in the low bin, per-batch c in the high
    bin), both measured on identical batches, noise and base uniforms.
    """
    rows, signs = {}, set()
    for rep, (c_low, c_high) in per_replicate.items():
        d = paired_difference(c_low, c_high, n_bootstrap, confidence, seed)
        excludes_zero = d["low"] > 0.0 or d["high"] < 0.0
        d["passed"] = abs(d["delta"]) >= min_delta - _SLACK and excludes_zero
        rows[rep] = d
        signs.add(math.copysign(1.0, d["delta"]))
    passed = bool(rows) and all(r["passed"] for r in rows.values()) and len(signs) == 1
    return {"replicates": rows, "same_sign": len(signs) == 1, "passed": passed}


# ---- reproducibility across seeds --------------------------------------------------------


def reproducible(pass_sets: dict[str, set[str]], layers: list[str], min_fraction: float) -> dict:
    """A criterion passes when >= ``min_fraction`` of the *same* layers pass in every seed.

    Also reports each seed's own pass fraction and the pairwise Jaccard overlaps.
    """
    if not pass_sets:
        raise ValueError("no seeds")
    universe = set(layers)
    stray = {s: sorted(p - universe) for s, p in pass_sets.items() if p - universe}
    if stray:
        raise ValueError(f"passing layers outside the evaluated set: {stray}")
    both = set.intersection(*(set(p) for p in pass_sets.values()))
    jaccard = {}
    for a, b in itertools.combinations(sorted(pass_sets), 2):
        union = pass_sets[a] | pass_sets[b]
        jaccard[f"{a}|{b}"] = (len(pass_sets[a] & pass_sets[b]) / len(union)) if union else None
    fraction = len(both) / len(layers)
    return {
        "per_seed_fraction": {s: len(p) / len(layers) for s, p in pass_sets.items()},
        "intersection": sorted(both),
        "fraction": fraction,
        "jaccard": jaccard,
        "min_fraction": min_fraction,
        "passed": fraction >= min_fraction - _SLACK,
    }


# ---- negative control ------------------------------------------------------------------


def gram_difference(n1: int, K1: torch.Tensor, n2: int, K2: torch.Tensor) -> float:
    """``||K1/N1 - K2/N2||_F / max(||K1/N1||_F, tiny)`` for two captures of one layer."""
    A = K1.to(torch.float64) / n1
    B = K2.to(torch.float64) / n2
    return float(torch.linalg.matrix_norm(A - B)) / max(float(torch.linalg.matrix_norm(A)), 1e-300)


def assert_s_independent(name: str, reference: tuple[int, torch.Tensor],
                         other: tuple[int, torch.Tensor], label: str, rel_tol: float) -> float:
    """Raise unless an s-independent layer's sufficient statistics match the reference.

    Its activation inputs do not depend on ``s`` or the action noise, so a different bin
    or replicate must see the same token count and the same normalized Gram. Returns the
    relative Gram difference.
    """
    (n_ref, K_ref), (n, K) = reference, other
    if n != n_ref:
        raise RuntimeError(f"negative control failed: {name} ({label}) has N={n}, reference "
                           f"N={n_ref}; token selection or data order is not shared")
    diff = gram_difference(n_ref, K_ref, n, K)
    if diff > rel_tol:
        raise RuntimeError(f"negative control failed: {name} ({label}) normalized Gram "
                           f"differs from the reference by {diff:.3e} > {rel_tol:.1e}; the "
                           "instrument lets s or the noise leak into s-independent inputs")
    return diff
