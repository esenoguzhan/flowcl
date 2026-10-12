"""Diagnostics for projected low-rank updates (projected LoRA, the InfLoRA adaptation; docs/runs/…).

* :func:`complement_basis` - an explicit orthonormal basis C of the complement of span(M).
* :func:`inflora_down` - InfLoRA's fixed down-projection: the top-r eigenvectors of C^T K C (the new
  task's input Gram restricted to the complement), mapped back through C, with eigenvalues at or below a
  relative tolerance dropped (arbitrary vectors from a zero eigenspace are never used).
* :func:`output_change` - the layer-output change on fixed historical inputs,
  E||D x||^2 = tr(D K_old D^T), relative to tr(W_0 K_old W_0^T). This general form is valid for ANY
  displacement D (an unprojected update included); the P⊥-sandwiched form tr(D P⊥ K P⊥ D^T) equals it
  only when D = D P⊥ and is reported only as a check for projected updates. It measures one layer's
  output on fixed inputs, not the whole network's change (upstream activations also move).
* :func:`subspace_movement` - principal angles between an initial row space Q_0 and the update's row
  space, and f_outside = ||D (I - Q_0 Q_0^T)||_F^2 / ||D||_F^2 (None for D = 0).
"""

from __future__ import annotations

import math

import torch


def complement_basis(M: torch.Tensor | None, d_in: int) -> torch.Tensor:
    """(d_in, d_in - k) orthonormal columns spanning span(M)^⊥ (float64); all of R^d_in for M = None."""
    eye = torch.eye(d_in, dtype=torch.float64)
    if M is None or M.shape[1] == 0:
        return eye
    M = M.to(torch.float64)
    k = M.shape[1]
    if k >= d_in:
        return torch.zeros(d_in, 0, dtype=torch.float64)
    evals, evecs = torch.linalg.eigh(eye - M @ M.T)
    order = torch.argsort(evals, descending=True)
    C = evecs[:, order[: d_in - k]]
    if float((C.T @ M).abs().max()) > 1e-8:
        raise RuntimeError("complement basis is not orthogonal to the memory")
    return C


def inflora_down(K: torch.Tensor, M: torch.Tensor | None, rank: int, rel_tol: float = 1e-8) -> tuple[torch.Tensor, dict]:
    """InfLoRA's down-projection (r_eff x d_in, float64): rows C u_i for the top eigenpairs of C^T K C with
    eigenvalue > rel_tol * trace(C^T K C); r_eff <= rank (0: the layer cannot be adapted)."""
    K = K.to(torch.float64)
    d_in = K.shape[0]
    C = complement_basis(M, d_in)
    info = {"free_dims": int(C.shape[1]), "trace_K": float(torch.trace(K))}
    if C.shape[1] == 0:
        return torch.zeros(0, d_in, dtype=torch.float64), {**info, "effective_rank": 0, "kept_energy": 0.0,
                                                           "complement_energy": 0.0}
    Kc = C.T @ K @ C
    Kc = 0.5 * (Kc + Kc.T)
    evals, evecs = torch.linalg.eigh(Kc)
    order = torch.argsort(evals, descending=True)
    evals, evecs = evals[order], evecs[:, order]
    total = float(torch.trace(Kc))
    keep = [i for i in range(min(rank, evals.numel())) if total > 0 and float(evals[i]) > rel_tol * total]
    rows = (C @ evecs[:, keep]).T.contiguous()
    info.update({"effective_rank": len(keep),
                 "complement_energy": total / info["trace_K"] if info["trace_K"] > 0 else 0.0,
                 "kept_energy": float(evals[keep].sum()) / total if keep and total > 0 else 0.0})
    return rows, info


def output_change(D: torch.Tensor, W0: torch.Tensor, K: torch.Tensor, P: torch.Tensor | None = None) -> dict:
    """tr(D K D^T) relative to tr(W0 K W0^T); with ``P`` (= P⊥) also the sandwiched form, as a check."""
    D, W0, K = D.to(torch.float64), W0.to(torch.float64), K.to(torch.float64)
    num = float(((D @ K) * D).sum())
    den = float(((W0 @ K) * W0).sum())
    out = {"energy": num, "base_energy": den, "relative": num / den if den > 0 else math.inf}
    if P is not None:
        P = P.to(torch.float64)
        out["sandwiched"] = float(((D @ P @ K @ P) * D).sum())
    return out


def subspace_movement(D: torch.Tensor, Q0: torch.Tensor, rel_tol: float = 1e-6) -> dict:
    """Principal angles (degrees) between the initial row space (the columns of Q0, d_in x r0,
    orthonormal) and D's row space, D's numerical rank, and f_outside (None for D = 0)."""
    D = D.to(torch.float64)
    Q0 = Q0.to(torch.float64)
    total = float((D * D).sum())
    if total == 0.0:
        return {"f_outside": None, "rank": 0, "principal_angles_deg": []}
    outside = D - (D @ Q0) @ Q0.T
    _, s, vh = torch.linalg.svd(D, full_matrices=False)
    rank = int((s > rel_tol * s[0]).sum())
    V = vh[:rank].T                                        # d_in x rank, D's row space
    cos = torch.linalg.svdvals(Q0.T @ V).clamp(max=1.0) if Q0.shape[1] and rank else torch.zeros(0)
    return {"f_outside": float((outside * outside).sum()) / total, "rank": rank,
            "principal_angles_deg": [float(torch.rad2deg(torch.arccos(c))) for c in cos]}


def orthonormal_rows(A: torch.Tensor, rel_tol: float = 1e-6) -> torch.Tensor:
    """An orthonormal basis (d_in x r) of A's row space (numerical rank)."""
    A = A.to(torch.float64)
    if A.numel() == 0:
        return torch.zeros(A.shape[1], 0, dtype=torch.float64)
    _, s, vh = torch.linalg.svd(A, full_matrices=False)
    if s.numel() == 0 or float(s[0]) == 0.0:
        return torch.zeros(A.shape[1], 0, dtype=torch.float64)
    rank = int((s > rel_tol * s[0]).sum())
    return vh[:rank].T.contiguous()
