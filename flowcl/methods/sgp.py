"""Scaled Gradient Projection, adapted to AdamW (§6 item 7; ``sgp_projected_adam``).

Source: Saha & Roy, "Continual Learning with Scaled Gradient Projection", AAAI 2023
(arXiv:2302.01386). SGP keeps GPM's memory ``M`` and gives each basis vector an
importance ``λ_i ∈ (0, 1]`` (their Eq. 2, 6-10; :mod:`flowcl.analysis.importance`); the
update is projected with ``I − M Λ Mᵀ`` instead of ``I − M Mᵀ``, so the most important
directions are blocked as in GPM and the rest are only scaled down by ``1 − λ``.

**Paper-faithful:** the memory rule (Eq. 8-9, :func:`~flowcl.analysis.subspace.extend_basis`,
shared with GPM), the importance ``λ = (α+1)σ / (ασ + max σ)`` (Eq. 2), the old bases'
surrogate ``σ'_i² = m_iᵀ K m_i`` (Eq. 6-8), the accumulation ``min(1, λ + λ_τ)`` (Eq. 10) and
the scaled projection. ``α`` is fixed a priori (25, the paper's Adam/RL setting).

**Our adaptation — never reported as "paper SGP".** Following :mod:`flowcl.methods.gpm`,
the method projects around AdamW twice:

1. ``modify_gradients`` removes only the **hard** bases (``λ_i == 1``) from the gradient, so
   Adam's moments never see fully protected directions;
2. ``after_step`` replaces AdamW's realised step ``ΔW`` (weight decay included) with
   ``ΔW (I − M Λ Mᵀ)`` and asserts, every step and layer, that each basis component kept
   exactly ``1 − λ_i`` of the raw step's.

At ``Λ = I`` this is bitwise ``gpm_projected_adam`` (tested). The update step uses the
paper's post-optimizer projection form (Adam-GP, their App. E), but because fully protected
components are removed *before* AdamW, which can change Adam's coordinate-wise moments and
so its output in soft directions, the complete optimizer coupling is a hybrid adaptation,
not exact Adam-GP. Without hard directions it reduces to post-optimizer scaled projection,
apart from AdamW's decoupled weight decay being projected with the step.

Other deviations and inferred details (§11):

* one fixed ``eps`` for every task and layer (the paper anneals it, e.g. 0.97 + 0.003 t), so
  the memory stays paired with plain GPM;
* when a task adds no basis, Eq. 2's max runs over the old bases' ``σ'`` only;
* basis-wise ``λ``, not README §6 item 7's per-layer ``α_l`` (a different method, not run);
* as GPM: Task 1 trains unconstrained, the §7.4 allowlist is frozen from Task 2, and the
  memory is also updated after the last task.

Artifacts keep GPM's names (``memory_task{τ}.pt``, ``gpm_logs_task{τ}.json``), so the
sequence report and the forgetting diagnostics read them unchanged; each layer's basis meta
adds ``importance`` (λ) and ``importance_update`` (``σ'``, ``σ̂``, ``λ_τ``, ``λ_prev``), from
which Eq. 2 and Eq. 10 can be recomputed independently.
"""

from __future__ import annotations

import math
import statistics
from pathlib import Path

import torch

from flowcl.analysis.importance import (
    importance_update,
    protected_energy_fraction,
    sgp_importance,
)
from flowcl.analysis.subspace import SubspaceBasis, load_bases
from flowcl.methods.base import register_method
from flowcl.methods.gpm import GPM

SGP_PROJECTION_NAMES = {"hard_gradient_scaled_update": "sgp_projected_adam"}
# The registered alpha (configs/method/sgp.yaml). Other values are named ``..._a<alpha>``, so the
# registered runs keep their ids.
DEFAULT_ALPHA = 25.0
IMPORTANCE_KEYS = ("sigma_old", "sigma_new", "lambda_task", "lambda_prev", "lambda")


@register_method
class SGP(GPM):
    """GPM memory plus basis-wise importance scaling; see the module docstring."""

    name = "sgp"
    projection_names = SGP_PROJECTION_NAMES

    def __init__(
        self,
        alpha: float = 25.0,
        eps: float = 0.95,
        projection: str = "hard_gradient_scaled_update",
        **kwargs,
    ) -> None:
        if not (isinstance(alpha, (int, float)) and math.isfinite(alpha) and alpha > 0):
            raise ValueError(f"alpha must be a positive finite number, got {alpha!r}")
        if kwargs.get("eps_later") is not None:
            # SGP is closed (docs/thesis_plan.md A7); its display name would not carry it.
            raise ValueError("eps_later is not supported for SGP")
        super().__init__(eps=eps, projection=projection, **kwargs)
        self.alpha = float(alpha)
        # Layer -> λ (float64, CPU), one per memory column, registry order.
        self._importance: dict[str, torch.Tensor] = {}
        # Layer -> the latest extension's record (IMPORTANCE_KEYS), and its task index.
        self._importance_updates: dict[str, dict[str, torch.Tensor]] = {}
        self._importance_task: int | None = None

    @property
    def display_name(self) -> str:
        name = self.projection_names[self.projection]
        if self.alpha != DEFAULT_ALPHA:
            name += f"_a{self.alpha:g}"
        if self.new_energy_fraction is not None:
            name += f"_ne{round(100 * self.new_energy_fraction)}"
        return name

    def config(self) -> dict:
        return {**super().config(), "alpha": self.alpha}

    # ---- memory and importance -------------------------------------------------

    def set_memory(self, bases: dict[str, SubspaceBasis], importance: dict | None = None) -> None:
        """A fixed memory; ``λ`` from each basis's own singular values (Eq. 2) unless given."""
        super().set_memory(bases)
        if importance is None:
            importance = {
                n: sgp_importance(b.singular_values[: b.ranks[self.eps]], self.alpha)
                for n, b in bases.items()
            }
        self._importance = {n: torch.as_tensor(v, dtype=torch.float64).cpu()
                            for n, v in importance.items()}
        self._importance_updates = {}
        self._importance_task = None

    def _update_memory(self, policy, task_idx: int, context) -> None:
        super()._update_memory(policy, task_idx, context)
        self._importance_task = task_idx

    def _after_extend(self, name, M_old, gram, M_new, info: dict, spectrum) -> None:
        prev = self._importance.get(name, torch.zeros(0, dtype=torch.float64))
        update = importance_update(M_old, gram, spectrum, info["k_added"], prev, self.alpha)
        lam = update["lambda"]
        if lam.numel() != M_new.shape[1]:
            raise RuntimeError(f"{name}: {lam.numel()} importances for {M_new.shape[1]} bases")
        self._importance[name] = lam
        self._importance_updates[name] = update
        info["importance"] = {
            "lambda_median": statistics.median(lam.tolist()),
            "lambda_min": float(lam.min()),
            "n_hard": int((lam == 1.0).sum()),
            "hard_fraction": float((lam == 1.0).sum()) / lam.numel(),
            "protected_energy_fraction": protected_energy_fraction(M_new, lam, gram),
        }

    def _layer_scales(self, name: str) -> torch.Tensor:
        lam = self._importance.get(name)
        if lam is None:
            raise RuntimeError(f"{name}: SGP memory has no importance")
        k = self._memory[name].shape[1]
        if lam.numel() != k:
            raise RuntimeError(f"{name}: {lam.numel()} importances for a {k}-column memory")
        return lam

    def _basis_meta(self, name: str) -> dict:
        meta = {"importance": self._importance[name].clone()}
        if name in self._importance_updates:
            meta["importance_update"] = {
                k: v.clone() for k, v in self._importance_updates[name].items()
            }
            meta["importance_update_task"] = self._importance_task
        return meta

    def restore_memory(self, path: str | Path, sha256: str, **kwargs) -> dict:
        """GPM's checks, plus ``alpha`` and one importance per stored basis vector."""
        meta = super().restore_memory(path, sha256, **kwargs)
        problems = []
        stored_alpha = meta.get("config", {}).get("alpha")
        if stored_alpha != self.alpha:
            problems.append(f"alpha {stored_alpha} != {self.alpha}")
        bases, _ = load_bases(path)
        importance = {}
        for name, basis in bases.items():
            lam = basis.meta.get("importance")
            if not isinstance(lam, torch.Tensor) or lam.numel() != basis.vectors.shape[1]:
                problems.append(f"{name}: importance missing or of the wrong length")
                continue
            importance[name] = lam.to(torch.float64).cpu()
        if problems:
            raise ValueError(f"{path}: SGP memory does not match: {problems}")
        self._importance = importance
        self._importance_updates = {}
        self._importance_task = None
        return meta

    # ---- reporting -------------------------------------------------------------

    def stored_bytes(self) -> int:
        """§8.2: the bases plus one importance per basis vector (float32 in use)."""
        return super().stored_bytes() + sum(lam.numel() * 4 for lam in self._importance.values())

    def describe(self) -> str:
        return (f"{self.display_name} (eps {self.eps}, alpha {self.alpha}, "
                f"stores {self.stored_bytes() / 1e6:.2f} MB)")
