"""Gradient Projection Memory with AdamW-safe update projection (§6 item 5, §7.2-§7.4).

Source: Saha, Garg & Roy, "Gradient Projection Memory for Continual Learning", ICLR 2021
(arXiv:2103.09762). For a fully connected layer the new task's gradient is projected onto
the orthogonal complement of the stored input basis (their Eq. 6)::

    ∇W L ← ∇W L − (∇W L) M M^T

and after each task the memory ``M`` is extended with the task's significant input
directions that it does not already contain (their Eq. 8-9,
:func:`flowcl.analysis.subspace.extend_basis`).

**Deviation, and the reported name.** The paper trains with plain SGD, where projecting
the gradient *is* projecting the update. This codebase trains with AdamW (the Gate 1
recipe). Adam's per-coordinate scaling does not preserve subspaces and AdamW's decoupled
weight decay shrinks weights in every direction, so a projected gradient does not give an
orthogonal *applied* update (Gate 3 §6.2; measured at 7-19% of the step's norm in the
pilot). This method therefore projects twice:

1. ``modify_gradients`` — ``G ← G P`` with ``P = I − M M^T``, so Adam's moments only see the
   free-subspace gradient;
2. ``after_step`` — the realised displacement ``ΔW = W − W_ref`` is replaced by ``ΔW P``, so
   the weights never move in protected directions whatever Adam or weight decay did.

Because that is stricter than canonical GPM, the method reports itself as
``gpm_projected_adam`` (``projection: gradient_and_update``), never as plain "GPM". A
gradient-only ablation would be ``gpm_grad_only``; it is not implemented yet.

Details inferred rather than taken from the paper (§11):

* the memory is built from §7.1/§7.2 Gram captures (primary, gradient-reachable view) at
  one fixed ``eps`` for every layer and every task (the paper tunes and anneals ``eps``);
* Task 1 trains unconstrained with every parameter trainable (paper Alg. 1); from Task 2
  on, projection acts on :meth:`~flowcl.models.policy.FlowPolicy.projectable_parameters`
  only and *every* other parameter is frozen (:func:`freeze_to_allowlist`): §7.4's list
  plus the encoder projections, context queries and positional embedding §7.4 does not
  name;
* the memory is updated after *every* task, including the last, so capacity is reported
  at the end of the sequence.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import torch

from flowcl.analysis.subspace import (
    SubspaceBasis,
    extend_basis,
    load_bases,
    save_bases,
)
from flowcl.methods.base import BaseMethod, register_method
from flowcl.utils.run import atomic_write_text, file_sha256

PROJECTION_NAMES = {"gradient_and_update": "gpm_projected_adam"}
MEMORY_KIND = "accumulated_memory"


# ---- §7.4 allowlist -----------------------------------------------------------


def allowlist(policy) -> tuple[str, ...]:
    """Parameters allowed to train in projection experiments: registry weights only."""
    return tuple(policy.projectable_parameters())


def freeze_to_allowlist(policy) -> dict:
    """Freeze every parameter outside :func:`allowlist`; return what was done.

    Must run before the optimiser is built, so frozen tensors never enter AdamW (and so
    never receive weight decay).
    """
    allowed = set(allowlist(policy))
    params = dict(policy.named_parameters())
    missing = sorted(allowed - set(params))
    if missing:
        raise RuntimeError(f"allowlisted parameters do not exist: {missing}")
    newly_frozen = []
    for name, param in params.items():
        if name in allowed:
            param.requires_grad_(True)
        elif param.requires_grad:
            param.requires_grad_(False)
            newly_frozen.append(name)
    assert_allowlist(policy)
    return {
        "trainable_tensors": len(allowed),
        "trainable_params": sum(params[n].numel() for n in allowed),
        "newly_frozen": newly_frozen,
        "newly_frozen_params": sum(params[n].numel() for n in newly_frozen),
    }


def assert_allowlist(policy) -> None:
    """Raise unless the trainable set is exactly :func:`allowlist` (§7.4)."""
    allowed = set(allowlist(policy))
    trainable = {n for n, p in policy.named_parameters() if p.requires_grad}
    extra, absent = sorted(trainable - allowed), sorted(allowed - trainable)
    if extra or absent:
        raise RuntimeError(
            "§7.4 projection protocol violated.\n"
            f"  trainable but not allowlisted (forgetting would leak through these): {extra}\n"
            f"  allowlisted but frozen: {absent}"
        )


# ---- runtime projection state -------------------------------------------------


class _ProjectedLayer:
    """Per-layer runtime state: projector, basis, reference weights, tolerances."""

    def __init__(self, name: str, weight: torch.nn.Parameter, M: torch.Tensor, weight_atol: float):
        d_in = weight.shape[1]
        if M.shape[0] != d_in:
            raise ValueError(f"{name}: memory d_in {M.shape[0]} != weight d_in {d_in}")
        k = M.shape[1]
        device = weight.device
        self.name = name
        self.weight = weight
        self.k = k
        self.full_rank = k == d_in
        if self.full_rank:
            # Exactly zero, not a numerically built near-zero matrix: nothing may move.
            self.P = torch.zeros(d_in, d_in, dtype=torch.float32, device=device)
        else:
            M64 = M.to(torch.float64)
            eye = torch.eye(d_in, dtype=torch.float64)
            self.P = (eye - M64 @ M64.T).to(device=device, dtype=torch.float32)
        self.M = M.to(device=device, dtype=torch.float32)
        self.W_ref = weight.detach().clone()
        self.atol = weight_atol * float(self.W_ref.norm())
        self.max_residual = 0.0
        self.max_residual_ratio = 0.0  # ||D M|| / (atol + rtol ||D||); must stay <= 1
        self.displacement = torch.zeros_like(self.W_ref)  # cumulative, for reporting


@register_method
class GPM(BaseMethod):
    """Hard projection against an accumulated input-subspace memory; see the module docstring."""

    name = "gpm"

    def __init__(
        self,
        eps: float = 0.95,
        projection: str = "gradient_and_update",
        residual_rtol: float = 1e-3,
        residual_weight_atol: float = 1e-6,
        log_interval: int = 100,
        update_memory: bool = False,
        capture_config: str = "subspace",
        new_energy_fraction: float | None = None,
    ) -> None:
        """``capture_config``: a ``configs/analysis/<name>.yaml`` name, or a YAML path.

        ``new_energy_fraction = f`` selects the adaptive memory target
        ``max(eps, p + f (1 - p))`` (:func:`flowcl.analysis.subspace.adaptive_target`), and
        a distinct display name (``..._ne90`` for ``f = 0.9``). ``None`` is plain eps.
        """
        super().__init__()
        if projection not in PROJECTION_NAMES:
            raise ValueError(
                f"projection {projection!r} not implemented; available "
                f"{sorted(PROJECTION_NAMES)}"
            )
        if not 0.0 < eps <= 1.0:
            raise ValueError(f"eps must lie in (0, 1], got {eps}")
        if log_interval < 1:
            raise ValueError(f"log_interval must be >= 1, got {log_interval}")
        if new_energy_fraction is not None and not 0.0 < new_energy_fraction < 1.0:
            raise ValueError(
                f"new_energy_fraction must lie in (0, 1), got {new_energy_fraction}"
            )
        self.new_energy_fraction = (
            None if new_energy_fraction is None else float(new_energy_fraction)
        )
        self.eps = float(eps)
        self.projection = projection
        self.residual_rtol = float(residual_rtol)
        self.residual_weight_atol = float(residual_weight_atol)
        self.log_interval = int(log_interval)
        self.update_memory = bool(update_memory)
        self.capture_config = capture_config

        # Memory: layer -> (d_in, k) float64 on CPU, registry order.
        self._memory: dict[str, torch.Tensor] = {}
        self._memory_samples: dict[str, int] = {}
        self._memory_spectra: dict[str, torch.Tensor] = {}
        self.memory_history: dict[int, dict[str, dict]] = {}
        self.memory_extended: dict[int, bool] = {}

        self._layers: list[_ProjectedLayer] = []
        self._active = False
        self._task: int | None = None
        # Per task: gradient_c / update_c (step -> layer -> c), residuals, displacement.
        self.task_logs: dict[int, dict] = {}

    # ---- identity and config ---------------------------------------------------

    @property
    def display_name(self) -> str:
        name = PROJECTION_NAMES[self.projection]
        if self.new_energy_fraction is not None:
            name += f"_ne{round(100 * self.new_energy_fraction)}"
        return name

    def config(self) -> dict:
        return {
            "eps": self.eps,
            "projection": self.projection,
            "residual_rtol": self.residual_rtol,
            "residual_weight_atol": self.residual_weight_atol,
            "log_interval": self.log_interval,
            "update_memory": self.update_memory,
            "capture_config": self.capture_config,
            "new_energy_fraction": self.new_energy_fraction,
        }

    # ---- memory ----------------------------------------------------------------

    def set_memory(self, bases: dict[str, SubspaceBasis]) -> None:
        """Install a fixed memory (e.g. the Task-1 bases Gate 2 persisted; the pilot)."""
        lacking = sorted(n for n, b in bases.items() if self.eps not in b.ranks)
        if lacking:
            raise ValueError(f"bases lack eps={self.eps} for {lacking}")
        self._memory = {n: b.basis(self.eps).to(torch.float64).cpu() for n, b in bases.items()}
        self._memory_samples = {n: b.n_samples for n, b in bases.items()}

    def memory_rho(self) -> dict[str, float]:
        """Capacity occupancy ``rho_l = k_l / d_l`` of the current memory."""
        return {n: M.shape[1] / M.shape[0] for n, M in self._memory.items()}

    def _update_memory(self, policy, task_idx: int, context) -> None:
        from flowcl.experiments.gate2 import capture_task_grams, load_subspace_config
        from flowcl.utils.libero_paths import repo_root
        from flowcl.utils.seeding import derive_seed

        ns = context.seed_namespace_run_id
        if ns is None:
            raise RuntimeError(
                "GPM memory update needs context.seed_namespace_run_id to seed the capture; "
                "run through run_continual, or set update_memory=false"
            )
        if context.dataset is None or context.task_key is None:
            raise RuntimeError("GPM memory update needs a single-task dataset and task key")
        config_path = Path(self.capture_config)
        if config_path.suffix != ".yaml":
            config_path = repo_root() / "configs" / "analysis" / f"{self.capture_config}.yaml"
        cfg = load_subspace_config(config_path)
        capture = capture_task_grams(
            policy,
            context.dataset,
            cfg,
            context.device,
            probe_seed=derive_seed(ns, f"gpm_memory_probe::{context.task_key}", task_idx),
            capture_seed=derive_seed(ns, f"gpm_memory::{context.task_key}", task_idx),
        )
        history = {}
        for entry in policy.projectable_layers():
            name = entry.name
            acc = capture.accumulators[name]
            view = capture.primary_view(name)
            M_new, info = extend_basis(
                self._memory.get(name),
                acc.gram[view],
                self.eps,
                name,
                neg_tol=cfg.neg_tol,
                rank_tol=cfg.rank_tol,
                new_energy_fraction=self.new_energy_fraction,
            )
            self._memory[name] = M_new
            self._memory_samples[name] = self._memory_samples.get(name, 0) + acc.n[view]
            self._memory_spectra[name] = info.pop("residual_spectrum")
            history[name] = {**info, "n_samples": acc.n[view], "view": view}
            acc.gram.clear()
        self.memory_history[task_idx] = history
        self.memory_extended[task_idx] = True

    def restore_memory(
        self,
        path: str | Path,
        sha256: str,
        *,
        method_run_id: str | None = None,
        task_idx: int | None = None,
    ) -> dict:
        """Reload a memory artifact after verifying its SHA-256 and identity.

        What a resumed run would call with the ``method_artifacts`` entry of the stage
        checkpoint it resumes from. Raises on any mismatch rather than projecting against
        a memory that is not the one the checkpoint was trained with.
        """
        path = Path(path)
        actual = file_sha256(path)
        if actual != sha256:
            raise ValueError(f"{path}: SHA-256 {actual} does not match the checkpoint's {sha256}")
        bases, meta = load_bases(path)
        problems = []
        if meta.get("kind") != MEMORY_KIND:
            problems.append(f"kind {meta.get('kind')!r}")
        if meta.get("eps") != self.eps:
            problems.append(f"eps {meta.get('eps')} != {self.eps}")
        stored_f = meta.get("config", {}).get("new_energy_fraction")
        if stored_f != self.new_energy_fraction:
            problems.append(f"new_energy_fraction {stored_f} != {self.new_energy_fraction}")
        if method_run_id is not None and meta.get("method_run_id") != method_run_id:
            problems.append(f"method_run_id {meta.get('method_run_id')!r} != {method_run_id!r}")
        if task_idx is not None and meta.get("task_idx") != task_idx:
            problems.append(f"task_idx {meta.get('task_idx')} != {task_idx}")
        if problems:
            raise ValueError(f"{path}: memory does not match: {problems}")
        self._memory = {n: b.vectors.to(torch.float64).cpu() for n, b in bases.items()}
        self._memory_samples = {n: b.n_samples for n, b in bases.items()}
        self.memory_history = {int(k): v for k, v in meta["memory_history"].items()}
        return meta

    # ---- §6 hooks --------------------------------------------------------------

    def on_task_start(self, policy, task_idx, *, context) -> None:
        self._task = task_idx
        self.task_logs[task_idx] = {"gradient_c": {}, "update_c": {}}
        if not self._memory:
            if task_idx == 0:
                self._active = False  # Task 1 trains unconstrained (paper Alg. 1)
                return
            raise RuntimeError(
                f"GPM at task {task_idx} has no memory; call set_memory() or train task 0 first"
            )
        freeze_to_allowlist(policy)
        registry = policy.projectable_layers()
        names = [e.name for e in registry]
        if sorted(self._memory) != sorted(names):
            raise ValueError(
                "memory layers do not match the registry: missing "
                f"{sorted(set(names) - set(self._memory))}, extra "
                f"{sorted(set(self._memory) - set(names))}"
            )
        self._layers = [
            _ProjectedLayer(e.name, e.module.weight, self._memory[e.name], self.residual_weight_atol)
            for e in registry
        ]
        self._active = True

    def modify_gradients(self, policy, batch_meta) -> None:
        if not self._active:
            return
        step = batch_meta["step"]
        record = step % self.log_interval == 0
        logged = {}
        with torch.no_grad():
            for layer in self._layers:
                G = layer.weight.grad
                if G is None:
                    raise RuntimeError(f"{layer.name}: no gradient at step {step}")
                if record:
                    logged[layer.name] = _ratio(G, layer.M)
                G.copy_(G @ layer.P)
        if record:
            self.task_logs[self._task]["gradient_c"][step] = logged

    def after_step(self, policy, step_meta) -> None:
        if not self._active:
            return
        step = step_meta["step"]
        record = step % self.log_interval == 0
        logged = {}
        residuals, norms, finite = [], [], []
        with torch.no_grad():
            for layer in self._layers:
                W = layer.weight
                delta = W - layer.W_ref
                if record:
                    logged[layer.name] = _ratio(delta, layer.M)
                W.copy_(torch.addmm(layer.W_ref, delta, layer.P))
                applied = W - layer.W_ref
                residuals.append((applied @ layer.M).norm())
                norms.append(applied.norm())
                finite.append(torch.isfinite(W).all())
                layer.displacement.add_(applied)
                layer.W_ref.copy_(W)
            # One device->host transfer per step instead of three per layer.
            residuals = torch.stack(residuals).cpu().tolist()
            norms = torch.stack(norms).cpu().tolist()
            finite = torch.stack(finite).cpu().tolist()
        for layer, residual, norm, ok in zip(self._layers, residuals, norms, finite):
            if not ok:
                raise RuntimeError(f"{layer.name}: non-finite weights at step {step}")
            bound = layer.atol + self.residual_rtol * norm
            if residual > bound:
                raise RuntimeError(
                    f"{layer.name}: applied update not orthogonal to the memory at step "
                    f"{step}: ||D M|| = {residual:.3e} > {bound:.3e} (atol {layer.atol:.3e} "
                    f"+ rtol {self.residual_rtol} * ||D|| {norm:.3e})"
                )
            layer.max_residual = max(layer.max_residual, residual)
            if bound > 0:
                layer.max_residual_ratio = max(layer.max_residual_ratio, residual / bound)
        if record:
            self.task_logs[self._task]["update_c"][step] = logged

    def on_task_end(self, policy, task_idx, *, context) -> None:
        logs = self.task_logs.setdefault(task_idx, {"gradient_c": {}, "update_c": {}})
        logs["projected"] = self._active
        logs["residuals"] = {
            layer.name: {
                "max_residual": layer.max_residual,
                "max_residual_over_bound": layer.max_residual_ratio,
                "atol": layer.atol,
                "k": layer.k,
                "full_rank": layer.full_rank,
            }
            for layer in self._layers
        }
        logs["cumulative_displacement_norm"] = {
            layer.name: float(layer.displacement.norm()) for layer in self._layers
        }
        # Release ~0.5 GB of projectors and reference weights; the logs stay.
        self._layers = []
        self._active = False
        if self.update_memory:
            self._update_memory(policy, task_idx, context)
        else:
            self.memory_extended[task_idx] = False

    def save_artifacts(self, directory, task_idx, *, context) -> list[Path]:
        """``memory_task{τ}.pt`` (if any memory) and ``gpm_logs_task{τ}.json``, atomically."""
        directory = Path(directory)
        paths: list[Path] = []
        if self._memory:
            eps = self.eps
            bases = {}
            for name, M in self._memory.items():
                d_in, k = M.shape
                spectrum = self._memory_spectra.get(name, torch.zeros(0, dtype=torch.float64))
                sigma = torch.zeros(d_in, dtype=torch.float64)
                sigma[: spectrum.numel()] = spectrum.clamp(min=0).sqrt()
                bases[name] = SubspaceBasis(
                    layer=name,
                    d_in=d_in,
                    n_samples=max(int(self._memory_samples.get(name, 0)), 1),
                    numerical_rank=k,
                    singular_values=sigma,
                    thresholds=(eps,),
                    ranks={eps: k},
                    rhos={eps: k / d_in},
                    vectors=M,
                    meta={
                        "kind": MEMORY_KIND,
                        "history": {
                            str(t): h[name] for t, h in self.memory_history.items() if name in h
                        },
                    },
                )
            meta = {
                "kind": MEMORY_KIND,
                "method": self.display_name,
                "eps": eps,
                "method_run_id": context.method_run_id,
                "seed_namespace_run_id": context.seed_namespace_run_id,
                "task_idx": task_idx,
                "task_key": context.task_key,
                "tasks_in_memory": sorted(self.memory_history),
                "memory_history": {str(t): h for t, h in self.memory_history.items()},
                "config": self.config(),
                "note": "singular_values hold sqrt of the latest residual spectrum (Eq. 8)",
            }
            paths.append(save_bases(directory / f"memory_task{task_idx}.pt", bases, meta))
        logs = {
            "method": self.display_name,
            "method_run_id": context.method_run_id,
            "seed_namespace_run_id": context.seed_namespace_run_id,
            "task_idx": task_idx,
            "task_key": context.task_key,
            "memory_extended": self.memory_extended.get(task_idx),
            "memory": self.memory_history.get(task_idx),
            **_json_logs(self.task_logs.get(task_idx, {})),
        }
        paths.append(
            atomic_write_text(
                directory / f"gpm_logs_task{task_idx}.json", json.dumps(logs, indent=2) + "\n"
            )
        )
        return paths

    # ---- reporting -------------------------------------------------------------

    def _latest_logs(self) -> dict:
        return self.task_logs[max(self.task_logs)] if self.task_logs else {}

    @property
    def gradient_c(self) -> dict:
        return self._latest_logs().get("gradient_c", {})

    @property
    def update_c(self) -> dict:
        return self._latest_logs().get("update_c", {})

    @property
    def residuals(self) -> dict:
        return self._latest_logs().get("residuals", {})

    @property
    def cumulative_displacement_norm(self) -> dict:
        return self._latest_logs().get("cumulative_displacement_norm", {})

    def stored_bytes(self) -> int:
        """§8.2: the memory is the bases, k x d_in floats per layer (float32 in use)."""
        return sum(M.numel() * 4 for M in self._memory.values())

    def state_dict(self) -> dict:
        """Summaries plus the latest task's logs (flat, pilot-compatible); no tensors.

        The memory itself is an artifact (``save_artifacts``) that the stage checkpoint
        references by path and SHA-256, so checkpoints do not grow by the basis size.
        """
        latest = self._latest_logs()
        return {
            "name": self.name,
            "display_name": self.display_name,
            "config": self.config(),
            "memory_layers": len(self._memory),
            "stored_mb": self.stored_bytes() / 1e6,
            "memory_extended": dict(self.memory_extended),
            "memory_summary": {
                str(t): _memory_summary(h) for t, h in self.memory_history.items()
            },
            "gradient_c": {str(k): v for k, v in latest.get("gradient_c", {}).items()},
            "update_c": {str(k): v for k, v in latest.get("update_c", {}).items()},
            "residuals": latest.get("residuals", {}),
            "cumulative_displacement_norm": latest.get("cumulative_displacement_norm", {}),
        }

    def describe(self) -> str:
        return f"{self.display_name} (eps {self.eps}, stores {self.stored_bytes() / 1e6:.2f} MB)"


def _json_logs(logs: dict) -> dict:
    return {
        "projected": logs.get("projected"),
        "gradient_c": {str(k): v for k, v in logs.get("gradient_c", {}).items()},
        "update_c": {str(k): v for k, v in logs.get("update_c", {}).items()},
        "residuals": logs.get("residuals", {}),
        "cumulative_displacement_norm": logs.get("cumulative_displacement_norm", {}),
    }


def _memory_summary(history: dict[str, dict]) -> dict:
    """Per-half medians of occupancy and energy fractions, plus exhausted-layer counts."""
    out = {}
    for half, prefix in (("trunk", "trunk."), ("decoder", "flow_head.")):
        rows = [h for n, h in history.items() if n.startswith(prefix)]
        if not rows:
            continue
        out[half] = {
            "median_rho_after": statistics.median(r["rho_after"] for r in rows),
            "median_proj_energy_fraction": statistics.median(r["proj_energy_fraction"] for r in rows),
            "k_added_total": sum(r["k_added"] for r in rows),
            "capacity_exhausted": sum(1 for r in rows if r["capacity_exhausted"]),
            "n_layers": len(rows),
        }
    return out


def _ratio(A: torch.Tensor, M: torch.Tensor) -> float:
    """``||A M|| / ||A||`` (NaN for a zero ``A``, e.g. a skipped AMP step)."""
    total = float(A.norm())
    if total == 0.0:
        return math.nan
    return float((A @ M).norm()) / total
