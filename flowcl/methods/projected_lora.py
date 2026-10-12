"""``projected_lora``: low-rank adapters confined to the complement of GPM's input memory.

Task 1 is full fine-tuning and builds GPM's memory exactly as ``gpm_projected_adam`` does (the same
capture, seeds and eps), so stage 0 is bitwise seq_ft's and the Task-1 memory bitwise
``gpm_projected_adam``'s. From Task 2 every base parameter is frozen and each registry layer gets a
projected adapter (:func:`flowcl.models.lora.attach_projected`): the effective down-projection
A_eff = A - (A M) M^T, so the update (scale) B A_eff satisfies Delta W M = 0 in exact arithmetic under
any optimiser. The update is merged into W at the task's end (fp32) and the memory is then extended
(GPM's rule). Three variants (the controlled comparison; no new projection mechanism is claimed):

* ``trained`` - A from the released LoRA initialisation, A and B trained, scale alpha / r;
* ``inflora`` - an InfLoRA adaptation (Liang & Li, CVPR 2024) under the shared protocol: A fixed to the
  top-r eigenvectors of Task t's input Gram restricted to the complement (captured at the task's start
  with the current weights), only B trained (from 0), scale 1. Our GPM memory rule (fixed eps) replaces
  DualGPM's update rule, a substantive protocol difference, disclosed;
* ``pca_trained`` - the matched ablation: A initialised to ``inflora``'s basis, A and B trained, scale 1.

A layer whose memory is full rank (or, for the PCA variants, whose complement holds no direction with
energy above tolerance) is not adapted; if no layer can be adapted the stage raises. Per task the
history records the cost (trainable parameters, optimiser-state estimate, stored subspace bytes), and after
the merge the constraint (intended and realized
||Delta W M|| against an absolute-plus-relative tolerance; a violation raises) and the subspace movement
(principal angles and f_outside against the initial A_eff row space). Wall-clock quantities (design
and training seconds, peak GPU memory) are kept apart in ``timings``, so the history is deterministic
(a resumed run's and the bracket path's must equal the runner's; ``timings`` is not compared).
"""

from __future__ import annotations

import math
import statistics
import time
from pathlib import Path

import torch
from torch import nn

from flowcl.analysis.projection_diagnostics import inflora_down, orthonormal_rows, subspace_movement
from flowcl.methods.base import register_method
from flowcl.methods.gpm import GPM
from flowcl.models import lora as adapters
from flowcl.utils.seeding import derive_seed

DISPLAY = {"trained": "projected_lora", "inflora": "inflora", "pca_trained": "pca_trained"}
STATE_BYTES_PER_PARAM = 16   # parameter, gradient and two AdamW moments, fp32


@register_method
class ProjectedLoRA(GPM):
    name = "projected_lora"

    def __init__(self, variant: str = "trained", rank: int | None = None, lr: float | None = None,
                 alpha: float = 16.0, residual_atol: float = 1e-5, residual_rtol: float = 1e-4,
                 design_rel_tol: float = 1e-8, **gpm_kwargs) -> None:
        gpm_kwargs.setdefault("update_memory", True)
        super().__init__(**gpm_kwargs)
        if variant not in DISPLAY:
            raise ValueError(f"variant must be one of {sorted(DISPLAY)}, got {variant!r}")
        if rank is None or lr is None:
            raise ValueError("projected_lora needs `rank` and `lr` (filled from its bracket's selection bundle)")
        if int(rank) != rank or int(rank) < 1 or not float(lr) > 0 or not float(alpha) > 0:
            raise ValueError(f"rank must be a positive integer and lr, alpha positive: {rank}, {lr}, {alpha}")
        self.variant, self.rank, self.lr, self.alpha = variant, int(rank), float(lr), float(alpha)
        self.scale = self.alpha / self.rank if variant == "trained" else 1.0
        self.residual_atol, self.residual_rtol = float(residual_atol), float(residual_rtol)
        self.design_rel_tol = float(design_rel_tol)
        self.history: dict[str, dict] = {}
        self.timings: dict[str, dict] = {}
        self._adapted: list[str] = []
        self._q0: dict[str, torch.Tensor] = {}
        self._started = 0.0

    @property
    def display_name(self) -> str:
        return DISPLAY[self.variant]

    def lora_config(self) -> dict:
        return {"variant": self.variant, "rank": self.rank, "lr": self.lr, "alpha": self.alpha,
                "scale": self.scale, "init": adapters.INIT_SPEC if self.variant == "trained"
                else "top eigenvectors of the task's input Gram within the memory's complement"}

    def stage_lr(self, task_idx: int) -> float | None:
        return None if task_idx == 0 else self.lr

    def install_memory(self, path: str | Path, sha256: str) -> dict:
        """Load a ``gpm_projected_adam`` memory artifact (verified) as this method's memory."""
        return self.restore_memory(path, sha256)

    # ---- the task ----------------------------------------------------------------------------------

    def on_task_start(self, policy, task_idx, *, context) -> None:
        self._task = task_idx
        self.task_logs[task_idx] = {"gradient_c": {}, "update_c": {}}
        self._active = False          # GPM's projected training never runs here
        if task_idx == 0:
            return
        if not self._memory:
            raise RuntimeError(f"{self.display_name} at task {task_idx} has no memory")
        if self._adapted:
            raise RuntimeError("adapters are still attached from an earlier task")
        ns, task = context.seed_namespace_run_id, context.task_key
        if ns is None or task is None:
            raise ValueError(f"{self.display_name} needs the run's seed namespace and a single task key")
        registry = policy.projectable_layers()
        if sorted(self._memory) != sorted(e.name for e in registry):
            raise ValueError("memory layers do not match the registry")
        t0 = time.perf_counter()   # before freezing: the capture's reachability probe needs gradients
        grams = self._design_grams(policy, task_idx, context) if self.variant != "trained" else None
        design_seconds = time.perf_counter() - t0
        for param in policy.parameters():
            param.requires_grad_(False)
            param.grad = None
        seed = derive_seed(ns, f"projected_lora_init::{task}", task_idx)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        per_layer, count, memory_bytes, fixed_bytes = {}, 0, 0, 0
        for entry in registry:
            M = self._memory[entry.name]
            d_in, k = entry.module.in_features, int(M.shape[1])
            memory_bytes += M.numel() * 8
            if k >= d_in:
                per_layer[entry.name] = {"skipped": "full_rank_memory", "k": k, "d_in": d_in}
                continue
            info = {}
            if self.variant == "trained":
                a = torch.empty(self.rank, d_in, dtype=torch.float32)
                nn.init.kaiming_uniform_(a, a=math.sqrt(5), generator=generator)
            else:
                rows, info = inflora_down(grams[entry.name], M, self.rank, self.design_rel_tol)
                if rows.shape[0] == 0:
                    per_layer[entry.name] = {"skipped": "no_free_direction", "k": k, "d_in": d_in, **info}
                    continue
                a = rows.to(torch.float32)
            count += adapters.attach_projected(entry.module, a, M.to(torch.float32), self.scale,
                                               train_a=self.variant != "inflora")
            if self.variant == "inflora":
                fixed_bytes += a.numel() * 4
            self._adapted.append(entry.name)
            self._q0[entry.name] = orthonormal_rows(adapters.effective_down(entry.module).detach().cpu())
            per_layer[entry.name] = {"k": k, "d_in": d_in, "rank": int(a.shape[0]), **info}
        if not self._adapted:
            raise RuntimeError(f"{self.display_name}: no registry layer has a free direction; nothing can be "
                               "adapted at this task")
        trainable = {n for n, p in policy.named_parameters() if p.requires_grad}
        expected = {f"{n}.lora_B" for n in self._adapted}
        if self.variant != "inflora":
            expected |= {f"{n}.lora_A" for n in self._adapted}
        if trainable != expected:
            raise RuntimeError(f"trainable set is not the adapters: {sorted(trainable ^ expected)[:5]}")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._started = time.perf_counter()
        self.history[str(task_idx)] = {
            "task_key": task, **self.lora_config(), "init_seed": seed,
            "adapted_layers": len(self._adapted),
            "skipped_layers": sorted(n for n, v in per_layer.items() if "skipped" in v),
            "trainable_params": count, "optimizer_state_bytes_estimate": STATE_BYTES_PER_PARAM * count,
            "memory_bytes": memory_bytes, "fixed_down_bytes": fixed_bytes, "per_layer": per_layer,
        }
        self.timings[str(task_idx)] = {"design_seconds": design_seconds}

    def _design_grams(self, policy, task_idx: int, context) -> dict[str, torch.Tensor]:
        """Task ``task_idx``'s input Gram per registry layer, at the current weights (GPM's capture)."""
        from flowcl.experiments.gate2 import capture_task_grams, load_subspace_config
        from flowcl.utils.libero_paths import repo_root

        path = Path(self.capture_config)
        if path.suffix != ".yaml":
            path = repo_root() / "configs" / "analysis" / f"{self.capture_config}.yaml"
        cfg = load_subspace_config(path)
        ns, task = context.seed_namespace_run_id, context.task_key
        capture = capture_task_grams(policy, context.dataset, cfg, context.device,
                                     probe_seed=derive_seed(ns, f"inflora_design_probe::{task}", task_idx),
                                     capture_seed=derive_seed(ns, f"inflora_design::{task}", task_idx))
        grams = {}
        for entry in policy.projectable_layers():
            acc = capture.accumulators[entry.name]
            grams[entry.name] = acc.gram[capture.primary_view(entry.name)].to(torch.float64).cpu().clone()
            acc.gram.clear()
        return grams

    def on_task_end(self, policy, task_idx, *, context) -> None:
        if task_idx >= 1 and self._adapted:
            modules = {e.name: e.module for e in policy.projectable_layers()}
            merges, ok = {}, True
            for name in self._adapted:
                intended, realized = adapters.merge_projected(modules[name])
                M = self._memory[name].to(device=realized.device, dtype=torch.float64)
                w_norm = float(modules[name].weight.detach().double().norm())
                d_norm = float(realized.double().norm())
                tol = self.residual_atol * w_norm + self.residual_rtol * d_norm
                res_i = float((intended.double() @ M).norm())
                res_r = float((realized.double() @ M).norm())
                ok = ok and res_i <= tol and res_r <= tol
                move = subspace_movement(realized.cpu(), self._q0[name])
                merges[name] = {"intended_residual": res_i, "realized_residual": res_r, "tolerance": tol,
                                "update_norm": d_norm, "relative_update": d_norm / w_norm if w_norm else math.inf,
                                **move}
            rel = [m["relative_update"] for m in merges.values()]
            outside = [m["f_outside"] for m in merges.values() if m["f_outside"] is not None]
            hist = self.history[str(task_idx)]
            hist["merge"] = {
                "per_layer": merges, "constraint_ok": ok,
                "max_residual_over_tolerance": max((m["realized_residual"] / m["tolerance"]) if m["tolerance"] > 0
                                                   else 0.0 for m in merges.values()),
                "median_relative_update": statistics.median(rel),
                "median_f_outside": statistics.median(outside) if outside else None,
            }
            self.timings[str(task_idx)].update({
                "train_seconds": time.perf_counter() - self._started,
                "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None})
            self._adapted, self._q0 = [], {}
            if not ok:
                raise RuntimeError(f"{self.display_name}: the protection constraint exceeds its tolerance at task "
                                   f"{task_idx} (see the history's merge record)")
        super().on_task_end(policy, task_idx, context=context)

    # ---- state -------------------------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {**super().state_dict(), "lora": self.lora_config(), "history": self.history, "timings": self.timings}

    def restore_state(self, checkpoint_extra: dict, source_run, task_idx: int) -> dict:
        out = super().restore_state(checkpoint_extra, source_run, task_idx)
        state = checkpoint_extra.get("method_state") or {}
        if state.get("lora") != self.lora_config():
            raise ValueError(f"the source's projected-LoRA config {state.get('lora')} is not {self.lora_config()}")
        history = state.get("history") or {}
        if sorted(int(k) for k in history) != list(range(1, task_idx + 1)):
            raise ValueError(f"the source's history covers tasks {sorted(history)}, expected 1..{task_idx}")
        self.history = dict(history)
        self.timings = dict(state.get("timings") or {})
        return {**out, "history_tasks": sorted(int(k) for k in self.history)}
