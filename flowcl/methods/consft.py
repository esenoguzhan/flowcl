"""``consft``: Conservative Supervised Fine-Tuning (docs/thesis_plan.md, Workstream B).

Source: Zhang et al., "Preserving Foundational Capabilities in Flow-Matching VLAs through
Conservative SFT", arXiv 2605.08879v2 (eq. 5, App. C.1, Table 5); the released code is
github.com/tyzhang2907/ConservativeSFT ``src/consft.py``. Every decision below is argued in
``docs/consft_notes.md``.

Each sample's flow-matching loss is scaled by a stop-gradient confidence weight

    ω_i = max(ω_min, exp(−κ · L_i / τ(t)))

with the temperature annealed upward over ``decay_steps`` (eq. 11–12)::

    ρ(t) = (exp(−λ · min(t/T, 1)^c) − e^{−λ}) / (1 − e^{−λ})
    τ(t) = min(τ_start / max(ρ(t), 1e-8), τ_end)

The fixed constants are Table 5's (κ = 25, λ = 0.8, c = 3.5, ω_min = 1e-3, τ_end = 5); ``tau_start``
and ``decay_steps`` are the two the paper tunes per task and come from the bracket
(``configs/analysis/baseline_selection_consft.yaml``).

**Our loss** (notes §1–2): ``L_i = S_i / (N_i·D)`` with ``S_i`` the sample's summed masked squared
error and ``N_i`` its valid steps, and the stage objective is ``Σ_i ω_i S_i / Σ_i N_i·D``, so that at
ω ≡ 1 it equals seq_ft's :func:`~flowcl.models.losses.masked_mse`. The weight is computed in fp32 with
autocast off under ``no_grad`` (the stop-gradient).

**Task 1 is seq_ft** (notes §3): the paper fine-tunes a pretrained base, and our T1 plays that role,
so at task 0 :meth:`ConSFT.modify_loss` returns the loss tensor untouched and stage 0 is bitwise
seq_ft's. **The schedule restarts at every task** (notes §4): the step counter is reset by
``on_task_start`` and advanced by ``after_step``, once per optimiser step (AMP-skipped steps included,
like the released code's ``global_step``).

Nothing persists across tasks but the weights (``stored_bytes`` is 0). ``state_dict`` carries the
configuration and the per-task logs (every ``log_interval`` steps: τ, raw and weighted loss, ω's mean,
floor fraction and its value over the highest-loss 5 % and 10 %, and ω and L per flow-time bin);
:meth:`ConSFT.restore_state` restores them on resume.
"""

from __future__ import annotations

import math

import torch

from flowcl.methods.base import BaseMethod, register_method

RHO_EPS = 1e-8          # eq. 12's ε
S_BINS = 5              # flow-time bins of the logged ω and L


def temperature(step: int, tau_start: float, decay_steps: int, decay_rate: float, curve: float,
                tau_end: float) -> float:
    """τ(t) of eq. 11–12 at optimiser step ``step`` (0-based) of the current task."""
    progress = min(max(float(step) / float(decay_steps), 0.0), 1.0)
    end = math.exp(-decay_rate)
    rho = (math.exp(-decay_rate * progress ** curve) - end) / (1.0 - end)
    return min(tau_start / max(rho, RHO_EPS), tau_end)


def per_sample_terms(velocity: torch.Tensor, target_velocity: torch.Tensor,
                     mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(S, N·D)`` per sample, fp32: the summed masked squared error and the valid element count."""
    err = (velocity.float() - target_velocity.float()).pow(2)
    m = mask.to(err.dtype)
    summed = (err * m.unsqueeze(-1)).sum(dim=(1, 2))
    count = m.sum(dim=1) * err.shape[-1]
    return summed, count


@register_method
class ConSFT(BaseMethod):
    """seq_ft on Task 1; from Task 2 on, each sample's loss scaled by sg[max(ω_min, exp(−κL/τ(t)))]."""

    name = "consft"

    def __init__(self, tau_start: float | None = None, decay_steps: int | None = None,
                 kappa: float = 25.0, decay_rate: float = 0.8, curve: float = 3.5,
                 omega_min: float = 1e-3, tau_end: float = 5.0, log_interval: int = 100,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        if tau_start is None or decay_steps is None:
            raise ValueError("consft needs `tau_start` and `decay_steps` (from the ConSFT bracket's selection)")
        if int(decay_steps) != decay_steps or int(decay_steps) < 1:
            raise ValueError(f"decay_steps must be a positive integer, got {decay_steps}")
        for key, value in (("tau_start", tau_start), ("kappa", kappa), ("decay_rate", decay_rate),
                           ("tau_end", tau_end)):
            if not (math.isfinite(float(value)) and float(value) > 0):
                raise ValueError(f"{key} must be positive and finite, got {value}")
        if not float(curve) >= 1.0:
            raise ValueError(f"curve must be >= 1 (the paper's c > 1), got {curve}")
        if not 0.0 < float(omega_min) <= 1.0:
            raise ValueError(f"omega_min must lie in (0, 1], got {omega_min}")
        if int(log_interval) != log_interval or int(log_interval) < 1:
            raise ValueError(f"log_interval must be a positive integer, got {log_interval}")
        self.tau_start, self.decay_steps = float(tau_start), int(decay_steps)
        self.kappa, self.decay_rate, self.curve = float(kappa), float(decay_rate), float(curve)
        self.omega_min, self.tau_end = float(omega_min), float(tau_end)
        self.log_interval = int(log_interval)
        self.logs: dict[str, dict] = {}
        self._task = 0
        self._step = 0

    def config(self) -> dict:
        return {"tau_start": self.tau_start, "decay_steps": self.decay_steps, "kappa": self.kappa,
                "decay_rate": self.decay_rate, "curve": self.curve, "omega_min": self.omega_min,
                "tau_end": self.tau_end}

    def temperature(self, step: int) -> float:
        return temperature(step, self.tau_start, self.decay_steps, self.decay_rate, self.curve, self.tau_end)

    def weights(self, per_sample_loss: torch.Tensor, step: int) -> torch.Tensor:
        """ω for fp32 per-sample losses at ``step``, without gradient."""
        with torch.no_grad():
            tau = self.temperature(step)
            return torch.exp(-self.kappa * per_sample_loss.detach().float() / tau).clamp(min=self.omega_min)

    # ---- §6 hooks --------------------------------------------------------------

    def on_task_start(self, policy, task_idx, *, context) -> None:
        self._task, self._step = int(task_idx), 0
        if task_idx > 0:
            self.logs[str(task_idx)] = {"task_key": context.task_key, "steps": {}}

    def modify_loss(self, loss, batch, policy, *, outputs=None):
        if self._task == 0:
            return loss  # T1: the identical tensor, so stage 0 is seq_ft's
        if outputs is None:
            raise ValueError("consft needs the forward outputs (velocity and target velocity)")
        with torch.autocast(device_type=loss.device.type, enabled=False):
            summed, count = per_sample_terms(outputs["velocity"], outputs["target_velocity"],
                                             batch["action_mask"])
            per_sample = summed.detach() / count
            omega = self.weights(per_sample, self._step)
            total = (omega * summed).sum() / count.sum()
        if self._step % self.log_interval == 0:
            self._log(per_sample, omega, outputs.get("s"), float(total.detach()))
        return total

    def after_step(self, policy, step_meta) -> None:
        self._step += 1

    def _log(self, per_sample: torch.Tensor, omega: torch.Tensor, s: torch.Tensor | None,
             weighted: float) -> None:
        n = per_sample.numel()
        order = torch.argsort(per_sample, descending=True)
        top = {f"omega_top{q}": float(omega[order[:max(int(n * q / 100), 1)]].mean()) for q in (5, 10)}
        record = {"tau": self.temperature(self._step), "raw_loss": float(per_sample.mean()),
                  "weighted_loss": weighted, "omega_mean": float(omega.mean()),
                  "omega_floor_fraction": float((omega <= self.omega_min).float().mean()), **top}
        if s is not None:
            bins = torch.clamp((s.detach().float() * S_BINS).long(), 0, S_BINS - 1).to(per_sample.device)
            record["by_s_bin"] = {
                str(b): {"n": int((bins == b).sum()),
                         "omega_mean": float(omega[bins == b].mean()) if (bins == b).any() else None,
                         "raw_loss": float(per_sample[bins == b].mean()) if (bins == b).any() else None}
                for b in range(S_BINS)}
        self.logs[str(self._task)]["steps"][str(self._step)] = record

    def state_dict(self) -> dict:
        return {"name": self.name, **self.config(), "log_interval": self.log_interval, "logs": self.logs}

    def restore_state(self, checkpoint_extra: dict, source_run, task_idx: int) -> dict:
        super().restore_state(checkpoint_extra, source_run, task_idx)
        state = checkpoint_extra.get("method_state") or {}
        recorded = {k: state.get(k) for k in self.config()}
        if state.get("name") != self.name or recorded != self.config():
            raise ValueError(f"the source checkpoint's method state {recorded} (name {state.get('name')!r}) "
                             f"is not this ConSFT's {self.config()}")
        logs = state.get("logs") or {}
        if sorted(int(k) for k in logs) != list(range(1, task_idx + 1)):
            raise ValueError(f"the source's ConSFT logs cover tasks {sorted(logs)}, expected 1..{task_idx}")
        self.logs = {k: v for k, v in logs.items()}
        return {"log_tasks": sorted(int(k) for k in self.logs)}
