"""Online EWC with a diagonal empirical Fisher (spec §6 item 4; docs/thesis_plan.md, Workstream B).

Source: Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks", PNAS 2017;
the online form (one running Fisher, one anchor) follows Schwarz et al., "Progress & Compress",
ICML 2018, with decay γ = 1.

After each task τ this method estimates

    F_τ = mean over n samples of (∂ℓ_i / ∂θ)²

for every trainable parameter, where ``ℓ_i`` is one training sample's masked flow-matching loss
(the policy's own ``outputs["loss"]`` at batch size 1, with that sample's ``s`` and ``A_0``). It
adds ``F_τ`` to the running Fisher and moves the anchor ``θ*`` to the post-task weights. From
the next task on, the loss is

    ℓ + (λ/2) Σ F (θ − θ*)²

with the penalty in fp32 (autocast off).

**What this Fisher is.** Squared per-sample gradients of the masked flow-matching MSE are an
*empirical-Fisher surrogate for that loss*. They are not the Fisher information of the policy's
action distribution (Kunstner et al., NeurIPS 2019). A weak result is a result about this
specified EWC baseline. The spec's §6 asked for a two-task probe; running the full sequence is a
recorded deviation (``docs/thesis_plan.md``).

The Fisher uses its own generator, ``derive_seed(ns, "ewc_fisher::<task>", τ)``, for sample
indices, ``s`` and ``A_0``: the training streams are never touched, so stage 0 is bitwise
seq_ft. It runs in ``train()`` mode whatever mode the policy arrives in (a loaded checkpoint is
in ``eval()``). The running Fisher and the anchor are an artifact
(``method/ewc_task{τ}.pt``) that the stage checkpoint names by SHA-256.
"""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path

import torch

from flowcl.methods.base import BaseMethod, register_method
from flowcl.utils.run import file_sha256

ARTIFACT_KIND = "ewc_state"
NORMALIZATION = ("mean over samples of the squared per-sample gradient of the masked-mean flow-"
                 "matching loss (policy outputs['loss'] at batch size 1), fp32, autocast off")


def tensor_fingerprint(tensors: dict[str, torch.Tensor]) -> str:
    """SHA-256 over names, dtypes, shapes and bytes, in name order (CPU copies)."""
    h = hashlib.sha256()
    for name in sorted(tensors):
        t = tensors[name].detach().to("cpu").contiguous()
        h.update(name.encode())
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def trainable_parameters(policy) -> list[tuple[str, torch.nn.Parameter]]:
    return [(n, p) for n, p in policy.named_parameters() if p.requires_grad]


def empirical_fisher(policy, dataset, n_samples: int, seed: int, device) -> tuple[dict, dict]:
    """``{name: F}`` over the trainable parameters, and its provenance.

    One sample per forward/backward (the per-sample gradient), fp32, ``train()`` mode;
    the previous mode is restored. Gradients are taken with ``torch.autograd.grad``, so no
    ``.grad`` attribute is touched.
    """
    from flowcl.data.dataset import collate_chunks
    from flowcl.train.trainer import move_batch

    device = torch.device(device)
    params = trainable_parameters(policy)
    if not params:
        raise ValueError("the policy has no trainable parameters")
    fisher = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in params}
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    indices = torch.randint(len(dataset), (int(n_samples),), generator=generator).tolist()
    was_training = policy.training
    policy.train()
    try:
        for i in indices:
            batch = move_batch(collate_chunks([dataset[i]]), device)
            with torch.autocast(device_type=device.type, enabled=False):
                loss = policy(batch, generator=generator)["loss"]
            grads = torch.autograd.grad(loss, [p for _, p in params], allow_unused=True)
            for (name, _), g in zip(params, grads):
                if g is not None:
                    fisher[name].add_(g.detach().float().pow(2))
    finally:
        policy.train(was_training)
    for f in fisher.values():
        f.div_(len(indices))
    nonfinite = [n for n, f in fisher.items() if not torch.isfinite(f).all()]
    if nonfinite:
        raise RuntimeError(f"non-finite Fisher entries in {nonfinite[:5]}")
    provenance = {"n_samples": len(indices), "seed": int(seed), "normalization": NORMALIZATION,
                  "n_parameters": int(sum(f.numel() for f in fisher.values())),
                  "fingerprint": tensor_fingerprint(fisher)}
    return fisher, provenance


@register_method
class EWC(BaseMethod):
    """Online EWC; see the module docstring."""

    name = "ewc"

    def __init__(self, lam: float | None, fisher_samples: int = 1024, log_interval: int = 100) -> None:
        super().__init__()
        if lam is None:
            raise ValueError("EWC's lam is unset: configs/method/ewc.yaml is filled from the λ "
                             "bracket's frozen selection bundle before any EWC run")
        if not (isinstance(lam, (int, float)) and math.isfinite(lam) and lam > 0):
            raise ValueError(f"lam must be a positive finite number, got {lam!r}")
        if int(fisher_samples) < 1:
            raise ValueError(f"fisher_samples must be >= 1, got {fisher_samples}")
        if int(log_interval) < 1:
            raise ValueError(f"log_interval must be >= 1, got {log_interval}")
        self.lam = float(lam)
        self.fisher_samples = int(fisher_samples)
        self.log_interval = int(log_interval)
        self._fisher: dict[str, torch.Tensor] = {}   # running Fisher, fp32
        self._anchor: dict[str, torch.Tensor] = {}   # θ*, fp32
        self.fisher_tasks: dict[int, dict] = {}      # task -> provenance of F_τ
        self.task_logs: dict[int, dict] = {}
        self._params: list[tuple[str, torch.nn.Parameter]] = []
        self._task: int | None = None
        self._calls = 0

    def config(self) -> dict:
        return {"lam": self.lam, "fisher_samples": self.fisher_samples, "log_interval": self.log_interval}

    @property
    def fisher(self) -> dict[str, torch.Tensor]:
        """The running Fisher (read-only use: calibration and reports)."""
        return self._fisher

    @property
    def anchor(self) -> dict[str, torch.Tensor]:
        return self._anchor

    # ---- the penalty -----------------------------------------------------------

    def penalty(self) -> torch.Tensor:
        """``Σ F (θ − θ*)²`` over the bound parameters, fp32."""
        total = None
        for name, p in self._params:
            term = (self._fisher[name] * (p.float() - self._anchor[name]).pow(2)).sum()
            total = term if total is None else total + term
        return total

    # ---- §6 hooks --------------------------------------------------------------

    def on_task_start(self, policy, task_idx, *, context) -> None:
        self._task = task_idx
        self._calls = 0
        self.task_logs[task_idx] = {"penalty": {}}
        if not self._fisher:
            self._params = []
            return
        params = trainable_parameters(policy)
        names = [n for n, _ in params]
        if sorted(names) != sorted(self._fisher):
            raise ValueError("trainable parameters differ from the Fisher's: missing "
                             f"{sorted(set(self._fisher) - set(names))[:5]}, extra "
                             f"{sorted(set(names) - set(self._fisher))[:5]}")
        device = params[0][1].device
        self._fisher = {n: f.to(device) for n, f in self._fisher.items()}
        self._anchor = {n: a.to(device) for n, a in self._anchor.items()}
        self._params = params

    def modify_loss(self, loss, batch, policy, *, outputs=None):
        if not self._params:
            return loss  # T1: the identical tensor, so stage 0 is seq_ft's
        with torch.autocast(device_type=loss.device.type, enabled=False):
            penalty = self.penalty()
            total = loss.float() + 0.5 * self.lam * penalty
        if self._calls % self.log_interval == 0:
            self.task_logs[self._task]["penalty"][self._calls] = {
                "loss": float(loss.detach()), "penalty": float(penalty.detach()),
                "weighted": float(0.5 * self.lam * penalty.detach())}
        self._calls += 1
        return total

    def on_task_end(self, policy, task_idx, *, context) -> None:
        from flowcl.utils.seeding import derive_seed

        ns = context.seed_namespace_run_id
        if ns is None or context.dataset is None or context.task_key is None:
            raise RuntimeError("EWC's Fisher needs the seed namespace, the task's dataset and its key")
        seed = derive_seed(ns, f"ewc_fisher::{context.task_key}", task_idx)
        fisher, provenance = empirical_fisher(policy, context.dataset, self.fisher_samples, seed,
                                              context.device)
        params = dict(trainable_parameters(policy))
        if self._fisher:
            if sorted(fisher) != sorted(self._fisher):
                raise ValueError("this task's trainable parameters differ from the running Fisher's")
            for name, f in fisher.items():
                self._fisher[name] = self._fisher[name].to(f.device) + f
        else:
            self._fisher = fisher
        self._anchor = {n: params[n].detach().float().clone() for n in fisher}
        self.fisher_tasks[task_idx] = {**provenance, "task_key": context.task_key}
        self._params = []

    # ---- artifacts -------------------------------------------------------------

    def save_artifacts(self, directory, task_idx, *, context) -> list[Path]:
        if not self._fisher:
            return []
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"ewc_task{task_idx}.pt"
        tmp = path.with_name(path.name + ".tmp")
        torch.save({
            "kind": ARTIFACT_KIND, "task_idx": task_idx, "config": self.config(),
            "fisher_tasks": {str(k): v for k, v in sorted(self.fisher_tasks.items())},
            "fisher": {n: f.detach().cpu() for n, f in self._fisher.items()},
            "anchor": {n: a.detach().cpu() for n, a in self._anchor.items()},
        }, tmp)
        os.replace(tmp, path)
        return [path]

    def restore_state(self, checkpoint_extra: dict, source_run: Path, task_idx: int) -> dict:
        artifacts = {a["path"]: a["sha256"] for a in checkpoint_extra.get("method_artifacts") or []}
        key = f"method/ewc_task{task_idx}.pt"
        if key not in artifacts:
            raise ValueError(f"the stage-{task_idx} checkpoint names no {key} (artifacts: {sorted(artifacts)})")
        path = Path(source_run) / key
        actual = file_sha256(path)
        if actual != artifacts[key]:
            raise ValueError(f"{path}: SHA-256 {actual} does not match the checkpoint's {artifacts[key]}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        problems = []
        if payload.get("kind") != ARTIFACT_KIND:
            problems.append(f"kind {payload.get('kind')!r}")
        if payload.get("task_idx") != task_idx:
            problems.append(f"task_idx {payload.get('task_idx')} != {task_idx}")
        if payload.get("config") != self.config():
            problems.append(f"config {payload.get('config')} != {self.config()}")
        if problems:
            raise ValueError(f"{path}: EWC state does not match: {problems}")
        self._fisher = dict(payload["fisher"])
        self._anchor = dict(payload["anchor"])
        self.fisher_tasks = {int(k): v for k, v in payload["fisher_tasks"].items()}
        return {"artifact": key, "sha256": actual, "fisher_tasks": sorted(self.fisher_tasks)}

    # ---- reporting -------------------------------------------------------------

    def stored_bytes(self) -> int:
        """The running Fisher plus the anchor (fp32 each)."""
        return sum(f.numel() * 4 for f in self._fisher.values()) + sum(a.numel() * 4 for a in self._anchor.values())

    def state_dict(self) -> dict:
        latest = self.task_logs.get(max(self.task_logs)) if self.task_logs else {}
        return {"name": self.name, "config": self.config(), "stored_mb": self.stored_bytes() / 1e6,
                "fisher_tasks": {str(k): v for k, v in sorted(self.fisher_tasks.items())},
                "penalty_log": {str(k): v for k, v in (latest or {}).get("penalty", {}).items()}}

    def describe(self) -> str:
        return f"ewc (lam {self.lam:g}, stores {self.stored_bytes() / 1e6:.1f} MB)"
