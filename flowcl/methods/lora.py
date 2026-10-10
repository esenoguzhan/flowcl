"""``lora``: sequential merged LoRA (docs/thesis_plan.md, Workstream B; Hu et al., ICLR 2022).

Task 1 trains every parameter (full fine-tuning), so the stage-0 model equals ``seq_ft``'s bit for
bit. From Task 2 on, every base parameter is frozen and each task gets fresh rank-``r`` adapters on
every §4.5 registry layer (:mod:`flowcl.models.lora`), trained at their own learning rate
(:meth:`LoRA.stage_lr`) and **merged into the weights at the task's end**. No per-task adapters are
kept: there is no task identity at inference, so the paper's deployment by swapping adapters is not
available, and each task's adapters start on the already-merged weights.

The adapted weights are exactly GPM's trainable set from Task 2 (:func:`flowcl.methods.gpm.allowlist`,
the registry weights; biases, LayerNorms, AdaLN, encoders frozen), so LoRA and GPM are two training
procedures on the same weight support. ``alpha`` is a constant in ``r`` and not tuned (§4.1).

Nothing persists across tasks but the weights (``stored_bytes`` is 0). The method's ``state_dict``
keeps a per-task history (the configuration, the adapter seed, the trainable count, an estimate of
the adapters' training-state memory, and the merge norms per layer); :meth:`LoRA.restore_state`
restores it on resume, so a resumed run reports the same history as its source.
"""

from __future__ import annotations

import statistics

import torch

from flowcl.methods.base import BaseMethod, register_method
from flowcl.models import lora as adapters
from flowcl.utils.seeding import derive_seed

# Adapters, their gradients and AdamW's two moments, fp32 (activations and temporaries excluded).
TRAINING_STATE_BYTES_PER_PARAM = 16


def init_seed(namespace: str, task_key: str, task_idx: int) -> int:
    return derive_seed(namespace, f"lora_init::{task_key}", task_idx)


@register_method
class LoRA(BaseMethod):
    """Full fine-tuning on Task 1; frozen base plus merged rank-``rank`` adapters from Task 2 on."""

    name = "lora"

    def __init__(self, rank: int | None = None, lr: float | None = None, alpha: float = 16.0,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        if rank is None or lr is None:
            raise ValueError("lora needs `rank` and `lr` (filled from the LoRA bracket's selection bundle)")
        if int(rank) != rank or int(rank) < 1:
            raise ValueError(f"rank must be a positive integer, got {rank}")
        if not float(lr) > 0 or not float(alpha) > 0:
            raise ValueError(f"lr and alpha must be positive, got {lr}, {alpha}")
        self.rank, self.lr, self.alpha = int(rank), float(lr), float(alpha)
        self.history: dict[str, dict] = {}
        self._adapted: list[str] = []

    def config(self) -> dict:
        return {"rank": self.rank, "alpha": self.alpha, "lr": self.lr, "init": adapters.INIT_SPEC}

    def stage_lr(self, task_idx: int) -> float | None:
        """The learning rate of stage ``task_idx``: the adapters' from Task 2 on (``None``: the recipe's)."""
        return None if task_idx == 0 else self.lr

    def on_task_start(self, policy, task_idx, *, context) -> None:
        if task_idx == 0:
            return
        if self._adapted:
            raise RuntimeError(f"adapters are still attached from an earlier task: {self._adapted[:3]}")
        if context.seed_namespace_run_id is None or context.task_key is None:
            raise ValueError("lora needs the run's seed namespace and a single task key for its adapter seed")
        for param in policy.parameters():
            param.requires_grad_(False)
        seed = init_seed(context.seed_namespace_run_id, context.task_key, task_idx)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        count = 0
        for entry in policy.projectable_layers():
            count += adapters.attach(entry.module, self.rank, self.alpha, generator)
            self._adapted.append(entry.name)
        trainable = {n for n, p in policy.named_parameters() if p.requires_grad}
        expected = {f"{name}.lora_{x}" for name in self._adapted for x in "AB"}
        if trainable != expected:
            raise RuntimeError(f"trainable set is not exactly the adapters: {sorted(trainable ^ expected)[:5]}")
        self.history[str(task_idx)] = {
            "task_key": context.task_key, **self.config(), "init_seed": seed,
            "adapted_layers": len(self._adapted), "trainable_params": count,
            "training_state_bytes_estimate": TRAINING_STATE_BYTES_PER_PARAM * count,
        }

    def on_task_end(self, policy, task_idx, *, context) -> None:
        if task_idx == 0:
            return
        modules = {entry.name: entry.module for entry in policy.projectable_layers()}
        merges = {name: adapters.merge_and_detach(modules[name]) for name in self._adapted}
        self._adapted = []
        relative = [m["relative"] for m in merges.values()]
        self.history[str(task_idx)]["merge"] = {
            "per_layer": merges,
            "relative_median": statistics.median(relative),
            "relative_max": max(relative),
        }

    def state_dict(self) -> dict:
        return {"name": self.name, **self.config(), "history": self.history}

    def restore_state(self, checkpoint_extra: dict, source_run, task_idx: int) -> dict:
        super().restore_state(checkpoint_extra, source_run, task_idx)
        state = checkpoint_extra.get("method_state") or {}
        recorded = {k: state.get(k) for k in ("rank", "alpha", "lr", "init")}
        if state.get("name") != self.name or recorded != self.config():
            raise ValueError(f"the source checkpoint's method state {recorded} (name {state.get('name')!r}) "
                             f"is not this LoRA's {self.config()}")
        history = state.get("history") or {}
        if sorted(int(k) for k in history) != list(range(1, task_idx + 1)):
            raise ValueError(f"the source's LoRA history covers tasks {sorted(history)}, expected 1..{task_idx}")
        self.history = {k: v for k, v in history.items()}
        return {"history_tasks": sorted(int(k) for k in self.history)}
