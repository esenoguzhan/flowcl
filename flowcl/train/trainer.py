"""Single-task training loop — one curriculum stage.

The continual runner (:mod:`flowcl.train.continual`) calls this once per stage rather
than reimplementing optimisation, so the §6 method hooks are threaded through here:

* ``build_batch`` — replay mixes in exemplars
* ``modify_loss`` — EWC, ConSFT
* ``modify_gradients`` — GPM, SGP, §7.5's s-binned method

Hook call order is fixed and identical for every method, including the no-op
``seq_ft``. That is deliberate: a difference between two methods' results must be
attributable to their mechanism and not to a different code path through the trainer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterable

import torch
from torch.utils.data import DataLoader

from flowcl.data.dataset import ChunkedActionDataset, collate_chunks
from flowcl.models.policy import FlowPolicy


@dataclass
class TrainConfig:
    """Optimisation settings for one task stage."""

    steps: int = 2000
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float | None = 1.0
    warmup_steps: int = 100
    log_every: int = 50
    num_workers: int = 0
    device: str = "cuda"
    # Gradient accumulation; §7.5 uses this to absorb the 4x backward cost.
    accumulation_steps: int = 1
    amp: bool = False


@dataclass
class TrainLog:
    """What one stage produced, for the run registry and the §8.2 systems table."""

    losses: list[float] = field(default_factory=list)
    steps: int = 0
    wall_clock_s: float = 0.0

    @property
    def final_loss(self) -> float:
        if not self.losses:
            raise ValueError("TrainLog has no recorded losses")
        return self.losses[-1]

    def mean_last(self, n: int = 50) -> float:
        if not self.losses:
            raise ValueError("TrainLog has no recorded losses")
        tail = self.losses[-n:]
        return sum(tail) / len(tail)


def _cycle(loader: Iterable):
    """Infinite iterator over a DataLoader, so training is step- not epoch-bounded."""
    while True:
        for batch in loader:
            yield batch


def build_dataloader(
    dataset: ChunkedActionDataset,
    batch_size: int,
    num_workers: int = 0,
    shuffle: bool = True,
    generator: torch.Generator | None = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_chunks,
        drop_last=False,
        generator=generator,
        persistent_workers=num_workers > 0,
    )


def build_optimizer(
    policy: FlowPolicy, cfg: TrainConfig
) -> torch.optim.Optimizer:
    """AdamW over trainable parameters, with no weight decay on 1-D parameters.

    Decaying LayerNorm gains, biases and positional embeddings is a well-known way to
    quietly hurt a transformer, so they go in a separate group.
    """
    decay, no_decay = [], []
    for name, param in policy.named_parameters():
        if not param.requires_grad:
            continue
        (no_decay if param.ndim <= 1 else decay).append(param)

    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer, cfg: TrainConfig
) -> torch.optim.lr_scheduler.LRScheduler:
    """Linear warmup then cosine decay."""
    import math

    def lr_lambda(step: int) -> float:
        if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
            return (step + 1) / cfg.warmup_steps
        if cfg.steps <= cfg.warmup_steps:
            return 1.0
        progress = (step - cfg.warmup_steps) / (cfg.steps - cfg.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def move_batch(batch: dict, device: torch.device) -> dict:
    """Move tensors to ``device``, leaving the string fields alone."""
    out = dict(batch)
    out["images"] = {k: v.to(device, non_blocking=True) for k, v in batch["images"].items()}
    for key in ("state", "actions", "action_mask"):
        out[key] = batch[key].to(device, non_blocking=True)
    return out


def train_one_task(
    policy: FlowPolicy,
    dataset: ChunkedActionDataset,
    cfg: TrainConfig,
    method=None,
    task_idx: int = 0,
    generator: torch.Generator | None = None,
    on_step: Callable[[int, dict], None] | None = None,
    context=None,
) -> TrainLog:
    """Train ``policy`` on one task's ``dataset`` for ``cfg.steps`` optimiser steps.

    Args:
        policy: The policy to train, already on the target device.
        dataset: Chunked dataset for this stage.
        cfg: Optimisation settings.
        method: A :class:`~flowcl.methods.base.ContinualMethod`. Defaults to a no-op
            ``seq_ft``, so the hook sequence is identical whether or not a method was
            passed.
        task_idx: Curriculum stage index, forwarded to the §6 hooks.
        generator: RNG for ``s`` and ``A_0``, so a stage is reproducible.
        on_step: Callback invoked as ``on_step(step, outputs)`` after each optimiser
            step. Used by the §7.1 analysis hooks to capture activations at an interval.
        context: :class:`~flowcl.methods.base.TaskContext` for the lifecycle hooks,
            built by the continual runner. Other callers get a minimal one with no seed
            namespace; a method that needs one raises.

    Returns:
        A :class:`TrainLog`.
    """
    from flowcl.methods.base import TaskContext
    from flowcl.methods.seq_ft import SeqFT

    if method is None:
        method = SeqFT()

    device = torch.device(cfg.device)
    policy.to(device)
    policy.train()

    if context is None:
        task_ids = getattr(dataset, "task_ids", ())
        context = TaskContext(
            task_key=task_ids[0] if len(task_ids) == 1 else None,
            dataset=dataset,
            device=str(device),
        )

    # Before the optimiser is built: a method may freeze parameters here (GPM's §7.4
    # allowlist), and a frozen parameter must never enter AdamW.
    method.on_task_start(policy, task_idx, context=context)

    optimizer = build_optimizer(policy, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    loader = build_dataloader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        generator=generator,
    )
    batches = _cycle(loader)

    scaler = torch.amp.GradScaler(device.type, enabled=cfg.amp and device.type == "cuda")
    log = TrainLog()
    started = time.perf_counter()

    for step in range(cfg.steps):
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0.0

        for micro in range(cfg.accumulation_steps):
            # §6: a method may own batch construction (replay mixing). Returning None
            # means "use the runner's dataloader", which is what every other method
            # does.
            batch = method.build_batch(dataset, task_idx)
            if batch is None:
                batch = next(batches)
            batch = move_batch(batch, device)

            with torch.autocast(
                device_type=device.type, enabled=cfg.amp and device.type == "cuda"
            ):
                outputs = policy(batch, generator=generator)
                loss = method.modify_loss(
                    outputs["loss"], batch, policy, outputs=outputs
                )
                # Scale so that accumulation is equivalent to one larger mean-reduced
                # batch, not to a sum over microbatches.
                loss = loss / cfg.accumulation_steps

            scaler.scale(loss).backward()
            accumulated += float(loss.detach()) * cfg.accumulation_steps

        # §7.3: gradient surgery happens after backward() and before step(), on
        # unscaled gradients — projecting AMP-scaled gradients would work by luck
        # (projection is linear) but clipping afterwards would not.
        scaler.unscale_(optimizer)
        method.modify_gradients(
            policy,
            {
                "step": step,
                "task_idx": task_idx,
                "s": outputs["s"].detach(),
                "batch": batch,
            },
        )

        if cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                [p for p in policy.parameters() if p.requires_grad], cfg.grad_clip
            )

        scaler.step(optimizer)
        scaler.update()
        # Documented §6 addition 3: the realised update is final only here, after Adam's
        # per-coordinate scaling and weight decay. A skipped AMP step leaves weights as
        # they were, so the method sees a zero update.
        method.after_step(policy, {"step": step, "task_idx": task_idx})
        scheduler.step()

        mean_loss = accumulated / cfg.accumulation_steps
        log.losses.append(mean_loss)
        log.steps = step + 1

        if on_step is not None:
            on_step(step, outputs)
        if cfg.log_every and (step % cfg.log_every == 0 or step == cfg.steps - 1):
            print(
                f"[flowcl] step {step + 1}/{cfg.steps} loss {mean_loss:.6f} "
                f"lr {scheduler.get_last_lr()[0]:.2e}",
                flush=True,
            )

    method.on_task_end(policy, task_idx, context=context)
    log.wall_clock_s = time.perf_counter() - started
    return log
