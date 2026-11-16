"""Old-task probe loss during a new task's training (status.md §5 item 5; optional).

Only stage-end checkpoints are kept, so the figures can show *that* a task was forgotten but not
*when*. With ``run_continual(probe_every=N)`` the runner measures, at the start of each stage and
every N steps of it, the flow loss of every previously trained task on a **fixed** probe batch:
fixed chunks, fixed flow times and fixed noise, all drawn once from
``derive_seed(namespace, "probe::<task>", 0)`` when that task's own stage ends (its dataset is
still loaded then, so nothing is reloaded later).

A probe must leave training bitwise unchanged:
- it draws nothing from the training generators or the global RNG (its draws are made once, from
  its own generator, outside the training ``DataLoader``);
- it runs under ``torch.no_grad`` in eval mode and restores every module's ``training`` flag;
- its batch has the training batch size, so no new input shape reaches cuDNN.
The guard smoke (probes every 50 steps against probes off) checks this on GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from flowcl.data.dataset import collate_chunks
from flowcl.models.flow_head import draw_with_generator
from flowcl.utils.seeding import derive_seed


@dataclass
class ProbeBatch:
    task_key: str
    batch: dict          # collated, on CPU
    s: torch.Tensor      # (B,) flow times
    noise: torch.Tensor  # (B, H, D) A_0


def task_indices(dataset, task_key: str) -> list[int]:
    """The dataset's sample indices that belong to ``task_key`` (a joint stage holds several tasks)."""
    return [i for i, idx in enumerate(dataset._index) if dataset.episodes[idx.episode_idx].task_id == task_key]


def make_probe_batch(dataset, task_key: str, namespace: str, policy, batch_size: int) -> ProbeBatch:
    indices = task_indices(dataset, task_key)
    if not indices:
        raise ValueError(f"no samples of {task_key} in this dataset")
    gen = torch.Generator().manual_seed(derive_seed(namespace, f"probe::{task_key}", 0))
    pick = torch.randint(len(indices), (batch_size,), generator=gen).tolist()
    batch = collate_chunks([dataset[indices[i]] for i in pick])
    s = policy.s_sampler.sample(batch_size, torch.device("cpu"), generator=gen)
    noise = draw_with_generator(tuple(batch["actions"].shape), device="cpu", generator=gen,
                                dtype=torch.float32, normal=True)
    return ProbeBatch(task_key, batch, s, noise)


def probe_losses(policy, probes: list[ProbeBatch], device, amp: bool) -> dict[str, float]:
    """``task_key -> flow loss`` of each probe batch; module modes are restored exactly."""
    from flowcl.train.trainer import move_batch

    if not probes:
        return {}
    device = torch.device(device)
    modes = [(module, module.training) for module in policy.modules()]
    policy.eval()
    out = {}
    try:
        with torch.no_grad():
            for probe in probes:
                batch = move_batch(probe.batch, device)
                with torch.autocast(device_type=device.type, enabled=bool(amp and device.type == "cuda")):
                    outputs = policy(batch, s=probe.s.to(device), noise=probe.noise.to(device))
                out[probe.task_key] = float(outputs["loss"].detach().float())
    finally:
        for module, flag in modes:
            module.training = flag
    return out
