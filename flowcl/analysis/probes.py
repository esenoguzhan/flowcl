"""Fixed-batch loss probes: a like-for-like loss read across checkpoints.

The masked flow-matching loss over fixed batches, with identical ``s`` and ``A_0`` for
every checkpoint: independent generator streams keyed on ``(tag, data task)``, fp32,
eval mode. First used by the GPM pilot (``docs/runs/2026-09-23_gpm_pilot.md`` §3); the
forgetting diagnostics reuse it with the pilot's tags, so seq_ft's stage-0/1 values
reproduce the pilot's recorded ones.
"""

from __future__ import annotations

import torch

from flowcl.models.flow_head import draw_with_generator
from flowcl.models.losses import valid_element_count
from flowcl.train.trainer import build_dataloader, move_batch
from flowcl.utils.seeding import derive_seed


def _generators(seed_tags: dict, task_key: str) -> dict[str, torch.Generator]:
    return {
        role: torch.Generator(device="cpu").manual_seed(derive_seed(tag, task_key, 0))
        for role, tag in seed_tags.items()
    }


@torch.no_grad()
def probe_loss(policy, dataset, probe: dict, device) -> float:
    """Masked flow-matching loss over fixed batches, weighted by valid elements.

    Identical batches, ``s`` and ``A_0`` for every checkpoint (streams keyed on the probe
    tags and the data task), fp32, eval mode — a like-for-like stability/plasticity read.
    """
    if len(dataset.task_ids) != 1:
        raise ValueError(f"probe needs a single-task dataset, got {dataset.task_ids}")
    device = torch.device(device)
    gens = _generators(probe["seed_tags"], dataset.task_ids[0])
    loader = build_dataloader(
        dataset, batch_size=probe["batch_size"], num_workers=0, shuffle=True,
        generator=gens["shuffle"],
    )
    policy.eval()
    total, weight = 0.0, 0.0
    for idx, batch in enumerate(loader):
        if idx >= probe["n_batches"]:
            break
        batch = move_batch(batch, device)
        size = batch["actions"].shape[0]
        s = policy.s_sampler.sample(size, device, generator=gens["flow_time"])
        noise = draw_with_generator(
            tuple(batch["actions"].shape), device=device, generator=gens["noise"],
            dtype=torch.float32, normal=True,
        )
        with torch.autocast(device_type=device.type, enabled=False):
            loss = float(policy(batch, s=s, noise=noise)["loss"])
        n_b = float(valid_element_count(batch["action_mask"], policy.d_action))
        total += loss * n_b
        weight += n_b
    if weight == 0:
        raise ValueError("probe saw no valid elements")
    return total / weight
