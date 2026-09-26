"""Per-batch probe losses: the refactored probe is bitwise the old one, and the probe-cells
step refuses a mean that does not reproduce the diagnostics."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from flowcl.analysis.probes import _generators, probe_batch_losses, probe_loss, weighted_mean
from flowcl.data.config import load_embodiment_spec
from flowcl.data.dataset import ChunkedActionDataset
from flowcl.data.episode import Episode
from flowcl.data.stats import compute_stats
from flowcl.experiments.probe_cells import check_reproduces
from flowcl.models.build import build_policy
from flowcl.models.flow_head import draw_with_generator
from flowcl.models.losses import valid_element_count
from flowcl.train.trainer import build_dataloader, move_batch

PROBE = {"batch_size": 4, "n_batches": 3,
         "seed_tags": {"shuffle": "t::shuffle", "flow_time": "t::flow_time", "noise": "t::noise"}}


def episode(spec, seed, length=18):
    rng = np.random.default_rng(seed)
    h, w = spec.observation.image_size
    return Episode(
        images={c: rng.integers(0, 255, (length, h, w, 3), dtype=np.uint8) for c in spec.cameras},
        state=rng.normal(size=(length, spec.d_state)).astype(np.float32),
        action=rng.uniform(-1, 1, size=(length, spec.d_action)).astype(np.float32),
        language="do toy/one", task_id="toy/one", embodiment=spec.name,
    )


@torch.no_grad()
def old_probe_loss(policy, dataset, probe, device):
    """The probe exactly as it was before the per-batch split (flowcl 1528e7e)."""
    device = torch.device(device)
    gens = _generators(probe["seed_tags"], dataset.task_ids[0])
    loader = build_dataloader(dataset, batch_size=probe["batch_size"], num_workers=0,
                              shuffle=True, generator=gens["shuffle"])
    policy.eval()
    total, weight = 0.0, 0.0
    for idx, batch in enumerate(loader):
        if idx >= probe["n_batches"]:
            break
        batch = move_batch(batch, device)
        size = batch["actions"].shape[0]
        s = policy.s_sampler.sample(size, device, generator=gens["flow_time"])
        noise = draw_with_generator(tuple(batch["actions"].shape), device=device,
                                    generator=gens["noise"], dtype=torch.float32, normal=True)
        with torch.autocast(device_type=device.type, enabled=False):
            loss = float(policy(batch, s=s, noise=noise)["loss"])
        n_b = float(valid_element_count(batch["action_mask"], policy.d_action))
        total += loss * n_b
        weight += n_b
    return total / weight


def test_probe_loss_is_bitwise_the_old_probe_and_the_batches_are_its_parts():
    spec = load_embodiment_spec("libero_franka")
    episodes = [episode(spec, s) for s in range(2)]
    stats = compute_stats(episodes, embodiment=spec.name, task_id="toy/one")
    data = ChunkedActionDataset(episodes, spec, stats)
    torch.manual_seed(0)
    policy = build_policy("flowpolicy_small", spec, pretrained=False)
    for p in policy.parameters():
        if p.requires_grad:
            nn.init.normal_(p, std=0.05)
    batches = probe_batch_losses(policy, data, PROBE, "cpu")
    assert len(batches) == PROBE["n_batches"] and all(w > 0 for _, w in batches)
    old = old_probe_loss(policy, data, PROBE, "cpu")
    assert probe_loss(policy, data, PROBE, "cpu") == old  # bitwise
    assert weighted_mean(batches) == old
    assert probe_batch_losses(policy, data, PROBE, "cpu") == batches  # fixed batches


def test_a_mean_that_does_not_reproduce_the_diagnostics_is_refused():
    assert check_reproduces(0.0135, 0.0135, 1e-6, "A L[3][3]") == 0.0
    assert check_reproduces(0.0135 * (1 + 5e-7), 0.0135, 1e-6, "A L[3][3]") < 1e-6
    with pytest.raises(RuntimeError, match="does not reproduce"):
        check_reproduces(0.0135 * (1 + 2e-6), 0.0135, 1e-6, "A L[3][3]")
    with pytest.raises(ValueError, match="no valid elements"):
        weighted_mean([(1.0, 0.0)])
