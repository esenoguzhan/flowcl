"""Gate 2 end to end on a tiny policy and an in-memory dataset (no sim, no GPU)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch
import torch.nn as nn

from flowcl.analysis.subspace import load_bases
from flowcl.data.config import load_embodiment_spec
from flowcl.data.dataset import ChunkedActionDataset
from flowcl.data.episode import Episode
from flowcl.data.stats import compute_stats
from flowcl.experiments.gate2 import (
    collect_bases,
    load_subspace_config,
    save_checkpoint_bases,
)
from flowcl.models.build import build_policy
from flowcl.train.checkpoint import LoadedCheckpoint

TASK = "toy_suite/toy_task"


@pytest.fixture(scope="module")
def spec():
    return load_embodiment_spec("libero_franka")


def make_episode(spec, length: int, seed: int) -> Episode:
    rng = np.random.default_rng(seed)
    h, w = spec.observation.image_size
    return Episode(
        images={
            c: rng.integers(0, 255, (length, h, w, 3), dtype=np.uint8)
            for c in spec.cameras
        },
        state=rng.normal(size=(length, spec.d_state)).astype(np.float32),
        action=rng.uniform(-1, 1, size=(length, spec.d_action)).astype(np.float32),
        language="pick up the toy",
        task_id=TASK,
        embodiment=spec.name,
    )


@pytest.fixture(scope="module")
def loaded(spec):
    episodes = [make_episode(spec, 20, seed) for seed in range(3)]
    stats = compute_stats(episodes, embodiment=spec.name, task_id=TASK)
    dataset = ChunkedActionDataset(episodes, spec, stats)

    torch.manual_seed(0)
    policy = build_policy("flowpolicy_small", spec, pretrained=False)
    for param in policy.parameters():
        if param.requires_grad:
            nn.init.normal_(param, std=0.05)
    checkpoint = LoadedCheckpoint(
        policy=policy,
        spec=spec,
        stats=stats,
        payload={"run_id": "toy_run", "stage": 0, "task_key": TASK},
    )
    return checkpoint, dataset


@pytest.fixture(scope="module")
def cfg():
    # Tiny data cannot reach N >= 10 d on fc2 (d = 1536); relax only that check.
    return dataclasses.replace(
        load_subspace_config(), min_samples_per_dim=0.01, num_workers=0, batch_size=16
    )


@pytest.fixture(scope="module")
def result(loaded, cfg):
    checkpoint, dataset = loaded
    return collect_bases(checkpoint, dataset, cfg, device="cpu")


def test_every_registry_layer_gets_bases(result, loaded):
    policy = loaded[0].policy
    assert list(result.layers) == list(policy.registry_names())
    for name, layer in result.layers.items():
        assert "all" in layer.views
        if layer.kind == "action_tokens":
            assert set(layer.views) == {"valid", "all"}
            assert layer.views["valid"].n_samples < layer.views["all"].n_samples
            assert layer.reachable is not None
        else:
            assert set(layer.views) == {"all"}
            assert layer.reachable is None
        for basis in layer.views.values():
            assert basis.vectors.shape[0] == layer.d_in
            values = [basis.rhos[eps] for eps in basis.thresholds]
            assert values == sorted(values)  # monotone in eps


def test_primary_view_follows_the_probe(result):
    assert not result.reachability_mismatches
    assert result.layers["flow_head.action_out"].primary_view == "valid"
    assert result.layers["flow_head.action_in"].primary_view == "all"
    assert result.layers["trunk.blocks.0.mlp.fc2"].primary_view == "all"


def test_verdict_and_summary_are_computable(result, cfg):
    verdict = result.verdict(cfg)
    assert verdict.gate == 2
    assert verdict.evidence["n_layers"] == len(result.layers)
    rows = result.summary(cfg)["layers"]
    assert rows[0]["n_primary_per_dim"] > 0


def test_too_few_samples_raises(loaded, cfg):
    checkpoint, dataset = loaded
    strict = dataclasses.replace(cfg, min_samples_per_dim=10)
    with pytest.raises(RuntimeError, match="samples per input dimension"):
        collect_bases(checkpoint, dataset, strict, device="cpu")


def test_bases_file_is_written_and_not_silently_overwritten(result, cfg, tmp_path):
    path = save_checkpoint_bases(result, cfg, results_root=tmp_path)
    assert path == tmp_path / "toy_run" / "bases" / "task0.pt"
    bases, meta = load_bases(path)
    assert list(bases) == list(result.layers)
    assert meta["task_key"] == TASK and meta["centered"] is False
    assert bases["flow_head.action_out"].meta["primary_view"] == "valid"
    with pytest.raises(FileExistsError):
        save_checkpoint_bases(result, cfg, results_root=tmp_path)


def test_config_rejects_bins_that_disagree_with_flow_head():
    base = load_subspace_config()
    with pytest.raises(ValueError, match="S_BINS"):
        dataclasses.replace(base, s_bin_edges=(0.0, 0.5, 1.0))
