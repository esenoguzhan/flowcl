"""Gate 3 end to end on a tiny policy with in-memory datasets (no sim, no GPU)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch
import torch.nn as nn

from flowcl.analysis.interference import projected_energies
from flowcl.data.config import load_embodiment_spec
from flowcl.data.dataset import ChunkedActionDataset
from flowcl.data.episode import Episode
from flowcl.data.stats import compute_stats
from flowcl.experiments.gate2 import collect_bases, load_subspace_config
from flowcl.experiments.gate3 import (
    batch_generators,
    load_interference_config,
    measure_gradient_interference,
    update_interference,
)
from flowcl.models.build import build_policy
from flowcl.models.flow_head import draw_with_generator
from flowcl.train.checkpoint import LoadedCheckpoint
from flowcl.train.trainer import build_dataloader

T1, T2 = "toy_suite/task_one", "toy_suite/task_two"


@pytest.fixture(scope="module")
def spec():
    return load_embodiment_spec("libero_franka")


def make_episode(spec, task: str, length: int, seed: int) -> Episode:
    rng = np.random.default_rng(seed)
    h, w = spec.observation.image_size
    return Episode(
        images={
            c: rng.integers(0, 255, (length, h, w, 3), dtype=np.uint8)
            for c in spec.cameras
        },
        state=rng.normal(size=(length, spec.d_state)).astype(np.float32),
        action=rng.uniform(-1, 1, size=(length, spec.d_action)).astype(np.float32),
        language=f"do {task}",
        task_id=task,
        embodiment=spec.name,
    )


@pytest.fixture(scope="module")
def setup(spec):
    t1_eps = [make_episode(spec, T1, 18, s) for s in range(2)]
    t2_eps = [make_episode(spec, T2, length, 10 + s) for s, length in enumerate((17, 19))]
    stats = compute_stats(t1_eps, embodiment=spec.name, task_id=T1)  # frozen on T1
    t1_data = ChunkedActionDataset(t1_eps, spec, stats)
    t2_data = ChunkedActionDataset(t2_eps, spec, stats)

    torch.manual_seed(0)
    policy = build_policy("flowpolicy_small", spec, pretrained=False)
    for param in policy.parameters():
        if param.requires_grad:
            nn.init.normal_(param, std=0.05)
    loaded = LoadedCheckpoint(
        policy=policy, spec=spec, stats=stats,
        payload={"run_id": "toy_run", "stage": 0, "task_key": T1},
    )
    sub_cfg = dataclasses.replace(
        load_subspace_config(), min_samples_per_dim=0.01, num_workers=0, batch_size=16
    )
    subspace = collect_bases(loaded, t1_data, sub_cfg, device="cpu")
    bases = {name: layer.primary for name, layer in subspace.layers.items()}
    meta = {
        "run_id": "toy_run", "task_idx": 0, "task_key": T1,
        "stats_fingerprint": stats.fingerprint(),
    }
    # batch 5 over 36 samples: unequal final batch, and chunks with padding.
    cfg = dataclasses.replace(load_interference_config(), batch_size=5, num_workers=0)
    return loaded, bases, meta, t1_data, t2_data, cfg


@pytest.fixture(scope="module")
def result(setup):
    loaded, bases, meta, _, t2_data, cfg = setup
    return measure_gradient_interference(
        loaded, bases, meta, t2_data, cfg, device="cpu", label="t2",
        keep_full_gradient=True,
    )


def test_every_layer_gets_per_batch_c_and_a_ci(result, setup):
    loaded, _, _, _, t2_data, cfg = setup
    assert list(result.layers) == list(loaded.policy.registry_names())
    assert result.n_samples == len(t2_data)
    assert result.n_batches == -(-len(t2_data) // cfg.batch_size)
    for layer in result.layers.values():
        assert len(layer.total) == result.n_batches
        row = layer.row(cfg)
        for eps in cfg.energy_thresholds:
            assert 0.0 <= row[f"c_mean@{eps}"] <= 1.0
        c = [row[f"c_mean@{eps}"] for eps in cfg.energy_thresholds]
        assert c == sorted(c)  # nested bases: c_l non-decreasing in eps
        assert row["ci_low"] <= row[f"c_mean@{cfg.default_eps}"] <= row["ci_high"]
    assert result.layers["flow_head.action_in"].mean_c(0.95) == pytest.approx(1.0)


def test_measurement_leaves_weights_and_grads_untouched(setup):
    loaded, bases, meta, _, t2_data, cfg = setup
    policy = loaded.policy
    weight = policy.flow_head.action_out.weight
    weight.grad = torch.full_like(weight, 3.0)
    before = {n: p.detach().clone() for n, p in policy.named_parameters()}
    measure_gradient_interference(loaded, bases, meta, t2_data, cfg, device="cpu")
    for n, p in policy.named_parameters():
        assert torch.equal(p, before[n])
    torch.testing.assert_close(weight.grad, torch.full_like(weight, 3.0))
    weight.grad = None
    assert policy.trunk.state_projection.weight.grad is None


def _replay_batches(setup):
    """Re-create the measurement's exact batches, s and noise."""
    loaded, _, _, _, t2_data, cfg = setup
    gens = batch_generators(cfg, T2)
    loader = build_dataloader(
        t2_data, batch_size=cfg.batch_size, num_workers=0, shuffle=True,
        generator=gens["shuffle"],
    )
    out = []
    for batch in loader:
        size = batch["actions"].shape[0]
        s = loaded.policy.s_sampler.sample(size, torch.device("cpu"), generator=gens["flow_time"])
        noise = draw_with_generator(
            tuple(batch["actions"].shape), device="cpu", generator=gens["noise"],
            dtype=torch.float32, normal=True,
        )
        out.append((batch, s, noise))
    return out


def _weight_grads(policy, batch, s, noise):
    policy.zero_grad(set_to_none=True)
    policy(batch, s=s, noise=noise)["loss"].backward()
    grads = {e.name: e.module.weight.grad.detach().clone() for e in policy.projectable_layers()}
    policy.zero_grad(set_to_none=True)
    return grads


def test_per_batch_energies_come_from_the_weight_gradient(result, setup):
    loaded, bases, _, _, _, cfg = setup
    batch, s, noise = _replay_batches(setup)[0]
    grads = _weight_grads(loaded.policy, batch, s, noise)
    for name in ("trunk.blocks.0.mlp.fc2", "flow_head.action_out"):
        basis = bases[name]
        ranks = [basis.ranks[e] for e in cfg.energy_thresholds]
        total, _ = projected_energies(grads[name], basis.vectors[:, : max(ranks)], ranks)
        assert result.layers[name].total[0] == pytest.approx(total, rel=1e-5)


def test_full_gradient_is_loss_weighted_not_a_plain_sum(result, setup):
    """Σ n_b G_b / Σ n_b equals the gradient of one batch holding the whole dataset."""
    loaded = setup[0]
    replay = _replay_batches(setup)
    assert len({int(b["action_mask"].sum()) for b, _, _ in replay}) > 1  # unequal n_b

    batches = [b for b, _, _ in replay]
    big = {
        "images": {c: torch.cat([b["images"][c] for b in batches]) for c in batches[0]["images"]},
        "state": torch.cat([b["state"] for b in batches]),
        "actions": torch.cat([b["actions"] for b in batches]),
        "action_mask": torch.cat([b["action_mask"] for b in batches]),
        "language": sum((b["language"] for b in batches), []),
    }
    s = torch.cat([s for _, s, _ in replay])
    noise = torch.cat([n for _, _, n in replay])
    reference = _weight_grads(loaded.policy, big, s, noise)

    per_batch = [_weight_grads(loaded.policy, b, s_, n_) for b, s_, n_ in replay]
    for name in ("trunk.blocks.1.attn.v_proj", "flow_head.blocks.0.mlp.fc1", "flow_head.action_out"):
        full = result.full_gradients[name].to(torch.float32)
        torch.testing.assert_close(full, reference[name], rtol=1e-4, atol=1e-7)
        unweighted = sum(g[name] for g in per_batch) / len(per_batch)
        assert not torch.allclose(unweighted, reference[name], rtol=1e-4, atol=1e-7)


def test_flow_time_and_noise_streams_ignore_the_shuffle_stream(setup):
    loaded, bases, meta, _, t2_data, cfg = setup
    reshuffled = dataclasses.replace(
        cfg, seed_tags={**cfg.seed_tags, "shuffle": "another::shuffle"}, n_batches=3
    )
    a = measure_gradient_interference(
        loaded, bases, meta, t2_data, dataclasses.replace(cfg, n_batches=3), device="cpu"
    )
    b = measure_gradient_interference(loaded, bases, meta, t2_data, reshuffled, device="cpu")
    for sa, sb in zip(a.s_trace, b.s_trace):
        torch.testing.assert_close(sa, sb)
    first = "flow_head.action_out"
    assert a.layers[first].total != b.layers[first].total  # different batches


def test_provenance_mismatch_raises(setup):
    loaded, bases, meta, _, t2_data, cfg = setup
    for key, value in (("stats_fingerprint", "0" * 32), ("run_id", "other_run"), ("task_idx", 3)):
        with pytest.raises(ValueError, match="do not belong"):
            measure_gradient_interference(
                loaded, bases, {**meta, key: value}, t2_data, cfg, device="cpu"
            )


def test_update_interference_is_exact_on_a_known_update(setup):
    loaded, bases, _, _, _, cfg = setup
    state = {k: v.clone() for k, v in loaded.policy.state_dict().items()}
    name = "trunk.blocks.0.attn.q_proj"
    basis = bases[name]
    M = basis.basis(0.95)
    q, _ = torch.linalg.qr(torch.cat([M, torch.randn(basis.d_in, 1, dtype=M.dtype)], dim=1))
    v_perp = q[:, -1].to(torch.float32)  # orthogonal to span(M)
    after = {k: v.clone() for k, v in state.items()}
    d_out = state[f"{name}.weight"].shape[0]
    after[f"{name}.weight"] += torch.outer(torch.randn(d_out), v_perp)
    out = update_interference(state, after, {name: basis}, cfg.energy_thresholds)
    assert out["per_layer"][name]["c"]["0.95"] == pytest.approx(0.0, abs=1e-5)
    assert out["per_layer"][name]["delta_norm"] > 0


def test_verdict_and_report_serialise(result, setup, tmp_path):
    import json

    from flowcl.experiments.gate3 import Gate3Report

    cfg = setup[-1]
    verdict = result.verdict(cfg)
    assert verdict.gate == 3
    assert set(verdict.evidence["per_half"]) == {"trunk", "decoder"}
    report = Gate3Report(result=verdict, t2=result, cfg=cfg, control=result, replicate=result)
    payload = json.loads(report.save(tmp_path / "gate3.json").read_text())
    assert payload["comparisons"]["replicate_verdict"]["same_verdict"] is True
    assert payload["comparisons"]["control_below_t2"]["layers"] == {}
    per_batch = payload["t2"]["layers"][0][f"c_per_batch@{cfg.default_eps}"]
    assert len(per_batch) == result.n_batches
