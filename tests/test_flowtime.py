"""Gate 4 (flow-time characterization): rules, reproducibility, controls and wiring."""

from __future__ import annotations

import dataclasses
import itertools
import json
import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from flowcl.analysis.flowtime import (
    angle_rule,
    assert_s_independent,
    c_rule,
    overlap,
    paired_difference,
    principal_cosines,
    reproducible,
    rho_rule,
)
from flowcl.analysis.gates import gate4
from flowcl.models.flow_head import S_BINS, BinnedSSampler

# ---- principal angles ----------------------------------------------------------------------


def orthonormal(d, k, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.linalg.qr(torch.randn(d, k, generator=g, dtype=torch.float64))[0]


def test_principal_angles_on_known_subspaces():
    Q = orthonormal(6, 4)
    assert principal_cosines(Q[:, :2], Q[:, :2]) == pytest.approx([1.0, 1.0])
    assert overlap(Q[:, :2], Q[:, :2]) == pytest.approx(1.0)
    assert overlap(Q[:, :2], Q[:, 2:4]) == pytest.approx(0.0, abs=1e-12)
    theta = 0.3
    Mb = (math.cos(theta) * Q[:, :1] + math.sin(theta) * Q[:, 1:2])
    assert principal_cosines(Q[:, :1], Mb) == pytest.approx([math.cos(theta)])
    # a sub-basis lies inside the larger one: overlap normalizes by the smaller k
    assert overlap(Q[:, :1], Q[:, :3]) == pytest.approx(1.0)


# ---- the per-layer rules -------------------------------------------------------------------


def test_rho_rule_uses_the_replicate_mean_and_the_noise_term():
    ok = rho_rule([0.10, 0.10, 0.12, 0.14], [0.10, 0.10, 0.12, 0.14], 0.02, 3.0)
    assert ok["passed"] and ok["delta"] == pytest.approx(0.04) and ok["eta"] == 0.0
    noisy = rho_rule([0.10, 0.10, 0.12, 0.14], [0.12, 0.10, 0.12, 0.14], 0.02, 3.0)
    assert noisy["eta"] == pytest.approx(0.02) and noisy["threshold"] == pytest.approx(0.06)
    assert not noisy["passed"]
    at = rho_rule([0.10, 0.12, 0.10, 0.10], [0.10, 0.12, 0.10, 0.10], 0.02, 3.0)
    assert at["passed"]  # delta == floor: inclusive
    assert rho_rule([0.1, 0.13], [0.1, 0.11], 0.02, 3.0)["rho_mean"] == pytest.approx([0.1, 0.12])


def test_angle_rule_averages_all_replicate_pairs():
    res = angle_rule([0.70, 0.72, 0.74, 0.76], [0.95, 0.89], 0.10)
    assert res["ov_across"] == pytest.approx(0.73) and res["ov_within"] == pytest.approx(0.92)
    assert res["passed"]
    assert angle_rule([0.82] * 4, [0.92, 0.92], 0.10)["passed"]      # boundary inclusive
    assert not angle_rule([0.83] * 4, [0.92, 0.92], 0.10)["passed"]
    with pytest.raises(ValueError):
        angle_rule([0.7, 0.7], [0.9, 0.9], 0.10)


def paired(delta, n=40, noise=0.02, seed=0, base=0.5):
    rng = np.random.default_rng(seed)
    shared = rng.normal(0, 0.2, n)  # batch-level variation common to both bins
    low = base + shared
    high = base + shared + delta + rng.normal(0, noise, n)
    return list(low), list(high)


BOOT = dict(n_bootstrap=2000, confidence=0.95, seed=0)


def test_c_rule_needs_both_replicates_with_the_same_sign():
    both = c_rule({"A": paired(0.08), "B": paired(0.07, seed=1)}, 0.05, **BOOT)
    assert both["passed"] and both["same_sign"]
    only_a = c_rule({"A": paired(0.08), "B": paired(0.01, seed=1)}, 0.05, **BOOT)
    assert not only_a["passed"]  # passes with A only -> fails
    opposite = c_rule({"A": paired(0.08), "B": paired(-0.08, seed=1)}, 0.05, **BOOT)
    assert not opposite["passed"] and not opposite["same_sign"]
    small = c_rule({"A": paired(0.03), "B": paired(0.03, seed=1)}, 0.05, **BOOT)
    assert not small["passed"]  # CI excludes 0 but |delta| < 0.05


def test_c_rule_ci_must_exclude_zero():
    noisy = c_rule({"A": paired(0.06, noise=0.5), "B": paired(0.06, noise=0.5, seed=1)}, 0.05, **BOOT)
    rows = noisy["replicates"]
    assert any(r["low"] <= 0.0 <= r["high"] for r in rows.values()) and not noisy["passed"]


def test_pairing_detects_what_marginal_intervals_miss():
    low, high = paired(0.06, noise=0.01)
    d = paired_difference(low, high, **BOOT)
    assert d["low"] > 0.0  # the paired CI excludes 0
    from flowcl.analysis.metrics import bootstrap_ci

    a, b = bootstrap_ci(low, **BOOT), bootstrap_ci(high, **BOOT)
    assert a.high >= b.low  # the marginal CIs overlap


# ---- reproducibility and the verdict ---------------------------------------------------------


LAYERS = [f"flow_head.l{i}" for i in range(10)]


def test_reproducibility_requires_the_same_layers():
    disjoint = {"s0": set(LAYERS[:6]), "s1": set(LAYERS[4:10]), "s2": set(LAYERS[:3] + LAYERS[7:10])}
    res = reproducible(disjoint, LAYERS, 0.5)
    assert all(v == 0.6 for v in res["per_seed_fraction"].values())
    assert res["intersection"] == [] and not res["passed"]
    same = reproducible({s: set(LAYERS[:5]) for s in ("s0", "s1", "s2")}, LAYERS, 0.5)
    assert same["passed"] and same["fraction"] == 0.5 and same["jaccard"]["s0|s1"] == 1.0
    assert reproducible(disjoint, LAYERS, 0.5)["jaccard"]["s0|s1"] == pytest.approx(2 / 10)
    with pytest.raises(ValueError, match="outside"):
        reproducible({"s0": {"trunk.x"}}, LAYERS, 0.5)


@pytest.mark.parametrize("flags", list(itertools.product([True, False], repeat=3)))
def test_gate4_passes_only_when_all_criteria_pass(flags):
    criteria = {q: {"passed": f} for q, f in zip(("rho", "angles", "c"), flags)}
    result = gate4(criteria, {"min_layer_fraction": 0.5})
    assert result.passed == all(flags)
    assert result.evidence["failed_criteria"] == [q for q, f in zip(("rho", "angles", "c"), flags) if not f]
    assert ("broad, layerwise-reproducible evidence" in result.notes) == (not all(flags))


# ---- controls ----------------------------------------------------------------------------------


def test_negative_control_raises_on_s_dependent_inputs():
    X = torch.randn(8, 40, dtype=torch.float64)
    K = X @ X.T
    assert assert_s_independent("trunk.a", (40, K), (40, K.clone()), "b0A", 1e-10) == 0.0
    with pytest.raises(RuntimeError, match="N="):
        assert_s_independent("trunk.a", (40, K), (41, K), "b0A", 1e-10)
    with pytest.raises(RuntimeError, match="normalized Gram"):
        assert_s_independent("trunk.a", (40, K), (40, K * (1 + 1e-6)), "b0A", 1e-10)


def test_binned_sampler_uses_common_random_numbers():
    u = []
    for b, (low, high) in enumerate(S_BINS):
        g = torch.Generator().manual_seed(123)
        s = BinnedSSampler(b).sample(64, "cpu", generator=g)
        assert ((s >= low) & (s <= high)).all()
        u.append((s.double() - low) / (high - low))
    for other in u[1:]:
        assert torch.allclose(other, u[0], atol=1e-6)


# ---- the instrument on a tiny policy (in-memory data, no sim) ----------------------------------


T1, T2 = "toy_suite/task_one", "toy_suite/task_two"


@pytest.fixture(scope="module")
def tiny():
    from flowcl.data.config import load_embodiment_spec
    from flowcl.data.dataset import ChunkedActionDataset
    from flowcl.data.episode import Episode
    from flowcl.data.stats import compute_stats
    from flowcl.experiments.gate2 import load_subspace_config
    from flowcl.experiments.gate3 import load_interference_config
    from flowcl.models.build import build_policy
    from flowcl.train.checkpoint import LoadedCheckpoint

    spec = load_embodiment_spec("libero_franka")

    def episode(task, length, seed):
        rng = np.random.default_rng(seed)
        h, w = spec.observation.image_size
        return Episode(
            images={c: rng.integers(0, 255, (length, h, w, 3), dtype=np.uint8) for c in spec.cameras},
            state=rng.normal(size=(length, spec.d_state)).astype(np.float32),
            action=rng.uniform(-1, 1, size=(length, spec.d_action)).astype(np.float32),
            language=f"do {task}", task_id=task, embodiment=spec.name,
        )

    t1 = [episode(T1, 18, s) for s in range(2)]
    t2 = [episode(T2, n, 10 + s) for s, n in enumerate((17, 19))]
    stats = compute_stats(t1, embodiment=spec.name, task_id=T1)
    torch.manual_seed(0)
    policy = build_policy("flowpolicy_small", spec, pretrained=False)
    for p in policy.parameters():
        if p.requires_grad:
            nn.init.normal_(p, std=0.05)
    loaded = LoadedCheckpoint(policy=policy, spec=spec, stats=stats,
                              payload={"run_id": "toy_run", "stage": 0, "task_key": T1})
    scfg = dataclasses.replace(load_subspace_config(), min_samples_per_dim=0.01, num_workers=0,
                               batch_size=16)
    icfg = dataclasses.replace(load_interference_config(), batch_size=5, num_workers=0)
    return loaded, ChunkedActionDataset(t1, spec, stats), ChunkedActionDataset(t2, spec, stats), scfg, icfg


def test_capture_streams_are_separate(tiny):
    """Different capture seeds: identical trunk Grams (shared tokens and data order),
    different s-dependent Grams (s and A_0 differ)."""
    from flowcl.experiments.gate2 import capture_task_grams
    from flowcl.experiments.gate4 import s_dependent_layers, s_sampler_as

    loaded, t1, _, scfg, _ = tiny
    policy = loaded.policy
    with s_sampler_as(policy, BinnedSSampler(0)):
        a = capture_task_grams(policy, t1, scfg, "cpu", probe_seed=1, capture_seed=11)
        b = capture_task_grams(policy, t1, scfg, "cpu", probe_seed=1, capture_seed=22)
    s_names = set(s_dependent_layers(policy))
    trunk = next(n for n in a.accumulators if n.startswith("trunk.blocks."))
    action = next(n for n in a.accumulators if n in s_names)
    va, vb = a.primary_view(trunk), b.primary_view(trunk)
    assert a.accumulators[trunk].n[va] == b.accumulators[trunk].n[vb]
    assert torch.equal(a.accumulators[trunk].gram[va], b.accumulators[trunk].gram[vb])
    va, vb = a.primary_view(action), b.primary_view(action)
    assert not torch.equal(a.accumulators[action].gram[va], b.accumulators[action].gram[vb])


def test_s_sampler_swap_is_restored_even_on_error(tiny):
    from flowcl.experiments.gate4 import s_sampler_as

    policy = tiny[0].policy
    original = policy.s_sampler
    with pytest.raises(RuntimeError):
        with s_sampler_as(policy, BinnedSSampler(3)):
            assert isinstance(policy.s_sampler, BinnedSSampler)
            raise RuntimeError("boom")
    assert policy.s_sampler is original


def test_extra_bases_leave_the_main_measurement_unchanged(tiny):
    from flowcl.experiments.gate2 import collect_bases
    from flowcl.experiments.gate3 import measure_gradient_interference

    loaded, t1, t2, scfg, icfg = tiny
    sub = collect_bases(loaded, t1, scfg, device="cpu")
    bases = {n: layer.primary for n, layer in sub.layers.items()}
    meta = {"run_id": "toy_run", "task_idx": 0, "task_key": T1,
            "stats_fingerprint": loaded.stats.fingerprint()}
    plain = measure_gradient_interference(loaded, bases, meta, t2, icfg, device="cpu")
    extra = measure_gradient_interference(loaded, bases, meta, t2, icfg, device="cpu",
                                          extra_bases={"same": bases})
    for name, layer in plain.layers.items():
        assert extra.layers[name].total == layer.total
        assert extra.layers[name].parallel == layer.parallel
        assert extra.layers[name].per_batch_c(0.95, "same") == layer.per_batch_c(0.95)
    assert not plain.layers[name].extra_parallel


# ---- end to end on tiny real runs ---------------------------------------------------------------


MILK = "libero_object/pick_up_the_milk_and_place_it_in_the_basket"
SAUCE = "libero_object/pick_up_the_tomato_sauce_and_place_it_in_the_basket"
TINY_POLICY = {"d_model": 384, "n_trunk_layers": 6, "n_heads": 8, "n_decoder_layers": 2,
               "n_context_tokens": 4, "pretrained": False, "euler_steps": 2}


def test_gate4_end_to_end_on_tiny_runs(dataset_dir, tmp_path):
    from omegaconf import OmegaConf

    from flowcl.data.config import load_embodiment_spec
    from flowcl.data.curriculum import load_curriculum
    from flowcl.envs.libero_env import EvalConfig
    from flowcl.experiments.gate4 import load_flowtime_config, run_gate4, s_dependent_layers
    from flowcl.models.build import build_policy
    from flowcl.train.continual import run_continual
    from flowcl.train.trainer import TrainConfig
    from flowcl.utils.libero_paths import repo_root

    analysis = repo_root() / "configs" / "analysis"
    capture = OmegaConf.load(analysis / "subspace.yaml")
    # One demo per task (as the tiny runs trained on); the real configs use every demo.
    capture.min_samples_per_dim, capture.num_workers, capture.n_demos = 0.01, 0, 1
    OmegaConf.save(capture, tmp_path / "capture.yaml")
    interference = OmegaConf.load(analysis / "interference.yaml")
    interference.n_batches, interference.num_workers, interference.batch_size = 2, 0, 2
    interference.n_demos = 1
    OmegaConf.save(interference, tmp_path / "interference.yaml")

    pair = load_curriculum({"name": "test_pair", "tasks": [
        {"task_key": MILK, "n_demos": 1}, {"task_key": SAUCE, "n_demos": 1}]})
    runs = []
    for seed in range(3):
        result = run_continual(
            pair, method_name="seq_ft", spec=load_embodiment_spec("libero_franka"),
            policy_config=TINY_POLICY,
            train_cfg=TrainConfig(steps=2, batch_size=2, num_workers=0, device="cpu",
                                  log_every=0, warmup_steps=1),
            eval_cfg=EvalConfig(n_episodes=1), seed=seed, dataset_dir=dataset_dir,
            results_root=tmp_path, pretrained=False, evaluate=False)
        runs.append(result.run_id)

    tiny_policy = build_policy(TINY_POLICY, load_embodiment_spec("libero_franka"), pretrained=False)
    cfg = load_flowtime_config()
    cfg.update({
        "runs": runs, "capture_config": str(tmp_path / "capture.yaml"),
        "interference_config": str(tmp_path / "interference.yaml"),
        "expected_s_dependent_layers": len(s_dependent_layers(tiny_policy)),
    })
    cfg["rule"]["c"]["n_bootstrap"] = 200
    report = run_gate4(cfg, results_root=tmp_path, dataset_dir=dataset_dir, device="cpu",
                       allow_dirty=True, out=tmp_path / "gate4.json")

    assert report["gate"]["gate"] == 4 and isinstance(report["gate"]["passed"], bool)
    assert set(report["gate"]["evidence"]["criteria"]) == {"rho", "angles", "c"}
    for run in runs:
        seed = report["seeds"][run]
        nc = seed["negative_control"]
        assert nc["compared"] > 0 and nc["max_rel_gram_diff"] <= 1e-10
        assert seed["crn_max_abs_du"] <= 1e-5
        layer = seed["layers"][seed["s_dependent_layers"][0]]
        assert set(layer["rules"]) == {"rho", "angles", "c"}
        assert len(layer["overlap_matrix"]["labels"]) == 8
    written = json.loads((tmp_path / "gate4.json").read_text())
    assert written["gate"]["passed"] == report["gate"]["passed"]
