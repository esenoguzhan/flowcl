"""Allocation rules at matched capacity: the matcher, the forced-k extension, and GPM's allocated mode."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from flowcl.analysis.allocation import (
    RULES,
    LayerStats,
    allocate,
    full_count,
    k_for_target,
    match_capacity,
    protected_count,
    reallocated,
    subspace_overlap,
)
from flowcl.analysis.subspace import basis_from_gram, extend_basis, residual_stats

NEG_TOL = 1e-8


def random_gram(d: int, n: int, seed: int, decay: float = 0.85) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    scales = decay ** torch.arange(d, dtype=torch.float64)
    X = torch.randn(n, d, generator=g, dtype=torch.float64) * scales
    return X.T @ X


def memory_for(d: int, seed: int, eps: float = 0.8) -> torch.Tensor:
    basis = basis_from_gram(random_gram(d, 4 * d, seed), "m", 4 * d, (eps,), neg_tol=NEG_TOL)
    return basis.basis(eps).to(torch.float64)


def stats_of(name, M, K, d_out=3):
    return LayerStats.from_residual(name, d_out, residual_stats(M, K, name, neg_tol=NEG_TOL))


def toy_stats(name, d_in, d_out, k_before, eigenvalues, proj, total=1.0, cap=None):
    lam = torch.tensor(eigenvalues, dtype=torch.float64)
    return LayerStats(name=name, d_in=d_in, d_out=d_out, k_before=k_before, proj=proj, total=total,
                      eigenvalues=lam, cap=len(eigenvalues) if cap is None else cap)


# ---- k_for_target and the forced-k extension ------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_k_for_target_is_extend_basis_arithmetic(seed):
    d = 24
    K, M = random_gram(d, 200, 100 + seed), memory_for(d, seed)
    s = stats_of("l", M, K)
    for eps in (0.95, 0.97, 0.99):
        _, info = extend_basis(M, K, eps, "l", neg_tol=NEG_TOL)
        assert k_for_target(s, eps) == info["k_added"], eps
        assert RULES["flat"].k_added(s, eps, 0.95) == info["k_added"]
    for f in (0.5, 0.7, 0.9):
        _, info = extend_basis(M, K, 0.95, "l", neg_tol=NEG_TOL, new_energy_fraction=f)
        assert RULES["novelty"].k_added(s, f, 0.95) == info["k_added"], f


@pytest.mark.parametrize("seed", [0, 1])
def test_forced_k_extension_is_bitwise_the_default_path_at_its_k(seed):
    d = 24
    K, M = random_gram(d, 200, 200 + seed), memory_for(d, 10 + seed)
    for kwargs in ({}, {"new_energy_fraction": 0.9}):
        M_def, info_def = extend_basis(M, K, 0.97, "l", neg_tol=NEG_TOL, **kwargs)
        M_k, info_k = extend_basis(M, K, 0.97, "l", neg_tol=NEG_TOL, k_added=info_def["k_added"])
        assert torch.equal(M_def, M_k)
        assert torch.equal(info_def.pop("residual_spectrum"), info_k.pop("residual_spectrum"))
        assert info_k.pop("target_fraction") is None
        info_def.pop("target_fraction")
        assert info_def == info_k


def test_forced_k_bounds_and_zero():
    d = 16
    K, M = random_gram(d, 120, 7), memory_for(d, 3)
    s = residual_stats(M, K, "l", neg_tol=NEG_TOL)
    M0, info = extend_basis(M, K, 0.95, "l", neg_tol=NEG_TOL, k_added=0)
    assert torch.equal(M0, M.to(torch.float64)) and info["k_added"] == 0
    assert torch.equal(info["residual_spectrum"], s["eigenvalues"])  # spectrum kept for k = 0
    with pytest.raises(ValueError, match="exceeds"):
        extend_basis(M, K, 0.95, "l", neg_tol=NEG_TOL, k_added=s["cap"] + 1)
    with pytest.raises(ValueError, match="non-negative"):
        extend_basis(M, K, 0.95, "l", neg_tol=NEG_TOL, k_added=-1)
    with pytest.raises(ValueError, match="exclusive"):
        extend_basis(M, K, 0.95, "l", neg_tol=NEG_TOL, k_added=1, new_energy_fraction=0.9)
    full, _ = extend_basis(M, K, 0.95, "l", neg_tol=NEG_TOL, k_added=s["cap"])
    assert full.shape[1] == M.shape[1] + s["cap"]


# ---- rules and the matcher --------------------------------------------------------------------------


def three_layers():
    out = []
    for i, (d, seed) in enumerate([(20, 1), (24, 2), (16, 3)]):
        out.append(stats_of(f"l{i}", memory_for(d, 30 + seed, eps=0.7), random_gram(d, 160, seed), d_out=4 + i))
    return out


@pytest.mark.parametrize("rule", sorted(RULES))
def test_every_rule_is_monotone_in_its_knob_direction(rule):
    stats, r = three_layers(), RULES[rule]
    if r.geometric:
        grid = np.geomspace(r.least, r.most, 60)
    else:
        grid = np.linspace(r.least, r.most, 60)
    counts = [protected_count(stats, allocate(r, stats, float(x))) for x in grid]
    assert counts == sorted(counts), rule
    assert counts[0] < counts[-1], rule


def test_size_knob_runs_reversed():
    stats = three_layers()
    small = protected_count(stats, allocate("size", stats, 1e-6))
    large = protected_count(stats, allocate("size", stats, 1e-1))
    assert small > large  # a larger delta protects less


@pytest.mark.parametrize("rule", sorted(RULES))
def test_match_picks_the_nearest_end_of_a_bracket(rule):
    stats = three_layers()
    r = RULES[rule]
    lo = protected_count(stats, allocate(r, stats, r.least))
    hi = protected_count(stats, allocate(r, stats, r.most))
    target = lo + (hi - lo) // 3
    m = match_capacity(rule, stats, target)
    assert m["status"] == "matched"
    assert m["lower"] < target <= m["upper"]
    assert m["count"] in (m["lower"], m["upper"])
    nearest = min((m["lower"], m["upper"]), key=lambda c: (abs(c - target), -c))
    assert m["count"] == nearest
    assert m["count"] == protected_count(stats, m["k_added"])
    assert m["error_pp"] == pytest.approx(100 * (m["count"] - target) / full_count(stats))
    assert m["jump_layers"]  # the layers whose k differs between the bracket's ends


def test_an_exact_tie_goes_to_the_upper_end():
    # proj 0.9: the 0.95 floor takes 2 directions, eps = 1 all 4; counts step by d_out = 2.
    s = toy_stats("a", d_in=6, d_out=2, k_before=0, eigenvalues=[0.04, 0.03, 0.02, 0.01], proj=0.9)
    lower = protected_count([s], {"a": 2})
    m = match_capacity("flat", [s], lower + 1)  # midway between two steps
    assert (m["lower"], m["upper"]) == (lower, lower + 2)
    assert m["count"] == lower + 2


def test_a_simultaneous_jump_widens_the_bracket():
    twin = dict(d_in=8, d_out=1, k_before=0, eigenvalues=[0.04, 0.03, 0.02, 0.01], proj=0.9)
    stats = [toy_stats("a", **twin), toy_stats("b", **twin)]
    m = match_capacity("flat", stats, 5)  # floor 2 + 2; both layers step together: 4 -> 6
    assert (m["lower"], m["upper"]) == (4, 6)
    assert m["jump_layers"] == ["a", "b"]
    assert abs(m["error_pp"]) > 0.05  # the 0.05 pp gate would catch this


def test_floor_and_ceiling_bounds_record_the_endpoint():
    # proj 0.5 under the 0.95 floor: the floor alone takes all three residual directions.
    s = toy_stats("a", d_in=10, d_out=1, k_before=2, eigenvalues=[0.3, 0.1, 0.05], proj=0.5, cap=3)
    floor = protected_count([s], allocate("novelty", [s], 0.0))
    ceiling = protected_count([s], allocate("novelty", [s], RULES["novelty"].most))
    assert floor == ceiling == 2 + 3
    m = match_capacity("novelty", [s], 4)
    assert m["status"] == "floor_bound" and m["count"] == 5 and m["error"] == 1
    m = match_capacity("novelty", [s], 8)  # below the registry's 10 but above what the rule reaches
    assert m["status"] == "ceiling_bound" and m["count"] == 5 and m["error"] == -3
    with pytest.raises(ValueError, match="below the current"):
        match_capacity("novelty", [s], 1)
    with pytest.raises(ValueError, match="exceeds"):
        match_capacity("novelty", [s], 11)


def test_floor_above_target_sets_floor_bound():
    # proj below the 0.95 floor: the floor alone adds directions, so the minimum exceeds a target
    # just above the current count.
    s = toy_stats("a", d_in=10, d_out=1, k_before=1, eigenvalues=[0.4, 0.3, 0.1, 0.05], proj=0.15, cap=4)
    m = match_capacity("room", [s], 2)
    assert m["floor_count"] > 2
    assert m["status"] == "floor_bound" and m["count"] == m["floor_count"]


def test_reallocated_counts_a_moved_direction_once_and_overlap():
    dims = {"a": (2, 4), "b": (2, 4)}
    assert reallocated({"a": 2, "b": 1}, {"a": 1, "b": 2}, dims) == pytest.approx(2 / 16)
    assert reallocated({"a": 2, "b": 1}, {"a": 2, "b": 1}, dims) == 0.0
    I = torch.eye(4, dtype=torch.float64)
    assert subspace_overlap({"a": I[:, :2]}, {"a": I[:, :3]}, {"a": 2}) == pytest.approx(1.0)
    assert subspace_overlap({"a": I[:, :2]}, {"a": I[:, 2:]}, {"a": 2}) == pytest.approx(0.0)


# ---- GPM's allocated mode -------------------------------------------------------------------------


@pytest.fixture(scope="module")
def spec():
    from flowcl.data.config import load_embodiment_spec

    return load_embodiment_spec("libero_franka")


@pytest.fixture()
def capture_yaml(tmp_path):
    from omegaconf import OmegaConf

    from flowcl.utils.libero_paths import repo_root

    cfg = OmegaConf.load(repo_root() / "configs" / "analysis" / "subspace.yaml")
    cfg.min_samples_per_dim = 0.01
    cfg.num_workers = 0
    cfg.batch_size = 16
    path = tmp_path / "capture.yaml"
    OmegaConf.save(cfg, path)
    return str(path)


def run_three_tasks(spec, capture_yaml, **method_kwargs):
    from flowcl.data.dataset import ChunkedActionDataset
    from flowcl.data.episode import Episode
    from flowcl.data.stats import compute_stats
    from flowcl.methods.base import TaskContext
    from flowcl.methods.gpm import GPM
    from flowcl.models.build import build_policy
    from flowcl.train.trainer import TrainConfig, train_one_task

    def episode(task, seed):
        rng = np.random.default_rng(seed)
        h, w = spec.observation.image_size
        return Episode(images={c: rng.integers(0, 255, (18, h, w, 3), dtype=np.uint8) for c in spec.cameras},
                       state=rng.normal(size=(18, spec.d_state)).astype(np.float32),
                       action=rng.uniform(-1, 1, size=(18, spec.d_action)).astype(np.float32),
                       language=f"do {task}", task_id=task, embodiment=spec.name)

    keys = ["toy/one", "toy/two", "toy/three"]
    episodes = {k: [episode(k, 10 * i + s) for s in range(2)] for i, k in enumerate(keys)}
    stats = compute_stats(episodes[keys[0]], embodiment=spec.name, task_id=keys[0])
    torch.manual_seed(0)
    policy = build_policy("flowpolicy_small", spec, pretrained=False)
    for p in policy.parameters():
        if p.requires_grad:
            nn.init.normal_(p, std=0.05)
    method = GPM(update_memory=True, capture_config=capture_yaml, log_interval=1, **method_kwargs)
    snaps = []
    for idx, key in enumerate(keys):
        data = ChunkedActionDataset(episodes[key], spec, stats)
        ctx = TaskContext(task_key=key, dataset=data, device="cpu", seed_namespace_run_id="toy_ns",
                          method_run_id="toy_run")
        cfg = TrainConfig(steps=3, batch_size=4, device="cpu", amp=False, num_workers=0, warmup_steps=1,
                          log_every=0)
        train_one_task(policy, data, cfg, method=method, task_idx=idx,
                       generator=torch.Generator().manual_seed(idx), context=ctx)
        snaps.append({n: p.detach().clone() for n, p in policy.named_parameters()})
    return policy, method, snaps, keys


def counts_of(method, task):
    return sum(h["d_out"] * h["k_after"] for h in method.memory_history[task].values())


def test_allocated_gpm_keeps_task1_and_hits_the_scheduled_counts(spec, capture_yaml, tmp_path):
    from flowcl.methods.base import TaskContext
    from flowcl.methods.gpm import GPM
    from flowcl.utils.run import file_sha256

    policy, plain, plain_snaps, keys = run_three_tasks(spec, capture_yaml)
    dims = {e.name: tuple(e.module.weight.shape) for e in policy.projectable_layers()}
    c0 = sum(dims[n][0] * plain.memory_history[0][n]["k_after"] for n in dims)
    full = sum(do * di for do, di in dims.values())
    schedule = {"toy_ns": [c0 + (full - c0) // 4, c0 + (full - c0) // 3]}
    for rule in ("novelty", "room"):
        _, alloc, snaps, _ = run_three_tasks(spec, capture_yaml, allocation_rule=rule,
                                             capacity_schedule=schedule)
        for stage in (0, 1):  # trained under no memory, then under T1's (0.95) memory
            for n, t in plain_snaps[stage].items():
                assert torch.equal(t, snaps[stage][n]), (rule, stage, n)
        for n in plain.memory_history[0]:
            assert plain.memory_history[0][n] == alloc.memory_history[0][n]
        assert sorted(alloc.allocation_history) == [1, 2]
        for task in (1, 2):
            a = alloc.allocation_history[task]
            target = schedule["toy_ns"][task - 1]
            assert a["target"] == target and a["count"] == counts_of(alloc, task)
            assert a["status"] in ("matched", "floor_bound", "ceiling_bound")
            if a["status"] == "matched":
                assert a["lower"] < target <= a["upper"]
        assert alloc.display_name == f"gpm_projected_adam_alloc_{rule}"

        ctx = TaskContext(task_key=keys[2], dataset=None, device="cpu", seed_namespace_run_id="toy_ns",
                          method_run_id="toy_run")
        memory_path = alloc.save_artifacts(tmp_path / rule, 2, context=ctx)[0]
        sha = file_sha256(memory_path)
        again = GPM(allocation_rule=rule, capacity_schedule=schedule)
        again.restore_memory(memory_path, sha, task_idx=2)
        assert again.allocation_history == alloc.allocation_history
        with pytest.raises(ValueError, match="allocation_rule"):
            GPM().restore_memory(memory_path, sha)
        with pytest.raises(ValueError, match="capacity_schedule"):
            GPM(allocation_rule=rule, capacity_schedule={"toy_ns": [schedule["toy_ns"][0] + 1,
                                                                    schedule["toy_ns"][1]]}
                ).restore_memory(memory_path, sha)


def test_allocated_gpm_needs_a_known_namespace(spec, capture_yaml):
    with pytest.raises(KeyError, match="no capacity schedule"):
        run_three_tasks(spec, capture_yaml, allocation_rule="novelty", capacity_schedule={"other_ns": [10**9]})


def test_allocation_kwargs_validation_name_and_config():
    from flowcl.methods.gpm import GPM

    sched = {"ns": [10, 20]}
    assert "allocation_rule" not in GPM().config()  # unset: the metadata plain GPM always had
    g = GPM(allocation_rule="size", capacity_schedule=sched)
    assert g.display_name == "gpm_projected_adam_alloc_size"
    assert g.config()["allocation_rule"] == "size" and g.config()["capacity_schedule"] == sched
    assert g.capacity_target("ns", 1) == 10 and g.capacity_target("ns", 2) == 20
    with pytest.raises(KeyError):
        g.capacity_target("ns", 3)
    with pytest.raises(KeyError):
        g.capacity_target("ns", 0)
    for bad in ({"allocation_rule": "novelty"}, {"capacity_schedule": sched},
                {"allocation_rule": "nope", "capacity_schedule": sched},
                {"allocation_rule": "novelty", "capacity_schedule": sched, "new_energy_fraction": 0.9},
                {"allocation_rule": "novelty", "capacity_schedule": sched, "eps_later": 0.99},
                {"allocation_rule": "novelty", "capacity_schedule": {"ns": [20, 10]}},
                {"allocation_rule": "novelty", "capacity_schedule": {"ns": [1.5]}},
                {"allocation_rule": "novelty", "capacity_schedule": {}}):
        with pytest.raises(ValueError):
            GPM(**bad)


@pytest.mark.parametrize("rule", ["novelty", "size", "room"])
def test_method_yamls_are_gpm_plus_the_rule_and_schedule(rule):
    from flowcl.data.config import load_method_config
    from flowcl.methods.gpm import GPM

    name, kwargs = load_method_config(f"gpm_alloc_{rule}")
    plain_name, plain = load_method_config("gpm")
    assert name == plain_name == "gpm"
    assert kwargs.pop("allocation_rule") == rule
    schedule = kwargs.pop("capacity_schedule")
    assert sorted(schedule) == [f"seq_hetero__seq_ft__seed{s}" for s in range(3)]
    assert all(len(c) == 3 for c in schedule.values())
    assert kwargs == plain
    assert GPM(**load_method_config(f"gpm_alloc_{rule}")[1]).display_name == f"gpm_projected_adam_alloc_{rule}"
