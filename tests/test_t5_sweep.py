"""The T5 saturation sweep: lambda recomputation, the replay's validity gate (sign flips,
permutations, degenerate rotations, full layers), the nominal pass-through, the per-alpha rule,
selection and verdict order, the runner text-cache state, the whole sweep end to end on the tiny
policy (replay -> gate -> arms -> summary, and a failing replay), and the queue."""

from __future__ import annotations

import copy
import json
import math
import os
import subprocess
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from flowcl.data.dataset import ChunkedActionDataset
from flowcl.experiments import low_update as lu
from flowcl.experiments import t5_sweep as ts
from flowcl.utils.libero_paths import repo_root
from flowcl.utils.run import file_sha256
from test_low_update import BOOT, KEYS, episodes, fake_world, spec, stub_evaluator, tiny_policy  # noqa: F401

N_RUN = "seq_hetero__gpm_projected_adam_ne90__seed0"
NS = "seq_hetero__seq_ft__seed0"
T5 = "libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy"
TOL = {"spectrum_rtol": 1e-6, "projector_rtol": 1e-6, "cosine_atol": 1e-6, "operator_rtol": 1e-6,
       "operator_floor": 1e-12}


# ---- lambda ----------------------------------------------------------------------------------------


def rec(old, new):
    return {"sigma_old": torch.tensor(old, dtype=torch.float64), "sigma_new": torch.tensor(new, dtype=torch.float64)}


def eq2(s, a):
    return [min(1.0, (a + 1) * (x / max(s)) / (a * (x / max(s)) + 1)) for x in s]


def test_lambda_for_alpha_is_eq2_then_eq10_and_monotone_in_alpha():
    records = {0: {"L": rec([], [4.0, 2.0, 1.0])}, 1: {"L": rec([3.0, 1.0, 0.5], [2.0])}}
    a = 25.0
    first = eq2([4.0, 2.0, 1.0], a)
    second = eq2([3.0, 1.0, 0.5, 2.0], a)
    expected = [min(1.0, first[i] + second[i]) for i in range(3)] + [second[3]]
    got = ts.lambda_for_alpha(records, a)["L"]
    assert got.tolist() == pytest.approx(expected, abs=1e-12)
    lo, hi = ts.lambda_for_alpha(records, 25.0)["L"], ts.lambda_for_alpha(records, 400.0)["L"]
    assert bool((hi >= lo).all()) and bool((hi > lo).any())
    with pytest.raises(ValueError, match="old sigmas"):
        ts.lambda_for_alpha({0: {"L": rec([1.0], [2.0])}}, a)


# ---- the replay's validity gate ------------------------------------------------------------------------


def orth(d, k, seed=0):
    q, _ = torch.linalg.qr(torch.randn(d, d, dtype=torch.float64, generator=torch.Generator().manual_seed(seed)))
    return q[:, :k]


def compare(M_rep, M_ref, lam, sigma=None, sigma_ref=None, k_added=(2, 2)):
    sigma = torch.linspace(3, 1, M_ref.shape[0], dtype=torch.float64) if sigma is None else sigma
    sigma_ref = sigma if sigma_ref is None else sigma_ref
    return ts.compare_layer(M_rep, sigma, k_added[0], M_ref, sigma_ref, k_added[1],
                            {"hard": torch.ones(M_rep.shape[1], dtype=torch.float64), "a": lam}, TOL)


def test_gate_bitwise_sign_flip_and_permutation_pass():
    M = orth(6, 4)
    lam = torch.tensor([1.0, 0.9, 0.5, 0.2], dtype=torch.float64)
    same = compare(M, M.clone(), lam)
    assert same["passed"] and same["bitwise"]
    flipped = M.clone()
    flipped[:, 1] *= -1
    r = compare(flipped, M, lam)
    assert r["passed"] and not r["bitwise"]
    perm = [2, 0, 3, 1]
    r = compare(M[:, perm], M, lam[perm])  # the replay's columns (and lambdas) in another order
    assert r["passed"] and r["operator"]["a"] < 1e-12


def test_gate_degenerate_rotation_passes_only_with_equal_importances():
    M = orth(6, 4, seed=1)
    c, s = math.cos(0.7), math.sin(0.7)
    rot = M.clone()
    rot[:, 1], rot[:, 2] = c * M[:, 1] + s * M[:, 2], -s * M[:, 1] + c * M[:, 2]
    equal = compare(rot, M, torch.tensor([1.0, 0.6, 0.6, 0.2], dtype=torch.float64))
    assert equal["passed"] and equal["min_cosine"] > 1 - 1e-9
    differ = compare(rot, M, torch.tensor([1.0, 0.9, 0.3, 0.2], dtype=torch.float64))
    assert not differ["passed"] and differ["operator"]["hard"] < 1e-12 and differ["operator"]["a"] > 1e-3


def test_gate_fails_another_subspace_rank_or_spectrum():
    M = orth(6, 3)
    lam = torch.ones(3, dtype=torch.float64)
    other = torch.cat([M[:, :2], orth(6, 6)[:, 5:]], dim=1)
    other = torch.linalg.qr(other)[0]
    assert not compare(other, M, lam)["passed"]
    r = compare(orth(6, 4)[:, :3], M, lam, k_added=(3, 2))
    assert not r["passed"] and "rank" in r["reason"]
    sigma = torch.linspace(3, 1, 6, dtype=torch.float64)
    r = compare(M.clone(), M, lam, sigma=sigma, sigma_ref=sigma * (1 + 1e-4))
    assert not r["passed"] and not r["bitwise"] and r["spectrum"] > 1e-6


def test_gate_compares_a_fully_occupied_hard_layer_without_a_zero_denominator():
    M = orth(5, 5)
    flipped = M.clone()
    flipped[:, 0] *= -1
    r = compare(flipped, M, torch.ones(5, dtype=torch.float64))  # P = I - M M^T = 0, Q = I
    assert r["passed"] and r["operator"]["hard"] < 1e-12


# ---- nominal pass-through, the rule, selection, the verdict ------------------------------------------


def test_nominal_passthrough_hand_case():
    M = torch.eye(3, dtype=torch.float64)[:, :2]                 # memory = the first two axes
    K = torch.diag(torch.tensor([2.0, 1.0, 1.0], dtype=torch.float64))
    lam = {"x": {"trunk.a": torch.tensor([1.0, 0.5], dtype=torch.float64)},
           "h": {"trunk.a": torch.ones(2, dtype=torch.float64)}}
    n = ts.nominal_passthrough({"trunk.a": M}, lam, {"trunk.a": K}, ratio=0.1)
    h = n["halves"]["trunk"]
    assert h["free_fraction"] == pytest.approx(0.25) and h["passthrough"]["x"] == pytest.approx(0.125)
    assert h["passthrough"]["h"] == 0.0
    assert n["effectively_hard"] == {"x": False, "h": True}


def paired(diff, low, high):
    return {"diff": diff, "low": low, "high": high}


def test_judge_alpha_rule_and_borderline():
    rule = {"t5_gain": 0.10, "retention_tolerance": 0.10}
    old = {"0": {"alpha": 0.86, "hard": 0.96, "paired": paired(-0.10, -0.2, 0.0)},   # exactly -10 pp
           "1": {"alpha": 0.80, "hard": 0.82, "paired": paired(-0.02, -0.12, 0.08)}}
    j = ts.judge_alpha(paired(0.10, 0.02, 0.2), {"high": -0.01}, old, rule)
    assert j["promising"] and j["retention_borderline"] == ["0", "1"]
    assert not ts.judge_alpha(paired(0.10, 0.0, 0.2), {"high": -0.01}, old, rule)["promising"]   # CI at 0
    assert not ts.judge_alpha(paired(0.08, 0.02, 0.2), {"high": -0.01}, old, rule)["promising"]  # too small
    assert not ts.judge_alpha(paired(0.20, 0.1, 0.3), {"high": 0.01}, old, rule)["promising"]    # loss
    bad = {**old, "2": {"alpha": 0.70, "hard": 0.82, "paired": paired(-0.12, -0.25, -0.01)}}
    j = ts.judge_alpha(paired(0.20, 0.1, 0.3), {"high": -0.1}, bad, rule)
    assert not j["promising"] and not j["retention"]["2"]["passed"]


def test_selection_ties_then_verdict_order():
    judged = {25.0: {"promising": True, "gain": 0.2, "l5": 0.02},
              100.0: {"promising": True, "gain": 0.2, "l5": 0.01},
              400.0: {"promising": True, "gain": 0.2, "l5": 0.01},
              800.0: {"promising": False, "gain": 0.5, "l5": 0.001}}
    assert ts.select_alpha(judged) == 400.0
    assert ts.select_alpha({25.0: {"promising": False, "gain": 0.3, "l5": 0.1}}) is None
    up, flat = {"low": 0.1, "high": 0.3}, {"low": -0.1, "high": 0.2}
    hard = {25.0: False, 100.0: True, 400.0: True}
    soft = {25.0: False, 100.0: False, 400.0: True}
    assert ts.classify(False, True, 100.0, up, hard, 100)["verdict"] == ts.REPLAY_INVALID
    assert ts.classify(True, False, 100.0, up, hard, 100)["verdict"] == ts.INCOMPLETE
    assert ts.classify(True, True, 100.0, flat, hard, 100)["verdict"] == ts.ALPHA_FOUND  # ahead of the premise
    assert ts.classify(True, True, None, flat, hard, 100)["verdict"] == ts.NO_COST
    assert ts.classify(True, True, None, up, hard, 100)["verdict"] == ts.ACCUMULATION
    assert ts.classify(True, True, None, up, soft, 100)["verdict"] == ts.NONE_PASSES


# ---- the runner's text-cache state ----------------------------------------------------------------------


def test_runner_cache_encodes_task_zero_under_amp_and_the_rest_in_fp32(spec, monkeypatch):
    calls = []
    real = torch.autocast

    class Recording(real):
        def __init__(self, device_type, enabled=True, **kw):
            calls.append((device_type, enabled))
            super().__init__(device_type="cpu", enabled=False)

    monkeypatch.setattr(torch, "autocast", Recording)
    policy = tiny_policy(spec)
    stats = __import__("flowcl.data.stats", fromlist=["compute_stats"]).compute_stats(
        episodes(spec, KEYS[0]), embodiment=spec.name, task_id=KEYS[0])
    data = {j: ChunkedActionDataset(episodes(spec, KEYS[j], offset=10 * j), spec, stats) for j in (0, 2)}
    out = lu.warm_runner_cache(policy, data, "cuda", amp=True)
    assert calls == [("cuda", True), ("cuda", False)]          # task 0 under AMP, task 2 in fp32
    assert out[0] == {"texts": [f"do {KEYS[0]}"], "precision": "amp"} and out[2]["precision"] == "fp32"
    assert sorted(policy.text_encoder._cache) == sorted([f"do {KEYS[0]}", f"do {KEYS[2]}"])
    calls.clear()
    lu.warm_runner_cache(tiny_policy(spec), data, "cuda", amp=False)
    assert calls == [("cuda", False), ("cuda", False)]


# ---- a tiny world: adaptive GPM's four stages, with real memories ---------------------------------------


def tiny_capture(tmp_path):
    cfg = OmegaConf.load(repo_root() / "configs" / "analysis" / "subspace.yaml")
    cfg.min_samples_per_dim = 0.01
    cfg.num_workers = 0
    cfg.batch_size = 4
    cfg.probe_batch_size = 4
    path = tmp_path / "tiny_subspace.yaml"
    OmegaConf.save(cfg, path)
    return path


def world(tmp_path, spec):
    """fake_world's references; adaptive GPM's four stages trained nowhere but holding a real
    adaptive memory, built stage by stage exactly as the runner's memory update does."""
    from flowcl.methods.base import TaskContext
    from flowcl.methods.gpm import GPM
    from flowcl.train.checkpoint import load_checkpoint, save_checkpoint

    luc, build4 = fake_world(tmp_path, spec, seeds=(0,))
    keys = [*KEYS, T5]

    def build(stage):
        offset = 10 * keys.index(stage.task_key)
        stats = load_checkpoint(tmp_path / NS / "checkpoints" / "stage0.pt").stats
        return ChunkedActionDataset(episodes(spec, stage.task_key, offset=offset), spec, stats)

    capture = tiny_capture(tmp_path)
    run = tmp_path / N_RUN
    base = load_checkpoint(tmp_path / NS / "checkpoints" / "stage0.pt")
    gpm = GPM(eps=0.95, new_energy_fraction=0.9, update_memory=True, capture_config=str(capture))
    stages = [type("S", (), {"task_key": k, "n_demos": 3})() for k in keys]
    artifacts = []
    for t in range(4):
        policy = tiny_policy(spec)
        with torch.no_grad():
            for i, p in enumerate(p for p in policy.parameters() if p.requires_grad):
                p.add_(0.01 * (t + 1) * torch.sin(torch.arange(p.numel(), dtype=torch.float32) + i).view_as(p))
        path = save_checkpoint(run / "checkpoints" / f"stage{t}.pt", policy=policy,
                               policy_config=base.payload["policy_config"], spec=base.spec, stats=base.stats,
                               run_id=N_RUN, stage=t, task_key=keys[t])
        policy = load_checkpoint(path).policy
        ds = build(stages[t])
        lu.warm_runner_cache(policy, {t: ds}, "cpu", amp=t == 0)
        ctx = TaskContext(task_key=keys[t], dataset=ds, device="cpu", seed_namespace_run_id=NS,
                          method_run_id=N_RUN)
        gpm._update_memory(policy, t, ctx)
        gpm.save_artifacts(run / "method", t, context=ctx)
        artifacts.append([{"path": f"method/memory_task{t}.pt",
                           "sha256": file_sha256(run / "method" / f"memory_task{t}.pt")}])
    train = OmegaConf.to_container(OmegaConf.load(tmp_path / NS / "config.yaml").train)
    OmegaConf.save(OmegaConf.create({"method": {"name": "gpm", "eps": 0.95, "new_energy_fraction": 0.9,
                                                "capture_config": capture.stem}, "train": train}),
                   run / "config.yaml")
    (run / "result.json").write_text(json.dumps({
        "seed_namespace_run_id": NS,
        "stages": [{"stage": t, "method_artifacts": artifacts[t]} for t in range(4)]}))
    (tmp_path / "gate0_t5").mkdir()
    (tmp_path / "gate0_t5" / "gate0.json").write_text(json.dumps(
        {"evidence": {"per_task": {T5: {"success_rate": 0.9}}}}))

    cfg = copy.deepcopy(ts.load_sweep_config())
    cfg["curriculum"] = {"name": "tiny_t5", "tasks": [{"task_key": k, "n_demos": 3} for k in keys]}
    cfg["recipe"] = {"steps": 3, "lr": 1e-3}
    cfg["source"]["capture_config"] = str(capture)
    paths = ts.source_paths(cfg, tmp_path)
    cfg["source"]["sha256"] = {k: file_sha256(paths[k]) for k in cfg["source"]["sha256"]}
    cfg["t5"]["gate0"]["sha256"] = file_sha256(paths["gate0"])
    return cfg, luc, build


def kwargs(tmp_path, build, luc):
    return dict(results_root=tmp_path, device="cpu", build_datasets=build, low_update_cfg=luc)


def test_replay_arms_and_summary_end_to_end(spec, tmp_path):
    cfg, luc, build = world(tmp_path, spec)
    kw = kwargs(tmp_path, build, luc)
    with pytest.raises(RuntimeError, match="replay gate"):
        ts.run_arm(cfg, "hard", allow_dirty=True, evaluator=stub_evaluator, bootstrap=BOOT, **kw)
    r = ts.run_replay(cfg, allow_dirty=True, **kw)
    assert r["gate"]["passed"] and r["gate"]["bitwise"] and r["gate"]["identity"]
    assert r["cache_state"]["0"]["texts"] == [f"do {KEYS[0]}"]
    assert set(r["nominal"]["effectively_hard"]) == {"hard", "25", "100", "400", "800", "1600"}
    assert r["nominal"]["effectively_hard"]["hard"] is True
    assert ts.replay_open(cfg, tmp_path)
    summary, records, bases = ts.load_replay(cfg, tmp_path)
    from flowcl.analysis.subspace import load_bases
    reference, _ = load_bases(tmp_path / N_RUN / "method" / "memory_task3.pt")
    assert sorted(bases) == sorted(reference)
    assert all(torch.equal(bases[n].vectors, reference[n].vectors) for n in reference)
    assert ts.run_replay(cfg, allow_dirty=True, **kw)["memory_sha256"] == summary["memory_sha256"]  # reused

    rows = {a: ts.run_arm(cfg, a, allow_dirty=True, evaluator=stub_evaluator, bootstrap=BOOT, **kw)
            for a in cfg["arms"]}
    for a, row in rows.items():
        assert row["stage"] == 4 and set(row["cells"]) == {"0", "1", "2", "3", "4"}
        assert row["warm_text_cache"] == "runner" and row["cells"]["4"]["paired"] == {}
        assert row["method"]["replay_memory_sha256"] == summary["memory_sha256"]
    assert rows["hard"]["update_split"]["trunk"]["memory"] < 1e-2 * rows["hard"]["update_split"]["trunk"]["total"]
    assert rows["free"]["update_split"]["trunk"]["memory"] > 0

    s = ts.run_summary(cfg, results_root=tmp_path, device="cpu", build_datasets=build,
                       low_update_cfg=luc, bootstrap=BOOT)
    v = s["verdict"]["verdict"]
    assert v in (ts.ALPHA_FOUND, ts.NO_COST, ts.ACCUMULATION, ts.NONE_PASSES) and not s["skipped_arms"]
    assert set(s["judged"]) == {"25", "100", "400", "800", "1600"}
    eh = {float(a): x for a, x in s["replay"]["nominal"]["effectively_hard"].items() if a != "hard"}
    sel = ts.select_alpha({float(a): j for a, j in s["judged"].items()})
    assert ts.classify(True, True, sel, s["premise_gap"], eh, 100)["verdict"] == v
    assert s["t5_threshold"]["gate0_success"] == 0.9 and set(s["arms"]["hard"]["change_from_start"]) == {"0", "1", "2", "3"}
    assert s["pareto"]["success"] and s["pareto"]["probe"]


def test_a_failing_replay_closes_the_sweep(spec, tmp_path):
    from flowcl.analysis.subspace import load_bases, save_bases

    cfg, luc, build = world(tmp_path, spec)
    path = tmp_path / N_RUN / "method" / "memory_task1.pt"
    bases, meta = load_bases(path)
    name = next(n for n, b in bases.items() if 0 < b.vectors.shape[1] < b.vectors.shape[0])
    import dataclasses
    v = bases[name].vectors.clone()
    v[:, 0] = torch.linalg.qr(torch.cat([v[:, 1:], torch.ones(v.shape[0], 1, dtype=v.dtype)], dim=1))[0][:, -1]
    bases[name] = dataclasses.replace(bases[name], vectors=v)       # a different subspace
    save_bases(path, bases, meta)
    cfg["source"]["sha256"]["memory1"] = file_sha256(path)
    result = json.loads((tmp_path / N_RUN / "result.json").read_text())
    result["stages"][1]["method_artifacts"][0]["sha256"] = file_sha256(path)
    (tmp_path / N_RUN / "result.json").write_text(json.dumps(result))
    cfg["source"]["sha256"]["result"] = file_sha256(tmp_path / N_RUN / "result.json")
    kw = kwargs(tmp_path, build, luc)
    r = ts.run_replay(cfg, allow_dirty=True, **kw)
    assert not r["gate"]["passed"] and name in r["gate"]["failed_layers"]["1"]
    assert not ts.replay_open(cfg, tmp_path)
    with pytest.raises(RuntimeError, match="replay gate"):
        ts.run_arm(cfg, "hard", allow_dirty=True, evaluator=stub_evaluator, bootstrap=BOOT, **kw)
    s = ts.run_summary(cfg, results_root=tmp_path, device="cpu", build_datasets=build, low_update_cfg=luc,
                       bootstrap=BOOT)
    assert s["verdict"]["verdict"] == ts.REPLAY_INVALID and len(s["skipped_arms"]) == 7


def test_start_checks(spec, tmp_path):
    cfg, luc, build = world(tmp_path, spec)
    ctx = lu.seed_context(luc, 0, tmp_path, extend_curriculum=cfg["curriculum"])
    assert ts.check_start(cfg, ctx, tmp_path)["passed"]
    for mutate, match in ((lambda c: c["t5"]["gate0"].__setitem__("sha256", "PENDING_GATE0"), "gate0"),
                          (lambda c: c["source"]["sha256"].__setitem__("stage2", "0" * 64), "sha256.stage2"),
                          (lambda c: c["t5"].__setitem__("task_key", KEYS[0]), "curriculum")):
        bad = copy.deepcopy(cfg)
        mutate(bad)
        with pytest.raises(ValueError, match=match):
            ts.check_start(bad, ctx, tmp_path)
    with pytest.raises(ValueError, match="does not extend"):
        lu.seed_context(luc, 0, tmp_path, extend_curriculum={"name": "x", "tasks": [
            {"task_key": k, "n_demos": 3} for k in [KEYS[1], KEYS[0], KEYS[2], KEYS[3], T5]]})


def test_config_is_consistent():
    cfg = ts.load_sweep_config()
    assert list(cfg["arms"]) == ["hard", "free", "sgp_a25", "sgp_a100", "sgp_a400", "sgp_a800", "sgp_a1600"]
    assert ts.alphas(cfg) == [25.0, 100.0, 400.0, 800.0, 1600.0] and cfg["t5"]["stage"] == 4
    assert set(cfg["source"]["sha256"]) == {f"stage{t}" for t in range(4)} | {f"memory{t}" for t in range(4)} | {"result"}
    cur = OmegaConf.load(repo_root() / "configs" / "curriculum" / f"{cfg['curriculum']}.yaml")
    base = OmegaConf.load(repo_root() / "configs" / "curriculum" / "seq_hetero.yaml")
    assert [t["task_key"] for t in cur.tasks][:4] == [t["task_key"] for t in base.tasks]
    assert cur.tasks[4]["task_key"] == cfg["t5"]["task_key"]
    used = {cfg["t5"]["task_key"], cfg["t5"]["backup"], *[t["task_key"] for t in base.tasks]}
    follow = [v for stage in cfg["follow_up"].values() for v in stage.values()]
    assert len(set(follow)) == 6 and not used & set(follow)
    for key in [cfg["t5"]["task_key"], cfg["t5"]["backup"], *follow]:
        suite, name = key.split("/")
        assert (repo_root() / "benchmark" / "LIBERO" / "libero" / "libero" / "bddl_files" / suite
                / f"{name}.bddl").is_file(), key


# ---- the queue -------------------------------------------------------------------------------------------

QUEUE = repo_root() / "scripts" / "queue_t5_sweep.sh"
STEPS = ["0_replay", "1_replay_gate", "2_arm_hard", "3_arm_free", "4_arm_sgp_a25", "5_arm_sgp_a100",
         "6_arm_sgp_a400", "7_arm_sgp_a800", "8_arm_sgp_a1600", "9_summary"]


def queue(tmp_path, args=(), fail="", present=()):
    results = tmp_path / "results"
    for rel in present:
        p = results / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}")
    env = {**os.environ, "QUEUE_DRY_RUN": "1", "QUEUE_FAIL_STEP": fail,
           "QUEUE_LOG_ROOT": str(tmp_path / "logs"), "QUEUE_RESULTS_ROOT": str(results)}
    done = subprocess.run(["bash", str(QUEUE), *args], env=env, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    (logdir,) = list((tmp_path / "logs").glob("queue_*_t5_sweep"))
    return (logdir / "queue.log").read_text(), logdir


def ran(log, step):
    return f"START {step}:" in log


def test_queue_order_gate_and_resume(tmp_path):
    assert subprocess.run(["bash", "-n", str(QUEUE)]).returncode == 0
    log, logdir = queue(tmp_path / "a")
    assert [log.index(f"START {s}:") for s in STEPS] == sorted(log.index(f"START {s}:") for s in STEPS)
    assert (logdir / "6_arm_sgp_a400.log").read_text().strip().endswith("--part arm --arm sgp_a400")
    log, _ = queue(tmp_path / "b", fail="1_replay_gate")
    assert not any(ran(log, s) for s in STEPS[2:-1]) and ran(log, "9_summary") and "SKIP 2-8" in log
    log, _ = queue(tmp_path / "c", fail="4_arm_sgp_a25")
    assert all(ran(log, s) for s in STEPS)
    log, _ = queue(tmp_path / "d", args=("--from-step", "4"),
                   present=["t5_sweep/replay/replay.json", "t5_sweep/arms/t5__hard__seed0/row.json",
                            "t5_sweep/arms/t5__free__seed0/row.json"])
    assert "PRIOR 0_replay ok" in log and "PRIOR 3_arm_free ok" in log and ran(log, "1_replay_gate")
    assert not ran(log, "2_arm_hard") and ran(log, "4_arm_sgp_a25")
    for bad in (("--from-step", "10"), ("--bogus",)):
        assert subprocess.run(["bash", str(QUEUE), *bad], capture_output=True).returncode == 2
