"""The optimizer study: the schedule option, the settings, reused-run verification, the probe
units and update size, the pure analyses, an end-to-end map on the tiny policy, the queue."""

from __future__ import annotations

import copy
import json
import math
import os
import subprocess

import pytest
import torch

from flowcl.experiments import optimizer_study as os_
from flowcl.train.trainer import TrainConfig, build_scheduler
from flowcl.utils.libero_paths import repo_root
from test_low_update import BOOT, KEYS, PROBE, fake_world, spec, stub_evaluator  # noqa: F401

# ---- the trainer's schedule option ---------------------------------------------------------------


def old_lambda(step, steps, warmup):
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    if steps <= warmup:
        return 1.0
    progress = (step - warmup) / (steps - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def lambdas(cfg):
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    sched = build_scheduler(opt, cfg)
    return [sched.lr_lambdas[0](s) for s in range(cfg.steps + 5)]


def test_cosine_is_unchanged_and_constant_holds_the_peak():
    cfg = TrainConfig(steps=300, warmup_steps=100)
    assert cfg.schedule == "cosine"
    assert lambdas(cfg) == [old_lambda(s, 300, 100) for s in range(305)]  # bitwise
    const = lambdas(TrainConfig(steps=300, warmup_steps=100, schedule="constant"))
    assert const[:100] == [(s + 1) / 100 for s in range(100)] and set(const[100:]) == {1.0}
    old = {k: v for k, v in TrainConfig().__dict__.items() if k != "schedule"}
    assert TrainConfig(**old).schedule == "cosine"  # configs written before the field load
    with pytest.raises(ValueError, match="schedule"):
        lambdas(TrainConfig(steps=10, schedule="linear"))


# ---- settings and reused runs --------------------------------------------------------------------


def test_settings_match_the_plan():
    cfg = os_.load_study_config()
    s = os_.settings(cfg)
    assert len(s["core"]) == 7 and [v["name"] for v in s["variants"]] == [
        "constant_same_peak", "wd0", "wd1e-2", "batch32", "batch128"]
    for v in s["variants"]:
        r = v["recipe"]
        if v["name"].startswith("batch"):
            assert r["batch_size"] * r["steps"] == 320_000 and r["lr"] == 3e-5
    assert os_.source_of(cfg, s["standard"], 1)["kind"] == "standard"
    assert os_.source_of(cfg, s["anchor"], 2) == {"kind": "reused", "run": "low_update/seed2/stage1"}
    counts = {sd: len(os_.runs_to_train(cfg, sd, "grid")) for sd in (0, 1, 2)}
    assert counts == {0: 3, 1: 5, 2: 5}
    assert len(os_.runs_to_train(cfg, 2, "variants")) == 0 and len(os_.runs_to_train(cfg, 1, "variants")) == 5


class Ctx:
    namespace = "seq_hetero__seq_ft__seed0"
    ref_dir = type("P", (), {"name": "seq_hetero__seq_ft__seed0"})()


def conf(**train):
    return {"train": {"lr": 3e-5, "steps": 5000, "batch_size": 64, "weight_decay": 1e-4, **train},
            "stream_seed": {"namespace": Ctx.namespace, "stage": 1},
            "reference_run": Ctx.namespace, "start_checkpoint": "stage0.pt"}


def test_reused_runs_are_verified_with_the_schedule_normalised():
    cfg = os_.load_study_config()
    r = os_.recipe(cfg, lr=3e-5, steps=5000)
    assert os_.verify_reused(conf(), r, Ctx) == {"schedule_normalised": True}
    assert os_.verify_reused(conf(schedule="cosine"), r, Ctx) == {"schedule_normalised": False}
    low_update_style = {**conf(), "parent": {"path": "/x/stage0.pt"}}
    del low_update_style["start_checkpoint"]
    os_.verify_reused(low_update_style, r, Ctx)
    for bad in (conf(schedule="constant"), conf(lr=1e-4), conf(batch_size=32), conf(steps=15000)):
        with pytest.raises(ValueError, match="does not match"):
            os_.verify_reused(bad, r, Ctx)
    with pytest.raises(ValueError, match="stream_seed"):
        os_.verify_reused({**conf(), "stream_seed": {"namespace": "x", "stage": 1}}, r, Ctx)


# ---- measurements -------------------------------------------------------------------------------


def test_probe_tolerance_accepts_reduction_noise_only():
    tol = {"rtol": 1e-5, "atol": 1e-8}
    os_.check_probe(0.0123 * (1 + 3e-6), 0.0123, tol, "x")
    with pytest.raises(RuntimeError, match="does not reproduce"):
        os_.check_probe(0.0123 * (1 + 1e-3), 0.0123, tol, "x")


def test_update_size_hand_case_and_frozen_guard():
    t = torch.tensor
    start = {"trunk.a.weight": t([3.0, 4.0]), "flow_head.b.weight": t([0.0, 1.0]),
             "trunk.norm.weight": t([1.0]), "encoder.w": t([5.0])}
    end = {"trunk.a.weight": t([3.0, 4.0]) + t([0.0, 5.0]), "flow_head.b.weight": t([0.0, 1.0]),
           "trunk.norm.weight": t([2.0]), "encoder.w": t([5.0])}
    trainable = {"trunk.a.weight", "flow_head.b.weight", "trunk.norm.weight"}
    registry = {"trunk.a.weight", "flow_head.b.weight"}
    u = os_.update_size(start, end, trainable, registry)
    assert u["registry_trunk"]["delta_norm"] == pytest.approx(5.0)
    assert u["registry_trunk"]["relative"] == pytest.approx(1.0)
    assert u["registry_decoder"]["delta_norm"] == 0.0
    assert u["non_registry"]["delta_norm"] == pytest.approx(1.0)
    assert u["trainable"]["delta_norm"] == pytest.approx(math.sqrt(26))
    assert u["trainable"]["relative"] == pytest.approx(math.sqrt(26 / 27))
    end["encoder.w"] = t([6.0])
    with pytest.raises(RuntimeError, match="frozen parameters moved"):
        os_.update_size(start, end, trainable, registry)


def test_matched_batch_difference():
    a = {"losses": [0.2, 0.4, 0.3], "weights": [10.0, 30.0, 20.0]}
    b = {"losses": [0.1, 0.1, 0.1], "weights": [10.0, 30.0, 20.0]}
    d = os_.matched_batch_diff(a, b, BOOT)
    assert d["diff"] == pytest.approx((0.1 * 10 + 0.3 * 30 + 0.2 * 20) / 60)
    assert d["low"] <= d["diff"] <= d["high"] and d["unit"] == "matched probe batch"
    with pytest.raises(ValueError, match="not matched"):
        os_.matched_batch_diff(a, {**b, "weights": [10.0, 31.0, 20.0]}, BOOT)


# ---- the pure analyses --------------------------------------------------------------------------------


def test_pareto_and_the_robust_frontier():
    pts = {"a": (0.9, 0.1), "b": (0.5, 0.5), "c": (0.4, 0.4), "d": (0.5, 0.5), "e": (0.1, 0.9)}
    assert os_.pareto(pts, (True, True)) == ["a", "b", "d", "e"]  # ties kept, c dominated
    assert os_.pareto({"x": (0.1, 0.2), "y": (0.2, 0.1), "z": (0.3, 0.3)}, (False, False)) == ["x", "y"]
    by = {"s1": {0: (0.9, 0.5, 0.01, 0.1), 1: (0.6, 0.3, 0.02, 0.2)},
          "s2": {0: (0.7, 0.4, 0.01, 0.1), 1: (0.7, 0.4, 0.01, 0.1)}}
    rf = os_.robust_frontier(by)
    assert rf["success"]["points"]["s1"] == (0.6, 0.3) and rf["success"]["pareto"] == ["s2"]
    assert rf["loss"]["points"]["s1"] == (0.02, 0.2) and rf["loss"]["pareto"] == ["s2"]


def vrow(obj, obj_ref, low):
    return {"cells": {"0": {"success": 0.3, "paired": {"seq_ft": {"diff": 0.3, "low": low, "high": 0.5}}},
                      "1": {"success": obj, "reference_success": {"seq_ft": obj_ref}}}}


def test_existence_needs_a_core_setting_on_every_seed():
    good = vrow(0.8, 0.9, 0.1)
    rows = {"core_ok": {0: good, 1: good, 2: good},
            "core_fail": {0: good, 1: good, 2: vrow(0.7, 0.9, 0.1)},
            "core_missing": {0: good, 1: good},
            "variant": {0: good, 1: good, 2: good}}
    e = os_.existence(rows, ["core_ok", "core_fail", "core_missing"], [0, 1, 2], 0.10)
    assert e["core_ok"]["achieves_both"] and not e["core_fail"]["achieves_both"]
    assert not e["core_missing"]["eligible"] and "missing seeds [2]" in e["core_missing"]["reason"]
    assert not e["variant"]["eligible"] and e["variant"]["reason"] == "not a core setting"


@pytest.mark.parametrize("intervals, expected", [
    ([(0.2, 0.1, 0.3), (0.1, 0.05, 0.2)], "supported_both_seeds"),
    ([(0.2, 0.1, 0.3), (0.1, -0.05, 0.2)], "supported_one_seed"),
    ([(-0.2, -0.3, -0.1), (-0.1, -0.2, 0.05)], "supported_one_seed"),
    ([(0.2, -0.1, 0.3), (0.1, -0.05, 0.2)], "directionally_consistent"),
    ([(0.2, 0.1, 0.3), (-0.1, -0.2, -0.05)], "inconclusive"),
    ([(0.0, -0.1, 0.1), (0.1, 0.05, 0.2)], "inconclusive"),
])
def test_one_factor_labels_are_exclusive(intervals, expected):
    iv = [{"diff": d, "low": lo, "high": hi} for d, lo, hi in intervals]
    assert os_.label(iv) == expected
    assert os_.one_factor({"v": {"m": iv}})["v"]["m"]["label"] == expected


# ---- end to end on the tiny policy ----------------------------------------------------------------


def small_study(tmp_path, spec):
    """A two-seed fake world with a standard stage-1 checkpoint whose probe losses are real."""
    from flowcl.analysis.probes import probe_loss
    from flowcl.train.checkpoint import load_checkpoint, save_checkpoint

    luc, build = fake_world(tmp_path, spec, seeds=(0, 1))
    for s in (0, 1):
        ref = tmp_path / f"seq_hetero__seq_ft__seed{s}"
        loaded = load_checkpoint(ref / "checkpoints" / "stage0.pt")
        save_checkpoint(ref / "checkpoints" / "stage1.pt", policy=loaded.policy,
                        policy_config=loaded.payload["policy_config"], spec=loaded.spec,
                        stats=loaded.stats, run_id=ref.name, stage=1, task_key=KEYS[1])
        stages = [type("S", (), {"task_key": k, "n_demos": 3})() for k in KEYS[:2]]
        real = [probe_loss(loaded.policy, build(stages[j]), PROBE, "cpu") for j in (0, 1)]
        path = tmp_path / luc["seeds"][s]["diagnostics"]["gpm"]
        diag = json.loads(path.read_text())
        diag["loss_matrix"]["reference"][1][:2] = real
        path.write_text(json.dumps(diag))
    cfg = copy.deepcopy(os_.load_study_config())
    cfg.update(seeds=[0, 1], variant_seeds=[0, 1], reused=[],
               base={"batch_size": 2, "weight_decay": 1e-4, "schedule": "cosine"},
               core=[{"lr": 3e-5, "steps": 2}, {"lr": 1e-5, "steps": 2}, {"lr": 1e-4, "steps": 4}],
               standard={"lr": 1e-4, "steps": 4}, anchor={"lr": 3e-5, "steps": 2},
               variants=[{"name": "constant_same_peak", "schedule": "constant"},
                         {"name": "batch4", "batch_size": 4, "steps": 1}],
               batch_bootstrap={"n_resamples": 200, "seed": 0, "confidence": 0.95})
    return cfg, luc, build


def test_map_end_to_end(spec, tmp_path):
    cfg, luc, build = small_study(tmp_path, spec)
    common = dict(results_root=tmp_path, device="cpu", allow_dirty=True, evaluator=stub_evaluator,
                  bootstrap=BOOT, build_datasets=build, low_update_cfg=luc)
    for seed in (0, 1):
        grid = os_.run_seed(cfg, seed, "grid", **common)
        assert len(grid) == 2 and all(r["train_overrides"]["batch_size"] == 2 for r in grid)
        variants = os_.run_seed(cfg, seed, "variants", **common)
        assert {r["train_overrides"]["schedule"] for r in variants} == {"constant", "cosine"}
        assert (tmp_path / "optimizer_study" / f"done_grid_seed{seed}.json").is_file()
    # A published run with a different recipe is refused, never overwritten.
    from flowcl.experiments.low_update import seed_context, train_stage

    run_id = os_.source_of(cfg, os_.settings(cfg)["anchor"], 0)["run_id"]
    ctx = seed_context(luc, 0, tmp_path)
    with pytest.raises(ValueError, match="incompatible"):
        train_stage(ctx, 1, 2, 3e-5, ctx.ref_dir / "checkpoints" / "stage0.pt",
                    tmp_path / "optimizer_study" / "seed0" / run_id, run_id,
                    train_overrides={"batch_size": 2, "weight_decay": 0.0, "schedule": "cosine"},
                    evaluator=stub_evaluator, bootstrap=BOOT, device="cpu", build_datasets=build)

    s = os_.run_summary(cfg, results_root=tmp_path, device="cpu", build_datasets=build,
                        low_update_cfg=luc, bootstrap=BOOT)
    core = [os_.setting_name(r) for r in os_.settings(cfg)["core"]]
    assert set(s["existence"]) == set(core) | {"constant_same_peak", "batch4"}
    assert all(s["existence"][n]["eligible"] for n in core)
    assert not s["existence"]["batch4"]["eligible"]
    assert set(s["frontiers"]["per_seed"]) == {"0", "1"} and "robust" in s["frontiers"]
    for metric in ("object_success", "spatial_success", "object_probe", "spatial_probe"):
        assert s["one_factor"]["batch4"][metric]["label"] in os_.LABELS
    row = s["rows"][core[0]]["0"]
    assert row["update_size"]["frozen_unchanged"] and row["update_size"]["trainable"]["delta_norm"] > 0
    std = s["rows"][s["standard"]]["1"]
    assert std["source"]["kind"] == "standard"
    units = s["probe_units"][core[0]]["0"]["0"]
    assert len(units["losses"]) == PROBE["n_batches"] and units["mean"] == pytest.approx(units["stored"])
    assert set(s["gpm_points"]) == {"0", "1"}


# ---- the queue ------------------------------------------------------------------------------------------

QUEUE = repo_root() / "scripts" / "queue_optimizer_study.sh"
STEPS = ["0_grid_s1", "1_grid_s2", "2_grid_s0", "3_variants_s0", "4_variants_s1", "5_summary"]


def queue(tmp_path, args=(), fail="", present=()):
    results = tmp_path / "results"
    for rel in present:
        path = results / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    env = {**os.environ, "QUEUE_DRY_RUN": "1", "QUEUE_FAIL_STEP": fail,
           "QUEUE_LOG_ROOT": str(tmp_path / "logs"), "QUEUE_RESULTS_ROOT": str(results)}
    done = subprocess.run(["bash", str(QUEUE), *args], env=env, capture_output=True, text=True,
                          timeout=60)
    assert done.returncode == 0, done.stderr
    (logdir,) = list((tmp_path / "logs").glob("queue_*_optimizer_study"))
    return (logdir / "queue.log").read_text(), logdir


def test_queue(tmp_path):
    log, logdir = queue(tmp_path / "a")
    positions = [log.index(f"START {s}:") for s in STEPS]
    assert positions == sorted(positions) and log.rstrip().endswith("QUEUE DONE")
    assert "--part variants --seed 1" in (logdir / "4_variants_s1.log").read_text()
    log, _ = queue(tmp_path / "b", fail="1_grid_s2")
    assert "START 2_grid_s0:" in log and "SKIP 5_summary" in log  # later steps still run
    log, _ = queue(tmp_path / "c", args=("--from-step", "3"),
                   present=[f"optimizer_study/done_grid_seed{s}.json" for s in (1, 2, 0)])
    assert "PRIOR 0_grid_s1 ok" in log and "START 5_summary:" in log and "START 0_grid_s1:" not in log
    log, _ = queue(tmp_path / "d", args=("--from-step", "3"))
    assert "PRIOR 0_grid_s1 missing" in log and "SKIP 5_summary" in log
    for bad in (("--from-step", "6"), ("--bogus",)):
        assert subprocess.run(["bash", str(QUEUE), *bad], capture_output=True).returncode == 2
