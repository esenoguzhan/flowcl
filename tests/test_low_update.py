"""Low-update continual fine-tuning and the projection pilot: the verdicts, the gates,
publishing and resuming, provenance, both parts end to end on the tiny policy, the queue."""

from __future__ import annotations

import copy
import dataclasses
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from flowcl.analysis.metrics import success_estimate
from flowcl.data.config import load_embodiment_spec
from flowcl.data.dataset import ChunkedActionDataset
from flowcl.data.episode import Episode
from flowcl.data.stats import compute_stats
from flowcl.envs.evaluation import EvaluationReport, TaskEvaluation
from flowcl.experiments import low_update as lu
from flowcl.experiments import projection_pilot as pp
from flowcl.models.build import build_policy
from flowcl.utils.libero_paths import repo_root
from flowcl.utils.run import file_sha256
from flowcl.utils.seeding import derive_seed

KEYS = ["libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
        "libero_object/pick_up_the_milk_and_place_it_in_the_basket",
        "libero_goal/open_the_middle_drawer_of_the_cabinet",
        "libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"]
TINY_POLICY = {"d_model": 384, "n_trunk_layers": 6, "n_heads": 8, "n_decoder_layers": 2,
               "n_context_tokens": 4, "pretrained": False, "euler_steps": 2}
TINY_TRAIN = {"steps": 4, "batch_size": 2, "lr": 1e-4, "weight_decay": 1e-4, "grad_clip": 1.0,
              "warmup_steps": 1, "log_every": 0, "num_workers": 0, "device": "cpu",
              "accumulation_steps": 1, "amp": False}
PROBE = {"batch_size": 2, "n_batches": 1,
         "seed_tags": {"shuffle": "t::shuffle", "flow_time": "t::flow_time", "noise": "t::noise"}}
BOOT = {"seed": 0, "n_resamples": 200, "confidence": 0.95}


class Ref:
    def __init__(self, task_key):
        self.task_key = task_key


def stub_evaluator(policy, refs, spec, stats, run_id, stage):
    """Deterministic 'rollouts': success from a hash of (namespace, task, episode) only, so
    any two evaluations under one namespace are identical (paired differences are 0)."""
    tasks = []
    for ref in refs:
        seeds = list(range(10))
        succ = [bool((derive_seed(run_id, ref.task_key, i) >> 3) % 2) for i in seeds]
        tasks.append(TaskEvaluation(ref.task_key, succ, [100] * 10, seeds, success_estimate(succ)))
    return EvaluationReport(run_id=run_id, stage=stage, tasks=tasks)


# ---- pure verdicts ---------------------------------------------------------------------------


def cell(success, seq_ft, low=0.1, diff=None):
    return {"success": success, "reference_success": {"seq_ft": seq_ft, "gpm": 0, "adaptive_gpm": 0},
            "paired": {"seq_ft": {"diff": success - seq_ft if diff is None else diff,
                                  "low": low, "high": 1.0}},
            "probe_loss": 0.1}


def stage1_row(obj, obj_ref, spa_low):
    return {"cells": {"0": cell(0.5, 0.0, low=spa_low), "1": cell(obj, obj_ref)}}


def test_validation_boundaries():
    assert lu.classify_validation(stage1_row(0.80, 0.90, 0.36), 0.10)["passed"]  # exactly -10 pp
    assert not lu.classify_validation(stage1_row(0.78, 0.90, 0.36), 0.10)["passed"]
    fail = lu.classify_validation(stage1_row(0.90, 0.90, 0.0), 0.10)          # CI low at 0 fails
    assert not fail["passed"] and not fail["retention"]["passed"] and fail["plasticity"]["passed"]


def test_verdict_counts_only_the_held_out_seeds():
    ok, bad = {"passed": True, "plasticity": {"passed": True}, "retention": {"passed": True}}, \
        {"passed": False, "plasticity": {"passed": True}, "retention": {"passed": False}}
    v = lu.classify_validation_across_seeds({0: bad, 1: ok, 2: ok}, [1, 2], 0)
    assert v["verdict"] == lu.VALIDATED and v["selection_seed"]["passed"] is False
    v = lu.classify_validation_across_seeds({0: ok, 1: ok, 2: bad}, [1, 2], 0)
    assert v["verdict"] == lu.NOT_VALIDATED and v["failed_held_out_seeds"] == {2: ["retention"]}
    with pytest.raises(ValueError, match="missing"):
        lu.classify_validation_across_seeds({0: ok, 1: ok}, [1, 2], 0)


def chain_rows(lows, goal=0.95, l10=0.95):
    return {s: {1: {}, 2: {"cells": {"2": cell(goal, 1.0)}},
                3: {"cells": {"0": cell(0.4, 0, low=lows[s][0]), "1": cell(0.4, 0, low=lows[s][1]),
                              "2": cell(0.4, 0, low=lows[s][2]), "3": cell(l10, 1.0)}}}
            for s in lows}


def test_chain_classification():
    pos, zero = [0.1, 0.1, 0.1], [0.0, -0.1, 0.0]
    assert lu.classify_chain(chain_rows({0: pos, 1: pos}), 0.10)["class"] == "cumulative"
    assert lu.classify_chain(chain_rows({0: zero, 1: zero}), 0.10)["class"] == "delay_only"
    assert lu.classify_chain(chain_rows({0: pos, 1: zero}), 0.10)["class"] == "partial"
    under = lu.classify_chain(chain_rows({0: pos, 1: pos}, goal=0.89), 0.10)
    assert under["class"] == "underlearns"  # perfect retention does not rescue it
    assert under["underlearns"][0]["stage"] == 2
    assert lu.classify_chain(chain_rows({0: pos}, l10=0.90), 0.10)["class"] == "cumulative"


def pilot_row(family, lr, obj, spa, probe=0.1, alpha=None):
    method = {"family": family, "lr": lr, **({"alpha": alpha} if alpha is not None else {})}
    return {"run_id": pp.run_name(method, 0), "method": method,
            "cells": {"0": {"success": spa, "probe_loss": probe}, "1": {"success": obj}}}


def test_selection_is_among_eligible_runs_only():
    cfg = pp.load_pilot_config()
    rows = [pilot_row("gpm", 3e-5, 0.9, 0.9),                    # best, but a control
            pilot_row("freeze_only", 3e-5, 0.9, 0.95),
            pilot_row("sgp", 3e-5, 0.9, 0.9, alpha=5),            # negative control
            pilot_row("sgp", 3e-5, 0.66, 0.6, probe=0.2, alpha=100),  # exactly at the floor
            pilot_row("sgp", 1e-4, 0.8, 0.6, probe=0.1, alpha=400),   # ties, lower probe loss
            pilot_row("sgp", 1e-4, 0.5, 0.99, alpha=1600)]            # does not qualify
    s = pp.select_configuration(rows, cfg["eligible"], 0.66)
    assert s["selected"]["run_id"] == "pilot__sgp400__lr0.0001__seed0"
    assert "pilot__sgp100__lr3e-05__seed0" in s["qualifying"]
    assert s["best_overall_projected"]["run_id"] == "pilot__gpm__lr3e-05__seed0"
    assert pp.select_configuration(rows[:3], cfg["eligible"], 0.66)["selected"] is None


def test_pilot_grid_matches_the_plan():
    cfg = pp.load_pilot_config()
    grid = cfg["grid"]
    assert len(grid) == 13
    assert sum(pp.is_eligible(e, cfg["eligible"]) for e in grid) == 8
    assert {(e["family"], e.get("alpha"), e["lr"]) for e in grid if not pp.is_eligible(e, cfg["eligible"])} \
        == {("freeze_only", None, 3e-5), ("freeze_only", None, 1e-4), ("gpm", None, 3e-5),
            ("gpm", None, 1e-4), ("sgp", 5, 3e-5)}
    assert cfg["steps"] == 5000 and cfg["selection"]["object_tolerance"] == 0.10
    luc = lu.load_low_update_config()
    assert luc["frozen"] == {"steps": 5000, "lr": 3e-5}
    assert luc["validation"]["held_out_seeds"] == [1, 2] and luc["validation"]["selection_seed"] == 0


# ---- a fake four-task world -------------------------------------------------------------------


@pytest.fixture(scope="module")
def spec():
    return load_embodiment_spec("libero_franka")


def episodes(spec, task, n=3, offset=0):
    out = []
    for i in range(n):
        rng = np.random.default_rng(offset + i)
        h, w = spec.observation.image_size
        out.append(Episode(
            images={c: rng.integers(0, 255, (18, h, w, 3), dtype=np.uint8) for c in spec.cameras},
            state=rng.normal(size=(18, spec.d_state)).astype(np.float32),
            action=rng.uniform(-1, 1, size=(18, spec.d_action)).astype(np.float32),
            language=f"do {task}", task_id=task, embodiment=spec.name))
    return out


def tiny_policy(spec):
    torch.manual_seed(0)
    policy = build_policy(TINY_POLICY, spec, pretrained=False)
    for p in policy.parameters():
        if p.requires_grad:
            nn.init.normal_(p, std=0.05)
    return policy


def fake_world(root, spec, seeds=(0, 1)):
    """seq_ft, GPM and adaptive-GPM runs with evaluations and diagnostics, per seed, plus
    seed 0's hparam selection run; returns the config and a dataset builder."""
    from flowcl.train.checkpoint import save_checkpoint

    stats = compute_stats(episodes(spec, KEYS[0]), embodiment=spec.name, task_id=KEYS[0])
    curriculum = {"name": "seq_hetero", "tasks": [{"task_key": k, "n_demos": 3} for k in KEYS]}
    cfg = copy.deepcopy(lu.load_low_update_config())
    cfg["frozen"] = {"steps": 2, "lr": 3e-5}
    cfg["validation"]["held_out_seeds"] = [s for s in seeds if s != 0]
    cfg["seeds"] = {}
    for s in seeds:
        ns = f"seq_hetero__seq_ft__seed{s}"
        runs = {"seq_ft": ns, "gpm": f"seq_hetero__gpm_projected_adam__seed{s}",
                "adaptive_gpm": f"seq_hetero__gpm_projected_adam_ne90__seed{s}"}
        for role, run in runs.items():
            for k in range(4):
                stub_evaluator(None, [Ref(key) for key in KEYS], spec, None, ns, k).save(
                    root / run / "eval" / f"stage{k}.json")
        ref_dir = root / ns
        OmegaConf.save(OmegaConf.create({"run_id": ns, "seed": s, "curriculum": curriculum,
                                         "train": TINY_TRAIN}), ref_dir / "config.yaml")
        (ref_dir / "result.json").write_text(json.dumps({"run_id": ns}))  # legacy: no field
        save_checkpoint(ref_dir / "checkpoints" / "stage0.pt", policy=tiny_policy(spec),
                        policy_config=TINY_POLICY, spec=spec, stats=stats, run_id=ns, stage=0,
                        task_key=KEYS[0])
        L = [[0.01 * (i + j + 1) for j in range(4)] for i in range(4)]
        diag = {}
        for key, method in (("gpm", runs["gpm"]), ("adaptive_gpm", runs["adaptive_gpm"])):
            path = root / f"diag_{key}_seed{s}.json"
            path.write_text(json.dumps({"reference_run_id": ns, "method_run_id": method,
                                        "config": {"probe": PROBE},
                                        "loss_matrix": {"reference": L, "method": L}}))
            diag[key] = path.name
        cfg["seeds"][s] = {"reference_run": ns, "gpm_run": runs["gpm"],
                           "adaptive_gpm_run": runs["adaptive_gpm"], "diagnostics": diag,
                           "stage1_source": None}
    # Seed 0's selection run, as the hparam study wrote it.
    src = root / "hparam_study" / "cl_lr" / "cl__libero_object__b5000__lr3e-05__seed0"
    save_checkpoint(src / "checkpoints" / "final.pt", policy=tiny_policy(spec),
                    policy_config=TINY_POLICY, spec=spec, stats=stats, run_id=src.name,
                    stage=1, task_key=KEYS[1])
    OmegaConf.save(OmegaConf.create({
        "reference_run": "seq_hetero__seq_ft__seed0", "start_checkpoint": "stage0.pt",
        "train": {**TINY_TRAIN, "steps": 2, "lr": 3e-5},
        "stream_seed": {"namespace": "seq_hetero__seq_ft__seed0", "stage": 1}}), src / "config.yaml")
    stub_evaluator(None, [Ref(KEYS[0]), Ref(KEYS[1])], spec, None, "seq_hetero__seq_ft__seed0",
                   1).save(src / "eval.json")
    (src / "row.json").write_text(json.dumps({"probe_loss": {"retained": 0.2, "new": 0.05}}))
    (src / "git_sha").write_text("8e6785a\n")
    cfg["seeds"][0]["stage1_source"] = str(src.relative_to(root))

    def build(stage):
        return ChunkedActionDataset(episodes(spec, stage.task_key, offset=KEYS.index(stage.task_key) * 10),
                                    spec, stats)
    return cfg, build


def test_validation_chain_and_summary_end_to_end(spec, tmp_path, capsys):
    cfg, build = fake_world(tmp_path, spec)
    common = dict(results_root=tmp_path, device="cpu", allow_dirty=True, evaluator=stub_evaluator,
                  bootstrap=BOOT, build_datasets=build)
    v1 = lu.run_validation_seed(cfg, 1, **common)
    row = v1["row"]
    assert row["stage"] == 1 and set(row["cells"]) == {"0", "1"} and row["namespace"] == \
        "seq_hetero__seq_ft__seed1"
    assert row["cells"]["0"]["paired"]["seq_ft"]["diff"] == 0.0  # stub: same namespace, same seeds
    assert row["cells"]["1"]["probe_reference"]["seq_ft"] == pytest.approx(0.03)
    stage1 = tmp_path / "low_update" / "seed1" / "stage1"
    assert (stage1 / "row.json").is_file() and not lu.abandoned_attempts(stage1.parent)
    assert row["parent"]["sha256"] == file_sha256(
        tmp_path / "seq_hetero__seq_ft__seed1" / "checkpoints" / "stage0.pt")
    mtime = (stage1 / "row.json").stat().st_mtime
    lu.run_validation_seed(cfg, 1, **common)  # published: reused, not retrained
    assert (stage1 / "row.json").stat().st_mtime == mtime and "reused" in capsys.readouterr().out

    v0 = lu.run_validation_seed(cfg, 0, **common)
    assert v0["row"]["selection_run"] and v0["row"]["source"]["git_sha"] == "8e6785a"
    verdict = lu.run_validation(cfg, results_root=tmp_path)
    assert verdict["verdict"] == lu.NOT_VALIDATED  # the stub gives no retention gain
    assert not lu.gate_open(cfg, tmp_path)
    with pytest.raises(RuntimeError, match="gate is closed"):
        lu.run_chain(cfg, 1, **common)

    # Open the gate by hand to exercise the chain.
    (tmp_path / "low_update" / "validation.json").write_text(json.dumps({"verdict": lu.VALIDATED}))
    rows1 = lu.run_chain(cfg, 1, **common)
    assert set(rows1[3]["cells"]) == {"0", "1", "2", "3"}
    assert rows1[2]["parent"]["sha256"] == file_sha256(stage1 / "checkpoints" / "final.pt")
    assert rows1[3]["parent"]["sha256"] == rows1[2]["checkpoint_sha256"]
    rows0 = lu.run_chain(cfg, 0, **common)
    assert rows0[2]["parent"]["sha256"] == v0["row"]["source"]["checkpoint_sha256"]
    summary = lu.run_chain_summary(cfg, results_root=tmp_path)
    assert summary["classification"]["class"] in ("delay_only", "underlearns")
    prov = summary["provenance"]["0"]
    assert prov["stage1"]["git_sha"] == "8e6785a" and prov["stage2"]["git_sha"] != "8e6785a"
    assert set(summary["final"]["1"]["final_row"]) == {"0", "1", "2", "3"}


def test_published_stages_are_validated_never_overwritten(spec, tmp_path):
    cfg, build = fake_world(tmp_path, spec, seeds=(0, 1))
    common = dict(results_root=tmp_path, device="cpu", allow_dirty=True, evaluator=stub_evaluator,
                  bootstrap=BOOT, build_datasets=build)
    lu.run_validation_seed(cfg, 1, **common)
    stage1 = tmp_path / "low_update" / "seed1" / "stage1"
    changed = copy.deepcopy(cfg)
    changed["frozen"]["lr"] = 1e-4
    with pytest.raises(ValueError, match="incompatible"):
        lu.run_validation_seed(changed, 1, **common)
    with open(stage1 / "checkpoints" / "final.pt", "ab") as f:
        f.write(b"x")
    with pytest.raises(ValueError, match="SHA-256"):
        lu.run_validation_seed(cfg, 1, **common)
    (stage1 / "row.json").unlink()
    with pytest.raises(RuntimeError, match="without row.json"):
        lu.run_validation_seed(cfg, 1, **common)
    with pytest.raises(FileExistsError):
        lu.publish(tmp_path / "anything", stage1)
    (stage1.parent / ".attempt_20260928_000000_1_stage2").mkdir()
    assert lu.abandoned_attempts(stage1.parent) == [".attempt_20260928_000000_1_stage2"]


def test_selection_run_must_match_the_frozen_configuration(spec, tmp_path):
    cfg, build = fake_world(tmp_path, spec, seeds=(0,))
    src = tmp_path / cfg["seeds"][0]["stage1_source"]
    conf = OmegaConf.load(src / "config.yaml")
    conf.train.lr = 1e-4
    OmegaConf.save(conf, src / "config.yaml")
    with pytest.raises(ValueError, match="frozen low-update configuration"):
        lu.run_validation_seed(cfg, 0, results_root=tmp_path, device="cpu", allow_dirty=True,
                               evaluator=stub_evaluator, bootstrap=BOOT)


def test_warm_cache_trains_from_the_runners_text_cache_state(spec, tmp_path):
    """The runner has encoded a task's instruction (fp32 rollouts) before the task trains; the
    opt-in warm-up recreates that before the method is set up, and is part of the recipe."""
    cfg, build = fake_world(tmp_path, spec, seeds=(0,))
    ctx = lu.seed_context(cfg, 0, tmp_path)
    start = tmp_path / "seq_hetero__seq_ft__seed0" / "checkpoints" / "stage0.pt"
    seen = {}

    def record(name):
        def setup(policy):
            seen[name] = sorted(policy.text_encoder._cache)
        return setup

    common = dict(evaluator=stub_evaluator, bootstrap=BOOT, device="cpu", build_datasets=build)
    warm = lu.train_stage(ctx, 1, 2, 3e-5, start, tmp_path / "w", "w", setup_method=record("warm"),
                          warm_cache=True, **common)
    lu.train_stage(ctx, 1, 2, 3e-5, start, tmp_path / "c", "c", setup_method=record("cold"), **common)
    assert seen == {"warm": [f"do {KEYS[1]}"], "cold": []}
    assert warm["warm_text_cache"] is True
    with pytest.raises(ValueError, match="incompatible"):  # a cold run is never reused as warm
        lu.train_stage(ctx, 1, 2, 3e-5, start, tmp_path / "c", "c", warm_cache=True, **common)
    assert lu.warm_text_cache(tiny_policy(spec), build(ctx.curriculum.stages[2]), "cpu") == [f"do {KEYS[2]}"]


# ---- the projection pilot ---------------------------------------------------------------------


def write_memory(root, spec, cfg, build):
    """A real T1 memory for the tiny policy (Gate 2 capture), saved where the pilot reads it."""
    from flowcl.analysis.subspace import save_bases
    from flowcl.experiments.gate2 import collect_bases, load_subspace_config
    from flowcl.train.checkpoint import load_checkpoint

    loaded = load_checkpoint(root / "seq_hetero__seq_ft__seed0" / "checkpoints" / "stage0.pt")
    stage0 = type("S", (), {"task_key": KEYS[0], "n_demos": 3})()
    sub = collect_bases(loaded, build(stage0),
                        dataclasses.replace(load_subspace_config(), min_samples_per_dim=0.01,
                                            num_workers=0, batch_size=4), device="cpu")
    bases = {n: layer.primary for n, layer in sub.layers.items()}
    path = root / "seq_hetero__gpm_projected_adam__seed0" / "method" / "memory_task0.pt"
    save_bases(path, bases, {"kind": "accumulated_memory"})
    return bases


def test_pilot_end_to_end(spec, tmp_path):
    from flowcl.analysis.importance import sgp_importance
    from flowcl.methods.gpm import allowlist
    from flowcl.train.checkpoint import load_checkpoint

    luc, build = fake_world(tmp_path, spec, seeds=(0,))
    pcfg = copy.deepcopy(pp.load_pilot_config())
    pcfg["steps"] = 2
    pcfg["grid"] = [{"family": "freeze_only", "lr": 3e-5}, {"family": "gpm", "lr": 3e-5},
                    {"family": "sgp", "alpha": 25, "lr": 3e-5}]
    pcfg["ft_references"] = [{"lr": 3e-5, "run": luc["seeds"][0]["stage1_source"]}]
    bases = write_memory(tmp_path, spec, luc, build)
    common = dict(results_root=tmp_path, device="cpu", allow_dirty=True, evaluator=stub_evaluator,
                  bootstrap=BOOT, build_datasets=build, low_update_cfg=luc)
    with pytest.raises(RuntimeError, match="pilot gate is closed"):
        pp.run_pilot(pcfg, 3e-5, **common)
    out = tmp_path / "low_update"
    out.mkdir(parents=True, exist_ok=True)
    (out / "validation.json").write_text(json.dumps({"verdict": lu.VALIDATED}))
    (out / "chain_summary.json").write_text(json.dumps({"classification": {"class": "underlearns"}}))
    assert not lu.pilot_gate_open(luc, tmp_path)
    (out / "chain_summary.json").write_text(json.dumps({"classification": {"class": "partial"}}))

    # SGP's T1 importance is Eq. 2 of the GPM spectrum at each alpha.
    policy = load_checkpoint(tmp_path / "seq_hetero__seq_ft__seed0" / "checkpoints" / "stage0.pt").policy
    for alpha in (25, 400):
        method = pp.make_method({"family": "sgp", "alpha": alpha, "lr": 3e-5}, bases, 0.95)(policy)
        for name, b in bases.items():
            k = b.ranks[0.95]
            assert torch.equal(method._importance[name], sgp_importance(b.singular_values[:k], alpha))

    rows = pp.run_pilot(pcfg, 3e-5, **common)
    assert [r["method"]["family"] for r in rows] == ["freeze_only", "gpm", "sgp"]
    frozen, gpm, sgp = rows
    for r in rows:
        assert set(r["update_split"]["halves"]) == {"trunk", "decoder"}
    assert "projection_logs" not in frozen and gpm["projection_logs"]["worst_residual_over_bound"] <= 1
    assert gpm["nominal_protected_scale"]["trunk"] == 0.0
    assert 0 < sgp["nominal_protected_scale"]["trunk"] <= 3e-5 and "nominal" in sgp["nominal_protected_scale"]["note"]
    # Freeze-only: exactly the allowlist complement stays put; nothing is projected.
    start = load_checkpoint(tmp_path / "seq_hetero__seq_ft__seed0" / "checkpoints" / "stage0.pt").policy
    end = load_checkpoint(tmp_path / "projection_pilot" / frozen["run_id"] / "checkpoints" / "final.pt").policy
    allowed = set(allowlist(start))
    s0, s1 = dict(start.named_parameters()), dict(end.named_parameters())
    assert all(torch.equal(s0[n], s1[n]) for n in s0 if n not in allowed)
    assert any(not torch.equal(s0[n], s1[n]) for n in allowed)
    assert frozen["update_split"]["halves"]["trunk"]["parallel_fraction"] > 0  # unprojected

    summary = pp.run_pilot_summary(pcfg, results_root=tmp_path, low_update_cfg=luc, bootstrap=BOOT)
    assert len(summary["rows"]) == 3 and "0" in summary["rows"][2]["paired_vs_ft_same_lr"]
    assert summary["selection"]["eligible"] == ["pilot__sgp25__lr3e-05__seed0"]


# ---- the queue ------------------------------------------------------------------------------------

QUEUE = repo_root() / "scripts" / "queue_low_update.sh"


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
    (logdir,) = list((tmp_path / "logs").glob("queue_*_low_update"))
    return (logdir / "queue.log").read_text(), logdir


STEPS = ["0_validation_s1", "1_validation_s2", "2_validation", "3_gate", "4_chain_s0",
         "5_chain_s1", "6_chain_s2", "7_chain_summary", "8_pilot_gate", "9_pilot_lr3e-5",
         "10_pilot_lr1e-4", "11_pilot_summary"]


def ran(log, step):
    return f"START {step}:" in log


def test_queue_runs_everything_when_both_gates_pass(tmp_path):
    log, logdir = queue(tmp_path)
    positions = [log.index(f"START {s}:") for s in STEPS]
    assert positions == sorted(positions) and log.rstrip().endswith("QUEUE DONE")
    assert "--part chain --seed 2" in (logdir / "6_chain_s2.log").read_text()
    assert "--lr 1e-4" in (logdir / "10_pilot_lr1e-4.log").read_text()


def test_queue_gates(tmp_path):
    log, _ = queue(tmp_path, fail="3_gate")
    assert not any(ran(log, s) for s in STEPS[4:]) and "SKIP 4-11" in log
    log, _ = queue(tmp_path / "b", fail="8_pilot_gate")
    assert all(ran(log, s) for s in STEPS[:9]) and not any(ran(log, s) for s in STEPS[9:])
    assert "SKIP 9-11" in log


def test_queue_resume_and_arguments(tmp_path):
    log, _ = queue(tmp_path, args=("--from-step", "5"),
                   present=["low_update/validation_seed1.json", "low_update/validation_seed2.json",
                            "low_update/seed0/stage3/row.json"])
    assert "PRIOR 4_chain_s0 ok" in log and ran(log, "3_gate") and ran(log, "5_chain_s1")
    assert not ran(log, "0_validation_s1") and not ran(log, "2_validation")
    for bad in (("--from-step", "12"), ("--bogus",)):
        assert subprocess.run(["bash", str(QUEUE), *bad], capture_output=True).returncode == 2
    for script, flags in (("low_update.py", ["--part", "--seed", "--allow-dirty"]),
                          ("projection_pilot.py", ["--lr", "--summary", "--allow-dirty"])):
        done = subprocess.run([sys.executable, str(repo_root() / "scripts" / script), "--help"],
                              capture_output=True, text=True, timeout=120)
        assert done.returncode == 0 and all(f in done.stdout for f in flags), script
