"""Instruction negatives on cabinet top and milk: the per-curriculum config, the milk endpoint labels,
the resumed-run checks, the projection basis, the mode-specific pins, and (tiny CPU world) that a
resumed gpm_neg / allowlist_ft_neg stage 1 equals the full run's bit for bit."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

import flowcl.train.continual as continual_module
from flowcl.analysis.metrics import success_estimate
from flowcl.data.config import load_embodiment_spec
from flowcl.data.curriculum import load_curriculum
from flowcl.envs.evaluation import EvaluationReport, TaskEvaluation
from flowcl.envs.libero_env import EvalConfig
from flowcl.experiments import instruction_negatives as inn
from flowcl.experiments import instruction_negatives_curricula as cur
from flowcl.methods.instruction_negatives import load_bank
from flowcl.train.continual import run_continual
from flowcl.train.trainer import TrainConfig
from flowcl.utils.libero_paths import repo_root
from flowcl.utils.run import file_sha256
from test_continual import MILK, SAUCE, TINY_POLICY, fake_evaluation

CFG = cur.load_config()
A = CFG["task_a"]
CAB, MLK = CFG["curricula"]["cabinet"]["task_b"], CFG["curricula"]["milk"]["task_b"]
BOOT = {"seed": 0, "n_resamples": 2000, "confidence": 0.95}


def s(p, n=50):
    k = int(round(p * n))
    return [True] * k + [False] * (n - k)


# ---- config ----------------------------------------------------------------------------------------------


def test_sub_config_has_the_stove_modules_shape_and_points_at_the_resumed_runs(tmp_path):
    sub = cur.sub_config(CFG, "cabinet")
    base = inn.load_config()
    for k in ("bank", "namespace", "panel", "probe", "swap", "chunks", "coverage", "drift", "acquisition",
              "retention", "production", "smoke"):
        assert sub[k] == base[k]
    assert sub["curriculum"] == "similar_ab_cabinet" and sub["task_b"] == CAB and sub["kind"] == "similar"
    assert inn.run_dir(sub, tmp_path, "gpm") == tmp_path / "similar_ab_cabinet__gpm_projected_adam_neg__seed0"
    assert inn.run_dir(sub, tmp_path, "allow") == tmp_path / "similar_ab_cabinet__allowlist_ft_neg__seed0"
    assert inn.out_dir(sub, tmp_path) == tmp_path / "instruction_negatives_curricula" / "cabinet"
    assert sub["runs"]["gpm"]["method"] == "gpm_neg" and sub["historical"]["allow"] == "similar_ab_cabinet__allowlist_ft__seed0"
    milk = cur.sub_config(CFG, "milk")
    assert milk["kind"] == "diverse" and milk["task_b"] == MLK and not CFG["curricula"]["milk"]["swaps"]


def test_the_rule_pins_the_stove_rule_as_committed():
    assert file_sha256(cur.base_path(CFG)) == CFG["stove"]["rule_sha256"]
    bad = copy.deepcopy(CFG)
    bad["namespace"] = "other"
    with pytest.raises(ValueError):
        cur.base_config(bad)


# ---- labels ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("d,label", [
    ({"diff": 0.0, "low": -0.06, "high": 0.06}, "no_cost_beyond_10pp"),
    ({"diff": -0.30, "low": -0.46, "high": -0.14}, "cost_beyond_10pp"),
    ({"diff": -0.12, "low": -0.24, "high": 0.02}, "observed_drop_over_10pp"),
    ({"diff": -0.02, "low": -0.16, "high": 0.10}, "within_10pp_on_observed_rollouts"),
    (None, "missing"),
])
def test_the_four_endpoint_labels(d, label):
    assert cur.endpoint_label(d, 0.10) == label


def milk_cells(gpm_a=0.92, gpm_b=0.76, allow_a=0.0, allow_b=0.86, hist_a=0.92, hist_b=0.76):
    return {"gpm": {0: {A: s(0.94)}, 1: {A: s(gpm_a), MLK: s(gpm_b)}},
            "allow": {0: {A: s(0.94)}, 1: {A: s(allow_a), MLK: s(allow_b)}},
            "hist_gpm": {0: {A: s(0.94)}, 1: {A: s(hist_a), MLK: s(hist_b)}},
            "hist_allow": {0: {A: s(0.94)}, 1: {A: s(0.0), MLK: s(0.86)}}}


def test_milk_no_harm_needs_both_endpoints_ruled_out():
    sub = cur.sub_config(CFG, "milk")
    same = cur.judge_diverse(sub, milk_cells(), CFG["diverse"], BOOT)
    assert same["labels"]["plate_after_B"] == same["labels"]["B"] == "no_cost_beyond_10pp"
    assert same["labels"]["milk_no_harm"] == "milk_no_harm" and same["labels"]["retains"] == "retains"
    assert same["labels"]["projection_contributes"] == "projection_contributes"
    assert same["labels"]["allowlist_without_projection_retains"] == "not_shown"
    worse = cur.judge_diverse(sub, milk_cells(gpm_a=0.50), CFG["diverse"], BOOT)
    assert worse["labels"]["plate_after_B"] == "cost_beyond_10pp" and worse["labels"]["milk_no_harm"] == "not_shown"
    assert worse["labels"]["B"] == "no_cost_beyond_10pp"


def test_milk_inconclusive_acquisition_assesses_nothing():
    j = cur.judge_diverse(cur.sub_config(CFG, "milk"), milk_cells(gpm_b=0.30), CFG["diverse"], BOOT)
    assert j["outcome"] == "inconclusive_acquisition" and set(j["labels"].values()) == {"not_assessed"}
    assert j["endpoints"]["B"]["diff"] < 0      # the raw comparison is still reported


def test_cabinet_uses_the_stove_judge_and_reports_each_arm_against_its_control():
    sub = cur.sub_config(CFG, "cabinet")
    cells = {"gpm": {0: {A: s(0.94)}, 1: {A: s(0.70), CAB: s(0.80)}},
             "allow": {0: {A: s(0.94)}, 1: {A: s(0.0), CAB: s(0.80)}},
             "hist_gpm": {0: {A: s(0.94)}, 1: {A: s(0.0), CAB: s(0.98)}},
             "hist_allow": {0: {A: s(0.94)}, 1: {A: s(0.0), CAB: s(0.96)}}}
    j = inn.judge(sub, cells, {"passed": True}, BOOT)
    assert j["acquisition"]["state"] == "clean" and j["labels"]["effect_present"] == "effect_present"
    vs = cur.vs_historical(sub, cells, BOOT)
    assert abs(vs["gpm"]["B"]["diff"] + 0.18) < 1e-9 and vs["gpm"]["B"]["high"] < 0   # the hidden drop shows
    assert vs["allow"]["plate_before_B"]["diff"] == 0.0


# ---- validity ----------------------------------------------------------------------------------------------


def report(stage, tasks, n=50, seeds=None):
    return EvaluationReport(run_id="similar_ab_stove__seq_ft__seed0", stage=stage, tasks=[
        TaskEvaluation(k, s(p, n), [10] * n, seeds or list(range(n)), success_estimate(s(p, n))) for k, p in tasks])


def fake_resumed(tmp_path, *, namespace="similar_ab_stove__seq_ft__seed0", own_stage0=False, b=CAB):
    src = tmp_path / "similar_ab_stove__gpm_projected_adam_neg__seed0"
    run = tmp_path / "similar_ab_cabinet__gpm_projected_adam_neg__seed0"
    (src / "checkpoints").mkdir(parents=True)
    torch.save({"state_dict": {"w": torch.ones(2)}}, src / "checkpoints" / "stage0.pt")
    report(0, [(A, 0.94)]).save(src / "eval" / "stage0.json")
    (run / "checkpoints").mkdir(parents=True)
    report(0, [(A, 0.94)]).save(run / "eval" / "stage0.json")
    report(1, [(A, 0.7), (b, 0.9)]).save(run / "eval" / "stage1.json")
    if own_stage0:
        torch.save({}, run / "checkpoints" / "stage0.pt")
    bank_sha = load_bank(inn.load_config()["bank"])["sha256"]
    torch.save({"state_dict": {}, "extra": {"method_state": {"instruction_negatives": {
        "negative_bank_sha256": bank_sha, "logs": {"1": {"active": False}}}}}}, run / "checkpoints" / "stage1.pt")
    OmegaConf.save({"seed_namespace_run_id": namespace, "method": {"name": "gpm_neg"},
                    "curriculum": {"tasks": [{"task_key": A}, {"task_key": CAB}]},
                    "train": {"steps": 30000, "batch_size": 64, "lr": 1.0e-4, "amp": True}}, run / "config.yaml")
    (run / "result.json").write_text(json.dumps({"task_keys": [A, CAB]}))
    (run / "git_sha").write_text("abc\n")
    sha = file_sha256(src / "checkpoints" / "stage0.pt")
    (run / "resume.json").write_text(json.dumps({"run": str(src), "start_stage": 1, "sha256": sha,
                                                 "method_state": {"sha256": "MEM"}}))
    return run, src, {"stage0": sha, "memory": "MEM"}, bank_sha


@pytest.fixture
def no_memory_check(monkeypatch):
    import flowcl.experiments.similar_cl as sc

    monkeypatch.setattr(sc, "memory_check", lambda run: {"passed": True})


def test_resumed_checks_pass_on_a_well_formed_run_and_fail_on_each_defect(tmp_path, no_memory_check):
    sub = cur.sub_config(CFG, "cabinet")
    run, src, pin, bank = fake_resumed(tmp_path / "ok")
    assert cur.run_checks_resumed(sub, run, "gpm", src, pin, False, bank, 50)["passed"]
    assert not cur.run_checks_resumed(sub, run, "gpm", src, {**pin, "stage0": "x"}, False, bank, 50)["passed"]
    assert not cur.run_checks_resumed(sub, run, "gpm", src, {**pin, "memory": "x"}, False, bank, 50)["passed"]
    run, src, pin, bank = fake_resumed(tmp_path / "own", own_stage0=True)
    c = cur.run_checks_resumed(sub, run, "gpm", src, pin, False, bank, 50)
    assert not c["passed"] and not c["resume"]["passed"]
    run, src, pin, bank = fake_resumed(tmp_path / "ns", namespace="other")
    assert not cur.run_checks_resumed(sub, run, "gpm", src, pin, False, bank, 50)["namespace"]
    run, src, pin, bank = fake_resumed(tmp_path / "b", b=MLK)
    c = cur.run_checks_resumed(sub, run, "gpm", src, pin, False, bank, 50)
    assert not c["passed"] and not c["b_task"]["eval_stage1"]


def test_cells_ask_stage0_for_A_only_since_a_resumed_stage0_is_the_stoves(tmp_path):
    run, _, _, _ = fake_resumed(tmp_path)
    succ, seeds, problems = cur.eval_cells(run, {0: [A], 1: [A, CAB]}, 50)
    assert not problems and set(succ[0]) == {A} and set(succ[1]) == {A, CAB}
    _, _, problems = cur.eval_cells(run, {0: [A, CAB]}, 50)
    assert problems == [f"{run.name}: stage 0 {CAB} has fewer than 50 rollouts"]


def test_negatives_must_be_inactive_at_stage1_with_the_bank(tmp_path):
    run, _, _, bank = fake_resumed(tmp_path)
    assert cur.negatives_inactive(run, bank)["passed"]
    assert not cur.negatives_inactive(run, "other")["passed"]


def test_projection_basis_needs_equal_start_set_and_optimizer(tmp_path, monkeypatch):
    import flowcl.experiments.similar_cl as sc
    import flowcl.train.checkpoint as ck

    class P:
        def projectable_parameters(self):
            return ["w"]

    monkeypatch.setattr(ck, "load_checkpoint", lambda path, device="cpu": type("L", (), {"policy": P()})())
    monkeypatch.setattr(sc, "frozen_check", lambda run, src, allowed, arm: {
        "changed": [], "trainable_set": {"report": {"trainable_tensors": 1, "newly_frozen": ["b"]}}})
    sources, runs = {}, {}
    for arm in cur.ARMS:
        sources[arm] = tmp_path / f"src_{arm}"
        (sources[arm] / "checkpoints").mkdir(parents=True)
        torch.save({"state_dict": {"w": torch.ones(2), "b": torch.zeros(1)}}, sources[arm] / "checkpoints" / "stage0.pt")
        runs[arm] = tmp_path / f"run_{arm}"
        runs[arm].mkdir()
        tc = {"optimizer": "adamw", "lr": 1e-4}
        (runs[arm] / "result.json").write_text(json.dumps({"seed_namespace_run_id": "ns", "stages": [
            {}, {"train_config": tc, "trainable_params": 2}]}))
    basis = cur.projection_basis(runs, sources)
    assert basis["passed"] and basis["allowlist"]["names"] == ["w"] and basis["start"]["n_different"] == 0
    (runs["allow"] / "result.json").write_text(json.dumps({"seed_namespace_run_id": "ns", "stages": [
        {}, {"train_config": {"optimizer": "adamw", "lr": 3e-4}, "trainable_params": 2}]}))
    basis = cur.projection_basis(runs, sources)
    assert not basis["passed"] and basis["train_config_diff"] == {"lr": (1e-4, 3e-4)}


def fake_sources(root: Path, gate_passed: bool):
    for arm in cur.ARMS:
        d = root / CFG["sources"][arm]["run"]
        (d / "checkpoints").mkdir(parents=True)
        torch.save({"state_dict": {"w": torch.ones(2)}}, d / "checkpoints" / "stage0.pt")
    (root / CFG["sources"]["gpm"]["run"] / "method").mkdir()
    (root / CFG["sources"]["gpm"]["run"] / "method" / "memory_task0.pt").write_bytes(b"memory")
    (root / "instruction_negatives" / "panel").mkdir(parents=True)
    (root / "instruction_negatives" / "summary.json").write_text(json.dumps(
        {"status": "valid", "judgment": {"gate": {"passed": gate_passed}}}))
    (root / "instruction_negatives" / "panel" / "gate.json").write_text("{}")


def fake_historical(cfg: dict, root: Path) -> dict:
    cfg = copy.deepcopy(cfg)
    for short in cur.CURRICULA:
        for arm in cur.ARMS:
            h = cfg["curricula"][short]["historical"][arm]
            for rel in h["sha256"]:
                f = root / h["run"] / rel
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(f"{short}{arm}{rel}")
                h["sha256"][rel] = file_sha256(f)
    return cfg


def test_pins_are_mode_specific(tmp_path):
    src, hist = tmp_path / "src", tmp_path / "hist"
    fake_sources(src, gate_passed=False)        # the stove smoke's gate fails at 200 steps
    cfg = fake_historical(CFG, hist)
    smoke = cur.pin_checks(cfg, src, hist, smoke=True)
    assert smoke["passed"] and smoke["pins"]["from"].startswith("smoke")
    assert smoke["pins"]["gpm"]["memory"] == file_sha256(src / CFG["sources"]["gpm"]["run"] / "method" / "memory_task0.pt")
    production = cur.pin_checks(cfg, src, hist, smoke=False)
    assert not production["passed"] and not production["source_gpm"] and not production["stove_summary"]["passed"]
    assert cur.pins(CFG, src, False)["gpm"]["stage0"] == CFG["sources"]["gpm"]["stage0_sha256"]
    (hist / CFG["curricula"]["milk"]["historical"]["gpm"]["run"] / "result.json").write_text("changed")
    assert not cur.pin_checks(cfg, src, hist, smoke=True)["historical_files"]["passed"]


# ---- the resumed path, tiny CPU world ---------------------------------------------------------------------


def tiny_cfg():
    return TrainConfig(steps=2, batch_size=2, num_workers=0, device="cpu", log_every=0, warmup_steps=1)


def pair():
    return load_curriculum({"name": "test_pair", "tasks": [{"task_key": MILK, "n_demos": 1},
                                                           {"task_key": SAUCE, "n_demos": 1}]})


NEG = {"negative_bank": "instruction_negatives_bank", "negative_weight": 1.0, "negative_stages": [0],
       "negative_log_interval": 1}


@pytest.fixture(scope="module")
def capture(tmp_path_factory):
    cfg = OmegaConf.load(repo_root() / "configs" / "analysis" / "subspace.yaml")
    cfg.min_samples_per_dim = 0.01
    cfg.num_workers = 0
    path = tmp_path_factory.mktemp("capture") / "capture.yaml"
    OmegaConf.save(cfg, path)
    return str(path)


@pytest.mark.parametrize("method", ["gpm_neg", "allowlist_ft_neg"])
def test_a_resumed_stage1_equals_the_full_run_bitwise(method, capture, dataset_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(continual_module, "evaluate_tasks", fake_evaluation([]))
    kwargs = {**NEG, **({"update_memory": True, "capture_config": capture, "log_interval": 1}
                        if method == "gpm_neg" else {})}
    common = dict(method_name=method, method_kwargs=kwargs, spec=load_embodiment_spec("libero_franka"),
                  policy_config=TINY_POLICY, train_cfg=tiny_cfg(), eval_cfg=EvalConfig(n_episodes=2), seed=0,
                  dataset_dir=dataset_dir, pretrained=False)
    full = run_continual(pair(), results_root=tmp_path / "a", **common)
    src = tmp_path / "a" / full.run_id
    resumed = run_continual(pair(), results_root=tmp_path / "b", resume_run=src, start_stage=1,
                            identity_reference_run=src, identity_stages=(1,), **common)
    assert resumed.identity_checks["1"]["passed"] and resumed.identity_checks["1"]["n_tensors"] > 0
    run = tmp_path / "b" / resumed.run_id
    assert not (run / "checkpoints" / "stage0.pt").exists()
    bank = load_bank("instruction_negatives_bank")["sha256"]
    full_state = torch.load(src / "checkpoints" / "stage0.pt", weights_only=False)["extra"]["method_state"]
    assert full_state["instruction_negatives"]["logs"]["0"]["active"] is True     # the source trained negatives
    assert cur.negatives_inactive(run, bank)["passed"]
    if method == "gpm_neg":
        from flowcl.experiments.resume_check import compare_resume

        assert compare_resume(src, run, 1)["passed"]
