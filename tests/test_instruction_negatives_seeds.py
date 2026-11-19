"""Instruction negatives at seeds 1-2: the per-seed configs, the labels per seed, the reading across seeds,
the control and resumed-control checks, the pins, and that the seeds can be narrowed in smoke only."""

from __future__ import annotations

import json

import pytest
import torch
from omegaconf import OmegaConf

from flowcl.experiments import instruction_negatives as inn
from flowcl.experiments import instruction_negatives_curricula as cur
from flowcl.experiments import instruction_negatives_seeds as ins
from flowcl.utils.libero_paths import repo_root
from flowcl.utils.run import file_sha256
from test_instruction_negatives_curricula import fake_resumed

CFG = ins.load_config()
A = CFG["task_a"]
STOVE, CAB = CFG["curricula"]["stove"]["task_b"], CFG["curricula"]["cabinet"]["task_b"]
RULE = CFG["labels"]
BOOT = {"seed": 0, "n_resamples": 2000, "confidence": 0.95}
PASSED = {"passed": True}


def s(p, n=50):
    k = int(round(p * n))
    return [True] * k + [False] * (n - k)


@pytest.fixture
def no_memory_check(monkeypatch):
    import flowcl.experiments.similar_cl as sc

    monkeypatch.setattr(sc, "memory_check", lambda run: {"passed": True})


# ---- config ----------------------------------------------------------------------------------------------


def test_sub_configs_name_the_seed_runs_and_keep_the_stove_rule(tmp_path):
    base = inn.load_config()
    st, cb = ins.stove_sub(CFG, 2), ins.cabinet_sub(CFG, 2)
    for k in ("bank", "panel", "gate", "probe", "swap", "chunks", "coverage", "drift", "production", "smoke"):
        assert st[k] == base[k] and cb[k] == base[k]
    assert st["seed"] == 2 and st["namespace"] == cb["namespace"] == "similar_ab_stove__seq_ft__seed2"
    assert st["held_out_tasks"] == base["held_out_tasks"] and st["task_b"] == STOVE and cb["task_b"] == CAB
    assert inn.run_dir(st, tmp_path, "gpm") == tmp_path / "similar_ab_stove__gpm_projected_adam_neg__seed2"
    assert inn.run_dir(cb, tmp_path, "gpm") == tmp_path / "similar_ab_cabinet__gpm_projected_adam_neg__seed2"
    assert inn.run_dir(cb, tmp_path, "ctrl") == tmp_path / "similar_ab_cabinet__gpm_projected_adam__seed2"
    assert inn.out_dir(st, tmp_path) == tmp_path / "instruction_negatives_seeds" / "seed2" / "stove"
    assert ins.source_of(CFG, tmp_path, 2, "cabinet", "ctrl") == tmp_path / "similar_ab_stove__gpm_projected_adam__seed2"
    assert st["runs"]["gpm"]["method"] == "gpm_neg" and st["runs"]["ctrl"]["method"] == "gpm"
    assert base["seed"] == 0  # the stove rule itself is untouched


def test_the_seeds_can_be_narrowed_in_smoke_only():
    assert ins.seeds_of(CFG, False, None) == [1, 2]
    assert ins.seeds_of(CFG, True, [1]) == [1]
    with pytest.raises(ValueError, match="smoke only"):
        ins.seeds_of(CFG, False, [1])


def test_the_rule_pins_the_earlier_rules_and_seed0s_artifacts():
    results = repo_root() / "results"
    if not (results / CFG["pins"]["seed0_stove_summary"]["path"]).is_file():
        pytest.skip("seed 0's artifacts are not on this machine")
    checks = ins.pin_checks(CFG, results)
    assert checks["passed"], checks


# ---- labels ----------------------------------------------------------------------------------------------


def cells(neg_a1, ctrl_a1, neg_b=0.96, ctrl_b=0.98, a0=0.94, b=STOVE):
    return {"neg": {0: {A: s(a0)}, 1: {A: s(neg_a1), b: s(neg_b)}},
            "ctrl": {0: {A: s(a0)}, 1: {A: s(ctrl_a1), b: s(ctrl_b)}}}


def test_judge_seed_labels():
    j = ins.judge_seed(RULE, cells(0.70, 0.0), PASSED, A, STOVE, BOOT)
    assert j["acquisition"]["state"] == "clean" and j["labels"] == {"effect_present": "effect_present",
                                                                     "retains": "not_shown"}
    assert j["projection_contributes"] == "not_run" and j["plate_before_B"]["n_rows_differ"] == 0
    j = ins.judge_seed(RULE, cells(0.04, 0.0), PASSED, A, STOVE, BOOT)
    assert j["labels"]["effect_present"] == "not_shown"
    j = ins.judge_seed(RULE, cells(0.90, 0.0), PASSED, A, STOVE, BOOT)
    assert j["labels"] == {"effect_present": "effect_present", "retains": "retains"}
    j = ins.judge_seed(RULE, cells(0.70, 0.0, neg_b=0.70, ctrl_b=0.96), PASSED, A, STOVE, BOOT)
    assert j["acquisition"]["state"] == "at_acquisition_cost"
    assert j["labels"]["effect_present"] == "effect_present_at_acquisition_cost"
    j = ins.judge_seed(RULE, cells(0.70, 0.0, neg_b=0.40), PASSED, A, STOVE, BOOT)
    assert j["labels"] == {"effect_present": "not_assessed", "retains": "not_assessed"}
    j = ins.judge_seed(RULE, cells(0.70, 0.0), {"passed": False}, A, STOVE, BOOT)
    assert j["outcome"] == "gate_failed" and j["labels"]["effect_present"] == "not_assessed"
    assert ins.judge_seed(RULE, cells(0.7, 0.0), None, A, STOVE, BOOT)["outcome"] == "gate_missing"
    j = ins.judge_seed(RULE, {"neg": cells(0.7, 0.0)["neg"]}, PASSED, A, STOVE, BOOT)
    assert j["acquisition"]["state"] == "missing" and j["labels"]["effect_present"] == "not_assessed"


def test_the_reading_across_seeds():
    assert ins.across_seeds({"1": "effect_present", "2": "effect_present_at_acquisition_cost"}) == "replicated"
    assert ins.across_seeds({"1": "effect_present", "2": "not_shown"}) == "mixed"
    assert ins.across_seeds({"1": "not_shown", "2": "not_shown"}) == "not_replicated"
    assert ins.across_seeds({"1": "effect_present", "2": "not_assessed"}) == "not_assessed"
    assert ins.across_seeds({"1": None}) == "not_assessed" and ins.across_seeds({}) == "not_assessed"


# ---- validity --------------------------------------------------------------------------------------------


def fake_control(path, *, method=None, ns="similar_ab_stove__seq_ft__seed1"):
    path.mkdir(parents=True)
    OmegaConf.save({"seed_namespace_run_id": ns, "method": method or {"name": "gpm", "eps": 0.95},
                    "train": {"steps": 30000, "batch_size": 64, "lr": 1.0e-4, "amp": True}}, path / "config.yaml")
    (path / "result.json").write_text("{}")
    (path / "git_sha").write_text("abc\n")
    return path


def test_control_checks(tmp_path, no_memory_check):
    sub = ins.stove_sub(CFG, 1)
    assert ins.control_checks(sub, fake_control(tmp_path / "ok"), False)["passed"]
    c = ins.control_checks(sub, fake_control(tmp_path / "neg", method={"name": "gpm", "negative_bank": "x"}), False)
    assert not c["passed"] and not c["no_negatives"]
    assert not ins.control_checks(sub, fake_control(tmp_path / "m", method={"name": "gpm_neg"}), False)["method"]
    assert not ins.control_checks(sub, fake_control(tmp_path / "ns", ns="other"), False)["namespace"]


def test_resumed_checks_without_negatives_for_the_plain_gpm_control(tmp_path, no_memory_check):
    sub = cur.sub_config(cur.load_config(), "cabinet")
    run, src, pin, bank = fake_resumed(tmp_path / "x")
    conf = OmegaConf.load(run / "config.yaml")
    conf.method.name = "gpm"
    OmegaConf.save(conf, run / "config.yaml")
    torch.save({"state_dict": {}, "extra": {"method_state": {}}}, run / "checkpoints" / "stage1.pt")
    ctrl_sub = {**sub, "runs": {"ctrl": {"method": "gpm", "name": run.name}}}
    c = cur.run_checks_resumed(ctrl_sub, run, "ctrl", src, pin, False, bank, 50, negatives=False)
    assert c["passed"] and "negatives" not in c and "memory" in c and "frozen" not in c
    c = cur.run_checks_resumed(ctrl_sub, run, "ctrl", src, pin, False, bank, 50)  # negatives expected: fails
    assert not c["passed"] and not c["negatives"]["passed"]


def test_seed0_rows_come_from_the_pinned_summaries(tmp_path):
    p = CFG["pins"]
    stove = {"status": "valid", "judgment": {"labels": {"effect_present": "effect_present"}},
             "cells": {"gpm": {"1": {A: 0.7, STOVE: 0.96}}, "hist_gpm": {"1": {A: 0.0, STOVE: 0.98}}}}
    curricula = {"curricula": {"cabinet": {"judgment": {"labels": {"effect_present": "not_shown"}},
                                           "cells": {"gpm": {"1": {A: 0.04, CAB: 0.96}},
                                                     "hist_gpm": {"1": {A: 0.0, CAB: 0.98}}}}}}
    for key, rec in (("seed0_stove_summary", stove), ("seed0_curricula_summary", curricula)):
        f = tmp_path / p[key]["path"]
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(rec))
    rows = ins.seed0_rows(CFG, tmp_path)
    assert rows["stove"]["effect_present"] == "effect_present" and rows["stove"]["neg"]["plate_after_B"] == 0.7
    assert rows["cabinet"]["ctrl"] == {"plate_after_B": 0.0, "B": 0.98}
    assert rows["cabinet"]["flag"] == "against_historical_control"


def test_probe_targets_use_each_arms_source(tmp_path):
    for arm, name in (("neg", "similar_ab_stove__gpm_projected_adam_neg__seed1"),
                      ("ctrl", "similar_ab_stove__gpm_projected_adam__seed1")):
        d = tmp_path / name
        (d / "checkpoints").mkdir(parents=True)
        torch.save({"extra": {"method_artifacts": [{"path": "method/memory_task0.pt", "sha256": f"M{arm}"}]}},
                   d / "checkpoints" / "stage0.pt")
    t = ins.probe_targets(CFG, tmp_path, 1)
    base = t[("cabinet", "neg_stage0")]
    assert base["checkpoint"] == tmp_path / "similar_ab_stove__gpm_projected_adam_neg__seed1" / "checkpoints" / "stage0.pt"
    assert base["memory"]["expected_sha256"] == "Mneg"
    ctrl = t[("cabinet", "ctrl_stage1")]
    assert ctrl["checkpoint"] == tmp_path / "similar_ab_cabinet__gpm_projected_adam__seed1" / "checkpoints" / "stage1.pt"
    assert ctrl["stage0"].parent.parent.name == "similar_ab_stove__gpm_projected_adam__seed1"
    assert ctrl["memory"]["expected_sha256"] == "Mctrl"
    assert t[("stove", "gpm_stage1")]["memory"]["expected_sha256"] == "Mneg"
