"""Instruction negatives on a second pair (rule ``wine``): the per-seed configs, the bank and pins, the panel
conditions, what is reported for every completed seed (a gate-failed seed included), the improvement flag,
the panel's own rows against the evaluation, the summary on fake runs (preliminary against final), and
probes skipped when a run is missing."""

from __future__ import annotations

import json

import pytest
import torch

from flowcl.analysis.metrics import success_estimate
from flowcl.envs.evaluation import EvaluationReport, TaskEvaluation
from flowcl.experiments import instruction_negatives as inn
from flowcl.experiments import instruction_negatives_pair as pair
from flowcl.experiments import instruction_negatives_seeds as ins
from flowcl.methods.instruction_negatives import load_bank
from flowcl.utils.run import file_sha256

CFG = pair.load_config("wine")
A, B = CFG["task_a"], CFG["task_b"]
BOOT = {"seed": 0, "n_resamples": 2000, "confidence": 0.95}


def s(p, n=50):
    k = int(round(p * n))
    return [True] * k + [False] * (n - k)


# ---- config, bank, pins ------------------------------------------------------------------------------------


def test_sub_config_names_the_seed_runs(tmp_path):
    sub = pair.sub_config(CFG, 2)
    assert sub["namespace"] == "similar_wine_rack__seq_ft__seed2" and sub["task_a"] == A and sub["task_b"] == B
    assert inn.run_dir(sub, tmp_path, "gpm") == tmp_path / "similar_wine_rack__gpm_projected_adam_neg__seed2"
    assert inn.run_dir(sub, tmp_path, "ctrl") == tmp_path / "similar_wine_rack__gpm_projected_adam__seed2"
    assert inn.out_dir(sub, tmp_path) == tmp_path / "instruction_negatives_wine" / "seed2"
    assert sub["runs"]["gpm"]["method"] == "gpm_neg" and sub["runs"]["ctrl"]["method"] == "gpm"
    hold = pair.sub_config(CFG, None)
    assert inn.out_dir(hold, tmp_path) == tmp_path / "instruction_negatives_wine" and hold["task_a"] == A
    stove = inn.load_config()
    for k in ("production", "smoke", "hold_check", "gate", "probe", "swap", "chunks", "coverage", "drift"):
        assert sub[k] == stove[k], k  # the bowl studies' recipe, thresholds and seed tags


def test_the_bank_keeps_the_held_out_words_out_and_the_pins_hold():
    bank = load_bank(CFG["bank"])
    assert bank["sha256"] == CFG["pins"]["bank_sha256"] and len(bank["strings"]) == 15
    assert bank["forbidden_words"] == ["rack", "bowl"]
    assert not any(w in x.split() for x in bank["strings"] for w in ("rack", "bowl"))
    held_out = ["put the wine bottle on the rack", "put the bowl on top of the cabinet"]
    for text in held_out + CFG["panel"]["paraphrases"] + [CFG["panel"]["held_out_other"][0]["text"]]:
        assert text not in bank["strings"], text
    assert all(t in bank["strings"] for t in CFG["panel"]["trained_negatives"])
    assert "put the wine bottle in the drawer" not in bank["strings"]  # GPT review: no drawer string next to A
    checks = pair.pin_checks(CFG)
    assert checks["passed"], checks


def test_the_panel_conditions_and_the_policys_instruction():
    from flowcl.data.tasks import TaskRef

    try:
        conds = inn.panel_conditions(pair.sub_config(CFG, 0))
    except FileNotFoundError:
        pytest.skip("the LIBERO-Goal demos are not on this machine")
    kinds = [c["kind"] for c in conds]
    assert kinds.count("held_out_task") == 2 and kinds.count("paraphrase") == 2 and kinds.count("own") == 1
    held = {c["requested"]: c["text"] for c in conds if c["kind"] == "held_out_task"}
    assert held == {B: "put the wine bottle on the rack",
                    "libero_goal/put_the_bowl_on_top_of_the_cabinet": "put the bowl on top of the cabinet"}
    own = next(c["text"] for c in conds if c["kind"] == "own")
    ref = TaskRef.from_key(A)
    assert own == "put the wine bottle on top of the cabinet" == ref.language  # training text = evaluation text
    ref.assert_consistent()


# ---- reported for every completed seed ---------------------------------------------------------------------


def cells(neg_a1, ctrl_a1, neg_a0=0.94, ctrl_a0=0.90, neg_b=0.96, ctrl_b=0.98, n=50):
    return {"neg": {0: {A: s(neg_a0, n)}, 1: {A: s(neg_a1, n), B: s(neg_b, n)}},
            "ctrl": {0: {A: s(ctrl_a0, n)}, 1: {A: s(ctrl_a1, n), B: s(ctrl_b, n)}}}


def test_a_gate_failed_seed_is_not_assessed_but_fully_reported():
    c = cells(0.30, 0.0, neg_a0=0.70)
    judged = ins.judge_seed(CFG["labels"], c, {"passed": False, "R0_A": 0.70}, A, B, BOOT)
    assert judged["labels"] == {"effect_present": "not_assessed", "retains": "not_assessed"}
    rep = pair.describe_seed(c, A, B, BOOT)
    assert rep["table"]["neg"] == {"A_before_B": 0.70, "A_after_B": 0.30, "B_after_B": 0.96,
                                   "forgetting": pytest.approx(0.40)}
    assert rep["table"]["ctrl"]["forgetting"] == pytest.approx(0.90)
    d = rep["neg_minus_ctrl"]["A_after_B"]
    assert d["diff"] == pytest.approx(0.30) and d["low"] > 0 and rep["improvement_ci_low_above_0"] is True
    assert rep["neg_minus_ctrl"]["A_before_B"]["diff"] == pytest.approx(-0.20)


def test_the_improvement_flag_needs_the_interval_above_zero():
    assert pair.describe_seed(cells(0.28, 0.0), A, B, BOOT)["improvement_ci_low_above_0"] is True
    crossing = pair.describe_seed(cells(0.04, 0.0), A, B, BOOT)
    assert crossing["neg_minus_ctrl"]["A_after_B"]["low"] <= 0 and crossing["improvement_ci_low_above_0"] is False
    below = pair.describe_seed(cells(0.0, 0.40), A, B, BOOT)
    assert below["neg_minus_ctrl"]["A_after_B"]["high"] < 0 and below["improvement_ci_low_above_0"] is False
    assert pair.describe_seed({"neg": cells(0.3, 0.0)["neg"]}, A, B, BOOT)["improvement_ci_low_above_0"] is None


def report(stage, tasks, n, seeds=None, steps=None):
    return EvaluationReport(run_id="similar_wine_rack__seq_ft__seed0", stage=stage, tasks=[
        TaskEvaluation(k, list(succ), list(steps or [100] * n), list(seeds or range(n)), success_estimate(list(succ)))
        for k, succ in tasks])


def test_the_panels_own_rows_must_reproduce_the_evaluation(tmp_path):
    ev = tmp_path / "stage0.json"
    report(0, [(A, [True, False, True])], 3, seeds=[7, 8, 9], steps=[90, 600, 95]).save(ev)
    rows = [{"seed": 7, "own_goal": True, "n_steps": 90}, {"seed": 8, "own_goal": False, "n_steps": 600}]
    assert pair.panel_own_reproduces_eval({"conditions": {"own": {"rows": rows}}}, ev, A)["passed"]
    rows[1]["n_steps"] = 599
    c = pair.panel_own_reproduces_eval({"conditions": {"own": {"rows": rows}}}, ev, A)
    assert not c["passed"] and not c["n_steps"] and c["successes"]
    assert not pair.panel_own_reproduces_eval({"conditions": {}}, ev, A)["passed"]


# ---- the summary on fake runs (smoke shape: seed 0, 1 rollout) ----------------------------------------------


def fake_study(root, *, ctrl_seeds=None):
    for arm in pair.ARMS:
        run = root / pair.run_name(CFG, arm, 0)
        (run / "checkpoints").mkdir(parents=True)
        for st in (0, 1):
            torch.save({"state_dict": {"w": torch.full((2,), float(st))},
                        "extra": {"method_artifacts": [{"path": "method/memory_task0.pt", "sha256": "M"}]}},
                       run / "checkpoints" / f"stage{st}.pt")
        report(0, [(A, [True])], 1).save(run / "eval" / "stage0.json")
        report(1, [(A, [arm == "neg"]), (B, [True])], 1,
               seeds=ctrl_seeds if arm == "ctrl" and ctrl_seeds else None).save(run / "eval" / "stage1.json")
        (run / "result.json").write_text(json.dumps({"stages": [{}, {}]}))
        (run / "git_sha").write_text("abc\n")
    neg = root / pair.run_name(CFG, "neg", 0)
    out = root / CFG["out"]
    (out / "hold").mkdir(parents=True)
    (out / "hold" / "hold.json").write_text(json.dumps({"passed": True, "git_sha": "abc"}))
    panel = {"git_sha": "abc", "checkpoint_sha256": file_sha256(neg / "checkpoints" / "stage0.pt"),
             "weights_unchanged": True, "gate": {"passed": False, "R0_A": 1.0},
             "text_cache": {"tasks": {A: {"text": "put the wine bottle on top of the cabinet", "precision": "amp"}}},
             "conditions": {"own": {"kind": "own", "text": "x", "n": 1, "own_goal_rate": 1.0, "requested_rate": None,
                                    "median_max_displacement_m": 0.3, "eval_seeds_match": True,
                                    "rows": [{"seed": 0, "own_goal": True, "n_steps": 100}]}}}
    (out / "seed0" / "panel").mkdir(parents=True)
    (out / "seed0" / "panel" / "panel.json").write_text(json.dumps(panel))
    for label, arm, st in (("neg_stage0", "neg", 0), ("gpm_stage1", "neg", 1), ("ctrl_stage1", "ctrl", 1)):
        ck = root / pair.run_name(CFG, arm, 0) / "checkpoints" / f"stage{st}.pt"
        d = out / "seed0" / "probes" / label
        d.mkdir(parents=True)
        (d / "probe.json").write_text(json.dumps({"git_sha": "abc", "checkpoint_sha256": file_sha256(ck),
                                                  "weights_unchanged": True, "probe_loss": {}, "instruction": {}}))
    return out


@pytest.fixture
def passing_run_checks(monkeypatch):
    monkeypatch.setattr(inn, "_run_checks", lambda sub, run, arm, smoke, bank: {"passed": True})
    monkeypatch.setattr(ins, "control_checks", lambda sub, run, smoke: {"passed": True})


def test_the_summary_preliminary_against_final_and_a_pairing_defect(tmp_path, passing_run_checks):
    out = fake_study(tmp_path / "ok")
    pre = pair.run_summary(CFG, results_root=tmp_path / "ok", smoke=True, seeds=[0])
    assert pre["status"] == "valid" and pre["stage"] == "preliminary", pre["incomplete"] + pre["invalid"]
    assert (out / "summary_preliminary.json").is_file()
    blk = pre["per_seed"]["0"]
    assert blk["validity"]["panel_own_reproduces_eval"]["passed"] and blk["commits"]["panel"] == "abc"
    assert blk["gate"]["passed"] is False and blk["judgment"]["gate"]["smoke_override"]  # smoke: gate not applied
    assert blk["reported"]["table"]["neg"]["A_after_B"] == 1.0 and blk["policy_texts"][A]["precision"] == "amp"
    final = pair.run_summary(CFG, results_root=tmp_path / "ok", smoke=True, seeds=[0], final=True)
    assert final["status"] == "incomplete" and final["stage"] == "final"
    assert any("swaps missing" in x for x in final["incomplete"]) and (out / "summary.json").is_file()
    fake_study(tmp_path / "bad", ctrl_seeds=[5])
    bad = pair.run_summary(CFG, results_root=tmp_path / "bad", smoke=True, seeds=[0])
    assert bad["status"] == "invalid" and any("paired rows differ" in x for x in bad["invalid"])


def test_the_seeds_are_narrowed_in_smoke_only():
    with pytest.raises(ValueError, match="smoke only"):
        pair.seeds_of(CFG, False, [0])
    assert pair.seeds_of(CFG, False, None) == [0, 1, 2]


def test_probes_of_a_missing_run_are_skipped_not_failed(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(inn, "run_probe", lambda sub, label, target, **kw: calls.append(label) or {})
    run = tmp_path / pair.run_name(CFG, "ctrl", 0)
    (run / "checkpoints").mkdir(parents=True)
    for st in (0, 1):
        torch.save({"extra": {"method_artifacts": [{"path": "method/memory_task0.pt", "sha256": "M"}]}},
                   run / "checkpoints" / f"stage{st}.pt")
    problems, skipped = pair.run_probes(CFG, results_root=tmp_path, smoke=True, seeds=[0])
    assert problems == [] and calls == ["ctrl_stage1"]
    assert skipped == ["seed 0 neg_stage0", "seed 0 gpm_stage1"]
