"""The sequence report's method-dependent provenance checks (flowcl/experiments/sequence_report.py):
explicit not-applicable records outside the GPM family, projected LoRA's merge records, fail-closed
identity and T1 pairing, the bracket-arm evaluation diagnostic, strict mode, and GPM reports that
keep exactly their old checks (frozen consumers read them)."""

from __future__ import annotations

import json
import runpy
import shutil

import pytest
from omegaconf import OmegaConf

import flowcl.train.continual as continual_module
from flowcl.analysis.metrics import success_estimate
from flowcl.data.config import load_embodiment_spec
from flowcl.envs.evaluation import EvaluationReport, TaskEvaluation
from flowcl.envs.libero_env import EvalConfig
from flowcl.experiments import sequence_report as sr
from flowcl.experiments.reverse_order import report_check
from flowcl.train.trainer import TrainConfig
from flowcl.utils.libero_paths import repo_root
from test_continual import TINY_POLICY, fake_evaluation
from test_resume import trio

REF = "test_trio__seq_ft__seed0"
LEGACY_KEYS = {"clean_git_sha", "seed_namespace", "t1_pairing", "occupancy_non_decreasing", "memory_chained",
               "residuals_within_bound", "artifact_hashes_match", "frozen_from_stage1"}


_load_report_config = sr.load_report_config


def report_config():
    cfg = _load_report_config()
    cfg["criteria"]["expected_thresholds_by_reference"] = {REF: {"mode": "derived", "margin": 0.15}}
    cfg["criteria"]["fallback_tasks"] = [2]
    return cfg


@pytest.fixture(scope="module")
def world(tmp_path_factory, dataset_dir):
    """Tiny three-task runs (fake evaluations): seq_ft, and ewc, lora, gpm and projected_lora paired with it."""
    root = tmp_path_factory.mktemp("world")
    capture_cfg = OmegaConf.load(repo_root() / "configs" / "analysis" / "subspace.yaml")
    capture_cfg.min_samples_per_dim = 0.01
    capture_cfg.num_workers = 0
    capture = root / "capture.yaml"
    OmegaConf.save(capture_cfg, capture)
    train = TrainConfig(steps=2, batch_size=2, num_workers=0, device="cpu", log_every=0, warmup_steps=1)
    common = dict(spec=load_embodiment_spec("libero_franka"), policy_config=TINY_POLICY, train_cfg=train,
                  eval_cfg=EvalConfig(n_episodes=2), seed=0, dataset_dir=dataset_dir, pretrained=False,
                  results_root=root)
    memory_kw = {"update_memory": True, "capture_config": str(capture), "log_interval": 1}
    runs = {}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(continual_module, "evaluate_tasks", fake_evaluation([]))
        ref = continual_module.run_continual(trio(), method_name="seq_ft", **common)
        ref_dir = root / ref.run_id
        paired = dict(t1_reference_run=ref_dir, identity_reference_run=ref_dir, identity_stages=(0,))
        for key, name, kw in (
            ("ewc", "ewc", {"lam": 1e-3, "fisher_samples": 4, "fisher_estimator": "sampled",
                            "fisher_normalization": "mean"}),
            ("lora", "lora", {"rank": 2, "lr": 5e-4, "alpha": 16}),
            ("gpm", "gpm", memory_kw),
            ("projected_lora", "projected_lora", {"variant": "trained", "rank": 2, "lr": 1e-3, **memory_kw}),
        ):
            extra = paired if key != "gpm" else {"t1_reference_run": ref_dir}
            result = continual_module.run_continual(trio(), method_name=name, method_kwargs=kw, **extra, **common)
            runs[key] = root / result.run_id
    return ref_dir, runs


def report(world, key, **kw):
    ref, runs = world
    return sr.build_report(runs[key], ref, config=report_config(), **kw)


# ---- the method families ------------------------------------------------------------------------------


def test_ewc_gets_explicit_not_applicable_records_and_passes_strict(world):
    r = report(world, "ewc")
    checks = r["provenance_checks"]
    for name in ("occupancy_non_decreasing", "memory_chained", "residuals_within_bound", "frozen_from_stage1"):
        assert checks[name]["passed"] is None and isinstance(checks[name]["not_applicable"], str), name
    assert checks["artifact_hashes_match"]["passed"] and checks["t1_pairing"]["passed"]
    assert checks["identity"]["passed"] and checks["identity"]["expected_stages"] == [0]
    assert r["pilot_comparison"] == {"skipped": "not a GPM-family method (ewc)"}
    assert r["method_registry_name"] == "ewc"
    assert sr.strict_failures(r, allow_dirty=True) == []


def test_lora_keeps_everything_outside_the_allowlist_frozen(world):
    checks = report(world, "lora")["provenance_checks"]
    assert checks["frozen_from_stage1"]["passed"]
    assert checks["frozen_from_stage1"]["n_tensors_outside_allowlist"] > 0
    assert checks["occupancy_non_decreasing"]["not_applicable"]


def test_projected_lora_reads_its_merge_records(world):
    r = report(world, "projected_lora")
    checks = r["provenance_checks"]
    res = checks["residuals_within_bound"]
    assert res["passed"] and res["source"] == "merge records" and sorted(res["stages"]) == ["1", "2"]
    assert checks["occupancy_non_decreasing"]["passed"] and checks["memory_chained"]["passed"]
    assert checks["frozen_from_stage1"]["passed"]
    assert sr.strict_failures(r, allow_dirty=True) == []


def payloads_with(records):
    out = [{"extra": {"method_state": {"history": {}}}}]
    for k, rec in enumerate(records, start=1):
        history = {} if rec is None else {str(k): {"merge": rec}}
        out.append({"extra": {"method_state": {"history": history}}})
    return out


def merge(ok=True, over=0.5):
    return {"constraint_ok": ok, "max_residual_over_tolerance": over,
            "per_layer": {"a": {"realized_residual": over, "tolerance": 1.0}}}


def test_merge_constraint_fails_on_a_violation_or_a_missing_stage():
    assert sr.merge_constraint_check(payloads_with([merge(), merge()]))["passed"]
    assert not sr.merge_constraint_check(payloads_with([merge(), merge(ok=False)]))["passed"]
    assert not sr.merge_constraint_check(payloads_with([merge(), merge(over=1.5)]))["passed"]
    missing = sr.merge_constraint_check(payloads_with([merge(), None]))
    assert not missing["passed"] and missing["problems"] == ["stage 2: no merge record"]


def test_a_gpm_run_with_a_memory_missing_fails(world, tmp_path):
    ref, runs = world
    copy = tmp_path / runs["gpm"].name
    shutil.copytree(runs["gpm"], copy)
    (copy / "method" / "memory_task1.pt").unlink()
    checks = sr.provenance_checks(sr.load_run(copy), sr.load_run(ref), verify_checkpoints=False)
    assert not checks["occupancy_non_decreasing"]["passed"]
    assert checks["occupancy_non_decreasing"]["missing_memories"] == [1]


def test_gpm_reports_keep_their_old_checks_for_the_frozen_consumers(world, tmp_path):
    r = report(world, "gpm")
    assert set(r["provenance_checks"]) == LEGACY_KEYS
    assert "skipped" not in r.get("pilot_comparison", {})
    path = tmp_path / "gpm.json"
    path.write_text(json.dumps(r, default=str))
    assert report_check(path, smoke=True)["passed"]


# ---- identity, T1 pairing, the bracket arm -------------------------------------------------------------


def test_identity_is_fail_closed():
    prov = {"identity_reference_run": "ref", "identity_stages": [0],
            "identity_reference_checkpoints": {"1": "arm/checkpoints/final.pt"}}
    good = {"0": {"reference_checkpoint": "ref/checkpoints/stage0.pt", "passed": True, "n_different": 0},
            "1": {"reference_checkpoint": "arm/checkpoints/final.pt", "passed": True, "n_different": 0}}
    assert sr.identity_check({"identity_checks": good}, prov)["passed"]
    assert sr.identity_check({"identity_checks": good}, {}) is None
    missing = sr.identity_check({"identity_checks": {"0": good["0"]}}, prov)
    assert not missing["passed"] and missing["problems"] == ["stage 1: no identity check recorded"]
    elsewhere = {**good, "1": {**good["1"], "reference_checkpoint": "other/checkpoints/final.pt"}}
    assert not sr.identity_check({"identity_checks": elsewhere}, prov)["passed"]
    differs = {**good, "0": {**good["0"], "passed": False, "n_different": 3}}
    assert not sr.identity_check({"identity_checks": differs}, prov)["passed"]


def test_t1_pairing_is_fail_closed_when_configured():
    assert not sr.t1_pairing_check({}, {"t1_reference_run": "ref"})["passed"]
    assert sr.t1_pairing_check({}, {}) is None
    assert sr.t1_pairing_check({"t1_pairing": {"passed": True}}, {"t1_reference_run": "ref"})["passed"]


def cell(key, succ, seeds=None, steps=None):
    seeds = list(range(len(succ))) if seeds is None else seeds
    steps = [10] * len(succ) if steps is None else steps
    return TaskEvaluation(key, succ, steps, seeds, success_estimate(succ))


def bracket_world(tmp_path, arm_cells, write_eval=True):
    keys = [f"suite{i}/task{i}" for i in range(4)]
    run = sr.RunView(tmp_path / "run", {"task_keys": keys}, {
        1: EvaluationReport(run_id="ns", stage=1, tasks=[cell(k, [True, False, True]) for k in keys])})
    arm = tmp_path / "arm"
    (arm / "checkpoints").mkdir(parents=True)
    if write_eval:
        EvaluationReport(run_id="ns", stage=1, tasks=arm_cells(keys)).save(arm / "eval.json")
    prov = {"identity_reference_checkpoints": {"1": str(arm / "checkpoints" / "final.pt")}}
    return run, prov


def test_bracket_eval_counts_disagreements_and_fails_only_a_broken_pairing(tmp_path):
    run, prov = bracket_world(tmp_path / "a", lambda k: [cell(k[0], [True, False, True]),
                                                       cell(k[1], [True, True, True])])
    check = sr.bracket_eval_check(run, prov)
    assert check["passed"] and sorted(check["tasks"]) == ["suite0/task0", "suite1/task1"]
    assert check["tasks"]["suite1/task1"]["success_disagreements"] == 1
    run, prov = bracket_world(tmp_path / "b", lambda k: [cell(k[0], [True, False, True], seeds=[0, 1, 5])])
    assert not sr.bracket_eval_check(run, prov)["passed"]
    run, prov = bracket_world(tmp_path / "c", lambda k: [cell("suite9/task9", [True, False, True])])
    assert not sr.bracket_eval_check(run, prov)["passed"]
    run, prov = bracket_world(tmp_path / "d", lambda k: [cell(k[0], [True]), cell(k[0], [True])])
    assert not sr.bracket_eval_check(run, prov)["passed"]
    run, prov = bracket_world(tmp_path / "e", None, write_eval=False)
    assert not sr.bracket_eval_check(run, prov)["passed"]
    assert sr.bracket_eval_check(run, {"identity_reference_checkpoints": {"1": "r/checkpoints/stage1.pt"}}) is None


# ---- strict mode --------------------------------------------------------------------------------------


def test_strict_accepts_only_passes_and_explicit_not_applicable():
    ok = {"provenance_required": ["a", "b"], "provenance_checks": {
        "a": {"passed": True}, "b": sr.not_applicable("reason")},
        "outcome": {"plasticity": {"0": {"ok": False}}}}
    assert sr.strict_failures(ok) == []
    assert sr.strict_failures({**ok, "provenance_checks": {**ok["provenance_checks"], "b": {"passed": None}}}) \
        == ["b: no verdict"]
    assert sr.strict_failures({**ok, "provenance_checks": {"a": {"passed": True}}}) == ["b: absent"]
    assert sr.strict_failures({**ok, "provenance_checks": {**ok["provenance_checks"], "c": None}}) \
        == ["c: absent"]
    dirty = {"provenance_required": ["clean_git_sha"], "provenance_checks": {"clean_git_sha": {"passed": False}}}
    assert sr.strict_failures(dirty) == ["clean_git_sha: failed"]
    assert sr.strict_failures(dirty, allow_dirty=True) == []


def test_strict_run_writes_the_report_then_raises(world, tmp_path, monkeypatch):
    ref, runs = world
    monkeypatch.setattr(sr, "load_report_config", report_config)
    monkeypatch.setattr(sr, "strict_failures", lambda report, allow_dirty=False: ["x: failed"])
    out = tmp_path / "r.json"
    with pytest.raises(sr.ProvenanceError, match="x: failed"):
        sr.run_sequence_report(runs["ewc"], ref, None, out, strict=True)
    assert json.loads(out.read_text())["method_registry_name"] == "ewc"


def test_the_cli_exits_3_on_a_strict_failure(monkeypatch):
    def fail(*args, **kwargs):
        assert kwargs["strict"] is True
        raise sr.ProvenanceError(["identity: failed"])

    monkeypatch.setattr(sr, "run_sequence_report", fail)
    monkeypatch.setattr("sys.argv", ["sequence_report.py", "--strict-provenance"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(repo_root() / "scripts" / "sequence_report.py"), run_name="__main__")
    assert exc.value.code == 3
