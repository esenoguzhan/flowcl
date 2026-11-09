"""The 4-task baseline queue (scripts/queue_baselines.sh) in dry runs: step order and flags, the arms only
on seed 0, the identity stop at stage 0 against the local skip at stage 1, --from-step's prior checks; and
the preflight's config and hash checks (flowcl/experiments/baseline_queue.py)."""

from __future__ import annotations

import os
import re
import subprocess

import pytest

from flowcl.experiments import baseline_queue as bq
from flowcl.utils.libero_paths import repo_root

QUEUE = repo_root() / "scripts" / "queue_baselines.sh"
ORDER = ["ewc", "lora", "replay", "consft", "inflora", "projected_lora"]


def dry_run(tmp_path, args=(), fail="", identity=""):
    env = {**os.environ, "QUEUE_DRY_RUN": "1", "QUEUE_FAIL_STEP": fail, "QUEUE_IDENTITY_FAIL": identity,
           "QUEUE_LOG_ROOT": str(tmp_path / "logs"), "QUEUE_RESULTS_ROOT": str(tmp_path / "res")}
    done = subprocess.run(["bash", str(QUEUE), *args], env=env, capture_output=True, text=True, timeout=300)
    (logdir,) = list((tmp_path / "logs").glob("queue_*"))
    return done.returncode, (logdir / "queue.log").read_text(), logdir


def started(log):
    return re.findall(r"^\S+ \S+ START (\S+): ", log, flags=re.M)


def test_queue_syntax():
    assert subprocess.run(["bash", "-n", str(QUEUE)]).returncode == 0
    assert tuple(ORDER) == bq.METHODS


def test_seed0_runs_every_method_in_order_with_the_arms(tmp_path):
    rc, log, logdir = dry_run(tmp_path)
    assert rc == 0 and log.rstrip().endswith("QUEUE DONE ok")
    expected = ["0_preflight"]
    for i, m in enumerate(ORDER):
        k = 1 + 3 * i
        expected += [f"{k}_run_{m}", f"{k + 1}_check_{m}", f"{k + 2}_report_{m}"]
    assert started(log) == expected
    ref = f"{tmp_path}/res/seq_hetero__seq_ft__seed0"
    for i, m in enumerate(ORDER):
        k = 1 + 3 * i
        run = (logdir / f"{k}_run_{m}.log").read_text()
        assert f"--method {m} " in run and f"--t1-reference-run {ref} --identity-reference-run {ref}" in run
        assert ("--identity-reference-checkpoint 1=" in run) == (m != "replay"), m
        if m != "replay":
            assert str(bq.arm_checkpoint(m)) in run
        check = (logdir / f"{k + 1}_check_{m}.log").read_text()
        assert ("--gpm-memory" in check) == (m in ("inflora", "projected_lora"))
        assert "--strict-provenance" in (logdir / f"{k + 2}_report_{m}.log").read_text()
    assert "--seed 0 --methods ewc lora replay consft inflora projected_lora" in \
        (logdir / "0_preflight.log").read_text()


def test_seed1_has_no_arm_flags(tmp_path):
    rc, log, logdir = dry_run(tmp_path, ["--seed", "1"])
    assert rc == 0
    for i, m in enumerate(ORDER):
        run = (logdir / f"{1 + 3 * i}_run_{m}.log").read_text()
        assert "--identity-reference-checkpoint" not in run and "seq_ft__seed1" in run
    assert "--seed 1 " in (logdir / "0_preflight.log").read_text()


def test_a_stage0_identity_failure_stops_every_later_method(tmp_path):
    rc, log, _ = dry_run(tmp_path, fail="4_run_lora", identity="4_run_lora:0")
    assert rc == 1
    steps = started(log)
    assert steps == ["0_preflight", "1_run_ewc", "2_check_ewc", "3_report_ewc", "4_run_lora"]
    assert "STOP: stage-0 identity failed for lora" in log
    assert "QUEUE DONE FAILED: 4_run_lora 7_run_replay(stopped)" in log


def test_a_stage1_identity_failure_skips_only_that_method(tmp_path):
    rc, log, _ = dry_run(tmp_path, fail="4_run_lora", identity="4_run_lora:1")
    assert rc == 1
    steps = started(log)
    assert "5_check_lora" not in steps and "6_report_lora" not in steps
    assert steps[-3:] == ["16_run_projected_lora", "17_check_projected_lora", "18_report_projected_lora"]
    assert "stage-1 identity failed for lora" in log
    assert log.rstrip().endswith("QUEUE DONE FAILED: 4_run_lora")


def test_a_failed_check_skips_only_its_report(tmp_path):
    rc, log, _ = dry_run(tmp_path, fail="2_check_ewc")
    steps = started(log)
    assert rc == 1 and "3_report_ewc" not in steps and "4_run_lora" in steps


def test_a_refused_preflight_runs_nothing(tmp_path):
    rc, log, _ = dry_run(tmp_path, fail="0_preflight")
    assert rc == 1 and started(log) == ["0_preflight"]
    assert "STOP: the preflight refused the launch" in log


def test_from_step_checks_prior_runs_and_reports(tmp_path):
    rc, log, _ = dry_run(tmp_path, ["--from-step", "4"])
    assert rc == 0
    assert started(log)[:4] == ["2_check_ewc", "3_prior_report_ewc", "4_run_lora", "5_check_lora"]
    rc, log, _ = dry_run(tmp_path / "b", ["--from-step", "4"], fail="2_check_ewc")
    assert rc == 1 and "3_prior_report_ewc" not in started(log) and "4_run_lora" in started(log)


def test_bad_arguments_are_rejected(tmp_path):
    for args in (["--seed", "3"], ["--from-step", "19"], ["--from-step", "x"], ["--smoke", "rel"]):
        env = {**os.environ, "QUEUE_DRY_RUN": "1", "QUEUE_LOG_ROOT": str(tmp_path)}
        done = subprocess.run(["bash", str(QUEUE), *args], env=env, capture_output=True, text=True, timeout=60)
        assert done.returncode == 2, args


# ---- the preflight (needs the bracket results on disk) -------------------------------------------------

needs_results = pytest.mark.skipif(
    not all((repo_root() / s["bundle"]).is_file() for s in bq.BRACKETS.values()),
    reason="the selection bundles are not on this machine")


@needs_results
def test_every_production_config_matches_its_arm():
    for method in bq.METHODS:
        check = bq.config_check(method)
        assert check["passed"], (method, check["problems"])


@needs_results
def test_a_doctored_config_is_refused(monkeypatch):
    real = bq.load_method_config

    def doctored(name):
        registry, kw = real(name)
        if name == "ewc":
            kw = {**kw, "fisher_samples": 512}
        if name == "inflora":
            kw = {**kw, "variant": "trained"}
        if name == "lora":
            kw = {**kw, "dropout": 0.1}
        return registry, kw

    monkeypatch.setattr(bq, "load_method_config", doctored)
    assert bq.config_check("ewc")["problems"] == ["fisher_samples: production 512, arm 1024"]
    assert any(p.startswith("variant:") for p in bq.config_check("inflora")["problems"])
    assert bq.config_check("lora")["problems"] == [
        "dropout: a production key the arm does not record and not runner-only"]
    assert bq.config_check("replay")["passed"]


@needs_results
def test_the_preflight_refuses_a_seed1_reference_hash_mismatch_and_a_busy_gpu(monkeypatch, tmp_path):
    monkeypatch.setitem(bq.SEQ_FT_STAGE0, 1, "0" * 64)
    out = bq.preflight(1, list(bq.METHODS), results=tmp_path, gpu_pids=lambda: ["123"])
    assert not out["passed"]
    refs = out["checks"]["references"]
    assert not refs["files"]["seq_ft_stage0"]["passed"] and refs["files"]["gpm_memory0"]["passed"]
    assert out["checks"]["gpu_idle"] == {"passed": False, "compute_pids": ["123"]}
    assert out["checks"]["brackets"]["passed"] and out["checks"]["configs"]["passed"]
    assert "arm_evals" not in out["checks"]
    (tmp_path / "seq_hetero__ewc__seed1").mkdir()
    out = bq.preflight(1, ["ewc"], results=tmp_path, gpu_pids=lambda: [])
    assert out["checks"]["targets_absent"]["existing"] == [str(tmp_path / "seq_hetero__ewc__seed1")]


def test_expected_identity_stages():
    assert bq.expected_identity_stages("ewc", 0) == [0, 1]
    assert bq.expected_identity_stages("replay", 0) == [0]
    assert bq.expected_identity_stages("inflora", 2) == [0]
