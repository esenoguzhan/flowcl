"""The f = 0.7 queue (scripts/queue_ne70.sh, flowcl/experiments/ne70_queue.py; pre-E3 plan, A3): the
method config, the dry-run step order and flags, the identity stop, the smoke mode, and the preflight and
per-run checks."""

from __future__ import annotations

import json
import os
import re
import subprocess

import pytest
from omegaconf import OmegaConf

from flowcl.experiments import ne70_queue as nq
from flowcl.utils.libero_paths import repo_root

QUEUE = repo_root() / "scripts" / "queue_ne70.sh"


def dry_run(tmp_path, args=(), fail="", identity=""):
    env = {**os.environ, "QUEUE_DRY_RUN": "1", "QUEUE_FAIL_STEP": fail, "QUEUE_IDENTITY_FAIL": identity,
           "QUEUE_LOG_ROOT": str(tmp_path / "logs"), "QUEUE_RESULTS_ROOT": str(tmp_path / "res")}
    done = subprocess.run(["bash", str(QUEUE), *args], env=env, capture_output=True, text=True, timeout=300)
    logs = list((tmp_path / "logs").glob("queue_*"))
    if not logs:
        return done.returncode, done.stderr, None
    return done.returncode, (logs[0] / "queue.log").read_text(), logs[0]


def started(log):
    return re.findall(r"^\S+ \S+ START (\S+): ", log, flags=re.M)


def test_the_config_is_gpm_with_f_07_and_names_the_run():
    from flowcl.data.config import load_method_config
    from flowcl.methods.gpm import GPM

    assert nq.config_check()["passed"]
    name, kwargs = load_method_config("gpm_ne70")
    assert name == "gpm" and GPM(**kwargs).display_name == nq.LABEL == "gpm_projected_adam_ne70"


def test_queue_syntax():
    assert subprocess.run(["bash", "-n", str(QUEUE)]).returncode == 0


def test_three_seeds_in_order_with_their_flags(tmp_path):
    rc, log, logdir = dry_run(tmp_path)
    assert rc == 0 and log.rstrip().endswith("QUEUE DONE ok")
    expected = ["0_preflight"]
    for i, s in enumerate((0, 1, 2)):
        k = 1 + 3 * i
        expected += [f"{k}_run_s{s}", f"{k + 1}_check_s{s}", f"{k + 2}_report_s{s}"]
    assert started(log) == expected
    res = tmp_path / "res"
    for i, s in enumerate((0, 1, 2)):
        k = 1 + 3 * i
        run = (logdir / f"{k}_run_s{s}.log").read_text()
        assert "--method gpm_ne70" in run and f"--seed {s}" in run and "--amp" in run
        assert f"--t1-reference-run {res}/seq_hetero__seq_ft__seed{s}" in run
        assert f"--identity-reference-run {res}/seq_hetero__gpm_projected_adam__seed{s} --identity-stages 0 1" in run
        assert "--steps-per-task" not in run and "--allow-dirty" not in run and "--no-eval-unseen" not in run
        check = (logdir / f"{k + 1}_check_s{s}.log").read_text()
        assert f"--gpm-run {res}/seq_hetero__gpm_projected_adam__seed{s}" in check
        report = (logdir / f"{k + 2}_report_s{s}.log").read_text()
        assert "--strict-provenance" in report and "--allow-dirty" not in report
        assert f"--out {res}/gpm_seq_ne70_seed{s}/report.json" in report
    assert "--preflight --seeds 0 1 2" in (logdir / "0_preflight.log").read_text()


def test_an_identity_failure_stops_every_later_seed(tmp_path):
    rc, log, _ = dry_run(tmp_path, fail="1_run_s0", identity="1_run_s0")
    assert rc == 1 and "STOP identity failure at seed 0" in log
    assert started(log) == ["0_preflight", "1_run_s0"]
    assert "seed1(identity_stop)" in log and "seed2(identity_stop)" in log


def test_another_run_failure_skips_only_that_seed(tmp_path):
    rc, log, _ = dry_run(tmp_path, fail="4_run_s1")
    assert rc == 1 and "7_run_s2" in started(log) and "5_check_s1" not in started(log)


def test_a_refused_preflight_stops_the_queue(tmp_path):
    rc, log, _ = dry_run(tmp_path, fail="0_preflight")
    assert rc == 1 and started(log) == ["0_preflight"]


def test_smoke_mode(tmp_path):
    smoke, refs = tmp_path / "smoke", tmp_path / "refs"
    rc, log, logdir = dry_run(tmp_path, ["--smoke", str(smoke), "--smoke-refs", str(refs)])
    assert rc == 0 and started(log) == ["0_preflight", "1_run_s2", "2_check_s2", "3_report_s2"]
    run = (logdir / "1_run_s2.log").read_text()
    assert f"--results-root {smoke} --steps-per-task 200 --n-episodes 1 --allow-dirty" in run
    assert f"--t1-reference-run {refs}/seq_hetero__seq_ft__seed2" in run
    assert "--strict-provenance --allow-dirty" in (logdir / "3_report_s2.log").read_text()
    assert f"--smoke --results-root {smoke} --refs {refs}" in (logdir / "0_preflight.log").read_text()
    assert dry_run(tmp_path / "x", ["--smoke", str(smoke)])[0] == 2       # refs required
    assert dry_run(tmp_path / "y", ["--smoke", "relative/dir", "--smoke-refs", str(refs)])[0] == 2


def _refs(root, seed=2):
    for name, files in ((f"seq_hetero__seq_ft__seed{seed}", ["checkpoints/stage0.pt"]),
                        (f"seq_hetero__gpm_projected_adam__seed{seed}",
                         ["checkpoints/stage0.pt", "checkpoints/stage1.pt", "method/memory_task0.pt"])):
        for f in files:
            (root / name / f).parent.mkdir(parents=True, exist_ok=True)
            (root / name / f).write_bytes(b"x")


def test_preflight_in_smoke_mode(tmp_path):
    kw = dict(results=tmp_path / "res", smoke=True, refs=tmp_path / "refs", gpu_pids=lambda: [],
              disk=lambda p: 100.0)
    (tmp_path / "res").mkdir()
    assert not nq.preflight([2], **kw)["checks"]["references"]["passed"]
    _refs(tmp_path / "refs")
    out = nq.preflight([2], **kw)
    assert out["passed"] and "clean_tree" not in out["checks"]
    assert not nq.preflight([2], **{**kw, "gpu_pids": lambda: ["123"]})["passed"]
    assert not nq.preflight([2], **{**kw, "disk": lambda p: 1.0})["passed"]
    (tmp_path / "res" / "seq_hetero__gpm_projected_adam_ne70__seed2").mkdir()
    assert not nq.preflight([2], **kw)["checks"]["targets_absent"]["passed"]


def test_production_preflight_pins_the_references(tmp_path):
    _refs(tmp_path / "res", seed=0)
    out = nq.preflight([0], results=tmp_path / "res", gpu_pids=lambda: [], disk=lambda p: 100.0)
    files = out["checks"]["references"]["files"]
    assert not out["checks"]["references"]["passed"]
    assert files["gpm_stage1_s0"]["expected"] == nq.GPM_STAGE[0][1]


def _run(tmp_path, *, stages=(0, 1), passed=True, t1=True, gpm=None):
    run = tmp_path / "run"
    run.mkdir(parents=True)
    gpm = gpm or tmp_path / "gpm"
    for k in (0, 1):
        (gpm / "checkpoints").mkdir(parents=True, exist_ok=True)
        (gpm / "checkpoints" / f"stage{k}.pt").write_bytes(f"stage{k}".encode())
    from flowcl.utils.run import file_sha256

    checks = {str(k): {"passed": passed, "reference_sha256": file_sha256(gpm / "checkpoints" / f"stage{k}.pt")}
              for k in stages}
    (run / "result.json").write_text(json.dumps({"identity_checks": checks, "t1_pairing": {"passed": t1}}))
    OmegaConf.save(OmegaConf.create({"provenance": {"identity_stages": list(stages),
                                                    "identity_reference_run": str(gpm)}}), run / "config.yaml")
    return run, gpm


@pytest.fixture
def memories_equal(monkeypatch):
    from flowcl.experiments import lora_selection

    state = {"passed": True}
    monkeypatch.setattr(lora_selection, "compare_memories",
                        lambda a, b: {"passed": state["passed"], "layers": 3,
                                      "different": [] if state["passed"] else ["x"]})
    return state


def test_check_run(tmp_path, memories_equal):
    run, gpm = _run(tmp_path / "ok")
    assert nq.check_run(run, gpm)["passed"]
    run, gpm = _run(tmp_path / "stages", stages=(0,))
    assert "identity stages configured [0]" in nq.check_run(run, gpm)["problems"][0]
    run, gpm = _run(tmp_path / "failed", passed=False)
    assert not nq.check_run(run, gpm)["passed"]
    run, gpm = _run(tmp_path / "t1", t1=False)
    assert "T1 pairing missing or failed" in nq.check_run(run, gpm)["problems"]
    run, gpm = _run(tmp_path / "changed")
    (gpm / "checkpoints" / "stage1.pt").write_bytes(b"other")
    assert any("has another" in p for p in nq.check_run(run, gpm)["problems"])
    memories_equal["passed"] = False
    run, gpm = _run(tmp_path / "memory")
    assert any("Task-1 memory" in p for p in nq.check_run(run, gpm)["problems"])
