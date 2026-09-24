"""The overnight queue never stops early: seq_ft seed 1 runs whatever fails before it."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from flowcl.utils.libero_paths import repo_root

QUEUE = repo_root() / "scripts" / "queue_2026-09-24_overnight.sh"


def dry_run(tmp_path, fail_step=""):
    env = {**os.environ, "QUEUE_DRY_RUN": "1", "QUEUE_FAIL_STEP": fail_step,
           "QUEUE_LOG_ROOT": str(tmp_path)}
    done = subprocess.run(["bash", str(QUEUE)], env=env, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    (logdir,) = list(tmp_path.glob("queue_*"))
    return (logdir / "queue.log").read_text()


def ran(log, step):
    return f"START {step}" in log


def test_queue_syntax():
    assert subprocess.run(["bash", "-n", str(QUEUE)]).returncode == 0


def test_all_steps_run_when_nothing_fails(tmp_path):
    log = dry_run(tmp_path)
    for step in ("1_gpm_ne90", "2_sequence_report", "3_diagnostics", "4_adaptive_report",
                 "5_seq_ft_seed1"):
        assert ran(log, step), step
    assert "SKIP" not in log and log.rstrip().endswith("QUEUE DONE")


def test_a_failed_variant_run_skips_its_analyses_but_not_seq_ft_seed1(tmp_path):
    log = dry_run(tmp_path, "1_gpm_ne90")
    assert "END 1_gpm_ne90 rc=1" in log
    assert "SKIP 2_sequence_report 3_diagnostics 4_adaptive_report" in log
    assert not ran(log, "2_sequence_report") and not ran(log, "4_adaptive_report")
    assert ran(log, "5_seq_ft_seed1") and log.rstrip().endswith("QUEUE DONE")


def test_a_failed_analysis_still_reaches_seq_ft_seed1(tmp_path):
    log = dry_run(tmp_path, "3_diagnostics")
    assert "SKIP 4_adaptive_report" in log
    assert ran(log, "5_seq_ft_seed1") and log.rstrip().endswith("QUEUE DONE")
    log2 = dry_run(tmp_path / "second", "2_sequence_report")
    assert ran(log2, "3_diagnostics") and ran(log2, "5_seq_ft_seed1")


@pytest.mark.parametrize("script, flags", [
    ("run_continual.py", ["--identity-reference-run", "--identity-stages", "--single-task-baseline"]),
    ("sequence_report.py", ["--method-run", "--out"]),
    ("forgetting_diagnostics.py", ["--method-run", "--out"]),
    ("adaptive_report.py", ["--config", "--out"]),
])
def test_every_step_accepts_its_flags(script, flags):
    done = subprocess.run([sys.executable, str(repo_root() / "scripts" / script), "--help"],
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    for flag in flags:
        assert flag in done.stdout, (script, flag)
