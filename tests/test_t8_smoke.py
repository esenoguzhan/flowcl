"""The 8-task resume smoke's queue (scripts/queue_t8_smoke.sh) in dry runs, and its matrix rule
(flowcl/experiments/t8_smoke.py; pre-E3 plan, B)."""

from __future__ import annotations

import os
import re
import subprocess

from flowcl.experiments import t8_smoke as ts
from flowcl.utils.libero_paths import repo_root

QUEUE = repo_root() / "scripts" / "queue_t8_smoke.sh"


def dry_run(tmp_path, fail=""):
    env = {**os.environ, "QUEUE_DRY_RUN": "1", "QUEUE_FAIL_STEP": fail, "QUEUE_LOG_ROOT": str(tmp_path / "logs")}
    args = ["--root", str(tmp_path / "t8"), "--new", str(tmp_path / "new"), "--ref", str(tmp_path / "ref")]
    done = subprocess.run(["bash", str(QUEUE), *args], env=env, capture_output=True, text=True, timeout=300)
    (logdir,) = list((tmp_path / "logs").glob("queue_*"))
    return done.returncode, (logdir / "queue.log").read_text(), logdir


def started(log):
    return re.findall(r"^\S+ \S+ START (\S+): ", log, flags=re.M)


def test_queue_syntax_and_arguments(tmp_path):
    assert subprocess.run(["bash", "-n", str(QUEUE)]).returncode == 0
    done = subprocess.run(["bash", str(QUEUE), "--root", "relative"], capture_output=True, text=True)
    assert done.returncode == 2


def test_the_steps_in_order_each_resume_checked_first(tmp_path):
    rc, log, logdir = dry_run(tmp_path)
    assert rc == 0 and log.rstrip().endswith("QUEUE DONE ok")
    names = ["new_seq_ft", "new_gpm", "new_replay", "new_ne70", "ref_seq_ft", "ref_gpm", "ref_replay",
             "probe_gpm", "probe_replay"]
    expected = ["0_ne70_smoke"]
    for k, name in enumerate(names, start=1):
        expected += [f"{k}_check_{name}", f"{k}_{name}"]
    expected += ["10_check_cut", "10_cut", "11_check_again", "11_again", "12_checks"]
    assert started(log) == expected
    run = (logdir / "2_new_gpm.log").read_text()
    for flag in ("--curriculum seq_hetero_t8", "--start-stage 4", "--no-eval-unseen", "--steps-per-task 200",
                 "--n-episodes 1", f"--results-root {tmp_path}/t8/from_new"):
        assert flag in run
    assert "--check-resume-only" in (logdir / "2_check_new_gpm.log").read_text()
    assert "--probe-every 50" in (logdir / "8_probe_gpm.log").read_text()
    again = (logdir / "11_again.log").read_text()
    assert "--start-stage 6" in again and f"{tmp_path}/t8/cut/seq_hetero_t8__gpm_projected_adam__seed2" in again
    assert "--allow-dirty" not in run and "--identity-stages" not in run


def test_a_failed_cut_skips_its_resume(tmp_path):
    rc, log, _ = dry_run(tmp_path, fail="10_cut")
    assert rc == 1 and "SKIP 11_again" in log and "12_checks" in started(log)


def test_the_matrix_rule():
    def matrix(fill):
        return {"retention_matrix": {"values": [[fill(i, j) for j in range(8)] for i in range(8)]}}

    good = matrix(lambda i, j: 0.5 if (j < 4 if i < 4 else j <= i) else None)
    assert ts.matrix_problems(good) == []
    bad = matrix(lambda i, j: 0.5 if (j < 4 if i < 4 else j <= i) or (i, j) == (5, 6) else None)
    assert ts.matrix_problems(bad) == ["cell (5,6) is set"]
