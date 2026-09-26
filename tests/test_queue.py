"""The queues never stop early; dependent steps are skipped explicitly and logged."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from flowcl.utils.libero_paths import repo_root

QUEUE = repo_root() / "scripts" / "queue_2026-09-24_overnight.sh"
SEED1 = repo_root() / "scripts" / "queue_2026-09-25_seed1.sh"


GENERIC = repo_root() / "scripts" / "queue_seed_pair.sh"


def dry_run(tmp_path, fail_step="", queue=QUEUE, args=()):
    env = {**os.environ, "QUEUE_DRY_RUN": "1", "QUEUE_FAIL_STEP": fail_step,
           "QUEUE_LOG_ROOT": str(tmp_path)}
    done = subprocess.run(["bash", str(queue), *args], env=env, capture_output=True, text=True,
                          timeout=60)
    assert done.returncode == 0, done.stderr
    (logdir,) = list(tmp_path.glob("queue_*"))
    return (logdir / "queue.log").read_text()


def ran(log, step):
    return f"START {step}" in log


def test_queue_syntax():
    for queue in (QUEUE, SEED1, GENERIC):
        assert subprocess.run(["bash", "-n", str(queue)]).returncode == 0


GEN_STEPS = ["0_seqft_s2", "1_gpm_s2", "2_seqrep_gpm_s2", "3_diag_gpm_s2", "4_gpm_ne90_s2",
             "5_seqrep_ne90_s2", "6_diag_ne90_s2", "7_adaptive_s2", "8_replication"]


def test_generic_queue_runs_seed2_with_seq_ft_in_order(tmp_path):
    log = dry_run(tmp_path, queue=GENERIC, args=("2", "--with-seq-ft"))
    positions = [log.index(f"START {s}") for s in GEN_STEPS]
    assert positions == sorted(positions) and "SKIP" not in log
    assert log.rstrip().endswith("QUEUE DONE")
    (logdir,) = list(tmp_path.glob("queue_*_seed2"))
    read = lambda s: (logdir / f"{s}.log").read_text()  # noqa: E731
    assert "--method seq_ft --seed 2 --amp" in read("0_seqft_s2")
    assert "--method gpm --seed 2" in read("1_gpm_s2")
    assert "--identity-reference-run results/seq_hetero__gpm_projected_adam__seed2" in read("4_gpm_ne90_s2")
    assert "--reference-run seq_hetero__seq_ft__seed2" in read("6_diag_ne90_s2")
    assert "--seed 2" in read("7_adaptive_s2")
    assert read("8_replication").rstrip().endswith("--replication 0 1 2")


def test_generic_queue_without_seq_ft_starts_at_the_gpm_run(tmp_path):
    log = dry_run(tmp_path, queue=GENERIC, args=("2",))
    assert not ran(log, "0_seqft_s2") and ran(log, "1_gpm_s2") and ran(log, "8_replication")


@pytest.mark.parametrize("fail, ran_steps, skipped", [
    ("0_seqft_s2", ["0_seqft_s2"], GEN_STEPS[1:]),
    ("1_gpm_s2", GEN_STEPS[:2], GEN_STEPS[2:]),
    ("4_gpm_ne90_s2", GEN_STEPS[:5], GEN_STEPS[5:]),
    ("2_seqrep_gpm_s2", GEN_STEPS[:7], ["7_adaptive_s2", "8_replication"]),  # 7 needs both seqreps
    ("5_seqrep_ne90_s2", GEN_STEPS[:7], ["7_adaptive_s2", "8_replication"]),
    ("7_adaptive_s2", GEN_STEPS[:8], ["8_replication"]),
])
def test_generic_queue_skips_only_what_depends_on_a_failure(tmp_path, fail, ran_steps, skipped):
    log = dry_run(tmp_path, fail, queue=GENERIC, args=("2", "--with-seq-ft"))
    for step in ran_steps:
        assert ran(log, step), step
    for step in skipped:
        assert not ran(log, step), step
    assert "SKIP" in log and log.rstrip().endswith("QUEUE DONE")


PRIOR_OUTPUTS = {  # step -> output that marks it done (relative to the results root)
    "0_seqft_s2": "seq_hetero__seq_ft__seed2/result.json",
    "1_gpm_s2": "seq_hetero__gpm_projected_adam__seed2/result.json",
    "2_seqrep_gpm_s2": "gpm_seq_seed2/report.json",
    "3_diag_gpm_s2": "forgetting_diag_seed2/report.json",
    "4_gpm_ne90_s2": "seq_hetero__gpm_projected_adam_ne90__seed2/result.json",
    "5_seqrep_ne90_s2": "gpm_seq_ne90_seed2/report.json",
    "6_diag_ne90_s2": "forgetting_diag_ne90_seed2/report.json",
    "7_adaptive_s2": "adaptive_gpm_seed2/report.json",
}


def resume(tmp_path, from_step, present, fail=""):
    """Dry-run the generic queue for seed 2 from ``from_step`` with ``present`` outputs."""
    results = tmp_path / "results"
    for step in present:
        path = results / PRIOR_OUTPUTS[step]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    env = {**os.environ, "QUEUE_DRY_RUN": "1", "QUEUE_FAIL_STEP": fail,
           "QUEUE_LOG_ROOT": str(tmp_path / "logs"), "QUEUE_RESULTS_ROOT": str(results)}
    done = subprocess.run(["bash", str(GENERIC), "2", "--with-seq-ft", "--from-step", str(from_step)],
                          env=env, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    (logdir,) = list((tmp_path / "logs").glob("queue_*"))
    return (logdir / "queue.log").read_text()


def test_from_step_reuses_prior_outputs_and_runs_the_rest(tmp_path):
    log = resume(tmp_path, 4, GEN_STEPS[:4])
    for step in GEN_STEPS[:4]:
        assert f"PRIOR {step} ok" in log and not ran(log, step), step
    for step in GEN_STEPS[4:]:
        assert ran(log, step), step
    assert "from_step=4" in log and log.rstrip().endswith("QUEUE DONE")


def test_from_step_blocks_dependents_of_a_missing_prior_output(tmp_path):
    log = resume(tmp_path, 4, ["0_seqft_s2", "1_gpm_s2", "2_seqrep_gpm_s2"])  # no gpm diagnostics
    assert "PRIOR 3_diag_gpm_s2 missing" in log
    for step in ("4_gpm_ne90_s2", "5_seqrep_ne90_s2", "6_diag_ne90_s2"):
        assert ran(log, step), step
    assert not ran(log, "7_adaptive_s2") and not ran(log, "8_replication")


def test_from_step_with_a_missing_baseline_run_skips_everything(tmp_path):
    log = resume(tmp_path, 4, ["0_seqft_s2"])
    assert "PRIOR 1_gpm_s2 missing" in log
    assert not any(ran(log, s) for s in GEN_STEPS)
    assert log.rstrip().endswith("QUEUE DONE")


def test_from_step_8_only_reruns_the_replication(tmp_path):
    log = resume(tmp_path, 8, GEN_STEPS[:8])
    assert [s for s in GEN_STEPS if ran(log, s)] == ["8_replication"]


def test_from_step_still_honours_failures_after_the_resume_point(tmp_path):
    log = resume(tmp_path, 4, GEN_STEPS[:4], fail="4_gpm_ne90_s2")
    assert "END 4_gpm_ne90_s2 rc=1" in log
    assert not ran(log, "5_seqrep_ne90_s2") and not ran(log, "7_adaptive_s2")


@pytest.mark.parametrize("args", [("2", "--from-step", "9"), ("2", "--from-step", "x"),
                                  ("2", "--from-step"), ("2", "--bogus")])
def test_bad_resume_arguments_are_rejected(tmp_path, args):
    env = {**os.environ, "QUEUE_DRY_RUN": "1", "QUEUE_LOG_ROOT": str(tmp_path)}
    done = subprocess.run(["bash", str(GENERIC), *args], env=env, capture_output=True, text=True)
    assert done.returncode == 2
    assert not list(tmp_path.glob("queue_*"))  # rejected before any log is written


def test_generic_queue_rejects_a_bad_seed_and_skips_replication_for_seed0(tmp_path):
    env = {**os.environ, "QUEUE_DRY_RUN": "1", "QUEUE_LOG_ROOT": str(tmp_path)}
    bad = subprocess.run(["bash", str(GENERIC), "two"], env=env, capture_output=True, text=True)
    assert bad.returncode == 2
    log = dry_run(tmp_path / "s0", queue=GENERIC, args=("0",))
    assert "SKIP 8_replication: replication needs at least two seeds" in log


SEED1_STEPS = ["1_gpm_s1", "2_seqrep_gpm_s1", "3_diag_gpm_s1", "4_gpm_ne90_s1",
               "5_seqrep_ne90_s1", "6_diag_ne90_s1", "7_adaptive_s1", "8_replication"]


def test_seed1_queue_runs_every_step_in_order(tmp_path):
    log = dry_run(tmp_path, queue=SEED1)
    positions = [log.index(f"START {s}") for s in SEED1_STEPS]
    assert positions == sorted(positions) and "SKIP" not in log
    assert log.rstrip().endswith("QUEUE DONE")
    (logdir,) = list(tmp_path.glob("queue_*"))
    ne90 = (logdir / "4_gpm_ne90_s1.log").read_text()
    assert "--identity-reference-run results/seq_hetero__gpm_projected_adam__seed1" in ne90
    assert "--identity-stages 0 1" in ne90
    assert "--reference-run seq_hetero__seq_ft__seed1" in (logdir / "3_diag_gpm_s1.log").read_text()


@pytest.mark.parametrize("fail, ran_steps, skipped", [
    ("1_gpm_s1", ["1_gpm_s1"], SEED1_STEPS[1:]),
    ("4_gpm_ne90_s1", SEED1_STEPS[:4], ["5_seqrep_ne90_s1", "6_diag_ne90_s1", "7_adaptive_s1",
                                         "8_replication"]),
    ("3_diag_gpm_s1", SEED1_STEPS[:6], ["7_adaptive_s1", "8_replication"]),
    ("7_adaptive_s1", SEED1_STEPS[:7], ["8_replication"]),
])
def test_seed1_queue_skips_only_what_depends_on_a_failure(tmp_path, fail, ran_steps, skipped):
    log = dry_run(tmp_path, fail, queue=SEED1)
    for step in ran_steps:
        assert ran(log, step), step
    for step in skipped:
        assert not ran(log, step), step
        assert step in log  # the skip is logged
    assert log.rstrip().endswith("QUEUE DONE")


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
    ("forgetting_diagnostics.py", ["--method-run", "--reference-run", "--out"]),
    ("adaptive_report.py", ["--config", "--out", "--seed", "--replication"]),
])
def test_every_step_accepts_its_flags(script, flags):
    done = subprocess.run([sys.executable, str(repo_root() / "scripts" / script), "--help"],
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    for flag in flags:
        assert flag in done.stdout, (script, flag)
