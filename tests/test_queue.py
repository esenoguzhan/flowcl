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


# ---- the SGP queue ---------------------------------------------------------------------

SGP_QUEUE = repo_root() / "scripts" / "queue_sgp_seed.sh"
SGP_STEPS = ["0_sgp_s0", "1_seqrep_sgp_s0", "2_diag_sgp_s0", "3_sgp_ne90_s0",
             "4_seqrep_sgp_ne90_s0", "5_diag_sgp_ne90_s0", "6_probe_cells_s0",
             "7_sgp_report_s0", "8_sgp_replication"]
SGP_INPUTS = {  # existing seed-0 inputs (seed 0's GPM analyses have no suffix)
    "reference": "seq_hetero__seq_ft__seed0/result.json",
    "plain_gpm": "seq_hetero__gpm_projected_adam__seed0/result.json",
    "adaptive_gpm": "seq_hetero__gpm_projected_adam_ne90__seed0/result.json",
    "seqrep_gpm": "gpm_seq/report.json",
    "seqrep_gpm_ne90": "gpm_seq_ne90/report.json",
    "diag_gpm": "forgetting_diag/report.json",
    "diag_gpm_ne90": "forgetting_diag_ne90/report.json",
}
SGP_OUTPUTS = {
    "0_sgp_s0": "seq_hetero__sgp_projected_adam__seed0/result.json",
    "1_seqrep_sgp_s0": "sgp_seq_seed0/report.json",
    "2_diag_sgp_s0": "forgetting_diag_sgp_seed0/report.json",
    "3_sgp_ne90_s0": "seq_hetero__sgp_projected_adam_ne90__seed0/result.json",
    "4_seqrep_sgp_ne90_s0": "sgp_seq_ne90_seed0/report.json",
    "5_diag_sgp_ne90_s0": "forgetting_diag_sgp_ne90_seed0/report.json",
    "6_probe_cells_s0": "sgp_probe_seed0/report.json",
    "7_sgp_report_s0": "sgp_seed0/report.json",
}


def sgp_queue(tmp_path, present=(), args=("0",), fail="", drop_inputs=()):
    results = tmp_path / "results"
    paths = [p for k, p in SGP_INPUTS.items() if k not in drop_inputs] + list(present)
    for rel in paths:
        path = results / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    env = {**os.environ, "QUEUE_DRY_RUN": "1", "QUEUE_FAIL_STEP": fail,
           "QUEUE_LOG_ROOT": str(tmp_path / "logs"), "QUEUE_RESULTS_ROOT": str(results)}
    done = subprocess.run(["bash", str(SGP_QUEUE), *args], env=env, capture_output=True,
                          text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    (logdir,) = list((tmp_path / "logs").glob("queue_*_sgp_seed*"))
    return (logdir / "queue.log").read_text(), logdir


def test_sgp_queue_syntax():
    assert subprocess.run(["bash", "-n", str(SGP_QUEUE)]).returncode == 0


def test_sgp_queue_runs_seed0_in_order_and_waits_for_the_other_seeds(tmp_path):
    log, _ = sgp_queue(tmp_path)
    positions = [log.index(f"START {s}") for s in SGP_STEPS[:-1]]
    assert positions == sorted(positions)
    assert "SKIP 8_sgp_replication: no SGP report yet for seed(s) 1 2" in log
    assert all(f"INPUT {k} ok" in log for k in SGP_INPUTS)
    assert log.rstrip().endswith("QUEUE DONE")


def test_sgp_queue_commands(tmp_path):
    _, logdir = sgp_queue(tmp_path)
    read = lambda s: (logdir / f"{s}.log").read_text()  # noqa: E731
    assert ("--method sgp --seed 0 --amp --identity-reference-run "
            "results/seq_hetero__gpm_projected_adam__seed0 --identity-stages 0") in read("0_sgp_s0")
    assert ("--method sgp_ne90 --seed 0 --amp --identity-reference-run "
            "results/seq_hetero__sgp_projected_adam__seed0 --identity-stages 0 1") in read("3_sgp_ne90_s0")
    assert "--out results/sgp_seq_seed0/report.json" in read("1_seqrep_sgp_s0")
    assert "--method-run seq_hetero__sgp_projected_adam_ne90__seed0" in read("5_diag_sgp_ne90_s0")
    assert "--out results/forgetting_diag_sgp_ne90_seed0/report.json" in read("5_diag_sgp_ne90_s0")
    assert "scripts/probe_cells.py --seed 0" in read("6_probe_cells_s0")
    assert "scripts/sgp_report.py --seed 0" in read("7_sgp_report_s0")


def test_sgp_queue_runs_the_replication_once_every_seed_has_a_report(tmp_path):
    log, logdir = sgp_queue(tmp_path, present=["sgp_seed1/report.json", "sgp_seed2/report.json"])
    assert ran(log, "8_sgp_replication")
    assert (logdir / "8_sgp_replication.log").read_text().rstrip().endswith("--replication")


@pytest.mark.parametrize("fail, ran_steps, skipped", [
    ("0_sgp_s0", SGP_STEPS[:1], SGP_STEPS[1:]),
    ("3_sgp_ne90_s0", SGP_STEPS[:4], SGP_STEPS[4:]),
    ("2_diag_sgp_s0", SGP_STEPS[:6], SGP_STEPS[6:]),   # the probe cells need both diagnostics
    ("4_seqrep_sgp_ne90_s0", SGP_STEPS[:7], SGP_STEPS[7:]),  # the report needs both seqreps
    ("6_probe_cells_s0", SGP_STEPS[:7], SGP_STEPS[7:]),
])
def test_sgp_queue_skips_only_what_depends_on_a_failure(tmp_path, fail, ran_steps, skipped):
    log, _ = sgp_queue(tmp_path, fail=fail)
    for step in ran_steps:
        assert ran(log, step), step
    for step in skipped:
        assert not ran(log, step), step
    assert "SKIP" in log and log.rstrip().endswith("QUEUE DONE")


def test_sgp_queue_missing_inputs(tmp_path):
    log, _ = sgp_queue(tmp_path / "a", drop_inputs=("plain_gpm",))
    assert "INPUT plain_gpm missing" in log and not any(ran(log, s) for s in SGP_STEPS)
    log, _ = sgp_queue(tmp_path / "b", drop_inputs=("diag_gpm_ne90",))
    assert all(ran(log, s) for s in SGP_STEPS[:6])
    assert not any(ran(log, s) for s in SGP_STEPS[6:])


def test_sgp_queue_from_step_reuses_prior_outputs(tmp_path):
    log, _ = sgp_queue(tmp_path, present=[SGP_OUTPUTS[s] for s in SGP_STEPS[:3]],
                       args=("0", "--from-step", "3"))
    for step in SGP_STEPS[:3]:
        assert f"PRIOR {step} ok" in log and not ran(log, step), step
    for step in SGP_STEPS[3:8]:
        assert ran(log, step), step


def test_sgp_queue_from_step_blocks_dependents_of_a_missing_prior_output(tmp_path):
    log, _ = sgp_queue(tmp_path, present=[SGP_OUTPUTS[s] for s in SGP_STEPS[:2]],
                       args=("0", "--from-step", "3"))
    assert "PRIOR 2_diag_sgp_s0 missing" in log
    assert all(ran(log, s) for s in SGP_STEPS[3:6])
    assert not ran(log, "6_probe_cells_s0") and not ran(log, "7_sgp_report_s0")


@pytest.mark.parametrize("args", [("3",), ("x",), ("0", "--from-step", "9"),
                                  ("0", "--from-step"), ("0", "--with-seq-ft")])
def test_sgp_queue_rejects_bad_arguments(tmp_path, args):
    env = {**os.environ, "QUEUE_DRY_RUN": "1", "QUEUE_LOG_ROOT": str(tmp_path)}
    done = subprocess.run(["bash", str(SGP_QUEUE), *args], env=env, capture_output=True, text=True)
    assert done.returncode == 2
    assert not list(tmp_path.glob("queue_*"))


@pytest.mark.parametrize("script, flags", [
    ("run_continual.py", ["--results-root", "--identity-reference-run", "--identity-stages"]),
    ("sgp_report.py", ["--config", "--out", "--seed", "--replication"]),
    ("probe_cells.py", ["--config", "--seed", "--device", "--out", "--allow-dirty"]),
])
def test_every_sgp_step_accepts_its_flags(script, flags):
    done = subprocess.run([sys.executable, str(repo_root() / "scripts" / script), "--help"],
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    for flag in flags:
        assert flag in done.stdout, (script, flag)


# ---- the exploratory alpha queue ---------------------------------------------------------

ALPHA_QUEUE = repo_root() / "scripts" / "queue_sgp_alpha.sh"
A_STEPS = ["0_sgp_s0", "1_seqrep_sgp_s0", "2_diag_sgp_s0", "3_sgp_ne90_s0",
           "4_seqrep_sgp_ne90_s0", "5_diag_sgp_ne90_s0", "6_probe_cells_s0", "7_sgp_report_s0"]


def alpha_queue(tmp_path, args=("0", "800"), fail="", present=()):
    results = tmp_path / "results"
    for rel in [*SGP_INPUTS.values(), *present]:
        path = results / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    env = {**os.environ, "QUEUE_DRY_RUN": "1", "QUEUE_FAIL_STEP": fail,
           "QUEUE_LOG_ROOT": str(tmp_path / "logs"), "QUEUE_RESULTS_ROOT": str(results)}
    done = subprocess.run(["bash", str(ALPHA_QUEUE), *args], env=env, capture_output=True,
                          text=True, timeout=60)
    return done, tmp_path / "logs"


def test_alpha_queue_runs_the_four_task_pair_with_its_own_names_and_rule(tmp_path):
    done, logs = alpha_queue(tmp_path)
    assert done.returncode == 0, done.stderr
    (logdir,) = list(logs.glob("queue_*_sgp_a800_seed0"))
    log = (logdir / "queue.log").read_text()
    positions = [log.index(f"START {s}:") for s in A_STEPS]
    assert positions == sorted(positions) and "replication" not in log
    read = lambda s: (logdir / f"{s}.log").read_text()  # noqa: E731
    assert ("--method sgp_a800 --seed 0 --amp --identity-reference-run "
            "results/seq_hetero__gpm_projected_adam__seed0 --identity-stages 0") in read("0_sgp_s0")
    assert ("--method sgp_a800_ne90 --seed 0 --amp --identity-reference-run "
            "results/seq_hetero__sgp_projected_adam_a800__seed0 --identity-stages 0 1") in read("3_sgp_ne90_s0")
    assert "--out results/sgp_a800_seq_seed0/report.json" in read("1_seqrep_sgp_s0")
    assert "--out results/forgetting_diag_sgp_a800_ne90_seed0/report.json" in read("5_diag_sgp_ne90_s0")
    assert "--config configs/analysis/sgp_a800.yaml --seed 0" in read("6_probe_cells_s0")
    assert "--config configs/analysis/sgp_a800.yaml --seed 0" in read("7_sgp_report_s0")


def test_alpha_queue_resume_and_refusals(tmp_path):
    done, logs = alpha_queue(tmp_path / "a", args=("0", "800", "--from-step", "3"),
                             present=["seq_hetero__sgp_projected_adam_a800__seed0/result.json",
                                      "sgp_a800_seq_seed0/report.json",
                                      "forgetting_diag_sgp_a800_seed0/report.json"])
    assert done.returncode == 0, done.stderr
    log = (next(logs.glob("queue_*")) / "queue.log").read_text()
    assert "PRIOR 0_sgp_s0 ok" in log and "START 3_sgp_ne90_s0:" in log and "START 0_sgp_s0:" not in log
    missing, logs_b = alpha_queue(tmp_path / "b", args=("0", "700"))
    assert missing.returncode == 2 and "alpha 700 is not configured" in missing.stderr
    assert not list(logs_b.glob("queue_*")) if logs_b.exists() else True
    for bad in (("0",), ("0", "x"), ("0", "800", "--from-step", "8"), ("0", "800", "--bogus")):
        assert subprocess.run(["bash", str(ALPHA_QUEUE), *bad], capture_output=True).returncode == 2
