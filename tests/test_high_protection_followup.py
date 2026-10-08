"""C1 follow-up: the reproduction and provenance gates, the C1-report gate, the verdict through C1's
classify at n = 150, the replication rows (including all unresolved) and refusals, and the queue's
dry run."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
import torch

from flowcl.experiments import high_protection as hp
from flowcl.experiments import high_protection_followup as fu
from flowcl.utils.libero_paths import repo_root
from flowcl.utils.run import file_sha256

CFG = fu.load_config()
C1 = fu.c1_config(CFG)
KEY = CFG["task_key"]


def cell(successes, offset=0, steps=100):
    from flowcl.analysis.metrics import success_estimate

    e = success_estimate(successes)
    n = len(successes)
    return {"task_key": KEY, "successes": list(successes), "n_steps": [steps] * n,
            "seeds": list(range(offset, offset + n)), "success_rate": e.value, "ci_low": e.low,
            "ci_high": e.high, "confidence": 0.95, "n_rollouts": n, "wall_clock_s": 0.0}


def write_eval(path: Path, run_id: str, stage: int, tasks: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"run_id": run_id, "stage": stage, "tasks": tasks}))


def successes(k, n):
    return [True] * k + [False] * (n - k)


def world(root: Path, seed=0, *, e99=(36, 108), adaptive=(34, 100), checks=None, c1_rule=None,
          ckpt_run=None, ns=None, n=150, drift=False):
    """C1-shaped runs with stored stage-2 cells and follow-up evals; ``(k50, k150)`` successes."""
    names = hp.run_names(C1, seed)
    namespace = ns or fu.namespace(C1, seed)
    for arm, (k50, k150) in (("e99", e99), ("adaptive", adaptive)):
        d = root / names[arm]
        first = successes(k50, 50)
        rest = successes(k150 - k50, n - 50) if n > 50 else []
        write_eval(d / "eval" / "stage2.json", namespace, 2, [cell(first)])
        (d / "checkpoints").mkdir(parents=True, exist_ok=True)
        torch.save({"run_id": ckpt_run or names[arm], "stage": 2}, d / "checkpoints" / "stage2.pt")
        new = first + rest
        if drift and arm == "e99":
            new = [not new[0]] + new[1:]
        write_eval(fu.eval_path(root, CFG, seed, arm), namespace, 2, [cell(new[:n])])
    report = {"seed": seed, "smoke": False, "runs": names,
              "inputs": {"rule": {"sha256": c1_rule or CFG["c1_rule_sha256"]}},
              "checks": checks or {k: True for k in ("identity", "energy", "valid", "premise", "reference",
                                                     "t3", "gain")},
              "primary": {"G": {"diff": 0.7, "low": 0.56, "high": 0.82},
                          "D": {"diff": 0.04, "low": -0.1, "high": 0.18}}}
    p = root / C1["out"].format(seed=seed)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report))
    return names


def build(root, seed=0, **kw):
    return fu.build_report(CFG, seed, runs_root=root, **kw)


def test_config_agrees_with_c1_and_the_queue():
    assert CFG["c1_rule_sha256"] == file_sha256(repo_root() / "configs" / "analysis" / "high_protection.yaml")
    i, j = CFG["cell"]
    assert [i, j] == C1["cells"]["primary"] and CFG["n_reproduce"] == C1["production"]["n_episodes"]
    from flowcl.data.curriculum import load_curriculum
    assert load_curriculum(C1["curriculum"]).stages[j].task_key == KEY
    queue = (repo_root() / "scripts" / "queue_high_protection_followup.sh").read_text()
    assert f"TASK={KEY}" in queue and f"N={CFG['n_episodes']}" in queue


def test_a_valid_seed_reaches_a_verdict_with_the_reported_views(tmp_path):
    names = world(tmp_path)
    r = build(tmp_path)
    assert all(g["passed"] for g in r["gates"].values())
    assert r["verdict"]["verdict"] in {"protection_suffices", "unresolved", "flat_control_inferior"}
    p = r["primary"]
    assert p["n"] == 150 and p["D"]["diff"] == pytest.approx((108 - 100) / 150)
    assert r["reported"]["D_first"]["diff"] == pytest.approx(2 / 50)
    assert r["reported"]["D_new_only"]["n"] == 100
    assert r["gates"]["provenance_e99"]["checkpoint_sha256"] == file_sha256(
        tmp_path / names["e99"] / "checkpoints" / "stage2.pt")
    assert r["rule"]["sha256"] == file_sha256(fu.rule_path())


def test_the_verdict_routes_through_c1_classify(tmp_path):
    world(tmp_path / "a", e99=(36, 108), adaptive=(36, 108))  # identical arms: D = 0 [0, 0]
    assert build(tmp_path / "a")["verdict"]["verdict"] == "protection_suffices"
    world(tmp_path / "b", e99=(20, 60), adaptive=(36, 108))   # e99 far behind on every new episode
    r = build(tmp_path / "b")
    assert r["verdict"]["verdict"] == "flat_control_inferior" and "adaptive_advantage" in r["verdict"]["flags"]


@pytest.mark.parametrize("kw,gate", [
    ({"drift": True}, "reproduction_e99"),
    ({"ckpt_run": "some_other_run"}, "provenance_e99"),
    ({"ns": "seq_hetero__seq_ft__seed7"}, "provenance_e99"),
    ({"n": 120}, "provenance_e99"),
    ({"c1_rule": "0" * 64}, "c1_report"),
    ({"checks": {"identity": True, "energy": False}}, "c1_report"),
])
def test_each_gate_rejects_its_failure(tmp_path, kw, gate):
    world(tmp_path, **kw)
    r = build(tmp_path)
    assert r["verdict"]["verdict"] == "invalid_followup" and not r["gates"][gate]["passed"]


def test_reproduction_check_catches_steps_and_seeds():
    from flowcl.envs.evaluation import TaskEvaluation
    from flowcl.analysis.metrics import success_estimate

    def te(succ, steps, seeds):
        return TaskEvaluation(task_key=KEY, successes=succ, n_steps=steps, seeds=seeds,
                              estimate=success_estimate(succ), wall_clock_s=0.0)
    base = te([True, False, True], [10, 600, 12], [1, 2, 3])
    assert fu.reproduction_check(te([True, False, True, True], [10, 600, 12, 9], [1, 2, 3, 4]), base, 3)["passed"]
    assert not fu.reproduction_check(te([True, False, True], [10, 600, 13], [1, 2, 3]), base, 3)["passed"]
    assert not fu.reproduction_check(te([True, False, True], [10, 600, 12], [1, 2, 4]), base, 3)["passed"]
    assert not fu.reproduction_check(te([True, False], [10, 600], [1, 2]), base, 3)["passed"]


def test_missing_inputs_and_unpaired_arms_are_invalid(tmp_path):
    world(tmp_path / "a")
    fu.eval_path(tmp_path / "a", CFG, 0, "adaptive").unlink()
    assert build(tmp_path / "a")["verdict"]["verdict"] == "invalid_followup"
    names = world(tmp_path / "b")
    p = fu.eval_path(tmp_path / "b", CFG, 0, "adaptive")
    data = json.loads(p.read_text())
    data["tasks"][0]["seeds"] = [s + 1 for s in data["tasks"][0]["seeds"]]
    p.write_text(json.dumps(data))
    r = build(tmp_path / "b")
    assert r["verdict"]["verdict"] == "invalid_followup" and not r["gates"]["episode_pairing"]["passed"]
    assert names


def test_smoke_compares_only_the_episodes_it_ran(tmp_path):
    world(tmp_path, n=3, e99=(36, 36), adaptive=(34, 34))
    r = build(tmp_path, smoke=True)
    assert r["gates"]["reproduction_e99"]["n_compared"] == 3 and all(g["passed"] for g in r["gates"].values())
    assert r["reported"]["D_new_only"] is None


@pytest.mark.parametrize("verdicts,outcome", [
    (["unresolved"] * 3, "unresolved_replicated"),
    (["protection_suffices"] * 3, "protection_suffices_replicated"),
    (["flat_control_inferior"] * 3, "flat_control_inferior_replicated"),
    (["unresolved", "protection_suffices", "unresolved"], "mixed"),
    (["invalid_followup", "unresolved", "unresolved"], "inconclusive_invalid"),
])
def test_replication_rows(verdicts, outcome):
    assert fu.classify_replication(dict(zip("012", verdicts)))["replication"] == outcome


def fake_seed_report(root, seed, verdict="unresolved", smoke=False, rule_sha=None):
    p = root / CFG["out_dir"] / f"seed{seed}" / "report.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"seed": seed, "smoke": smoke,
                             "rule": {"sha256": rule_sha or file_sha256(fu.rule_path())},
                             "verdict": {"verdict": verdict, "flags": ["within_margin_by_point_only"]},
                             "primary": {"D": {"diff": 0.0, "low": -0.08, "high": 0.08}}}))


def test_replication_reads_every_seed_and_refuses_bad_inputs(tmp_path):
    for s in (0, 1, 2):
        fake_seed_report(tmp_path, s)
    out = fu.run_replication(CFG, out_root=tmp_path)
    assert out["replication"] == "unresolved_replicated"
    assert out["flags"]["within_margin_by_point_only"] == [0, 1, 2]
    for name, bad in (("smoke", {"smoke": True}), ("rule", {"rule_sha": "0" * 64})):
        root = tmp_path / name
        for s in (0, 1, 2):
            fake_seed_report(root, s, **(bad if s == 2 else {}))
        with pytest.raises(ValueError, match="replication refused"):
            fu.build_replication(CFG, out_root=root)
    root = tmp_path / "missing"
    fake_seed_report(root, 0)
    with pytest.raises(ValueError, match="seed 1: no report"):
        fu.build_replication(CFG, out_root=root)


# ---- the queue ------------------------------------------------------------------------------------

QUEUE = repo_root() / "scripts" / "queue_high_protection_followup.sh"


def run_queue(tmp_path, args=(), fail="", present=()):
    results = tmp_path / "results"
    results.mkdir(parents=True, exist_ok=True)
    for rel in present:
        p = results / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}")
    env = {**os.environ, "QUEUE_DRY_RUN": "1", "QUEUE_FAIL_STEP": fail,
           "QUEUE_LOG_ROOT": str(tmp_path / "logs"), "QUEUE_RESULTS_ROOT": str(results)}
    done = subprocess.run(["bash", str(QUEUE), *args], env=env, capture_output=True, text=True, timeout=120)
    (logdir,) = list((tmp_path / "logs").glob("queue_*"))
    return done.returncode, (logdir / "queue.log").read_text(), logdir


def test_queue_order_commands_and_replication(tmp_path):
    assert subprocess.run(["bash", "-n", str(QUEUE)]).returncode == 0
    reports = [f"high_protection_followup/seed{s}/report.json" for s in (0, 1, 2)]
    rc, log, logdir = run_queue(tmp_path, present=reports)
    steps = [f"{3 * i + k}_{name}_s{s}" for i, s in enumerate((0, 1, 2))
             for k, name in ((0, "eval_e99"), (1, "eval_adaptive"), (2, "report"))] + ["9_replication"]
    assert rc == 0 and log.rstrip().endswith("QUEUE DONE ok")
    assert [log.index(f"START {s}:") for s in steps] == sorted(log.index(f"START {s}:") for s in steps)
    res = str(tmp_path / "results")
    cmd = (logdir / "3_eval_e99_s1.log").read_text()
    assert f"--checkpoint {res}/seq_hetero__gpm_projected_adam_e99__seed1/checkpoints/stage2.pt" in cmd
    assert f"--tasks {KEY} --run-id seq_hetero__seq_ft__seed1 --n-episodes 150" in cmd
    assert f"--out {res}/high_protection_followup/seed1/e99.eval.json" in cmd
    assert "seq_hetero__gpm_projected_adam_ne90__seed2/checkpoints/stage2.pt" in (logdir / "7_eval_adaptive_s2.log").read_text()


def test_queue_failures_resume_and_bad_arguments(tmp_path):
    rc, log, _ = run_queue(tmp_path / "a", fail="1_eval_adaptive_s0")
    assert rc == 1 and "START 2_report_s0" not in log and "START 3_eval_e99_s1" in log
    prior = ["high_protection_followup/seed0/e99.eval.json", "high_protection_followup/seed0/adaptive.eval.json",
             "high_protection_followup/seed0/report.json"]
    rc, log, _ = run_queue(tmp_path / "b", ["--from-step", "3"], present=prior)
    assert "PRIOR 0_eval_e99_s0 ok" in log and "PRIOR 2_report_s0 ok" in log and "START 3_eval_e99_s1" in log
    for bad in (["--from-step", "10"], ["--bogus"], ["--smoke", "rel/dir"]):
        assert subprocess.run(["bash", str(QUEUE), *bad], capture_output=True).returncode == 2


def test_queue_smoke_mode(tmp_path):
    smoke = tmp_path / "scratch"
    rc, log, logdir = run_queue(tmp_path, ["--smoke", str(smoke)])
    assert rc == 0 and logdir.name.endswith("_smoke")
    names = {p.name[:-4] for p in logdir.glob("*.log")} - {"queue"}
    assert names == {"0_eval_e99_s0", "1_eval_adaptive_s0", "2_report_s0"}
    cmd = (logdir / "0_eval_e99_s0.log").read_text()
    assert "--n-episodes 3" in cmd and f"--out {smoke}/high_protection_followup/seed0/e99.eval.json" in cmd
    assert f"--checkpoint {tmp_path}/results/seq_hetero__gpm_projected_adam_e99__seed0" in cmd
    assert f"--seed 0 --results-root {smoke} --smoke" in (logdir / "2_report_s0.log").read_text()


@pytest.mark.parametrize("script, flags", [
    ("evaluate.py", ["--checkpoint", "--tasks", "--run-id", "--n-episodes", "--out"]),
    ("high_protection_followup.py", ["--config", "--seed", "--replication", "--results-root", "--smoke"]),
])
def test_every_queue_step_accepts_its_flags(script, flags):
    import sys

    done = subprocess.run([sys.executable, str(repo_root() / "scripts" / script), "--help"],
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    for flag in flags:
        assert flag in done.stdout, (script, flag)
