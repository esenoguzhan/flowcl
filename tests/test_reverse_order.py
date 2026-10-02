"""C2 reverse-order summary: the cell judgments and the verdict order, the joint contrast bootstrap,
every validity condition on fake run directories (invalid, incomplete), the adaptive comparison's
own completeness and validity, smoke mode, and the queue's dry run."""

from __future__ import annotations

import copy
import json
import os
import subprocess
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from flowcl.experiments import reverse_order as ro
from flowcl.utils.libero_paths import repo_root
from flowcl.utils.run import file_sha256

KEYS = [s.task_key for s in __import__("flowcl.data.curriculum", fromlist=["load_curriculum"])
        .load_curriculum("seq_hetero_reverse").stages]
N_EP = 50


def est(v, lo=None, hi=None):
    return {"value": v, "low": v - 0.1 if lo is None else lo, "high": v + 0.1 if hi is None else hi}


# ---- pure pieces --------------------------------------------------------------------------------------


def test_cell_states_and_borderline():
    assert ro.judge_cell(est(0.9), est(0.2), 0.6)["state"] == ro.FORGOTTEN
    assert ro.judge_cell(est(0.9), est(0.8), 0.6)["state"] == ro.RETAINED
    assert ro.judge_cell(est(0.5), est(0.1), 0.6)["state"] == ro.NOT_LEARNED
    c = ro.judge_cell(est(0.65, 0.55, 0.75), est(0.6, 0.5, 0.7), 0.6)
    assert c["state"] == ro.RETAINED and c["retention"]["borderline"] and c["diagonal"]["borderline"]
    assert ro.thresholds([0.9, 0.78], 15) == pytest.approx([0.75, 0.63])


@pytest.mark.parametrize("pos,task,verdict", [
    (ro.FORGOTTEN, ro.FORGOTTEN, ro.BOTH), (ro.FORGOTTEN, ro.RETAINED, ro.POSITION),
    (ro.RETAINED, ro.FORGOTTEN, ro.TASK), (ro.RETAINED, ro.RETAINED, ro.NEITHER),
    (ro.NOT_LEARNED, ro.FORGOTTEN, ro.INCONCLUSIVE)])
def test_classify(pos, task, verdict):
    v = ro.classify("ok", {"position": {"state": pos}, "task": {"state": task}})
    assert v["verdict"] == verdict and v["exit_code"] == 0
    assert ro.classify(ro.INVALID, None)["exit_code"] == 3 and ro.classify(ro.INCOMPLETE, None)["exit_code"] == 4


def test_joint_contrast_keeps_the_pairing():
    boot = {"seed": 0, "n_resamples": 2000, "confidence": 0.95}
    a_before, a_after = [1] * 20, [1] * 20          # adaptive keeps everything
    p_before, p_after = [1] * 20, [1] * 10 + [0] * 10
    c = ro.joint_contrast(a_before, a_after, p_before, p_after, boot)
    assert c["diff"] == pytest.approx(-0.5) and c["high"] < 0 and c["n"] == 20
    with pytest.raises(ValueError):
        ro.joint_contrast([1], [1, 0], [1], [1], boot)


# ---- fake runs ----------------------------------------------------------------------------------------


def write_run(root: Path, name: str, role: str, cfg: dict, matrix, *, seed=0, smoke=False, fp="fp1",
              fitted=KEYS[0], dirty=False, n=N_EP, stages=4, t1=True, seeds_offset=0):
    from flowcl.analysis.metrics import success_estimate

    d = root / name
    (d / "eval").mkdir(parents=True)
    (d / "checkpoints").mkdir()
    method_name, kwargs, label = ro.method_spec(cfg, role)
    want = cfg["smoke"] if smoke else cfg["production"]
    OmegaConf.save(OmegaConf.create({
        "run_id": name, "seed": seed, "seed_namespace_run_id": f"seq_hetero_reverse__seq_ft__seed{seed}",
        "method": {"name": method_name, "display_name": label, **kwargs},
        "curriculum": {"name": "seq_hetero_reverse"},
        "train": {**want["train"], "batch_size": 64, "lr": 1e-4},
        "eval": {"n_episodes": want["n_episodes"], "max_steps": 600}}), d / "config.yaml")
    (d / "git_sha").write_text("abc" + ("-dirty" if dirty else "") + "\n")
    (d / "stats.json").write_text(json.dumps({"fingerprint": fp, "fitted_on_task_id": fitted}))
    for i in range(stages):
        torch.save({"stats_fingerprint": fp}, d / "checkpoints" / f"stage{i}.pt")
        tasks = []
        for j, key in enumerate(KEYS):
            k = round(matrix[i][j] * n)
            succ = [True] * k + [False] * (n - k)
            e = success_estimate(succ)
            tasks.append({"task_key": key, "successes": succ, "n_steps": [100] * n,
                          "seeds": list(range(seeds_offset + 1000 * j, seeds_offset + 1000 * j + n)),
                          "success_rate": e.value, "ci_low": e.low, "ci_high": e.high, "confidence": 0.95,
                          "n_rollouts": n, "wall_clock_s": 0.0})
        (d / "eval" / f"stage{i}.json").write_text(json.dumps(
            {"run_id": "ns", "stage": i, "tasks": tasks}))
    if stages == 4:
        (d / "result.json").write_text(json.dumps({
            "task_keys": KEYS, "seed_namespace_run_id": f"seq_hetero_reverse__seq_ft__seed{seed}",
            "t1_pairing": {"passed": t1} if role != "seq_ft" else None, "metrics": {}}))
    return d


SEQFT = [[0.9, 0, 0, 0], [0.1, 0.9, 0, 0], [0, 0.1, 0.9, 0], [0, 0, 0.1, 0.9]]
PLAIN = [[0.9, 0, 0, 0], [0.8, 0.9, 0, 0], [0.8, 0.2, 0.9, 0], [0.8, 0.2, 0.85, 0.9]]   # position
ADAPT = [[0.9, 0, 0, 0], [0.8, 0.9, 0, 0], [0.8, 0.8, 0.9, 0], [0.8, 0.8, 0.85, 0.9]]


def report(root: Path, out: str, seqft_dir: Path, theta, ok=True, dirty_check=True):
    checks = {"clean_git_sha": {"passed": dirty_check}, "seed_namespace": {"passed": True},
              "t1_pairing": {"passed": True}, "occupancy_non_decreasing": {"passed": ok},
              "memory_chained": {"passed": True}, "residuals_within_bound": {"passed": True},
              "artifact_hashes_match": {"passed": True}, "frozen_from_stage1": {"passed": True}}
    p = root / out / "report.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"provenance_checks": checks, "threshold_block": {
        "thresholds": theta, "reference_result_sha256": file_sha256(seqft_dir / "result.json")}}))
    return p


def world(tmp_path, *, smoke=False, adaptive="ok", **plain_kw):
    cfg = copy.deepcopy(ro.load_config())
    names = ro.run_names(cfg, 0)
    n = 1 if smoke else N_EP
    seqft = write_run(tmp_path, names["seq_ft"], "seq_ft", cfg, SEQFT, smoke=smoke, n=n,
                      dirty=smoke)
    theta = ro.thresholds([SEQFT[j][j] for j in range(4)], cfg["margin_pp"])
    if n == 1:  # one rollout: the diagonal is 1.0 or 0.0
        theta = ro.thresholds([1.0, 1.0, 1.0, 1.0], cfg["margin_pp"])
    plain_kw.setdefault("dirty", smoke)
    write_run(tmp_path, names["plain"], "plain", cfg, PLAIN, smoke=smoke, n=n, **plain_kw)
    report(tmp_path, cfg["reports"]["plain"].format(seed=0), seqft, theta, dirty_check=not smoke)
    if adaptive == "ok":
        d = write_run(tmp_path, names["adaptive"], "adaptive", cfg, ADAPT, smoke=smoke, n=n, dirty=smoke)
        for s in (0, 1):
            (d / f"identity_stage{s}.json").write_text(json.dumps({"passed": True}))
        report(tmp_path, cfg["reports"]["adaptive"].format(seed=0), seqft, theta, dirty_check=not smoke)
    elif adaptive == "stopped":   # identity failed at stage 1: no result.json, no later evals, no report
        d = write_run(tmp_path, names["adaptive"], "adaptive", cfg, ADAPT, stages=2)
        (d / "identity_stage0.json").write_text(json.dumps({"passed": True}))
        (d / "identity_stage1.json").write_text(json.dumps({"passed": False}))
    elif adaptive == "bad_identity":
        d = write_run(tmp_path, names["adaptive"], "adaptive", cfg, ADAPT)
        (d / "identity_stage0.json").write_text(json.dumps({"passed": True}))
        (d / "identity_stage1.json").write_text(json.dumps({"passed": False}))
        report(tmp_path, cfg["reports"]["adaptive"].format(seed=0), seqft, theta)
    return cfg, names


def summarize(tmp_path, cfg, smoke=False):
    return ro.run_summary(cfg, 0, results_root=tmp_path, out=tmp_path / "out", smoke=smoke)


def test_valid_position_verdict_with_everything_reported(tmp_path):
    cfg, _ = world(tmp_path)
    s = summarize(tmp_path, cfg)
    assert s["verdict"]["verdict"] == ro.POSITION and s["verdict"]["exit_code"] == 0, s["checks"]
    assert all(c["passed"] for c in s["checks"].values())
    assert s["verdict"]["cells"]["position"]["state"] == ro.FORGOTTEN
    assert s["verdict"]["cells"]["task"]["state"] == ro.RETAINED
    d = s["reported"]["drops"]["plain"]
    assert d["goal"]["diff"] == pytest.approx(0.7) and d["object"]["diff"] == pytest.approx(0.06)
    a = s["adaptive_comparison"]
    assert a["status"] == "valid" and a["contrasts"]["goal_drop"]["diff"] == pytest.approx(-0.6)
    assert a["contrasts"]["goal_drop"]["high"] < 0
    assert json.loads((tmp_path / "out" / "summary.json").read_text())["verdict"]["verdict"] == ro.POSITION


@pytest.mark.parametrize("kw,check", [
    ({"fp": "other"}, "statistics"), ({"fitted": KEYS[3]}, "statistics"), ({"dirty": True}, "production_plain"),
    ({"t1": False}, "t1_pairing"), ({"seeds_offset": 7}, "episode_pairing")])
def test_each_validity_condition_gives_invalid(tmp_path, kw, check):
    cfg, _ = world(tmp_path, **kw)
    s = summarize(tmp_path, cfg)
    assert s["verdict"]["verdict"] == ro.INVALID and s["verdict"]["exit_code"] == 3
    assert not s["checks"][check]["passed"]


def test_checkpoint_stats_mismatch_and_threshold_block_give_invalid(tmp_path):
    cfg, names = world(tmp_path)
    torch.save({"stats_fingerprint": "zzz"}, tmp_path / names["plain"] / "checkpoints" / "stage2.pt")
    assert summarize(tmp_path, cfg)["verdict"]["verdict"] == ro.INVALID
    cfg, names = world(tmp_path / "b")
    p = tmp_path / "b" / cfg["reports"]["plain"].format(seed=0) / "report.json"
    r = json.loads(p.read_text())
    r["threshold_block"]["thresholds"][1] += 0.01
    p.write_text(json.dumps(r))
    s = summarize(tmp_path / "b", cfg)
    assert s["verdict"]["verdict"] == ro.INVALID and not s["checks"]["thresholds"]["passed"]
    r["provenance_checks"]["memory_chained"]["passed"] = False
    r["threshold_block"]["thresholds"][1] -= 0.01
    p.write_text(json.dumps(r))
    assert summarize(tmp_path / "b", cfg)["checks"]["sequence_report"]["failed"] == ["memory_chained"]


def test_missing_primary_inputs_give_incomplete(tmp_path):
    cfg, names = world(tmp_path)
    (tmp_path / names["plain"] / "eval" / "stage3.json").unlink()
    s = summarize(tmp_path, cfg)
    assert s["verdict"]["verdict"] == ro.INCOMPLETE and s["verdict"]["exit_code"] == 4
    cfg, names = world(tmp_path / "b")
    (tmp_path / "b" / cfg["reports"]["plain"].format(seed=0) / "report.json").unlink()
    assert summarize(tmp_path / "b", cfg)["verdict"]["verdict"] == ro.INCOMPLETE
    assert (tmp_path / "b" / "out" / "summary.json").is_file()


def test_adaptive_stopped_early_is_incomplete_and_the_plain_verdict_stands(tmp_path):
    cfg, _ = world(tmp_path, adaptive="stopped")
    s = summarize(tmp_path, cfg)
    assert s["verdict"]["verdict"] == ro.POSITION and s["verdict"]["exit_code"] == 0
    assert s["adaptive_comparison"]["status"] == ro.INCOMPLETE
    assert s["adaptive_comparison"]["identity"]["stages"] == {"0": True, "1": False}
    cfg, _ = world(tmp_path / "b", adaptive="absent")
    assert summarize(tmp_path / "b", cfg)["adaptive_comparison"]["status"] == ro.INCOMPLETE


def test_present_but_failing_adaptive_evidence_is_invalid(tmp_path):
    cfg, _ = world(tmp_path, adaptive="bad_identity")
    s = summarize(tmp_path, cfg)
    assert s["verdict"]["verdict"] == ro.POSITION and s["adaptive_comparison"]["status"] == ro.INVALID


def test_smoke_relaxes_only_what_it_says(tmp_path):
    cfg, _ = world(tmp_path, smoke=True)
    s = summarize(tmp_path, cfg, smoke=True)
    assert s["smoke"] and s["verdict"]["exit_code"] == 0, (s["problems"], s["checks"])
    assert s["checks"]["production_plain"]["dirty_excused"]
    assert s["adaptive_comparison"]["status"] == "valid"
    # without --smoke, the same 1-rollout runs are incomplete (the configured 50 rollouts are missing)
    assert summarize(tmp_path, cfg)["verdict"]["verdict"] == ro.INCOMPLETE


def test_weak_reference_label(tmp_path):
    cfg, names = world(tmp_path)
    s = summarize(tmp_path, cfg)
    assert not any(l.startswith("weak_reference") for l in s["verdict"]["labels"])
    cfg2 = {**cfg, "weak_reference_below": 0.95}
    assert "weak_reference_task0" in summarize(tmp_path, cfg2)["verdict"]["labels"]


def test_config_is_consistent():
    cfg = ro.load_config()
    assert cfg["cells"] == {"position": {"task": 1, "stage": 2}, "task": {"task": 2, "stage": 3}}
    names = ro.run_names(cfg, 0)
    assert names == {"seq_ft": "seq_hetero_reverse__seq_ft__seed0",
                     "plain": "seq_hetero_reverse__gpm_projected_adam__seed0",
                     "adaptive": "seq_hetero_reverse__gpm_projected_adam_ne90__seed0"}
    keys = [s.task_key for s in __import__("flowcl.data.curriculum", fromlist=["x"]).load_curriculum(
        cfg["curriculum"]).stages]
    assert keys[1].startswith("libero_goal/") and keys[2].startswith("libero_object/")
    seq = OmegaConf.load(repo_root() / "configs" / "analysis" / "sequence_report.yaml")
    for s in (0, 1, 2):
        entry = seq.criteria.expected_thresholds_by_reference[f"seq_hetero_reverse__seq_ft__seed{s}"]
        assert entry.mode == "derived" and entry.margin == cfg["margin_pp"] / 100
    for role in ("plain", "adaptive"):
        assert "reverse" in cfg["reports"][role]


# ---- the queue ----------------------------------------------------------------------------------------

QUEUE = repo_root() / "scripts" / "queue_reverse_order.sh"


def run_queue(tmp_path, args=(), fail="", present=()):
    results = tmp_path / "results"
    for rel in present:
        p = results / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}")
    env = {**os.environ, "QUEUE_DRY_RUN": "1", "QUEUE_FAIL_STEP": fail,
           "QUEUE_LOG_ROOT": str(tmp_path / "logs"), "QUEUE_RESULTS_ROOT": str(results)}
    done = subprocess.run(["bash", str(QUEUE), *args], env=env, capture_output=True, text=True, timeout=120)
    (logdir,) = list((tmp_path / "logs").glob("queue_*"))
    return done.returncode, (logdir / "queue.log").read_text(), logdir


def test_queue_commands_order_and_paths(tmp_path):
    assert subprocess.run(["bash", "-n", str(QUEUE)]).returncode == 0
    rc, log, logdir = run_queue(tmp_path, ["0"])
    steps = ["0_seqft", "1_gpm", "2_ne90", "3_seqrep_gpm", "4_seqrep_ne90", "5_summary"]
    assert rc == 0 and log.rstrip().endswith("QUEUE DONE ok")
    assert [log.index(f"START {s}:") for s in steps] == sorted(log.index(f"START {s}:") for s in steps)
    res = str(tmp_path / "results")
    cmd = {s: (logdir / f"{s}.log").read_text() for s in steps}
    assert all("--curriculum seq_hetero_reverse" in cmd[s] for s in steps[:3])
    assert all("--single-task-baseline" in cmd[s] for s in steps[:3])
    assert f"--t1-reference-run {res}/seq_hetero_reverse__seq_ft__seed0" in cmd["1_gpm"]
    assert (f"--identity-reference-run {res}/seq_hetero_reverse__gpm_projected_adam__seed0 "
            "--identity-stages 0 1") in cmd["2_ne90"]
    for s, out in (("3_seqrep_gpm", "gpm_seq_reverse_seed0"), ("4_seqrep_ne90", "gpm_seq_ne90_reverse_seed0")):
        assert f"--out {res}/{out}/report.json" in cmd[s]
    forward = {"/gpm_seq/", "/gpm_seq_ne90/", "/gpm_seq_seed0/", "/gpm_seq_ne90_seed0/", "adaptive_gpm",
               "forgetting_diag"}
    assert not any(f in " ".join(cmd.values()) for f in forward)
    rc, log, logdir = run_queue(tmp_path / "s1", ["1"])
    assert "--single-task-baseline" not in (logdir / "0_seqft.log").read_text()


def test_queue_failures_skips_and_resume(tmp_path):
    rc, log, _ = run_queue(tmp_path / "a", ["0"], fail="1_gpm")
    assert rc == 1 and "START 2_ne90" not in log and "START 5_summary" in log and "FAILED: 1_gpm" in log
    rc, log, _ = run_queue(tmp_path / "b", ["0"], fail="2_ne90")
    assert rc == 1 and "START 3_seqrep_gpm" in log and "START 4_seqrep_ne90" not in log
    rc, log, _ = run_queue(tmp_path / "c", ["0", "--from-step", "2"], present=[
        "seq_hetero_reverse__seq_ft__seed0/result.json", "seq_hetero_reverse__gpm_projected_adam__seed0/result.json"])
    assert rc == 0 and "PRIOR 0_seqft ok" in log and "PRIOR 1_gpm ok" in log and "START 2_ne90" in log
    for bad in (["0", "--from-step", "6"], ["x"], ["0", "--bogus"], ["0", "--smoke", "rel/dir"]):
        assert subprocess.run(["bash", str(QUEUE), *bad], capture_output=True).returncode == 2


def test_queue_smoke_mode(tmp_path):
    smoke = tmp_path / "scratch"
    rc, log, logdir = run_queue(tmp_path, ["0", "--smoke", str(smoke)])
    assert rc == 0 and logdir.name.endswith("_smoke")
    cmd = (logdir / "0_seqft.log").read_text()
    assert f"--results-root {smoke}" in cmd and "--steps-per-task 200 --n-episodes 1 --allow-dirty" in cmd
    assert "--single-task-baseline" not in cmd
    assert f"--t1-reference-run {smoke}/seq_hetero_reverse__seq_ft__seed0" in (logdir / "1_gpm.log").read_text()
    summ = (logdir / "5_summary.log").read_text()
    assert f"--results-root {smoke}" in summ and "--smoke" in summ and f"--out {smoke}/reverse_order/seed0" in summ
    bad = subprocess.run(["bash", str(QUEUE), "0", "--smoke", str(repo_root() / "results" / "x")],
                         capture_output=True, env={**os.environ, "QUEUE_LOG_ROOT": str(tmp_path / "l2")})
    assert bad.returncode == 2
