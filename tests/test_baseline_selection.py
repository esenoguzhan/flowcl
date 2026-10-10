"""EWC λ bracket: the selection rule (best, tie, none eligible, edge extension owed and resolved), the
calibration and its refusals, runner-vs-bracket equivalence on the tiny policy (bitwise), the whole
pipeline on a fake world (calibrate, arms, select, verify, smoke arm, compatibility refusals), the
queue's dry run; and the compatibility fixes: compare_resume over every artifact, an explicit identity
reference checkpoint, and its CLI parsing."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from flowcl.experiments import baseline_selection as bs
from flowcl.methods.base import TaskContext
from flowcl.methods.ewc import EWC
from flowcl.train.trainer import train_one_task
from flowcl.utils.libero_paths import repo_root
from flowcl.utils.run import file_sha256
from test_gpm import fresh_policy, small_cfg  # noqa: F401
from test_low_update import BOOT, KEYS, TINY_POLICY, fake_world, spec, stub_evaluator, tiny_policy  # noqa: F401
from test_replay import TASKS, demos

L_SEQFT = 0.0075858106359373775


def arm(ratio, lam, spatial, obj, extension=False):
    return {"ratio": ratio, "lam": lam, "spatial": spatial, "object": obj, "extension": extension}


GRID = [arm(0.01, 1, 0.10, 0.78), arm(0.1, 10, 0.30, 0.76), arm(1, 100, 0.60, 0.70),
        arm(10, 1000, 0.80, 0.60), arm(100, 10000, 0.90, 0.20)]


def test_selection_rule_branches():
    out = bs.select_lambda(GRID, floor=0.68)
    assert out["status"] == "selected" and out["selected"]["ratio"] == 1 and out["flags"] == []
    assert out["eligible"] == [0.01, 0.1, 1]
    tie = [arm(0.01, 1, 0.5, 0.8), arm(0.1, 10, 0.5, 0.8), arm(1, 100, 0.4, 0.8)]
    assert bs.select_lambda(tie, 0.68)["selected"]["ratio"] == 0.01           # ties -> smaller λ
    none = bs.select_lambda(GRID, floor=0.95)
    assert none["selected"]["ratio"] == 0.01 and none["flags"] == ["no_qualifying_setting"]
    edge = [arm(0.01, 1, 0.1, 0.8), arm(100, 10000, 0.9, 0.8)]
    assert bs.select_lambda(edge, 0.68)["status"] == "needs_extension"
    resolved = bs.select_lambda(edge + [arm(1000, 1e5, 0.95, 0.8, extension=True)], 0.68)
    assert resolved["selected"]["ratio"] == 1000 and resolved["flags"] == ["edge_extension", "optimum_at_boundary"]
    inside = bs.select_lambda(edge + [arm(1000, 1e5, 0.5, 0.8, extension=True)], 0.68)
    assert inside["selected"]["ratio"] == 100 and inside["flags"] == ["edge_extension"]
    assert bs.select_lambda([arm(0.01, 1, 0.1, 0.68)], 0.68)["status"] == "needs_extension"  # inclusive floor


def test_calibration_lambdas_and_refusals():
    lam = bs.calibration_lambdas([0.01, 1, 100], D=2.0, L=L_SEQFT)
    assert lam == {"0.01": 0.01 * L_SEQFT, "1": L_SEQFT, "100": 100 * L_SEQFT}
    for D, L in ((0.0, 1.0), (-1.0, 1.0), (float("nan"), 1.0), (1.0, float("inf")), (1.0, 0.0)):
        with pytest.raises(ValueError):
            bs.calibration_lambdas([1], D, L)
    cfg = bs.load_config()
    assert cfg["ewc"]["ratios"] == [0.01, 0.1, 1, 10, 100] and cfg["ewc"]["calibration"] in (None, cfg["ewc"]["calibration"])
    assert bs.extension_ratio(cfg) == 1000.0 and bs.arm_run_id(0.01, 0) == "ewcb__r0.01__seed0"
    queue = (repo_root() / "scripts" / "queue_ewc_bracket.sh").read_text()
    assert 'RATIOS="' + " ".join(bs.ratio_tag(r) for r in cfg["ewc"]["ratios"]) + '"' in queue


def test_bracket_path_equals_runner_path_bitwise(spec):
    """Runner: train T1 -> on_task_end (Fisher) -> train T2. Bracket: load stage 0 in eval() mode ->
    ewc_t1_setup -> train T2. Same Fisher, same anchor, same final weights."""
    from flowcl.data.dataset import ChunkedActionDataset
    from flowcl.data.stats import compute_stats

    eps = {t: demos(spec, t, offset=10 * j) for j, t in enumerate(TASKS[:2])}
    stats = compute_stats(eps[TASKS[0]], embodiment=spec.name, task_id=TASKS[0])
    data = {j: ChunkedActionDataset(eps[t], spec, stats) for j, t in enumerate(TASKS[:2])}
    ns, cfg = "seq_hetero__seq_ft__seed0", small_cfg(steps=3)

    def context(j):
        return TaskContext(task_key=TASKS[j], dataset=data[j], device="cpu", seed_namespace_run_id=ns,
                           batch_size=cfg.batch_size)

    runner_policy, runner = fresh_policy(spec), EWC(lam=50.0, fisher_samples=4)
    train_one_task(runner_policy, data[0], cfg, method=runner, task_idx=0,
                   generator=torch.Generator().manual_seed(0), context=context(0))
    stage0 = {k: v.clone() for k, v in runner_policy.state_dict().items()}
    t1 = dict(runner.fisher_tasks[0])
    train_one_task(runner_policy, data[1], cfg, method=runner, task_idx=1,
                   generator=torch.Generator().manual_seed(1), context=context(1))

    bracket_policy = fresh_policy(spec)
    bracket_policy.load_state_dict(stage0)
    bracket_policy.eval()  # as load_checkpoint leaves it
    ctx = SimpleNamespace(namespace=ns, curriculum=SimpleNamespace(stages=[SimpleNamespace(task_key=t) for t in TASKS[:2]]))
    bracket = bs.ewc_t1_setup(ctx, 50.0, 4, "cpu")(bracket_policy, data)
    assert bracket.fisher_tasks[0]["fingerprint"] == t1["fingerprint"]
    train_one_task(bracket_policy, data[1], cfg, method=bracket, task_idx=1,
                   generator=torch.Generator().manual_seed(1), context=context(1))
    a, b = dict(runner_policy.named_parameters()), dict(bracket_policy.named_parameters())
    assert all(torch.equal(a[n], b[n]) for n in a)
    assert all(torch.equal(runner.fisher[n], bracket.fisher[n]) for n in runner.fisher)


# ---- the whole pipeline on a fake world ----------------------------------------------------------


def calibrated_world(tmp_path, spec):
    from flowcl.train.checkpoint import load_checkpoint, save_checkpoint

    luc, build = fake_world(tmp_path, spec, seeds=(0,))
    ref = tmp_path / "seq_hetero__seq_ft__seed0"
    stats = load_checkpoint(ref / "checkpoints" / "stage0.pt").stats
    moved = tiny_policy(spec)
    with torch.no_grad():
        for p in moved.parameters():
            if p.requires_grad:
                p.add_(0.01)
    save_checkpoint(ref / "checkpoints" / "stage1.pt", policy=moved, policy_config=TINY_POLICY, spec=spec,
                    stats=stats, run_id=ref.name, stage=1, task_key=KEYS[1])
    (ref / "result.json").write_text(json.dumps({"run_id": ref.name,
                                                 "stages": [{}, {"mean_last_50_loss": L_SEQFT}]}))
    cfg = copy.deepcopy(bs.load_config())
    cfg["ewc"]["calibration"] = None
    cfg["ewc"]["fisher_samples"] = 3
    cfg["ewc"]["recipe"] = {"steps": 2, "lr": 1e-4}
    kw = dict(results_root=tmp_path, device="cpu", build_datasets=build, low_update_cfg=luc)
    payload = bs.calibrate(cfg, **kw)
    cfg["ewc"]["calibration"] = {"file": "baseline_selection/ewc_calibration.json",
                                 "sha256": file_sha256(bs.calibration_path(cfg, tmp_path)),
                                 "lambdas": payload["lambdas"]}
    return cfg, kw, payload


def test_calibrate_arms_select_verify_and_refusals(spec, tmp_path, capsys):
    cfg, kw, payload = calibrated_world(tmp_path, spec)
    assert payload["D"] > 0 and payload["L"]["value"] == L_SEQFT
    assert payload["lambdas"] == bs.calibration_lambdas(cfg["ewc"]["ratios"], payload["D"], L_SEQFT)
    assert payload["text_cache"] and payload["fisher_t1"]["n_samples"] == 3
    with pytest.raises(RuntimeError, match="already calibrated"):
        bs.calibrate(cfg, **kw)
    train = dict(kw, evaluator=stub_evaluator, bootstrap=BOOT, allow_dirty=True)
    rows = [bs.run_arm(cfg, r, **train) for r in cfg["ewc"]["ratios"]]
    assert [r["run_id"] for r in rows] == [bs.arm_run_id(r, 0) for r in cfg["ewc"]["ratios"]]
    assert all(r["ewc"]["t1_fisher"]["fingerprint"] == payload["fisher_t1"]["fingerprint"] for r in rows)
    first = bs.arm_dir(cfg, tmp_path, 0.01, 0) / "row.json"
    mtime = first.stat().st_mtime
    bs.run_arm(cfg, 0.01, **train)  # published: reused, not retrained
    assert first.stat().st_mtime == mtime and "reused" in capsys.readouterr().out
    with pytest.raises(RuntimeError, match="no edge extension is owed"):
        bs.run_extension(cfg, **train)
    sel = bs.run_select(cfg, results_root=tmp_path)
    # The stub gives every arm seq_ft's successes: all eligible, Spatial tied -> the smallest λ.
    assert sel["status"] == "selected" and sel["selected"]["ratio"] == 0.01 and sel["flags"] == []
    assert sel["rule_sha256"] == file_sha256(bs.rule_path()) and sel["calibration_sha256"] == \
        cfg["ewc"]["calibration"]["sha256"]
    assert bs.verify(cfg, results_root=tmp_path)["passed"]
    with pytest.raises(ValueError, match="another rule or calibration"):
        bs.check_bundle(sel, "0" * 64, sel["calibration_sha256"])
    bad = copy.deepcopy(cfg)
    bad["ewc"]["calibration"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="differs from the rule"):
        bs.run_select(bad, results_root=tmp_path)
    assert not bs.verify(bad, results_root=tmp_path)["passed"]
    bad = copy.deepcopy(cfg)
    bad["ewc"]["calibration"]["lambdas"]["1"] *= 1.0000001
    with pytest.raises(ValueError, match="registered λ"):
        bs.run_arm(bad, 1, **train)
    smoke = bs.smoke_arm(cfg, 1, stop_after=1, episodes=1, out_root=tmp_path / "scratch",
                         **{k: v for k, v in train.items() if k != "allow_dirty"})
    assert smoke["passed"], smoke
    method_yaml = bs.write_smoke_method(cfg, tmp_path / "ewc_smoke.yaml", results_root=tmp_path)
    from flowcl.data.config import load_method_config
    name, mkw = load_method_config(method_yaml)
    assert name == "ewc" and mkw["lam"] == payload["lambdas"]["1"] and mkw["fisher_samples"] == 256


# ---- the queue ----------------------------------------------------------------------------------

QUEUE = repo_root() / "scripts" / "queue_ewc_bracket.sh"


def run_queue(tmp_path, args=(), fail="", step_rc="", present=()):
    results = tmp_path / "results"
    for rel in present:
        p = results / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}")
    env = {**os.environ, "QUEUE_DRY_RUN": "1", "QUEUE_FAIL_STEP": fail, "QUEUE_STEP_RC": step_rc,
           "QUEUE_LOG_ROOT": str(tmp_path / "logs"), "QUEUE_RESULTS_ROOT": str(results)}
    done = subprocess.run(["bash", str(QUEUE), *args], env=env, capture_output=True, text=True, timeout=120)
    (logdir,) = list((tmp_path / "logs").glob("queue_*"))
    return done.returncode, (logdir / "queue.log").read_text(), logdir


ARMS = ["0_arm_r0.01", "1_arm_r0.1", "2_arm_r1", "3_arm_r10", "4_arm_r100"]


def order(log, steps):
    return [log.index(f"START {s}:") for s in steps] == sorted(log.index(f"START {s}:") for s in steps)


def test_queue_order_with_and_without_the_extension(tmp_path):
    assert subprocess.run(["bash", "-n", str(QUEUE)]).returncode == 0
    rc, log, logdir = run_queue(tmp_path / "a")
    assert rc == 0 and log.rstrip().endswith("QUEUE DONE ok") and order(log, ARMS + ["5_select", "8_verify"])
    assert "START 6_extension" not in log and "START 7_select" not in log
    assert (logdir / "2_arm_r1.log").read_text().strip().endswith("scripts/baseline_selection.py --arm 1")
    rc, log, _ = run_queue(tmp_path / "b", step_rc="5_select=10")
    assert rc == 0 and order(log, ARMS + ["5_select", "6_extension", "7_select", "8_verify"])
    rc, log, _ = run_queue(tmp_path / "c", step_rc="5_select=1")
    assert rc == 1 and "FAILED: 5_select" in log and "START 8_verify" in log
    rc, log, _ = run_queue(tmp_path / "d", fail="8_verify")
    assert rc == 1 and log.rstrip().endswith("QUEUE DONE FAILED: 8_verify")


def test_queue_failures_resume_and_arguments(tmp_path):
    rc, log, _ = run_queue(tmp_path / "a", fail="2_arm_r1")
    assert rc == 1 and "START 3_arm_r10" in log and "SKIP 5-8" in log and "START 5_select" not in log
    rows = [f"baseline_selection/ewc/seed0/ewcb__r{r}__seed0/row.json" for r in ("0.01", "0.1")]
    rc, log, _ = run_queue(tmp_path / "b", ["--from-step", "2"], present=rows)
    assert rc == 0 and "PRIOR 0_arm_r0.01 ok" in log and "START 2_arm_r1" in log
    rc, log, _ = run_queue(tmp_path / "c", ["--from-step", "2"], present=rows[:1])
    assert rc == 1 and "PRIOR 1_arm_r0.1 missing" in log and "START 5_select" not in log
    for bad in (["--from-step", "6"], ["--bogus"], ["--smoke", "rel/dir"]):
        assert subprocess.run(["bash", str(QUEUE), *bad], capture_output=True).returncode == 2


def test_queue_smoke_mode(tmp_path):
    smoke = tmp_path / "scratch"
    rc, log, logdir = run_queue(tmp_path, ["--smoke", str(smoke)])
    steps = ["s0_smoke_method", "s1_seqft", "s2_replay", "s3_ewc", "s4_replay_resume", "s5_replay_compare",
             "s6_ewc_resume", "s7_ewc_compare", "s8_bracket_arm"]
    assert rc == 0 and logdir.name.endswith("_smoke") and order(log, steps)
    cmd = {s: (logdir / f"{s}.log").read_text() for s in steps}
    ref = f"{smoke}/seq_hetero__seq_ft__seed0"
    for s in ("s2_replay", "s3_ewc"):
        assert f"--t1-reference-run {ref} --identity-reference-run {ref} --identity-stages 0" in cmd[s]
        assert "--steps-per-task 200 --n-episodes 1 --allow-dirty" in cmd[s]
    assert f"--method {smoke}/ewc_smoke.yaml" in cmd["s3_ewc"]
    assert f"--resume-run {smoke}/seq_hetero__ewc__seed0 --start-stage 3" in cmd["s6_ewc_resume"]
    assert "--stage 3 --all-artifacts" in cmd["s5_replay_compare"]
    assert f"--smoke-arm 1 --out-root {smoke}/bracket" in cmd["s8_bracket_arm"]


@pytest.mark.parametrize("script, flags", [
    ("baseline_selection.py", ["--calibrate", "--arm", "--select", "--extension", "--verify", "--smoke-arm",
                               "--write-smoke-method", "--results-root", "--out-root", "--allow-dirty"]),
    ("compare_resume.py", ["--source", "--resumed", "--stage", "--all-artifacts", "--out"]),
    ("run_continual.py", ["--identity-reference-checkpoint", "--resume-run", "--start-stage",
                          "--t1-reference-run", "--identity-reference-run", "--identity-stages"]),
])
def test_every_queue_step_accepts_its_flags(script, flags):
    done = subprocess.run([sys.executable, str(repo_root() / "scripts" / script), "--help"],
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    for flag in flags:
        assert flag in done.stdout, (script, flag)


# ---- compatibility fixes ------------------------------------------------------------------------


def fake_run(root: Path, name: str, buffers: dict[str, bytes], weight=1.0):
    run = root / name
    (run / "method").mkdir(parents=True)
    (run / "eval").mkdir()
    (run / "checkpoints").mkdir()
    arts = []
    for fname, content in buffers.items():
        torch.save({"payload": torch.tensor(list(content), dtype=torch.uint8)}, run / "method" / fname)
        arts.append({"path": f"method/{fname}", "sha256": file_sha256(run / "method" / fname)})
    torch.save({"state_dict": {"w": torch.full((2,), weight)}, "extra": {"method_artifacts": arts}},
               run / "checkpoints" / "stage3.pt")
    (run / "eval" / "stage3.json").write_text(json.dumps({"tasks": [{"task_key": "t", "successes": [True],
                                                                     "n_steps": [5]}]}))
    (run / "resume.json").write_text(json.dumps({"checkpoint": "c", "sha256": "s", "start_stage": 3,
                                                 "method_state": {}, "text_cache": {}}))
    return run


def test_compare_resume_checks_every_named_artifact(tmp_path):
    from flowcl.experiments.resume_check import compare_resume

    files = {"replay_task0.pt": b"a", "replay_task1.pt": b"b", "replay_task3.pt": b"d"}
    src = fake_run(tmp_path / "src", "run", files)
    same = fake_run(tmp_path / "same", "run", files)
    assert compare_resume(src, same, 3, all_artifacts=True)["passed"]
    differ = fake_run(tmp_path / "differ", "run", {**files, "replay_task0.pt": b"z"})
    assert compare_resume(differ, src, 3)["passed"]                 # the *_task3 glob misses it
    report = compare_resume(differ, src, 3, all_artifacts=True)
    assert not report["passed"] and not report["method_artifacts"]["replay_task0.pt"]["passed"]
    fewer = fake_run(tmp_path / "fewer", "run", {k: v for k, v in files.items() if k != "replay_task1.pt"})
    assert not compare_resume(src, fewer, 3, all_artifacts=True)["passed"]


def test_identity_check_against_an_explicit_reference_checkpoint(tmp_path):
    from flowcl.train.continual import stage_identity_check

    a = tmp_path / "a.pt"
    torch.save({"state_dict": {"w": torch.ones(3)}}, a)
    b = tmp_path / "b.pt"
    torch.save({"state_dict": {"w": torch.ones(3)}}, b)
    check = stage_identity_check(a, None, 1, reference_checkpoint=b)
    assert check["passed"] and check["reference_checkpoint"] == str(b) and check["reference_run"] is None
    torch.save({"state_dict": {"w": torch.tensor([1.0, 1.0, 2.0])}}, b)
    assert not stage_identity_check(a, None, 1, reference_checkpoint=b)["passed"]
    with pytest.raises(ValueError, match="needs a reference"):
        stage_identity_check(a, None, 1)


def test_identity_checkpoint_cli_parsing():
    import argparse

    spec_ = importlib.util.spec_from_file_location("run_continual_cli", repo_root() / "scripts" / "run_continual.py")
    mod = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(mod)
    parser = argparse.ArgumentParser()
    assert mod.parse_identity_checkpoints(parser, ["1=/x/final.pt"]) == {1: Path("/x/final.pt")}
    for bad in (["1"], ["a=/x"], ["1="], ["1=/x", "1=/y"]):
        with pytest.raises(SystemExit):
            mod.parse_identity_checkpoints(parser, bad)


def test_review_fixes_rule_path_inputs_and_fingerprint_precheck(spec, tmp_path):
    """(1) An alternative --config is the rule that is hashed: arms published under the default rule
    are not reused under it. (2) A calibration made from other inputs (Fisher sample count, seed, a
    replaced reference checkpoint) is refused, and an arm whose T1 Fisher differs from the
    calibration's stops before training."""
    cfg, kw, payload = calibrated_world(tmp_path, spec)
    train = dict(kw, evaluator=stub_evaluator, bootstrap=BOOT, allow_dirty=True)
    bs.run_arm(cfg, 0.01, **train)
    alt = tmp_path / "alt_rule.yaml"
    alt.write_text(bs.rule_path().read_text() + "\n# an alternative rule\n")
    for r in cfg["ewc"]["ratios"][1:]:
        bs.run_arm(cfg, r, **train)
    with pytest.raises(ValueError, match="incompatible"):
        bs.run_select(cfg, results_root=tmp_path, rule=alt)  # rows carry the default rule's SHA
    with pytest.raises(ValueError, match="incompatible"):
        bs.run_arm(cfg, 0.01, rule=alt, **train)
    assert bs.run_select(cfg, results_root=tmp_path)["status"] == "selected"

    for mutate, match in ((lambda c: c["ewc"].__setitem__("fisher_samples", 4), "Fisher samples"),
                          (lambda c: c.__setitem__("bracket_seed", 1), "seed")):
        bad = copy.deepcopy(cfg)
        mutate(bad)
        with pytest.raises(ValueError, match=match):
            bs.validate_calibration(bad, tmp_path)
    stage1 = tmp_path / "seq_hetero__seq_ft__seed0" / "checkpoints" / "stage1.pt"
    original = stage1.read_bytes()
    stage1.write_bytes(original + b"x")
    with pytest.raises(ValueError, match="no longer has the calibrated SHA-256"):
        bs.validate_calibration(cfg, tmp_path)
    stage1.write_bytes(original)
    bs.validate_calibration(cfg, tmp_path)

    from flowcl.train.checkpoint import load_checkpoint
    from flowcl.experiments.low_update import seed_context

    ctx = seed_context(kw["low_update_cfg"], 0, tmp_path)
    start = load_checkpoint(ctx.ref_dir / "checkpoints" / "stage0.pt")
    datasets = {j: kw["build_datasets"](ctx.curriculum.stages[j]) for j in (0, 1)}
    with pytest.raises(RuntimeError, match="differs from the calibration"):
        bs.ewc_t1_setup(ctx, 1.0, 3, "cpu", expected_fingerprint="0" * 64)(start.policy, datasets)
    ok = bs.ewc_t1_setup(ctx, 1.0, 3, "cpu", expected_fingerprint=payload["fisher_t1"]["fingerprint"])
    assert ok(start.policy, datasets).fisher_tasks[0]["fingerprint"] == payload["fisher_t1"]["fingerprint"]


# ---- v2: sampled Fisher, mean normalization, probe tie-break, diagnostics, queue -----------------

V2 = repo_root() / "configs" / "analysis" / "baseline_selection_ewc_v2.yaml"


def test_bracket_path_equals_runner_path_bitwise_v2(spec):
    """The same equivalence with the sampled estimator and mean normalization."""
    from flowcl.data.dataset import ChunkedActionDataset
    from flowcl.data.stats import compute_stats

    eps = {t: demos(spec, t, offset=10 * j) for j, t in enumerate(TASKS[:2])}
    stats = compute_stats(eps[TASKS[0]], embodiment=spec.name, task_id=TASKS[0])
    data = {j: ChunkedActionDataset(eps[t], spec, stats) for j, t in enumerate(TASKS[:2])}
    ns, cfg = "seq_hetero__seq_ft__seed0", small_cfg(steps=3)
    opts = {"fisher_estimator": "sampled", "fisher_normalization": "mean"}

    def context(j):
        return TaskContext(task_key=TASKS[j], dataset=data[j], device="cpu", seed_namespace_run_id=ns,
                           batch_size=cfg.batch_size)

    runner_policy, runner = fresh_policy(spec), EWC(lam=50.0, fisher_samples=4, **opts)
    train_one_task(runner_policy, data[0], cfg, method=runner, task_idx=0,
                   generator=torch.Generator().manual_seed(0), context=context(0))
    stage0 = {k: v.clone() for k, v in runner_policy.state_dict().items()}
    t1 = dict(runner.fisher_tasks[0])
    train_one_task(runner_policy, data[1], cfg, method=runner, task_idx=1,
                   generator=torch.Generator().manual_seed(1), context=context(1))
    bracket_policy = fresh_policy(spec)
    bracket_policy.load_state_dict(stage0)
    bracket_policy.eval()
    ctx = SimpleNamespace(namespace=ns, curriculum=SimpleNamespace(stages=[SimpleNamespace(task_key=t) for t in TASKS[:2]]))
    bracket = bs.ewc_t1_setup(ctx, 50.0, 4, "cpu", expected_fingerprint=t1["fingerprint"], **opts)(bracket_policy, data)
    assert bracket.fisher_tasks[0]["normalized_fingerprint"] == t1["normalized_fingerprint"]
    train_one_task(bracket_policy, data[1], cfg, method=bracket, task_idx=1,
                   generator=torch.Generator().manual_seed(1), context=context(1))
    a, b = dict(runner_policy.named_parameters()), dict(bracket_policy.named_parameters())
    assert all(torch.equal(a[n], b[n]) for n in a)
    assert all(torch.equal(runner.fisher[n], bracket.fisher[n]) for n in runner.fisher)


def parm(ratio, lam, spatial, obj, probe, extension=False):
    return {**arm(ratio, lam, spatial, obj, extension), "spatial_probe": probe}


def test_v2_tie_break_zero_flag_and_extension_through_the_tie_break():
    v2 = {"tie_break": "spatial_probe_loss", "flag_zero": True}
    grid = [parm(0.01, 1, 0.0, 0.84, 0.73), parm(1, 100, 0.0, 0.86, 0.56), parm(100, 1e4, 0.0, 0.70, 0.17),
            parm(1e4, 1e6, 0.0, 0.40, 0.02)]                       # the strongest is ineligible
    out = bs.select_lambda(grid, 0.68, **v2)
    assert out["selected"]["ratio"] == 100 and out["flags"] == ["all_eligible_spatial_zero"]
    v1 = bs.select_lambda(grid, 0.68)                               # v1's rule on the same arms
    assert v1["selected"]["ratio"] == 0.01 and v1["flags"] == []
    edge = [parm(0.01, 1, 0.0, 0.84, 0.73), parm(1e4, 1e6, 0.0, 0.70, 0.05)]
    owed = bs.select_lambda(edge, 0.68, **v2)                      # largest selected via the probe tie-break
    assert owed["status"] == "needs_extension" and "all_eligible_spatial_zero" in owed["flags"]
    done = bs.select_lambda(edge + [parm(1e5, 1e7, 0.0, 0.69, 0.01, extension=True)], 0.68, **v2)
    assert done["selected"]["ratio"] == 1e5 and done["flags"] == ["all_eligible_spatial_zero", "edge_extension",
                                                                   "optimum_at_boundary"]
    success = [parm(0.01, 1, 0.0, 0.84, 0.73), parm(1, 100, 0.10, 0.80, 0.56), parm(10, 1e3, 0.10, 0.75, 0.30),
               parm(100, 1e4, 0.0, 0.72, 0.17)]                    # r = 10 is not the boundary here
    out = bs.select_lambda(success, 0.68, **v2)
    assert out["selected"]["ratio"] == 10 and out["flags"] == []  # tied successes -> lower probe loss
    with pytest.raises(ValueError):
        bs.select_lambda(success, 0.68, tie_break="bogus")


def test_v1_spec_unchanged_and_v2_spec_names_the_estimator():
    v1, v2 = bs.load_config(), bs.load_config(V2)
    assert set(bs.method_spec(v1, 1, 1.0, "r", "c")) == {"family", "ratio", "lam", "fisher_samples",
                                                          "rule_sha256", "calibration_sha256"}
    spec2 = bs.method_spec(v2, 1, 1.0, "r", "c")
    assert spec2["fisher_estimator"] == "sampled" and spec2["fisher_normalization"] == "mean"
    assert bs.fisher_options(v1) == {"fisher_estimator": "empirical", "fisher_normalization": "none"}
    assert bs.selection_options(v1) == {"tie_break": "smaller_lambda", "flag_zero": False}
    assert bs.selection_options(v2) == {"tie_break": "spatial_probe_loss", "flag_zero": True}
    assert v2["ewc"]["ratios"] == [0.01, 0.1, 1, 10, 100, 1000, 10000] and v2["out"] != v1["out"]
    assert bs.extension_ratio(v2) == 1e5
    assert v2["ewc"]["v1_t1_fingerprint"].startswith("320120a7")
    queue = (repo_root() / "scripts" / "queue_ewc_bracket_v2.sh").read_text()
    assert 'RATIOS="' + " ".join(bs.ratio_tag(r) for r in v2["ewc"]["ratios"]) + '"' in queue


def test_diagnostics_helpers():
    g = torch.Generator().manual_seed(0)
    a = torch.rand(20000, generator=g, dtype=torch.float64) ** 4
    assert bs.log_correlation(a, a) == pytest.approx(1.0)
    assert abs(bs.log_correlation(a, torch.rand(20000, generator=g, dtype=torch.float64) ** 4)) < 0.05
    assert bs.topk_jaccard(a, a) == 1.0
    b = torch.zeros(1000, dtype=torch.float64)
    b[:10] = 1.0
    c = torch.zeros(1000, dtype=torch.float64)
    c[-10:] = 1.0
    assert bs.topk_jaccard(b, c) == 0.0
    con = bs.concentration(b + 1e-12)
    assert con["top_0.01_share"] == pytest.approx(1.0, rel=1e-6) and con["near_zero_importance_fraction"] == 0.99
    shares = bs.module_shares({"trunk.a": torch.ones(3), "trunk.b": torch.ones(1), "flow_head.c": torch.ones(4)})
    assert shares == {"flow_head": 0.5, "trunk": 0.5}


def test_v2_pipeline_on_a_fake_world(spec, tmp_path):
    from flowcl.train.checkpoint import load_checkpoint, save_checkpoint

    luc, build = fake_world(tmp_path, spec, seeds=(0,))
    ref = tmp_path / "seq_hetero__seq_ft__seed0"
    stats = load_checkpoint(ref / "checkpoints" / "stage0.pt").stats
    moved = tiny_policy(spec)
    with torch.no_grad():
        for p in moved.parameters():
            if p.requires_grad:
                p.add_(0.01)
    save_checkpoint(ref / "checkpoints" / "stage1.pt", policy=moved, policy_config=TINY_POLICY, spec=spec,
                    stats=stats, run_id=ref.name, stage=1, task_key=KEYS[1])
    (ref / "result.json").write_text(json.dumps({"run_id": ref.name, "stages": [{}, {"mean_last_50_loss": L_SEQFT}]}))
    cfg = copy.deepcopy(bs.load_config(V2))
    cfg["ewc"]["fisher_samples"] = 3
    cfg["ewc"]["recipe"] = {"steps": 2, "lr": 1e-4}
    kw = dict(results_root=tmp_path, device="cpu", build_datasets=build, low_update_cfg=luc, rule=V2)
    payload = bs.calibrate(cfg, **kw)
    assert payload["fisher_estimator"] == "sampled" and payload["fisher_normalization"] == "mean"
    assert payload["fisher_t1"]["estimator"] == "sampled" and payload["fisher_t1"]["mean_normalized_by"] > 0
    cfg["ewc"]["calibration"] = {"file": "baseline_selection_v2/ewc_calibration.json",
                                 "sha256": file_sha256(bs.calibration_path(cfg, tmp_path)),
                                 "lambdas": payload["lambdas"]}
    diag = bs.fisher_diagnostics(cfg, **kw)
    assert set(diag["reliability"]) == {"sampled", "empirical"}
    assert diag["reproduction"]["sampled_A_equals_calibration"] is True
    assert diag["reproduction"]["empirical_A_equals_v1"] is False  # a fake world, not v1's model
    train = dict(kw, evaluator=stub_evaluator, bootstrap=BOOT, allow_dirty=True)
    rows = [bs.run_arm(cfg, r, **train) for r in cfg["ewc"]["ratios"]]
    assert all(r["method"]["fisher_estimator"] == "sampled" for r in rows)
    sel = bs.run_select(cfg, results_root=tmp_path, rule=V2)
    if sel["status"] == "needs_extension":
        bs.run_extension(cfg, **train)
        sel = bs.run_select(cfg, results_root=tmp_path, rule=V2)
        assert "edge_extension" in sel["flags"]
    assert sel["status"] == "selected"
    assert all("spatial_probe" in a for a in sel["arms"].values())
    assert bs.verify(cfg, results_root=tmp_path, rule=V2)["passed"]
    bad = copy.deepcopy(cfg)
    bad["ewc"]["fisher_normalization"] = "none"
    with pytest.raises(ValueError, match="fisher_normalization"):
        bs.validate_calibration(bad, tmp_path)


QUEUE2 = repo_root() / "scripts" / "queue_ewc_bracket_v2.sh"


def run_queue2(tmp_path, args=(), fail="", step_rc="", present=()):
    results = tmp_path / "results"
    for rel in present:
        p = results / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}")
    env = {**os.environ, "QUEUE_DRY_RUN": "1", "QUEUE_FAIL_STEP": fail, "QUEUE_STEP_RC": step_rc,
           "QUEUE_LOG_ROOT": str(tmp_path / "logs"), "QUEUE_RESULTS_ROOT": str(results)}
    done = subprocess.run(["bash", str(QUEUE2), *args], env=env, capture_output=True, text=True, timeout=120)
    (logdir,) = list((tmp_path / "logs").glob("queue_*"))
    return done.returncode, (logdir / "queue.log").read_text(), logdir


ARMS2 = [f"{k}_arm_r{r}" for k, r in enumerate(["0.01", "0.1", "1", "10", "100", "1000", "10000"], start=1)]


def test_v2_queue_order_extension_failures_and_resume(tmp_path):
    assert subprocess.run(["bash", "-n", str(QUEUE2)]).returncode == 0
    rc, log, logdir = run_queue2(tmp_path / "a")
    assert rc == 0 and order(log, ["0_diagnostics"] + ARMS2 + ["8_select", "11_verify"])
    assert "START 9_extension" not in log
    assert all("--config configs/analysis/baseline_selection_ewc_v2.yaml" in (logdir / f"{s}.log").read_text()
               for s in ["0_diagnostics", "7_arm_r10000", "8_select", "11_verify"])
    rc, log, _ = run_queue2(tmp_path / "b", step_rc="8_select=10")
    assert rc == 0 and order(log, ARMS2 + ["8_select", "9_extension", "10_select", "11_verify"])
    rc, log, _ = run_queue2(tmp_path / "c", fail="0_diagnostics")   # descriptive: never blocks the arms
    assert rc == 1 and "START 1_arm_r0.01" in log and "START 8_select" in log and "FAILED: 0_diagnostics" in log
    rc, log, _ = run_queue2(tmp_path / "d", fail="3_arm_r1")
    assert rc == 1 and "START 4_arm_r10" in log and "SKIP 8-11" in log
    rows = [f"baseline_selection_v2/ewc/seed0/ewcb__r{r}__seed0/row.json" for r in ("0.01", "0.1")]
    rc, log, _ = run_queue2(tmp_path / "e", ["--from-step", "3"], present=rows)
    assert rc == 0 and "SKIP 0_diagnostics" in log and "PRIOR 2_arm_r0.1 ok" in log and "START 3_arm_r1" in log
    for bad in (["--from-step", "9"], ["--bogus"], ["--smoke", "rel/dir"]):
        assert subprocess.run(["bash", str(QUEUE2), *bad], capture_output=True).returncode == 2


def test_v2_queue_smoke_mode(tmp_path):
    smoke = tmp_path / "scratch"
    rc, log, logdir = run_queue2(tmp_path, ["--smoke", str(smoke)])
    steps = ["s0_smoke_method", "s1_seqft", "s2_ewc", "s3_ewc_resume", "s4_ewc_compare", "s5_diagnostics",
             "s6_bracket_arm"]
    assert rc == 0 and order(log, steps)
    cmd = {s: (logdir / f"{s}.log").read_text() for s in steps}
    assert f"--method {smoke}/ewc_smoke.yaml" in cmd["s2_ewc"] and "--identity-stages 0" in cmd["s2_ewc"]
    assert "--stage 3 --all-artifacts" in cmd["s4_ewc_compare"]
    assert f"--diagnostics --diag-samples 64 --out-root {smoke}/bracket" in cmd["s5_diagnostics"]
    assert all("--config configs/analysis/baseline_selection_ewc_v2.yaml" in cmd[s]
               for s in ("s0_smoke_method", "s5_diagnostics", "s6_bracket_arm"))
