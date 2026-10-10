"""LoRA's rank × lr bracket (flowcl/experiments/lora_selection.py): the pure rule (orderings, the
fallback, the flags, every edge and a corner), the bracket path against the runner bit for bit, the
whole pipeline on a fake world (arms, extensions, the bundle, verify and its refusals, the smoke
check's plumbing), and the queue's dry runs (order, extensions, failures, resume, the deadline guard,
smoke commands)."""

from __future__ import annotations

import copy
import dataclasses
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from flowcl.experiments import lora_selection as ls
from flowcl.methods.lora import LoRA
from flowcl.train.continual import method_stage_config
from flowcl.train.trainer import train_one_task
from flowcl.utils.libero_paths import repo_root
from test_gpm import fresh_policy, small_cfg  # noqa: F401
from test_lora import ctx, data_for
from test_low_update import BOOT, fake_world, spec, stub_evaluator  # noqa: F401

RANKS, LRS = [4, 16, 64], [1e-4, 3e-4, 1e-3]
EXT = {"rank": {"low": 1, "high": 256}, "lr": {"low": 3e-5, "high": 3e-3}}


def arm(rank, lr, spatial=0.0, obj=0.8, sp=0.5, op=0.01, extension=False):
    return {"rank": rank, "lr": lr, "extension": extension, "run_id": f"{rank}/{lr}", "spatial": spatial,
            "object": obj, "spatial_probe": sp, "object_probe": op}


def grid_arms(overrides=None):
    """Nine arms; ``overrides`` maps (rank, lr) -> field overrides."""
    overrides = overrides or {}
    out = []
    for r in RANKS:
        for lr in LRS:
            out.append(arm(r, lr, **overrides.get((r, lr), {})))
    return out


def select(arms, floor=0.68):
    return ls.select_lora(arms, floor, RANKS, LRS, EXT)


def test_rule_orderings_flags_and_fallback():
    # interior winner on Spatial success: nothing owed
    out = select(grid_arms({(16, 3e-4): {"spatial": 0.2}, (4, 1e-4): {"spatial": 0.3, "obj": 0.5}}))
    assert out["status"] == "selected" and (out["selected"]["rank"], out["selected"]["lr"]) == (16, 3e-4)
    assert out["flags"] == [] and out["owed_extensions"] == [] and (4, 1e-4) not in out["eligible"]
    # Spatial tied -> lower Spatial probe loss
    out = select(grid_arms({(16, 3e-4): {"spatial": 0.2, "sp": 0.4}, (64, 3e-4): {"spatial": 0.2, "sp": 0.3}}))
    assert (out["status"], out["grid_selection"]) == ("needs_extension", (64, 3e-4))
    # probe tied -> smaller rank, then smaller lr; all at zero -> flagged
    out = select(grid_arms({(16, 3e-4): {"sp": 0.1}, (64, 3e-4): {"sp": 0.1}}))
    assert out["status"] == "selected" and out["selected"]["rank"] == 16 and out["flags"] == ["all_eligible_spatial_zero"]
    out = select(grid_arms({(16, 3e-4): {"sp": 0.1}, (16, 1e-3): {"sp": 0.1}}))
    assert out["status"] == "selected" and (out["selected"]["rank"], out["selected"]["lr"]) == (16, 3e-4)
    # none eligible -> the highest Object, then the lower Object probe loss, then smaller rank; no extension
    out = select(grid_arms({(r, lr): {"obj": 0.5} for r in RANKS for lr in LRS} |
                 {(64, 1e-3): {"obj": 0.6, "op": 0.02}, (4, 1e-4): {"obj": 0.6, "op": 0.03}}))
    assert out["status"] == "selected" and (out["selected"]["rank"], out["selected"]["lr"]) == (64, 1e-3)
    assert out["flags"] == ["no_qualifying_setting"] and out["owed_extensions"] == [] and out["eligible"] == []
    # inclusive floor
    out = select(grid_arms({(16, 3e-4): {"spatial": 0.1, "obj": 0.68}}))
    assert out["selected"]["rank"] == 16


@pytest.mark.parametrize("cell, owed", [
    ((4, 3e-4), [(1, 3e-4)]), ((64, 3e-4), [(256, 3e-4)]), ((16, 1e-4), [(16, 3e-5)]),
    ((16, 1e-3), [(16, 3e-3)]), ((64, 1e-3), [(256, 1e-3), (64, 3e-3)]), ((4, 1e-4), [(1, 1e-4), (4, 3e-5)]),
])
def test_edges_and_corners(cell, owed):
    arms = grid_arms({cell: {"spatial": 0.1}})
    out = select(arms)
    assert out["status"] == "needs_extension" and out["owed_extensions"] == owed and out["missing"] == owed
    assert "edge_extension_owed" in out["flags"]
    # one of two present: still owed
    if len(owed) == 2:
        out = select(arms + [arm(*owed[0], extension=True)])
        assert out["status"] == "needs_extension" and out["missing"] == owed[1:]
    # all present: re-applied once over all arms
    beaten = select(arms + [arm(r, lr, extension=True) for r, lr in owed])
    assert beaten["status"] == "selected" and beaten["selected"]["extension"] is False
    assert beaten["flags"] == ["edge_extension"]
    winner = select(arms + [arm(r, lr, spatial=0.3, extension=True) for r, lr in owed])
    assert winner["selected"]["extension"] and winner["flags"] == ["edge_extension", "optimum_at_boundary"]
    # an extension selected at a new edge owes nothing more (the rule is re-applied once)
    assert winner["owed_extensions"] == owed


def test_rule_refusals():
    with pytest.raises(ValueError, match="not the rule's grid"):
        select(grid_arms()[:-1])
    with pytest.raises(ValueError, match="not owed"):
        select(grid_arms({(16, 3e-4): {"spatial": 0.1}}) + [arm(1, 3e-4, extension=True)])
    assert ls.arm_run_id(4, 1e-4, 0) == "lorab__r4__lr0.0001__seed0" and ls.lr_tag(3e-5) == "3e-05"
    cfg = ls.load_config()
    assert ls.grid(cfg) == [(r, lr) for r in RANKS for lr in LRS] and cfg["lora"]["extension"] == EXT
    assert cfg["lora"]["alpha"] == 16 and cfg["lora"]["object_tolerance"] == 0.10
    queue = (repo_root() / "scripts" / "queue_lora_bracket.sh").read_text()
    assert 'ARMS="' + " ".join(f"{r}:{ls.lr_tag(lr)}" for r, lr in ls.grid(cfg)) + '"' in queue


# ---- the bracket path against the runner ----------------------------------------------------------


def test_bracket_path_equals_runner_path_bitwise(spec):
    """Runner: T1 with LoRA (a no-op), then T2 at the method's stage lr. Bracket: stage 0 loaded in
    eval() mode, a fresh LoRA, T2 at the arm's lr. Same adapters, same final weights, same history."""
    data = data_for(spec)
    cfg = small_cfg(steps=3)
    runner_policy, runner = fresh_policy(spec), LoRA(rank=2, lr=2e-3)
    train_one_task(runner_policy, data[0], method_stage_config(cfg, runner, 0), method=runner, task_idx=0,
                   generator=torch.Generator().manual_seed(0), context=ctx(data, 0))
    stage0 = {k: v.clone() for k, v in runner_policy.state_dict().items()}
    train_one_task(runner_policy, data[1], method_stage_config(cfg, runner, 1), method=runner, task_idx=1,
                   generator=torch.Generator().manual_seed(1), context=ctx(data, 1))
    bracket_policy = fresh_policy(spec)
    bracket_policy.load_state_dict(stage0)
    bracket_policy.eval()  # as load_checkpoint leaves it
    bracket = LoRA(rank=2, lr=2e-3)
    train_one_task(bracket_policy, data[1], dataclasses.replace(cfg, lr=2e-3), method=bracket, task_idx=1,
                   generator=torch.Generator().manual_seed(1), context=ctx(data, 1))
    a, b = runner_policy.state_dict(), bracket_policy.state_dict()
    assert list(a) == list(b) and all(torch.equal(a[n], b[n]) for n in a)
    assert runner.history["1"] == bracket.history["1"]


# ---- the pipeline on a fake world -----------------------------------------------------------------


def world(tmp_path, spec):
    luc, build = fake_world(tmp_path, spec, seeds=(0,))
    cfg = copy.deepcopy(ls.load_config())
    cfg["lora"]["recipe"] = {"steps": 2}
    kw = dict(results_root=tmp_path, device="cpu", build_datasets=build, low_update_cfg=luc)
    train = dict(kw, evaluator=stub_evaluator, bootstrap=BOOT, allow_dirty=True)
    return cfg, kw, train


def test_pipeline_arms_extensions_select_verify_and_refusals(spec, tmp_path, capsys, monkeypatch):
    cfg, kw, train = world(tmp_path, spec)
    rows = [ls.run_arm(cfg, r, lr, **train) for r, lr in ls.grid(cfg)]
    assert [r["run_id"] for r in rows] == [ls.arm_run_id(r, lr, 0) for r, lr in ls.grid(cfg)]
    first = rows[0]["lora"]
    assert first["history"]["trainable_params"] > 0 and first["train_config"]["lr"] == 1e-4
    assert first["inputs"]["start_checkpoint"]["path"].endswith("seq_hetero__seq_ft__seed0/checkpoints/stage0.pt")
    assert all(r["lora"]["inputs"] == first["inputs"] for r in rows)
    path = ls.arm_dir(cfg, tmp_path, 4, 1e-4, 0) / "row.json"
    mtime = path.stat().st_mtime
    ls.run_arm(cfg, 4, 1e-4, **train)  # published: reused, not retrained
    assert path.stat().st_mtime == mtime and "reused" in capsys.readouterr().out
    with pytest.raises(ValueError, match="not in the rule's grid"):
        ls.run_arm(cfg, 8, 1e-4, **train)
    sel_kw = dict(results_root=tmp_path, low_update_cfg=kw["low_update_cfg"], allow_dirty=True)
    out = ls.run_select(cfg, **sel_kw)
    n_ext = 0
    while out["status"] == "needs_extension":
        ls.run_extension(cfg, **train)
        n_ext += 1
        out = ls.run_select(cfg, **sel_kw)
    assert n_ext == len(out["owed_extensions"]) <= 2
    if n_ext == 0:
        with pytest.raises(RuntimeError, match="no edge extension is owed"):
            ls.run_extension(cfg, **train)
    assert out["status"] == "selected" and out["rule_sha256"] == ls.file_sha256(ls.rule_path())
    assert out["selected"]["checkpoint_sha256"] == out["arms"][ls.arm_key(out["selected"]["rank"],
                                                                         out["selected"]["lr"])]["checkpoint_sha256"]
    assert ls.verify(cfg, **sel_kw)["passed"]
    # provenance, not HEAD: a later commit does not break verification
    monkeypatch.setattr(ls, "git_sha", lambda: "f" * 40)
    assert ls.verify(cfg, **sel_kw)["passed"]
    # dirty provenance is refused unless allowed (the arms here were trained on a test tree)
    if rows[0]["git_sha"].endswith("-dirty"):
        assert not ls.verify(cfg, results_root=tmp_path, low_update_cfg=kw["low_update_cfg"])["passed"]
    with pytest.raises(ValueError, match="another rule"):
        ls.check_bundle(out, "0" * 64)
    # tampering, one at a time
    row = json.loads(path.read_text())
    tampered = copy.deepcopy(row)
    tampered["lora"]["inputs"]["namespace"] = "other"
    path.write_text(json.dumps(tampered))
    v = ls.verify(cfg, **sel_kw)
    assert not v["passed"] and any("shared inputs" in p for p in v["problems"])
    tampered = copy.deepcopy(row)
    tampered["lora"]["train_config"]["warmup_steps"] = 7
    path.write_text(json.dumps(tampered))
    v = ls.verify(cfg, **sel_kw)
    assert not v["passed"] and any("TrainConfig" in p for p in v["problems"])
    tampered = copy.deepcopy(row)
    tampered["lora"]["state_fingerprint"] = "0" * 64
    path.write_text(json.dumps(tampered))
    v = ls.verify(cfg, **sel_kw)
    assert not v["passed"] and any("fingerprint" in p for p in v["problems"])
    path.write_text(json.dumps(row))
    assert ls.verify(cfg, **sel_kw)["passed"]
    ref_result = tmp_path / "seq_hetero__seq_ft__seed0" / "result.json"
    saved = ref_result.read_text()
    ref_result.write_text(saved + " ")
    v = ls.verify(cfg, **sel_kw)
    assert not v["passed"] and any("reference_result" in p for p in v["problems"])
    ref_result.write_text(saved)
    bundle = ls._out(cfg, tmp_path) / "lora" / "selection" / "selection.json"
    payload = json.loads(bundle.read_text())
    payload["selected"]["rank"] = 999
    bundle.write_text(json.dumps(payload, indent=2) + "\n")
    assert not ls.verify(cfg, **sel_kw)["passed"]


def test_recomputed_selection_must_match_the_bundle(spec, tmp_path):
    """A bundle whose selection the validated rows do not reproduce fails verify (here: a bundle
    rewritten consistently with its done-file, so only the recomputation can catch it)."""
    from flowcl.utils.run import file_sha256

    cfg, kw, train = world(tmp_path, spec)
    for r, lr in ls.grid(cfg):
        ls.run_arm(cfg, r, lr, **train)
    sel_kw = dict(results_root=tmp_path, low_update_cfg=kw["low_update_cfg"], allow_dirty=True)
    out = ls.run_select(cfg, **sel_kw)
    while out["status"] == "needs_extension":
        ls.run_extension(cfg, **train)
        out = ls.run_select(cfg, **sel_kw)
    directory = ls._out(cfg, tmp_path) / "lora" / "selection"
    payload = json.loads((directory / "selection.json").read_text())
    payload["flags"] = payload["flags"] + ["made_up"]
    (directory / "selection.json").write_text(json.dumps(payload, indent=2) + "\n")
    (directory / "selection.done.json").write_text(json.dumps(
        {"selection_sha256": file_sha256(directory / "selection.json")}, indent=2) + "\n")
    v = ls.verify(cfg, **sel_kw)
    assert not v["passed"] and any("recomputed flags" in p for p in v["problems"])


def test_smoke_arm_checks_the_trainconfig_and_the_tensors(spec, tmp_path):
    cfg, kw, train = world(tmp_path, spec)
    ref = tmp_path / "seq_hetero__seq_ft__seed0"
    # a stand-in for the runner's LoRA run: a bracket arm's outputs in the runner's layout
    row = ls.run_arm(cfg, 4, 1e-3, out_root=tmp_path / "src", **train)
    arm = ls.arm_dir(cfg, tmp_path / "src", 4, 1e-3, 0)
    runner = tmp_path / "runner"
    (runner / "checkpoints").mkdir(parents=True)
    payload = torch.load(arm / "checkpoints" / "final.pt", weights_only=False)
    payload["extra"] = {"method_state": {"history": {"1": row["lora"]["history"]}}}
    torch.save(payload, runner / "checkpoints" / "stage1.pt")
    stages = [{}, {"train_config": row["lora"]["train_config"]}]
    (runner / "result.json").write_text(json.dumps({"stages": stages}))
    smoke_kw = {k: v for k, v in train.items() if k != "allow_dirty"}
    out = ls.smoke_arm(cfg, 4, 1e-3, reference_run=ref, identity_run=runner, steps=2, episodes=1,
                       out_root=tmp_path / "scratch", **smoke_kw)
    assert out["passed"] and out["identity"]["n_different"] == 0 and out["history_equal"], out
    stages[1]["train_config"] = {**row["lora"]["train_config"], "warmup_steps": 5}
    (runner / "result.json").write_text(json.dumps({"stages": stages}))
    shutil.rmtree(tmp_path / "scratch")
    out = ls.smoke_arm(cfg, 4, 1e-3, reference_run=ref, identity_run=runner, steps=2, episodes=1,
                       out_root=tmp_path / "scratch", **smoke_kw)
    assert not out["passed"] and "TrainConfig" in out["reason"]


# ---- the queue ------------------------------------------------------------------------------------

QUEUE = repo_root() / "scripts" / "queue_lora_bracket.sh"
ARMS = [f"{k}_arm_r{r}_lr{ls.lr_tag(lr)}" for k, (r, lr) in enumerate((r, lr) for r in RANKS for lr in LRS)]


def run_queue(tmp_path, args=(), fail="", step_rc="", present=(), fake_now="", step_s=""):
    results = tmp_path / "results"
    for rel in present:
        p = results / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}")
    env = {**os.environ, "QUEUE_DRY_RUN": "1", "QUEUE_FAIL_STEP": fail, "QUEUE_STEP_RC": step_rc,
           "QUEUE_LOG_ROOT": str(tmp_path / "logs"), "QUEUE_RESULTS_ROOT": str(results),
           "QUEUE_FAKE_NOW": str(fake_now), "QUEUE_FAKE_STEP_S": str(step_s)}
    done = subprocess.run(["bash", str(QUEUE), *args], env=env, capture_output=True, text=True, timeout=120)
    (logdir,) = list((tmp_path / "logs").glob("queue_*"))
    return done.returncode, (logdir / "queue.log").read_text(), logdir


def order(log, steps):
    return [log.index(f"START {s}:") for s in steps] == sorted(log.index(f"START {s}:") for s in steps)


def test_queue_order_extensions_and_failures(tmp_path):
    assert subprocess.run(["bash", "-n", str(QUEUE)]).returncode == 0
    rc, log, logdir = run_queue(tmp_path / "a")
    assert rc == 0 and log.rstrip().endswith("QUEUE DONE ok") and order(log, ARMS + ["9_select", "12_verify"])
    assert "START 10_extension" not in log
    assert (logdir / "4_arm_r16_lr0.0003.log").read_text().strip().endswith("--arm 16 0.0003")
    rc, log, _ = run_queue(tmp_path / "b", step_rc="9_select=10")
    assert rc == 0 and order(log, ARMS + ["9_select", "10_extension_1", "11_select_1", "12_verify"])
    rc, log, _ = run_queue(tmp_path / "c", step_rc="9_select=10,11_select_1=10")
    assert rc == 0 and order(log, ["9_select", "10_extension_1", "11_select_1", "10_extension_2", "11_select_2",
                                   "12_verify"])
    rc, log, _ = run_queue(tmp_path / "d", step_rc="9_select=10,11_select_1=10,11_select_2=10")
    assert rc == 1 and "extensions still owed" in log and "START 12_verify" not in log
    rc, log, _ = run_queue(tmp_path / "e", step_rc="9_select=1")
    assert rc == 1 and "FAILED: select(rc 1)" in log
    rc, log, _ = run_queue(tmp_path / "f", step_rc="9_select=10", fail="10_extension_1")
    assert rc == 1 and "FAILED: 10_extension_1" in log and "START 11_select_1" not in log
    rc, log, _ = run_queue(tmp_path / "g", fail="12_verify")
    assert rc == 1 and log.rstrip().endswith("QUEUE DONE FAILED: 12_verify")
    rc, log, _ = run_queue(tmp_path / "h", fail=ARMS[2])
    assert rc == 1 and f"START {ARMS[3]}" in log and "SKIP 9-12: an arm failed" in log


def test_queue_resume_and_arguments(tmp_path):
    rows = [f"baseline_selection_lora/lora/seed0/lorab__r4__lr{t}__seed0/row.json" for t in ("0.0001", "0.0003")]
    rc, log, _ = run_queue(tmp_path / "a", ["--from-step", "2"], present=rows)
    assert rc == 0 and f"PRIOR {ARMS[0]} ok" in log and f"START {ARMS[2]}" in log and f"START {ARMS[0]}" not in log
    rc, log, _ = run_queue(tmp_path / "b", ["--from-step", "2"], present=rows[:1])
    assert rc == 1 and f"PRIOR {ARMS[1]} missing" in log and "START 9_select" not in log
    for bad in (["--from-step", "10"], ["--bogus"], ["--smoke", "rel/dir"], ["--deadline", "25:99"],
                ["--deadline", "noon"], ["--arm-estimate-min", "1.5"]):
        assert subprocess.run(["bash", str(QUEUE), *bad], capture_output=True).returncode == 2, bad


def test_queue_deadline_guard(tmp_path):
    # arms take an hour each on the fake clock; the deadline admits three
    now = 1_000_000
    rc, log, _ = run_queue(tmp_path / "a", ["--deadline", f"@{now + 3 * 3600 + 10}", "--arm-estimate-min", "60"],
                           fake_now=now, step_s=3600)
    assert rc == 3 and all(f"START {a}" in log for a in ARMS[:3]) and f"START {ARMS[3]}" not in log
    assert f"DEFERRED {ARMS[3]}" in log and "START 9_select" not in log and "SKIP 9-12: arms deferred" in log
    assert log.rstrip().endswith("QUEUE DONE DEFERRED: " + " ".join(ARMS[3:]))
    # the longest arm so far counts even when the estimate is shorter
    rc, log, _ = run_queue(tmp_path / "b", ["--deadline", f"@{now + 2 * 3600 + 10}", "--arm-estimate-min", "1"],
                           fake_now=now, step_s=3600)
    assert rc == 3 and f"START {ARMS[1]}" in log and f"START {ARMS[2]}" not in log
    # an owed extension that cannot finish is deferred; nothing is verified
    rc, log, _ = run_queue(tmp_path / "c", ["--deadline", f"@{now + 9 * 3600 + 10}", "--arm-estimate-min", "60",
                                            "--extension-estimate-min", "60"],
                           fake_now=now, step_s=3600, step_rc="9_select=10")
    assert rc == 3 and "DEFERRED 10_extension_1" in log and "START 12_verify" not in log
    # no deadline: everything runs
    rc, log, _ = run_queue(tmp_path / "d", fake_now=now, step_s=3600)
    assert rc == 0 and "QUEUE DONE ok" in log


def test_queue_smoke_mode(tmp_path):
    smoke = tmp_path / "scratch"
    rc, log, logdir = run_queue(tmp_path, ["--smoke", str(smoke)])
    steps = ["s0_smoke_method", "s1_seqft", "s2_lora", "s3_lora_resume", "s4_lora_compare", "s5_bracket_identity",
             "s6_benchmark"]
    assert rc == 0 and logdir.name.endswith("_smoke") and order(log, steps)
    cmd = {s: (logdir / f"{s}.log").read_text() for s in steps}
    ref, run = f"{smoke}/seq_hetero__seq_ft__seed0", f"{smoke}/seq_hetero__lora__seed0"
    assert "--write-smoke-method" in cmd["s0_smoke_method"] and "--rank 4 --lr 0.001" in cmd["s0_smoke_method"]
    assert f"--t1-reference-run {ref} --identity-reference-run {ref} --identity-stages 0" in cmd["s2_lora"]
    assert "--steps-per-task 200 --n-episodes 1 --allow-dirty" in cmd["s2_lora"]
    assert f"--resume-run {run} --start-stage 3" in cmd["s3_lora_resume"]
    assert "--stage 3 --all-artifacts --method-state" in cmd["s4_lora_compare"]
    assert f"--smoke-arm 4 0.001 --reference-run {ref} --identity-run {run} --smoke-steps 200" in cmd["s5_bracket_identity"]
    assert "--benchmark --bench-ranks 0 4 64 256" in cmd["s6_benchmark"]


@pytest.mark.parametrize("script, flags", [
    ("lora_selection.py", ["--arm", "--select", "--extension", "--verify", "--smoke-arm", "--benchmark",
                           "--write-smoke-method", "--reference-run", "--identity-run", "--smoke-steps",
                           "--bench-ranks", "--results-root", "--out-root", "--allow-dirty"]),
    ("compare_resume.py", ["--method-state", "--all-artifacts"]),
])
def test_every_queue_step_accepts_its_flags(script, flags):
    done = subprocess.run([sys.executable, str(repo_root() / "scripts" / script), "--help"],
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    for flag in flags:
        assert flag in done.stdout, flag


def test_write_smoke_method(tmp_path):
    from flowcl.data.config import load_method_config

    path = ls.write_smoke_method(tmp_path / "m.yaml", 4, 1e-3)
    assert load_method_config(path) == ("lora", {"rank": 4, "lr": 1e-3, "alpha": 16.0})


def test_pipeline_through_two_extensions_at_a_corner(spec, tmp_path, monkeypatch):
    """The grid's selection forced to the (64, 1e-3) corner: both extensions train through run_arm,
    nothing is published before them, and verify recomputes the same result."""
    original = ls._summary

    def summary(row):
        s = original(row)
        s["spatial_probe"] = 0.0 if (s["rank"], s["lr"]) == (64, 1e-3) else 1.0
        return s

    monkeypatch.setattr(ls, "_summary", summary)
    cfg, kw, train = world(tmp_path, spec)
    for r, lr in ls.grid(cfg):
        ls.run_arm(cfg, r, lr, **train)
    sel_kw = dict(results_root=tmp_path, low_update_cfg=kw["low_update_cfg"], allow_dirty=True)
    out = ls.run_select(cfg, **sel_kw)
    assert out["status"] == "needs_extension" and out["missing"] == [(256, 1e-3), (64, 3e-3)]
    assert not (ls._out(cfg, tmp_path) / "lora" / "selection").exists()
    with pytest.raises(ValueError, match="not an extension cell"):
        ls.run_arm(cfg, 256, 3e-3, extension=True, **train)   # the combined outer corner is never run
    first = ls.run_extension(cfg, **train)
    assert (first["lora"]["rank"], first["lora"]["lr"], first["lora"]["extension"]) == (256, 1e-3, True)
    assert ls.run_select(cfg, **sel_kw)["missing"] == [(64, 3e-3)]
    second = ls.run_extension(cfg, **train)
    assert (second["lora"]["rank"], second["lora"]["lr"]) == (64, 3e-3)
    out = ls.run_select(cfg, **sel_kw)
    assert out["status"] == "selected" and (out["selected"]["rank"], out["selected"]["lr"]) == (64, 1e-3)
    assert out["flags"][-1] == "edge_extension" and len(out["arms"]) == 11
    assert out["owed_extensions"] == [[256, 1e-3], [64, 3e-3]]
    with pytest.raises(RuntimeError, match="no edge extension is owed"):
        ls.run_extension(cfg, **train)
    assert ls.verify(cfg, **sel_kw)["passed"]
