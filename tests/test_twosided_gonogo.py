"""E1 (two-sided go/no-go): the Hutchinson output-Gram estimator on a linear net, the capacity
accounting, the forecast and overlap, the verdict order, the whole measurement end to end on the
tiny policy (memory replay and forward identity bitwise, incomplete -> complete, smoke, invalid),
the config, and both queues (E1 and Gate 0 T6-T8, with the backup rule)."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import time
from pathlib import Path

import pytest
import torch
from torch import nn

from flowcl.analysis.hooks import KIND_STATE, VIEW_ALL, CaptureLayer
from flowcl.analysis.sensitivity import (
    OutputGradientCapture,
    forecast_k,
    fraction_table,
    protected_fraction,
    subspace_overlap,
)
from flowcl.experiments import gate0_t8 as g8
from flowcl.experiments import t5_sweep as ts
from flowcl.experiments import twosided_gonogo as tg
from flowcl.utils.libero_paths import repo_root
from flowcl.utils.run import file_sha256
from test_low_update import spec  # noqa: F401
from test_t5_sweep import world

# ---- the estimator -----------------------------------------------------------------------------------


def two_layer(d0=5, d1=6, d2=4, seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(d0, d1, bias=False), nn.Linear(d1, d2, bias=False))


def test_hutchinson_output_gram_estimates_w2t_w2():
    net = two_layer()
    cap = OutputGradientCapture(net, [CaptureLayer("first", net[0], KIND_STATE)], views={"first": VIEW_ALL},
                                n_probes=2, tokens_per_sample=1, subsample_seed=0)
    generator = torch.Generator().manual_seed(1)
    with cap:
        for _ in range(10):
            x = torch.randn(4000, 5)
            with cap.recording():
                v = net(x)
            cap.backward(v, None, generator)
    acc = cap.accumulators["first"]
    W2 = net[1].weight.detach().to(torch.float64)
    expected = W2.T @ W2                         # E[J^T J] for y1 -> v = W2 y1
    for r in range(2):
        est = acc.gram[r] / acc.n[r]
        assert est.shape == (6, 6) and acc.n[r] == 40000
        assert float((est - expected).norm() / expected.norm()) < 0.05
    assert not torch.equal(acc.gram[0], acc.gram[1])          # independent probes
    assert net[0].weight.grad is None and net[1].weight.grad is None   # outputs only


def test_the_probe_mask_and_the_orientation():
    net = two_layer()
    cap = OutputGradientCapture(net, [CaptureLayer("first", net[0], KIND_STATE)], views={"first": VIEW_ALL},
                                n_probes=1, tokens_per_sample=1, subsample_seed=0)
    with cap:
        with cap.recording():
            v = net(torch.randn(8, 5))
        cap.backward(v, torch.zeros_like(v), torch.Generator().manual_seed(0))
    assert cap.accumulators["first"].n[0] == 0              # a fully masked probe: zero rows skipped

    class Transposed(nn.Linear):
        def forward(self, x):
            return super().forward(x).T

    bad = Transposed(5, 6, bias=False)
    cap = OutputGradientCapture(bad, [CaptureLayer("bad", bad, KIND_STATE)], views={"bad": VIEW_ALL},
                                n_probes=1, tokens_per_sample=1, subsample_seed=0)
    with cap, pytest.raises(RuntimeError, match="output space"):
        with cap.recording():
            bad(torch.randn(3, 5))
    with pytest.raises(RuntimeError, match="no_grad"):
        cap = OutputGradientCapture(net, [CaptureLayer("first", net[0], KIND_STATE)],
                                    views={"first": VIEW_ALL}, n_probes=1, tokens_per_sample=1,
                                    subsample_seed=0)
        with cap, torch.no_grad():
            with cap.recording():
                v = net(torch.randn(3, 5))
            cap.backward(v, None, torch.Generator().manual_seed(0))


# ---- capacity accounting -----------------------------------------------------------------------------


def test_protected_fraction_by_hand_and_the_gpm_limit():
    f = protected_fraction([2, 1], [3, 5], [4, 2], [3, 5])
    assert f["two_sided"] == pytest.approx((2 * 3 + 1 * 5) / 22)
    assert f["gpm"] == pytest.approx((4 * 3 + 2 * 5) / 22) and f["n_params"] == 22
    assert f["rho_out"] == pytest.approx((2 * 3 + 1 * 5) / 22)
    full = protected_fraction([4, 2], [1, 2], [4, 2], [3, 5])
    assert full["two_sided"] == pytest.approx(full["gpm"])                  # U = I is GPM
    with pytest.raises(ValueError, match="out of range"):
        protected_fraction([5], [1], [4], [3])
    dims = {"trunk.a": (4, 3), "trunk.b": (2, 5), "flow_head.c": (3, 3)}
    groups = {"trunk.a": "trunk_attn", "trunk.b": "trunk_mlp", "flow_head.c": "decoder_mlp"}
    t = fraction_table({"trunk.a": 2, "trunk.b": 1, "flow_head.c": 0},
                       {"trunk.a": 3, "trunk.b": 5, "flow_head.c": 3}, dims, groups)
    assert set(t) == {"all", "trunk", "decoder", "trunk_attn", "trunk_mlp", "decoder_mlp"}
    assert t["trunk"]["two_sided"] == pytest.approx(f["two_sided"]) and t["decoder"]["two_sided"] == 0


def test_forecast_and_overlap():
    assert forecast_k(10, 20, 100, 4, 2, [5, 6, 7, 8]) == {5: 25.0, 6: 30.0, 7: 35.0, 8: 40.0}
    assert forecast_k(10, 20, 27, 4, 2, [5, 6, 8]) == {5: 25.0, 6: 27.0, 8: 27.0}
    with pytest.raises(ValueError, match="shrink"):
        forecast_k(20, 10, 100, 4, 2, [5])
    E = torch.eye(4, dtype=torch.float64)
    assert subspace_overlap(E[:, :2], E[:, :2]) == 1.0 and subspace_overlap(E[:, :2], E[:, 2:]) == 0.0
    assert subspace_overlap(E[:, :2], E[:, :1]) == 0.5


def test_classify_order_and_exit_codes():
    assert tg.classify(False, True, 0.1, 0.5)["verdict"] == tg.INVALID
    assert tg.classify(True, False, 0.1, 0.5)["verdict"] == tg.INCOMPLETE
    assert tg.classify(True, True, None, 0.5)["verdict"] == tg.INCOMPLETE
    assert tg.classify(True, True, 0.5, 0.5)["verdict"] == tg.PASS               # inclusive
    assert tg.classify(True, True, 0.51, 0.5)["verdict"] == tg.FAIL
    assert {v: tg.EXIT_CODES[v] for v in tg.EXIT_CODES} == {"pass": 0, "fail": 0, "invalid": 3, "incomplete": 4}


def test_compare_gram():
    K = torch.eye(3, dtype=torch.float64)
    assert tg.compare_gram(K, K.clone()) == {"bitwise": True, "finite": True, "rel": 0.0}
    c = tg.compare_gram(K, K * (1 + 1e-4))
    assert not c["bitwise"] and c["rel"] == pytest.approx(1e-4)
    bad = K.clone()
    bad[0, 0] = float("nan")
    assert not tg.compare_gram(K, bad)["finite"] and tg.compare_gram(K, bad)["rel"] == float("inf")


# ---- end to end on the tiny policy ---------------------------------------------------------------------


def e1_world(tmp_path, spec):
    t5cfg, luc, build = world(tmp_path, spec)
    cfg = copy.deepcopy(tg.load_config())
    cfg["source"]["capture_config"] = t5cfg["source"]["capture_config"]
    paths = tg.source_paths(cfg, tmp_path)
    cfg["source"]["sha256"] = {k: file_sha256(paths[k]) for k in cfg["source"]["sha256"]}
    kw = dict(source_root=tmp_path, device="cpu", allow_dirty=True, build_datasets=build, low_update_cfg=luc)
    return cfg, kw


def test_capture_and_summary_end_to_end(spec, tmp_path):
    cfg, kw = e1_world(tmp_path, spec)
    res = tmp_path / "res"
    recs = []
    for i in range(3):
        rec = tg.run_capture(cfg, i, results_root=res, **kw)
        recs.append(rec)
        c = rec["checks"]
        assert rec["valid"] and not rec["smoke"]
        assert c["replay"]["passed"] and c["replay"]["bitwise"] and c["replay"]["status"] == "verified"
        assert c["identity"]["passed"] and c["identity"]["bitwise"]   # grad-enabled == forward-only
        assert c["weights_unchanged"] and c["grads_unchanged"] and c["outputs_finite"]
        assert rec["cache_state"]["precision"] == "fp32"               # CPU: no AMP
        assert all(r["n_out"][0] > 0 and r["n_out"][1] > 0 for r in rec["layers"])
    assert tg.run_capture(cfg, 0, results_root=res, **kw)["files"] == recs[0]["files"]   # reused
    s = tg.run_summary(cfg, results_root=res, source_root=tmp_path)
    assert s["verdict"]["verdict"] == tg.INCOMPLETE and s["available_tasks"] == [0, 1, 2]
    assert s["verdict"]["decisive"] is not None                       # computed, but not a finding

    tg.run_capture(cfg, 3, results_root=res, **kw)
    s = tg.run_summary(cfg, results_root=res, source_root=tmp_path)
    v = s["verdict"]
    assert v["verdict"] in (tg.PASS, tg.FAIL) and v["exit_code"] == 0
    dec = s["tables"]["adaptive"]["0.95"]["2"]["fractions"]["trunk"]
    assert v["decisive"] == dec["two_sided"] and v["gpm_at_decisive"] == dec["gpm"]
    assert set(s["tables"]) == {"adaptive", "fixed"} and set(s["tables"]["fixed"]) == {"0.9", "0.95", "0.99"}
    # first task: the two rules coincide
    assert s["tables"]["adaptive"]["0.95"]["0"]["k_out"] == s["tables"]["fixed"]["0.95"]["0"]["k_out"]
    # the adaptive union never shrinks, and is never smaller than the fixed one at T2
    k = s["tables"]["adaptive"]["0.95"]
    assert all(k[str(i + 1)]["k_out"][n] >= k[str(i)]["k_out"][n] for i in range(3) for n in k["0"]["k_out"])
    assert all(k["1"]["k_out"][n] >= s["tables"]["fixed"]["0.95"]["1"]["k_out"][n] for n in k["0"]["k_out"])
    # GPM's fraction from the memory files themselves
    from flowcl.analysis.subspace import load_bases
    mem, _ = load_bases(tmp_path / cfg["source"]["run"] / "method" / "memory_task2.pt")
    rows = {r["name"]: r for r in tg.load_capture(cfg, res, 2)["layers"] if r["name"].startswith("trunk.")}
    gpm = sum(r["d_out"] * mem[n].vectors.shape[1] for n, r in rows.items()) / sum(
        r["d_out"] * r["d_in"] for r in rows.values())
    assert v["gpm_at_decisive"] == pytest.approx(gpm)
    assert s["forecast"]["label"].startswith("FORECAST") and set(s["forecast"]["tasks"]) == {"5", "6", "7", "8"}
    assert set(s["stability"]["fractions"]) == {"A", "B"} and "estimator_sensitive" in s["stability"]
    assert all(0.0 <= x <= 1.0 + 1e-12 for x in s["stability"]["overlap_per_layer"].values())
    for i in range(4):
        name = tg.decisive_bases_name(cfg, i)
        assert s["bases"][name] == file_sha256(res / "twosided_gonogo" / "bases" / name)
        payload = torch.load(res / "twosided_gonogo" / "bases" / name, weights_only=False)
        U = payload["bases"]["trunk.state_projection"]
        assert U.shape[1] == k[str(i)]["k_out"]["trunk.state_projection"]
        assert torch.allclose(U.T @ U, torch.eye(U.shape[1], dtype=torch.float64), atol=1e-10)
    assert tg.run_summary(cfg, results_root=res, source_root=tmp_path)["bases"] == s["bases"]   # rerun


def test_smoke_is_incomplete_not_invalid(spec, tmp_path):
    cfg, kw = e1_world(tmp_path, spec)
    rec = tg.run_capture(cfg, 0, results_root=tmp_path / "smoke", smoke_batches=2, **kw)
    assert rec["smoke"] and rec["valid"] and rec["n_captured_samples"] == 8 < rec["n_dataset_samples"]
    assert rec["checks"]["replay"]["status"] == "unverified" and rec["checks"]["samples"]["status"] == "unverified"
    assert rec["checks"]["identity"]["passed"]                      # enforced even in a smoke run
    s = tg.run_summary(cfg, results_root=tmp_path / "smoke", source_root=tmp_path)
    assert s["verdict"]["verdict"] == tg.INCOMPLETE and s["verdict"]["exit_code"] == 4 and s["smoke_tasks"] == [0]


def test_failed_checks_make_the_verdict_invalid(spec, tmp_path, monkeypatch):
    cfg, kw = e1_world(tmp_path, spec)
    strict = copy.deepcopy(cfg)
    strict["identity"]["gram_rtol"] = -1.0          # no Gram can pass: the identity gate fires
    rec = tg.run_capture(strict, 0, results_root=tmp_path / "a", **kw)
    assert not rec["valid"] and not rec["checks"]["identity"]["passed"]
    for i in (1, 2, 3):
        tg.run_capture(cfg, i, results_root=tmp_path / "a", **kw)
    s = tg.run_summary(cfg, results_root=tmp_path / "a", source_root=tmp_path)
    assert s["verdict"]["verdict"] == tg.INVALID and s["verdict"]["exit_code"] == 3

    real = tg.replay_memory
    monkeypatch.setattr(tg, "replay_memory", lambda *a, **k: {**real(*a, **k), "passed": False})
    assert not tg.run_capture(cfg, 0, results_root=tmp_path / "b", **kw)["valid"]
    # a wrong reference really fails the replay
    from flowcl.analysis.subspace import load_bases
    ref1, meta1 = load_bases(tmp_path / cfg["source"]["run"] / "method" / "memory_task1.pt")
    ref0, meta0 = load_bases(tmp_path / cfg["source"]["run"] / "method" / "memory_task0.pt")
    K = {n: torch.eye(b.vectors.shape[0], dtype=torch.float64) for n, b in ref0.items()}
    out = real(K, None, ref1, meta1["memory_history"]["1"], cfg["source"], tg._capture_config(cfg), cfg["replay"])
    assert not out["passed"]


def test_a_tampered_capture_is_refused(spec, tmp_path):
    cfg, kw = e1_world(tmp_path, spec)
    tg.run_capture(cfg, 0, results_root=tmp_path / "r", **kw)
    path = tmp_path / "r" / "twosided_gonogo" / "capture" / "task0" / "grams_A.pt"
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(RuntimeError, match="changed after publication"):
        tg.run_summary(cfg, results_root=tmp_path / "r", source_root=tmp_path)


def test_source_mismatch_is_refused(spec, tmp_path):
    cfg, kw = e1_world(tmp_path, spec)
    bad = copy.deepcopy(cfg)
    bad["source"]["sha256"]["memory1"] = "0" * 64
    with pytest.raises(ValueError, match="source does not match"):
        tg.run_capture(bad, 0, results_root=tmp_path / "x", **kw)


def test_config_is_consistent():
    cfg = tg.load_config()
    t5 = ts.load_sweep_config()
    assert cfg["source"] == t5["source"]                    # the same pinned files
    dec = cfg["u_rule"]["decisive"]
    assert dec["rule"] in cfg["u_rule"]["rules"] and dec["eps_out"] in cfg["u_rule"]["eps_out"]
    assert cfg["u_rule"]["rules"]["adaptive"] == cfg["source"]["new_energy_fraction"]   # M's own rule
    assert cfg["u_rule"]["rules"]["fixed"] is None and cfg["capture"]["n_probes"] == 2
    assert cfg["rule"] == {"decisive_task": 2, "half": "trunk", "threshold": 0.5, "near_full_rank": 0.9}
    assert cfg["replay"] == {k: v for k, v in t5["replay"].items() if k != "alpha"}
    assert cfg["identity"]["gram_rtol"] == 1e-5


# ---- Gate 0 T6-T8: the backup rule ---------------------------------------------------------------------


def report(path: Path, task: str, passed) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"passed": passed, "evidence": {"per_task": {task: {"success_rate": 0.5}}}}))
    return path


def test_backup_decision(tmp_path):
    key = "libero_10/X"
    since = time.time() - 5
    assert g8.backup_decision(tmp_path / "none.json", key, since) == g8.NO_REPORT
    assert g8.backup_decision(report(tmp_path / "f.json", key, False), key, since) == g8.RUN_BACKUP
    assert g8.backup_decision(report(tmp_path / "p.json", key, True), key, since) == g8.PRIMARY_PASSED
    assert g8.backup_decision(report(tmp_path / "w.json", "libero_10/Y", False), key, since) == g8.WRONG_TASK
    assert g8.backup_decision(report(tmp_path / "n.json", key, None), key, since) == g8.MALFORMED
    (tmp_path / "m.json").write_text("{")
    assert g8.backup_decision(tmp_path / "m.json", key, since) == g8.MALFORMED
    old = report(tmp_path / "o.json", key, False)
    os.utime(old, (since - 100, since - 100))
    assert g8.backup_decision(old, key, since) == g8.STALE


def test_slots_and_stale_paths(tmp_path):
    s = g8.slots()
    t5 = ts.load_sweep_config()
    assert list(s) == ["T6", "T7", "T8"]
    assert all(s[k] == {"primary": t5["follow_up"][k]["primary"], "backup": t5["follow_up"][k]["backup"]} for k in s)
    key = s["T6"]["primary"]
    assert g8.stale_paths(tmp_path, tmp_path / "gate0_t8" / "T6", key) == []
    (tmp_path / g8.single_task_run_id(key, 0)).mkdir()
    assert len(g8.stale_paths(tmp_path, tmp_path / "gate0_t8" / "T6", key)) == 1


def test_gate0_cli_refuses_smoke_into_results(tmp_path):
    script = repo_root() / "scripts" / "gate0.py"
    for extra in ([], ["--results-root", str(tmp_path)],
                  ["--results-root", str(tmp_path), "--out-dir", str(repo_root() / "results" / "x")]):
        done = subprocess.run(["uv", "run", "python", str(script), "--tasks", "libero_10/X", "--n-episodes", "1",
                               *extra], capture_output=True, text=True, timeout=300, cwd=repo_root())
        assert done.returncode == 2 and "smoke run needs" in done.stderr, done.stderr[-500:]


# ---- the queues ------------------------------------------------------------------------------------------

E1_QUEUE = repo_root() / "scripts" / "queue_twosided_gonogo.sh"
G8_QUEUE = repo_root() / "scripts" / "queue_gate0_t8.sh"


def run_queue(script, tmp_path, args=(), env_extra=None, present=()):
    results = tmp_path / "results"
    for rel in present:
        p = results / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}")
    env = {**os.environ, "QUEUE_DRY_RUN": "1", "QUEUE_LOG_ROOT": str(tmp_path / "logs"),
           "QUEUE_RESULTS_ROOT": str(results), **(env_extra or {})}
    done = subprocess.run(["bash", str(script), *args], env=env, capture_output=True, text=True, timeout=300)
    (logdir,) = list((tmp_path / "logs").glob("queue_*"))
    return done.returncode, (logdir / "queue.log").read_text(), logdir


def test_e1_queue_order_status_and_resume(tmp_path):
    assert subprocess.run(["bash", "-n", str(E1_QUEUE)]).returncode == 0
    rc, log, logdir = run_queue(E1_QUEUE, tmp_path / "a")
    steps = ["0_capture_t0", "1_capture_t1", "2_capture_t2", "3_capture_t3", "4_summary"]
    assert rc == 0 and log.rstrip().endswith("QUEUE DONE ok")
    assert [log.index(f"START {s}:") for s in steps] == sorted(log.index(f"START {s}:") for s in steps)
    assert (logdir / "2_capture_t2.log").read_text().strip().endswith("--part capture --task 2")
    rc, log, _ = run_queue(E1_QUEUE, tmp_path / "b", env_extra={"QUEUE_FAIL_STEP": "1_capture_t1"})
    assert rc == 1 and "QUEUE DONE FAILED: 1_capture_t1" in log and "START 4_summary:" in log
    rc, log, _ = run_queue(E1_QUEUE, tmp_path / "c", env_extra={"QUEUE_FAIL_STEP": "4_summary"})
    assert rc == 1 and "QUEUE DONE FAILED: 4_summary" in log
    rc, log, _ = run_queue(E1_QUEUE, tmp_path / "d", args=("--from-step", "2"),
                           present=["twosided_gonogo/capture/task0/capture.json"])
    assert rc == 1 and "PRIOR 0_capture_t0 ok" in log and "1_capture_t1(missing)" in log
    assert "START 2_capture_t2:" in log
    for bad in (("--from-step", "5"), ("--bogus",)):
        assert subprocess.run(["bash", str(E1_QUEUE), *bad], capture_output=True).returncode == 2


def test_gate0_queue_backups_staleness_and_status(tmp_path):
    assert subprocess.run(["bash", "-n", str(G8_QUEUE)]).returncode == 0
    s = g8.slots()
    rc, log, logdir = run_queue(G8_QUEUE, tmp_path / "a")
    assert rc == 0 and log.rstrip().endswith("QUEUE DONE ok")
    assert all(f"START {k}_{slot}:" in log for k, slot in enumerate(s)) and "_backup:" not in log
    cmd = (logdir / "1_T7.log").read_text()
    assert f"--tasks {s['T7']['primary']}" in cmd and "--amp" in cmd and "--train-steps 30000" in cmd
    assert cmd.strip().endswith(str(tmp_path / "a" / "results" / "gate0_t8" / "T7"))

    rc, log, logdir = run_queue(G8_QUEUE, tmp_path / "b", env_extra={"QUEUE_DRY_FAILED_SLOTS": "T7"})
    assert rc == 0 and "BACKUP 1_T7" in log and "START 1_T7_backup:" in log
    assert "START 0_T6_backup:" not in log and "START 2_T8_backup:" not in log
    assert f"--tasks {s['T7']['backup']}" in (logdir / "1_T7_backup.log").read_text()

    rc, log, _ = run_queue(G8_QUEUE, tmp_path / "c",
                           env_extra={"QUEUE_DRY_FAILED_SLOTS": "T7", "QUEUE_FAIL_STEP": "1_T7"})
    assert rc == 1 and "CRASH 1_T7" in log and "_backup:" not in log and "QUEUE DONE FAILED: 1_T7" in log

    stale = g8.single_task_run_id(s["T8"]["primary"], 0)
    rc, log, _ = run_queue(G8_QUEUE, tmp_path / "d", present=[f"{stale}/config.yaml"])
    assert rc == 1 and "REFUSE 2_T8" in log and "START 2_T8:" not in log and "2_T8(stale)" in log

    rc, log, _ = run_queue(G8_QUEUE, tmp_path / "e", args=("--from-step", "1"), present=["gate0_t8/T6/gate0.json"])
    assert rc == 0 and "PRIOR 0_T6 ok" in log and "START 0_T6:" not in log and "START 1_T7:" in log


# ---- Gate 0 round 2 (LIBERO-90 candidates for T6 and T8) -------------------------------------------------

R2 = repo_root() / "configs" / "analysis" / "gate0_t8_r2.yaml"


def test_round2_config_meets_its_criteria():
    import re

    out, slots = g8.load_slots(R2)
    assert out == "gate0_t8_r2" and list(slots) == ["T6", "T8"]
    keys = [k for v in slots.values() for k in v.values()]
    round1 = {k for v in g8.slots().values() for k in v.values()}
    assert len(set(keys)) == 4 and not round1 & set(keys)
    scenes = []
    for key in keys:
        suite, name = key.split("/")
        assert suite == "libero_90"
        assert (repo_root() / "benchmark" / "LIBERO" / "libero" / "libero" / "bddl_files" / suite
                / f"{name}.bddl").is_file(), key
        scene = re.match(r"^((?:KITCHEN|STUDY)_SCENE\d+)_", name).group(1)
        assert scene not in {"KITCHEN_SCENE3", "KITCHEN_SCENE4", "STUDY_SCENE1"}      # T4, T7, T5
        assert not any(o in name for o in ("black_bowl", "milk", "moka_pot", "book"))  # T1-T7's objects
        scenes.append(scene)
    assert len(set(scenes)) == 4
    from flowcl.data.libero_setup import task_file_pattern
    assert task_file_pattern(keys[0]) == f"libero_90/{keys[0].split('/')[1]}_demo.hdf5"
    with pytest.raises(ValueError, match="Unknown suite"):
        task_file_pattern("libero_1000/x")
    with pytest.raises(ValueError, match="must be"):
        task_file_pattern("libero_90")


def test_load_slots_refuses_round1_output_and_duplicates(tmp_path):
    assert g8.load_slots(None)[0] == "gate0_t8"
    bad = tmp_path / "bad.yaml"
    bad.write_text("out: gate0_t8\nslots: {T6: {primary: a/b, backup: a/c}}\n")
    with pytest.raises(ValueError, match="own output"):
        g8.load_slots(bad)
    dup = tmp_path / "dup.yaml"
    dup.write_text("out: r9\nslots: {T6: {primary: a/b, backup: a/c}, T8: {primary: a/b, backup: a/d}}\n")
    with pytest.raises(ValueError, match="twice"):
        g8.load_slots(dup)


def test_gate0_queue_round2_config(tmp_path):
    _, slots = g8.load_slots(R2)
    rc, log, logdir = run_queue(G8_QUEUE, tmp_path / "a", args=("--config", str(R2)),
                                env_extra={"QUEUE_DRY_FAILED_SLOTS": "T8"})
    assert rc == 0 and logdir.name.endswith("_gate0_t8_r2") and log.rstrip().endswith("QUEUE DONE ok")
    assert "START 0_T6:" in log and "START 1_T8:" in log and "START 1_T8_backup:" in log
    assert "START 0_T6_backup:" not in log and "T7" not in log
    cmd = (logdir / "0_T6.log").read_text()
    assert f"--tasks {slots['T6']['primary']}" in cmd
    assert cmd.strip().endswith(str(tmp_path / "a" / "results" / "gate0_t8_r2" / "T6"))
    assert f"--tasks {slots['T8']['backup']}" in (logdir / "1_T8_backup.log").read_text()


def test_seq_hetero_t8_is_t5_plus_the_gate0_slots():
    from flowcl.data.curriculum import load_curriculum

    t8 = [st.task_key for st in load_curriculum("seq_hetero_t8").stages]
    t5 = [st.task_key for st in load_curriculum("seq_hetero_t5").stages]
    _, r2 = g8.load_slots(R2)
    assert t8[:5] == t5 and len(t8) == 8 and len(set(t8)) == 8
    assert t8[5] == r2["T6"]["backup"]               # round 2: the primary failed, the backup passed
    assert t8[6] == g8.slots()["T7"]["primary"]      # round 1
    assert t8[7] == r2["T8"]["primary"]              # round 2
    assert all(st.n_demos == 50 for st in load_curriculum("seq_hetero_t8").stages)
