"""The SGP update-rule coupling test: the pure pieces (rate matching, pilot rule, split, log-ratio
intervals, the verdict order), the read-only recorder, the reproduction gate, the whole study end
to end on the tiny policy (including the skip paths), the CLI's smoke guard and the queue."""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from flowcl.experiments import low_update as lu
from flowcl.experiments import sgp_coupling as sc
from flowcl.methods.gpm import GPM
from flowcl.utils.libero_paths import repo_root
from flowcl.utils.run import file_sha256
from test_low_update import BOOT, KEYS, PROBE, Ref, fake_world, spec, stub_evaluator  # noqa: F401

A800 = "seq_hetero__sgp_projected_adam_a800_ne90__seed0"
NS = "seq_hetero__seq_ft__seed0"
LABELS = {sc.INVALID, sc.NO_PROJECTION_COST, sc.NO_COUPLING_COST, sc.INCONCLUSIVE, sc.SUPPORTED,
          sc.MIXED, sc.INCOMPLETE}


# ---- pure pieces ------------------------------------------------------------------------------------


def rec(step, g, u, f=0.5):
    return {"step": step, "grad_norm": g, "update_norm": u, "f": f}


def test_matched_rate_is_the_median_ratio_over_the_window():
    records = [rec(0, 1.0, 9.0), rec(1, 2.0, 1.0), rec(2, 1.0, 1.0), rec(3, 4.0, 1.0),
               rec(4, math.inf, 1.0), rec(5, 1.0, 0.0), rec(9, 1.0, 5.0)]
    m = sc.matched_rate(records, (1, 9))
    # usable ratios u / (f g): 1.0, 2.0, 0.5 -> median 1.0; the inf and the skipped step excluded
    assert m["eta"] == 1.0 and m["n_steps"] == 3 and m["n_skipped"] == 2
    with pytest.raises(ValueError, match="no usable step"):
        sc.matched_rate(records, (20, 30))


def test_layer_rates_are_quantiles_per_half():
    q = sc.layer_rates({"trunk.a.weight": 1.0, "trunk.b.weight": 3.0, "flow_head.c.weight": 2.0,
                        "trunk.d.weight": None})
    assert q["trunk"]["n_layers"] == 2 and q["trunk"]["median"] == 2.0 and q["decoder"]["max"] == 2.0


def pilot(start, end, finite=True):
    return {"probe": {"start": start, "end": end}, "losses_finite": finite}


def test_pilot_verdicts():
    assert sc.pilot_verdict(pilot(1.4, 0.05), 0.04, 2.0)["verdict"] == "viable"
    assert sc.pilot_verdict(pilot(1.4, 0.08), 0.04, 2.0)["verdict"] == "viable"  # exactly 2x
    assert sc.pilot_verdict(pilot(1.4, 0.09), 0.04, 2.0)["verdict"] == "slow"
    assert sc.pilot_verdict(pilot(1.4, 1.4), 0.04, 2.0)["verdict"] == "diverged"  # no improvement
    assert sc.pilot_verdict(pilot(1.4, math.nan), 0.04, 2.0)["verdict"] == "diverged"
    assert sc.pilot_verdict(pilot(1.4, 0.05, finite=False), 0.04, 2.0)["verdict"] == "diverged"


@pytest.mark.parametrize("verdicts, chosen", [
    ({"0.5": "viable", "1": "viable", "2": "viable"}, 1.0),
    ({"0.5": "viable", "1": "diverged", "2": "diverged"}, 0.5),
    ({"0.5": "slow", "1": "slow", "2": "viable"}, 2.0),
    ({"0.5": "viable", "1": "slow", "2": "viable"}, 0.5),   # a tie on the log scale: the smaller
    ({"0.5": "diverged", "1": "slow", "2": "diverged"}, None),
])
def test_rate_selection(verdicts, chosen):
    assert sc.select_rate(verdicts) == chosen


def test_importance_split_hard_scaled_and_free():
    q, _ = torch.linalg.qr(torch.randn(5, 5, dtype=torch.float64, generator=torch.Generator().manual_seed(0)))
    M = q[:, :3]
    lam = torch.tensor([1.0, 0.5, 0.2], dtype=torch.float64)
    coeffs = torch.tensor([[3.0, 4.0, 0.0, 12.0, 0.0]], dtype=torch.float64)  # rows of dW in q's basis
    delta = coeffs @ q.T
    start = {"trunk.x.weight": torch.zeros(1, 5), "flow_head.y.weight": torch.zeros(1, 5)}
    end = {"trunk.x.weight": delta.float(), "flow_head.y.weight": torch.zeros(1, 5)}
    s = sc.importance_split(start, end, {"trunk.x": M, "flow_head.y": M},
                            {"trunk.x": lam, "flow_head.y": lam})
    t = s["trunk"]
    assert t["hard"] == pytest.approx(3.0) and t["scaled"] == pytest.approx(4.0)
    assert t["memory"] == pytest.approx(5.0) and t["free"] == pytest.approx(12.0, rel=1e-6)
    assert t["total"] == pytest.approx(13.0) and "decoder" not in s  # nothing moved there


def units(losses, weights=None):
    return {"losses": losses, "weights": weights or [1.0] * len(losses)}


def test_log_ratio_ci_point_interaction_and_matching():
    u = {"a": units([0.2, 0.4]), "b": units([0.1, 0.2]), "c": units([0.3, 0.3]), "d": units([0.3, 0.3])}
    boot = {"n_resamples": 500, "seed": 0, "confidence": 0.95}
    r = sc.log_ratio_ci(u, [(1, "a", "b")], boot)
    assert r["value"] == pytest.approx(math.log(2)) and r["low"] <= r["value"] <= r["high"]
    assert r["low"] == pytest.approx(math.log(2))  # every batch has ratio 2
    i = sc.log_ratio_ci(u, [(1, "a", "b"), (-1, "c", "d")], boot)
    assert i["value"] == pytest.approx(math.log(2))
    with pytest.raises(ValueError, match="not matched"):
        sc.log_ratio_ci({**u, "b": units([0.1, 0.2], [2.0, 1.0])}, [(1, "a", "b")], boot)


def ci(low, high):
    return {"low": low, "high": high, "value": (low + high) / 2}


def test_verdict_order():
    up, down, flat = ci(0.1, 0.3), ci(-0.3, -0.1), ci(-0.1, 0.1)
    ok, bad = {"passed": True}, {"passed": False}
    assert sc.classify(False, up, up, down, True, ok)["verdict"] == sc.INVALID
    assert sc.classify(True, None, up, down, True, ok)["verdict"] == sc.INCOMPLETE
    assert sc.classify(True, flat, up, down, True, ok)["verdict"] == sc.NO_PROJECTION_COST
    assert sc.classify(True, up, flat, down, True, ok)["verdict"] == sc.NO_COUPLING_COST
    assert sc.classify(True, up, down, down, True, ok)["verdict"] == sc.NO_COUPLING_COST  # benefit
    assert sc.classify(True, up, up, down, False, ok)["verdict"] == sc.INCONCLUSIVE
    assert sc.classify(True, up, up, None, True, None)["verdict"] == sc.INCOMPLETE
    assert sc.classify(True, up, up, down, True, ok)["verdict"] == sc.SUPPORTED
    assert sc.classify(True, up, up, down, True, bad)["verdict"] == sc.MIXED  # gain by forgetting
    assert sc.classify(True, up, up, flat, True, ok)["verdict"] == sc.MIXED
    assert sc.sign_label(down, "cost", "gain") == "gain" and sc.sign_label(flat, "a", "b") == "none"


def test_retention_guard_boundary():
    cells = lambda *s: {str(j): {"success": v} for j, v in enumerate(s)}  # noqa: E731
    assert sc.retention_guard(cells(0.8, 0.6, 0.9), cells(0.9, 0.7, 0.9), 0.10)["passed"]
    g = sc.retention_guard(cells(0.8, 0.58, 0.9), cells(0.9, 0.7, 0.9), 0.10)
    assert not g["passed"] and not g["per_task"]["1"]["passed"]


def test_step_log_parsing_and_logged_steps(tmp_path):
    assert sc.logged_steps(30000, 50)[:3] == [1, 51, 101] and len(sc.logged_steps(30000, 50)) == 601
    assert sc.logged_steps(6, 2) == [1, 3, 5, 6] and sc.logged_steps(6, 0) == []
    log = tmp_path / "q.log"
    log.write_text("\n".join([
        "[flowcl] === r stage 2: t2 ===", "[flowcl] step 1/6 loss 9.000000 lr 1e-04",
        "[flowcl] === r stage 3: t3 ===", "[flowcl] step 1/6 loss 0.500000 lr 1e-04",
        "[flowcl] t3 rollout 1/50: fail", "[flowcl] step 6/6 loss 0.100000 lr 0.00e+00"]))
    assert sc.parse_step_log(log, "r", 3, 6) == {1: "0.500000", 6: "0.100000"}
    with pytest.raises(ValueError, match="no section"):
        sc.parse_step_log(log, "r", 4, 6)


def test_c_logs_compare_exactly_with_nan_as_nan():
    ref = {"gradient_c": {"0": {"a": 0.5, "b": math.nan}}, "update_c": {"0": {"a": 0.1, "b": 0.2}}}
    mine = {"gradient_c": {0: {"a": 0.5, "b": math.nan}}, "update_c": {0: {"a": 0.1, "b": 0.2}}}
    assert sc.c_logs_equal(mine, ref)["equal"]
    mine["update_c"][0]["b"] = 0.2000000001
    r = sc.c_logs_equal(mine, ref)
    assert not r["equal"] and r["n_mismatch"] == 1
    assert not sc.c_logs_equal(None, ref)["equal"]


def test_recorder_is_read_only_and_measures_the_sgd_gradient_and_the_realised_step():
    from test_sgp import CTX, Toy, basis_for

    basis, _ = basis_for(6, 3)

    def run(record):
        model = Toy()
        gpm = GPM(eps=0.95)
        gpm.set_memory({"a": basis})
        method = sc.StepNormRecorder(gpm, (1, 3), lambda t: 0.5) if record else gpm
        method.on_task_start(model, 1, context=CTX)
        opt = torch.optim.AdamW(model.parameters(), lr=0.05, weight_decay=0.0)
        g, expected = torch.Generator().manual_seed(0), []
        for step in range(4):
            x, y = torch.randn(32, 6, generator=g), torch.randn(32, 4, generator=g)
            opt.zero_grad()
            ((model.a(x) - y) ** 2).mean().backward()
            method.modify_gradients(model, {"step": step, "task_idx": 1})
            projected = float(model.a.weight.grad.norm())  # after the projection, before the clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-3)
            before = model.a.weight.detach().clone()
            opt.step()
            method.after_step(model, {"step": step, "task_idx": 1})
            expected.append((step, projected, float((model.a.weight.detach() - before).norm())))
        return model, method, gpm, expected

    plain, *_ = run(False)
    model, recorder, gpm, expected = run(True)
    assert torch.equal(plain.a.weight, model.a.weight)
    assert [r["step"] for r in recorder.records] == [1, 2]
    for r, (_, g, u) in zip(recorder.records, expected[1:3]):
        assert r["grad_norm"] == pytest.approx(g, rel=1e-5) and g > 1e-3  # unclipped
        assert r["update_norm"] == pytest.approx(u, rel=1e-5) and r["f"] == 0.5
    rates = [u / (0.5 * g) for _, g, u in expected[1:3]]
    assert recorder.summary()["layer_rates"]["a.weight"] == pytest.approx(sum(rates) / 2, rel=1e-5)
    assert recorder.task_logs is gpm.task_logs  # everything else is the wrapped method's


def test_config_is_consistent():
    cfg = sc.load_coupling_config()
    rates = cfg["calibration"]["rates"]
    assert tuple(cfg["arms"]) == sc.ARMS and cfg["seed"] == 0
    assert cfg["arms"]["sgd_hard"]["rate"] == cfg["arms"]["sgd_soft"]["rate"] == "proj"
    assert cfg["arms"][rates["proj"]]["projection"] == "hard"
    assert cfg["arms"][rates["free"]]["projection"] == "none"
    for rate, arm in cfg["pilot"]["sgd_arms"].items():
        assert cfg["arms"][arm]["rate"] == rate
        assert cfg["arms"][arm]["projection"] == cfg["arms"][rates[rate]]["projection"]
    assert cfg["sgd_overrides"] == {"optimizer": "sgd", "weight_decay": 0.0, "grad_clip": None}
    assert set(cfg["source"]["sha256"]) == set(sc.source_paths(cfg, Path("/r"))) - {"run"}
    assert cfg["calibration"]["window"][1] <= cfg["pilot"]["steps"] and 1.0 in cfg["pilot"]["scales"]
    assert cfg["source"]["alpha"] == 800 and cfg["recipe"] == {"steps": 30000, "lr": 1e-4}
    assert cfg["source"]["run"] == A800 and cfg["source"]["stage"] == 3


def test_smoke_needs_a_scratch_results_root():
    script = str(repo_root() / "scripts" / "sgp_coupling.py")
    for extra in ([], ["--results-root", str(repo_root() / "results")]):
        done = subprocess.run([sys.executable, script, "--part", "pilot", "--arm", "aw_free",
                               "--smoke-stop-after", "5", *extra], capture_output=True, text=True,
                              timeout=120)
        assert done.returncode == 2 and "results-root" in done.stderr


# ---- a tiny world with an A800 stand-in ----------------------------------------------------------------


def world(tmp_path, spec):
    """fake_world's references plus an A800 stand-in: stage2 = seq_ft's stage0 weights, a real
    SGP (alpha 800) memory for task index 2, and T4 references produced by the same stage path."""
    from flowcl.analysis.probes import probe_loss
    from flowcl.experiments.gate2 import collect_bases, load_subspace_config
    from flowcl.methods.base import TaskContext
    from flowcl.methods.sgp import SGP
    from flowcl.train.checkpoint import load_checkpoint, save_checkpoint

    luc, build = fake_world(tmp_path, spec, seeds=(0,))
    ref_dir = tmp_path / NS
    conf = OmegaConf.load(ref_dir / "config.yaml")
    conf.train.log_every = 2
    OmegaConf.save(conf, ref_dir / "config.yaml")
    run = tmp_path / A800
    loaded = load_checkpoint(ref_dir / "checkpoints" / "stage0.pt")
    save_checkpoint(run / "checkpoints" / "stage2.pt", policy=loaded.policy,
                    policy_config=loaded.payload["policy_config"], spec=loaded.spec, stats=loaded.stats,
                    run_id=A800, stage=2, task_key=KEYS[2])
    stages = [type("S", (), {"task_key": k, "n_demos": 3})() for k in KEYS]
    sub = collect_bases(loaded, build(stages[0]), dataclasses.replace(
        load_subspace_config(), min_samples_per_dim=0.01, num_workers=0, batch_size=4), device="cpu")
    sgp = SGP(alpha=800.0, eps=0.95, new_energy_fraction=0.9)
    sgp.set_memory({n: layer.primary for n, layer in sub.layers.items()})
    sgp.save_artifacts(run / "method", 2, context=TaskContext(
        task_key=KEYS[2], dataset=None, device="cpu", seed_namespace_run_id=NS, method_run_id=A800))
    OmegaConf.save(OmegaConf.create({
        "method": {"name": "sgp", "alpha": 800, "eps": 0.95, "new_energy_fraction": 0.9},
        "train": OmegaConf.to_container(conf.train)}), run / "config.yaml")
    for k, n in ((2, 3), (3, 4)):
        stub_evaluator(None, [Ref(key) for key in KEYS[:n]], spec, None, NS, k).save(
            run / "eval" / f"stage{k}.json")

    cfg = copy.deepcopy(sc.load_coupling_config())
    cfg["recipe"] = {"steps": 6, "lr": 1e-3}
    cfg["calibration"]["window"] = [0, 3]
    cfg["pilot"]["steps"] = 3
    cfg["source"].update({"diagnostics": "diag_a800.json", "sequence_report": "seq_a800.json",
                          "step_log": "logs/a800.log"})
    paths = sc.source_paths(cfg, tmp_path)
    cfg["source"]["sha256"] = {"memory": file_sha256(paths["memory"])}  # for the stand-in's own run

    # A800's "T4": the aw_soft configuration trained once through the same stage path.
    ctx = lu.seed_context(luc, 0, tmp_path)
    captured, base = {}, sc.arm_extra(cfg, paths)

    def extra(policy, method, start_state):
        captured.update(method.task_logs[3])
        return base(policy, method, start_state)

    boot = tmp_path / "standin" / "t4"
    row = lu.train_stage(ctx, 3, 6, 1e-3, paths["start"], boot, "standin", method_spec={"standin": 1},
                         setup_method=sc.make_arm_method(cfg["arms"]["aw_soft"], cfg, paths),
                         extra_row=extra, evaluator=stub_evaluator, bootstrap=BOOT, device="cpu",
                         build_datasets=build, warm_cache=True)  # the runner's text-cache state
    shutil.copy(boot / "checkpoints" / "final.pt", paths["end"])
    paths["gpm_logs"].write_text(json.dumps({key: {str(s): v for s, v in captured[key].items()}
                                             for key in ("gradient_c", "update_c")}))
    losses = json.loads((boot / "losses.json").read_text())
    tail = losses[-50:]
    paths["result"].write_text(json.dumps({"seed_namespace_run_id": NS, "stages": [
        {}, {}, {"method_artifacts": [{"path": "method/memory_task2.pt",
                                       "sha256": file_sha256(paths["memory"])}]},
        {"final_loss": losses[-1], "mean_last_50_loss": sum(tail) / len(tail),
         "evaluation": json.loads((boot / "eval.json").read_text())}]}))
    paths["step_log"].parent.mkdir(parents=True, exist_ok=True)
    paths["step_log"].write_text("\n".join(
        [f"[flowcl] === {A800} stage 3: {KEYS[3]} ==="]
        + [f"[flowcl] step {n}/6 loss {losses[n - 1]:.6f} lr 1.00e-03" for n in sc.logged_steps(6, 2)]) + "\n")
    L = [[0.1] * 4 for _ in range(4)]
    L[2][3] = probe_loss(load_checkpoint(paths["start"]).policy, build(stages[3]), PROBE, "cpu")
    L[3] = [row["cells"][str(j)]["probe_loss"] for j in range(4)]
    paths["diagnostics"].write_text(json.dumps({"method_run_id": A800, "reference_run_id": NS,
                                                "config": {"probe": PROBE}, "loss_matrix": {"method": L}}))
    paths["sequence_report"].write_text(json.dumps({"criteria": {"thresholds": [0.75, 0.63, 0.85, 0.83]}}))
    cfg["source"]["sha256"] = {k: file_sha256(paths[k]) for k in
                               ("start", "memory", "end", "gpm_logs", "result", "diagnostics",
                                "sequence_report", "step_log")}
    return cfg, luc, build


def kwargs(tmp_path, build, luc):
    return dict(results_root=tmp_path, device="cpu", build_datasets=build, low_update_cfg=luc)


def arm(cfg, name, kw):
    return sc.run_arm(cfg, name, allow_dirty=True, evaluator=stub_evaluator, bootstrap=BOOT, **kw)


def force(monkeypatch, verdict):
    def fake(row, adamw_end, slow_factor):
        return {"verdict": verdict, "probe_start": row["probe"]["start"],
                "probe_end": row["probe"]["end"], "adamw_end": adamw_end}
    monkeypatch.setattr(sc, "pilot_verdict", fake)


def pilots(cfg, kw):
    for a in ("aw_free", "aw_hard"):
        sc.run_pilot(cfg, a, allow_dirty=True, **kw)
    return [sc.run_pilot(cfg, a, s, allow_dirty=True, **kw)
            for a in ("sgd_free", "sgd_hard") for s in (0.5, 1.0, 2.0)]


def test_gate_pilots_selection_arms_and_summary_end_to_end(spec, tmp_path, monkeypatch):
    cfg, luc, build = world(tmp_path, spec)
    kw = kwargs(tmp_path, build, luc)
    with pytest.raises(RuntimeError, match="reproduction gate"):
        sc.run_pilot(cfg, "aw_free", allow_dirty=True, **kw)
    arm(cfg, "aw_soft", kw)
    gate = sc.gate_status(cfg, tmp_path)
    assert gate["passed"], gate["checks"]
    assert gate["tensors"]["n_different"] == 0 and gate["training"]["step_log"]["n_expected"] == 4
    assert gate["training"]["c_logs"]["n_compared"] > 0
    assert all(r["same_seeds"] and r["n_differing_episodes"] == 0
               for r in gate["rollouts_descriptive"].values())

    force(monkeypatch, "viable")
    rows = pilots(cfg, kw)
    cal = sc.calibration(cfg, tmp_path)
    assert cal["free"]["n_steps"] == 3 and cal["free"]["eta"] > 0 and cal["proj"]["eta"] > 0
    for p in rows:
        assert p["lr"] == pytest.approx(cal[p["method"]["rate"]]["eta"] * p["method"]["scale"])
        assert "cells" not in p and set(p["probe"]) == {"start", "end"}  # T4 only
        assert p["training"]["steps"] == 3 and p["train_overrides"]["grad_clip"] is None
        assert p["warm_text_cache"] is True
    sel = sc.run_select(cfg, results_root=tmp_path)
    assert all(r["selected_scale"] == 1.0 and r["status"] == "selected" for r in sel["rates"].values())
    _, sha = sc.load_selection(tmp_path / "sgp_coupling")
    assert sc.rate_open(cfg, "free", tmp_path) and sc.rate_open(cfg, "proj", tmp_path)

    done = {a: arm(cfg, a, kw) for a in ("aw_hard", "aw_free", "sgd_free", "sgd_hard", "sgd_soft")}
    for a in ("sgd_free", "sgd_hard", "sgd_soft"):
        m = done[a]["method"]
        assert m["selection_sha256"] == sha and done[a]["train_overrides"] == cfg["sgd_overrides"]
        assert m["lr"] == sel["rates"][m["rate"]]["lr"]
    assert done["sgd_hard"]["lr"] == done["sgd_soft"]["lr"]
    assert all(r["warm_text_cache"] is True for r in done.values())
    assert done["aw_hard"]["update_split"]["trunk"]["memory"] < 1e-2 * done["aw_hard"]["update_split"]["trunk"]["total"]
    assert done["sgd_soft"]["update_split"]["trunk"]["hard"] < 1e-2 * done["sgd_soft"]["update_split"]["trunk"]["total"]
    assert done["aw_free"]["update_split"]["trunk"]["memory"] > 0 and "projection_logs" not in done["aw_free"]

    s = sc.run_summary(cfg, results_root=tmp_path, device="cpu", build_datasets=build,
                       low_update_cfg=luc, bootstrap=BOOT)
    q = s["quantities"]
    assert s["verdict"]["verdict"] in LABELS and not s["skipped_arms"] and s["selection_sha256"] == sha
    assert all(q[k] is not None for k in ("gap_aw", "c_aw", "c_sgd", "delta"))
    again = sc.classify(True, q["gap_aw"], q["c_aw"], q["delta"], s["sgd_viable"]["viable"], s["retention_guard"])
    assert again["verdict"] == s["verdict"]["verdict"]
    assert set(s["arms"]) == set(sc.ARMS) and set(s["arms"]["sgd_soft"]["change_from_stage2"]) == {"0", "1", "2"}
    assert s["arms"]["aw_soft"]["paired_vs"]["a800_stage3"]["3"]["diff"] == 0.0
    assert set(s["soft_vs_hard_rollouts"]) == {"adamw", "sgd"}


def test_the_gate_fails_on_any_divergence_and_then_closes_the_study(spec, tmp_path):
    cfg, luc, build = world(tmp_path, spec)
    kw = kwargs(tmp_path, build, luc)
    row = arm(cfg, "aw_soft", kw)
    out_dir, paths = sc.arm_dir(cfg, tmp_path, "aw_soft"), sc.source_paths(cfg, tmp_path)
    assert sc.reproduction_gate(cfg, tmp_path, row, out_dir)["passed"]

    def gate_after(path, mutate):
        backup = path.with_suffix(".bak")
        shutil.copy(path, backup)
        mutate(path)
        try:
            return sc.reproduction_gate(cfg, tmp_path, row, out_dir)
        finally:
            shutil.move(backup, path)

    def bump_tensor(path):
        payload = torch.load(path, weights_only=False)
        name = next(n for n, t in payload["state_dict"].items() if t.is_floating_point())
        payload["state_dict"][name] = payload["state_dict"][name] + 1e-6
        torch.save(payload, path)

    def bump_final_loss(path):
        result = json.loads(path.read_text())
        result["stages"][3]["final_loss"] = math.nextafter(result["stages"][3]["final_loss"], 1.0)
        path.write_text(json.dumps(result))

    def bump_printed(path):
        text = path.read_text().splitlines()
        text[-1] = text[-1].replace(" loss ", " loss 1")
        path.write_text("\n".join(text) + "\n")

    def bump_probe(path):
        d = json.loads(path.read_text())
        d["loss_matrix"]["method"][3][0] *= 1.01
        path.write_text(json.dumps(d))

    for path, mutate, check in ((paths["end"], bump_tensor, "tensors"),
                                (paths["result"], bump_final_loss, "final_loss"),
                                (paths["step_log"], bump_printed, "step_log"),
                                (paths["diagnostics"], bump_probe, "probes")):
        g = gate_after(path, mutate)
        assert not g["passed"] and not g["checks"][check]
        assert sum(not v for v in g["checks"].values()) == 1, g["checks"]
    assert row["c_log_reproduction"]["equal"]  # the c-log comparison itself: test_c_logs_...

    (tmp_path / "sgp_coupling" / sc.GATE_FILE).write_text(json.dumps({"passed": False}))
    assert not sc.gate_open(cfg, tmp_path)
    with pytest.raises(RuntimeError, match="reproduction gate"):
        arm(cfg, "aw_hard", kw)
    s = sc.run_summary(cfg, results_root=tmp_path, device="cpu", build_datasets=build,
                       low_update_cfg=luc, bootstrap=BOOT)
    assert s["verdict"]["verdict"] == sc.INVALID and s["skipped_arms"] == list(sc.ARMS[1:])


def test_a_rate_without_a_viable_scale_skips_its_arms(spec, tmp_path, monkeypatch):
    cfg, luc, build = world(tmp_path, spec)
    kw = kwargs(tmp_path, build, luc)
    arm(cfg, "aw_soft", kw)
    force(monkeypatch, "diverged")
    pilots(cfg, kw)
    sel = sc.run_select(cfg, results_root=tmp_path)
    assert all(r["selected_scale"] is None and r["status"] == "not_viable" for r in sel["rates"].values())
    assert not sc.rate_open(cfg, "proj", tmp_path)
    with pytest.raises(RuntimeError, match="not viable"):
        arm(cfg, "sgd_hard", kw)
    arm(cfg, "aw_hard", kw)
    arm(cfg, "aw_free", kw)
    s = sc.run_summary(cfg, results_root=tmp_path, device="cpu", build_datasets=build,
                       low_update_cfg=luc, bootstrap=BOOT)
    assert s["skipped_arms"] == ["sgd_free", "sgd_hard", "sgd_soft"] and not s["sgd_viable"]["viable"]
    assert s["verdict"]["verdict"] in (sc.NO_PROJECTION_COST, sc.NO_COUPLING_COST, sc.INCONCLUSIVE)
    assert s["quantities"]["c_sgd"] is None and s["verdict"]["q2_scaled_under_sgd"] is None


def test_start_checks_refuse_a_changed_reference(spec, tmp_path):
    cfg, luc, build = world(tmp_path, spec)
    ctx = lu.seed_context(luc, 0, tmp_path)
    assert sc.check_start(cfg, ctx, tmp_path)["passed"]
    bad = copy.deepcopy(cfg)
    bad["source"]["sha256"]["step_log"] = "0" * 64
    with pytest.raises(ValueError, match="sha256.step_log"):
        sc.check_start(bad, ctx, tmp_path)
    bad = copy.deepcopy(cfg)
    bad["source"]["alpha"] = 400
    with pytest.raises(ValueError, match="method"):
        sc.check_start(bad, ctx, tmp_path)
    bad = copy.deepcopy(cfg)
    bad["verdict"]["sgd_t4_threshold"] = 0.9
    with pytest.raises(ValueError, match="t4_threshold"):
        sc.check_start(bad, ctx, tmp_path)


# ---- the queue ---------------------------------------------------------------------------------------

QUEUE = repo_root() / "scripts" / "queue_sgp_coupling.sh"
STEPS = ["0_arm_aw_soft", "1_repro_gate", "2_pilot_aw_free", "3_pilot_aw_hard",
         "4_pilot_sgd_free_x0.5", "5_pilot_sgd_free_x1", "6_pilot_sgd_free_x2",
         "7_pilot_sgd_hard_x0.5", "8_pilot_sgd_hard_x1", "9_pilot_sgd_hard_x2", "10_select",
         "11_arm_aw_hard", "12_arm_aw_free", "13_rate_gate_free", "14_arm_sgd_free",
         "15_rate_gate_proj", "16_arm_sgd_hard", "17_arm_sgd_soft", "18_summary"]


def queue(tmp_path, args=(), fail="", present=()):
    results = tmp_path / "results"
    for rel in present:
        path = results / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    env = {**os.environ, "QUEUE_DRY_RUN": "1", "QUEUE_FAIL_STEP": fail,
           "QUEUE_LOG_ROOT": str(tmp_path / "logs"), "QUEUE_RESULTS_ROOT": str(results)}
    done = subprocess.run(["bash", str(QUEUE), *args], env=env, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    (logdir,) = list((tmp_path / "logs").glob("queue_*_sgp_coupling"))
    return (logdir / "queue.log").read_text(), logdir


def ran(log, step):
    return f"START {step}:" in log


def test_queue_order_commands_and_gates(tmp_path):
    assert subprocess.run(["bash", "-n", str(QUEUE)]).returncode == 0
    log, logdir = queue(tmp_path / "a")
    assert [log.index(f"START {s}:") for s in STEPS] == sorted(log.index(f"START {s}:") for s in STEPS)
    assert (logdir / "7_pilot_sgd_hard_x0.5.log").read_text().split("sgp_coupling.py ")[1].strip() == \
        "--part pilot --arm sgd_hard --scale 0.5"
    assert "--part rate_gate --rate proj" in (logdir / "15_rate_gate_proj.log").read_text()
    log, _ = queue(tmp_path / "b", fail="1_repro_gate")
    assert not any(ran(log, s) for s in STEPS[2:-1]) and ran(log, "18_summary") and "SKIP 2-17" in log
    log, _ = queue(tmp_path / "c", fail="13_rate_gate_free")
    assert not ran(log, "14_arm_sgd_free") and ran(log, "16_arm_sgd_hard") and ran(log, "17_arm_sgd_soft")
    log, _ = queue(tmp_path / "d", fail="15_rate_gate_proj")
    assert ran(log, "14_arm_sgd_free") and not ran(log, "16_arm_sgd_hard") and ran(log, "18_summary")
    log, _ = queue(tmp_path / "e", fail="2_pilot_aw_free")  # a failed step never stops the queue
    assert all(ran(log, s) for s in STEPS if s != "2_pilot_aw_free") and ran(log, "2_pilot_aw_free")


def test_queue_resume(tmp_path):
    out = "sgp_coupling"
    present = [f"{out}/arms/coup__aw_soft__seed0/row.json", f"{out}/selection/selection.done.json",
               *[f"{out}/pilots/pilot__{p}__seed0/row.json" for p in
                 ("aw_free", "aw_hard", *[f"{a}_x{s}" for a in ("sgd_free", "sgd_hard") for s in ("0.5", "1", "2")])]]
    log, _ = queue(tmp_path / "a", args=("--from-step", "11"), present=present)
    assert "PRIOR 0_arm_aw_soft ok" in log and "PRIOR 9_pilot_sgd_hard_x2 ok" in log
    assert "PRIOR 10_select ok" in log and ran(log, "1_repro_gate") and ran(log, "11_arm_aw_hard")
    assert not ran(log, "0_arm_aw_soft") and not ran(log, "10_select")
    log, _ = queue(tmp_path / "b", args=("--from-step", "11"), present=present[:1])
    assert "PRIOR 10_select missing" in log
    for bad in (("--from-step", "19"), ("--from-step", "x"), ("--bogus",)):
        assert subprocess.run(["bash", str(QUEUE), *bad], capture_output=True).returncode == 2
