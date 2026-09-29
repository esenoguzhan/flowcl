"""SGP sensitivity: the selection and validation rules, the immutable selection bundle, reused-arm
and pairing checks, nominal attenuation, a bracket -> select -> validate -> summary run on the
tiny policy, the summary when nothing qualifies, and the queue."""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess

import pytest
import torch
from omegaconf import OmegaConf

from flowcl.analysis.subspace import SubspaceBasis
from flowcl.experiments import sgp_sensitivity as ss
from flowcl.utils.libero_paths import repo_root
from test_low_update import BOOT, KEYS, PROBE, fake_world, spec, stub_evaluator  # noqa: F401

RULE = {"object_tolerance": 0.10, "spatial_tolerance": 0.10}

# ---- pure rules --------------------------------------------------------------------------------------


def arm(family, obj, spa, alpha=None):
    return {"family": family, "object": obj, "spatial": spa, **({"alpha": alpha} if alpha else {})}


def test_select_alpha_picks_the_softest_qualifying_eligible_strength():
    arms = {"gpm": arm("gpm", 0.78, 0.94), "fine_tuning": arm("ft", 0.78, 0.0),
            "freeze_only": arm("freeze_only", 0.80, 0.99),
            "a25": arm("sgp", 0.80, 0.50, 25), "a100": arm("sgp", 0.68, 0.84, 100),  # exactly both floors
            "a400": arm("sgp", 0.78, 0.94, 400), "a1600": arm("sgp", 0.78, 0.96, 1600)}
    s = ss.select_alpha(arms, [25, 100, 400, 1600], 0.78, 0.94, RULE)
    assert s["selected_alpha"] == 100.0 and s["selected_arm"] == "a100"
    assert s["floors"] == pytest.approx({"object": 0.68, "spatial": 0.84})
    assert not s["per_arm"]["freeze_only"]["eligible"] and s["per_arm"]["freeze_only"]["qualifies"]
    assert "freeze_only" not in s["qualifying"] and "gpm" not in s["qualifying"]
    arms["a100"]["spatial"] = 0.83
    assert ss.select_alpha(arms, [25, 100, 400, 1600], 0.78, 0.94, RULE)["selected_alpha"] == 400.0
    for n in ("a100", "a400", "a1600"):
        arms[n]["spatial"] = 0.5
    none = ss.select_alpha(arms, [25, 100, 400, 1600], 0.78, 0.94, RULE)
    assert none["selected_alpha"] is None and none["qualifying"] == []


def vseed(obj, spa, ft, gpm):
    return {"object": obj, "spatial": spa, "ft_object": ft, "gpm_spatial": gpm,
            "object_tolerance": 0.10, "spatial_tolerance": 0.10}


def test_validation_needs_every_held_out_seed_with_per_seed_floors():
    v = ss.classify_validation({1: vseed(0.80, 0.88, 0.90, 0.98), 2: vseed(0.82, 0.84, 0.92, 0.94)}, [1, 2])
    assert v["verdict"] == ss.VALIDATED and "No rescue tuning" in v["no_rescue"]
    v = ss.classify_validation({1: vseed(0.80, 0.87, 0.90, 0.98), 2: vseed(0.9, 0.9, 0.92, 0.94)}, [1, 2])
    assert v["verdict"] == ss.NOT_VALIDATED and v["failed_seeds"] == ["1"]
    assert not v["per_seed"]["1"]["protection"] and v["per_seed"]["1"]["plasticity"]
    with pytest.raises(ValueError, match="missing"):
        ss.classify_validation({1: vseed(1, 1, 1, 1)}, [1, 2])


def test_nominal_attenuation_hand_case_and_monotone():
    s = torch.tensor([4.0, 2.0, 1.0, 0.1], dtype=torch.float64)
    b = SubspaceBasis("trunk.x", 4, 10, 4, s, (0.95,), {0.95: 3}, {0.95: 0.75},
                      torch.eye(4, dtype=torch.float64)[:, :3])
    n = ss.nominal_attenuation({"trunk.x": b}, 1.0, 1e-4, 0.95)
    lam = torch.tensor([1.0, 2 / 3, 0.4], dtype=torch.float64)
    expected = float(((1 - lam) * s[:3] ** 2).sum() / (s[:3] ** 2).sum())
    assert n["trunk"]["energy_weighted_unprotected"] == pytest.approx(expected)
    assert n["trunk"]["nominal_protected_scale"] == pytest.approx(1e-4 * float((1 - lam).mean()))
    values = [ss.nominal_attenuation({"trunk.x": b}, a, 1e-4, 0.95)["trunk"]["energy_weighted_unprotected"]
              for a in (5, 25, 100, 400, 1600)]
    assert values == sorted(values, reverse=True)
    assert ss.nominal_attenuation({"trunk.x": b}, None, 1e-4, 0.95)["trunk"]["energy_weighted_unprotected"] == 0.0
    assert "nominal" in n["note"]


def test_selection_bundle_is_atomic_immutable_and_verified(tmp_path):
    out = tmp_path / "sens"
    (out / ".selection_attempt_20260929_000000_1").mkdir(parents=True)  # an abandoned attempt
    sel, sha = ss.publish_selection(out, {"selected_alpha": 100.0})
    assert sel["selected_alpha"] == 100.0 and (out / "selection" / "selection.done.json").is_file()
    mtime = (out / "selection" / "selection.json").stat().st_mtime_ns
    again, sha2 = ss.publish_selection(out, {"selected_alpha": 400.0})  # never rewritten
    assert again["selected_alpha"] == 100.0 and sha2 == sha
    assert (out / "selection" / "selection.json").stat().st_mtime_ns == mtime
    assert ss.load_selection(out) == (sel, sha)
    (out / "selection" / "selection.json").write_text(json.dumps({"selected_alpha": 1600.0}))
    with pytest.raises(RuntimeError, match="marker records"):
        ss.load_selection(out)
    with pytest.raises(RuntimeError, match="no published selection"):
        ss.load_selection(tmp_path / "elsewhere")
    (out / "selection" / "selection.done.json").unlink()
    with pytest.raises(RuntimeError, match="incomplete"):
        ss.load_selection(out)


# ---- a fake world with GPM and SGP references ----------------------------------------------------------


def world(tmp_path, spec, seeds=(0, 1)):
    """seq_ft/GPM references per seed (stage0 = stage1 weights, real T1 memory, real probe
    references), plus the SGP alpha-25 run on seed 0."""
    import dataclasses

    from flowcl.analysis.probes import probe_loss
    from flowcl.analysis.subspace import save_bases
    from flowcl.experiments.gate2 import collect_bases, load_subspace_config
    from flowcl.train.checkpoint import load_checkpoint, save_checkpoint

    luc, build = fake_world(tmp_path, spec, seeds=seeds)
    stages = [type("S", (), {"task_key": k, "n_demos": 3})() for k in KEYS[:2]]
    for s in seeds:
        ns = f"seq_hetero__seq_ft__seed{s}"
        g = tmp_path / f"seq_hetero__gpm_projected_adam__seed{s}"
        loaded = load_checkpoint(tmp_path / ns / "checkpoints" / "stage0.pt")
        for run, stage in ((ns, 1), (g.name, 0), (g.name, 1)):
            save_checkpoint(tmp_path / run / "checkpoints" / f"stage{stage}.pt", policy=loaded.policy,
                            policy_config=loaded.payload["policy_config"], spec=loaded.spec,
                            stats=loaded.stats, run_id=run, stage=stage, task_key=KEYS[stage])
        OmegaConf.save(OmegaConf.create({"seed_namespace_run_id": ns, "train": dict(
            OmegaConf.load(tmp_path / ns / "config.yaml").train),
            "method": {"name": "gpm", "eps": 0.95, "projection": "gradient_and_update"}}), g / "config.yaml")
        (g / "git_sha").write_text("gpmsha\n")
        (tmp_path / ns / "git_sha").write_text("ftsha\n")
        sub = collect_bases(loaded, build(stages[0]), dataclasses.replace(
            load_subspace_config(), min_samples_per_dim=0.01, num_workers=0, batch_size=4), device="cpu")
        save_bases(g / "method" / "memory_task0.pt", {n: l.primary for n, l in sub.layers.items()},
                   {"kind": "accumulated_memory"})
        real = [probe_loss(loaded.policy, build(stages[j]), PROBE, "cpu") for j in (0, 1)]
        for key in ("gpm", "adaptive_gpm"):
            path = tmp_path / luc["seeds"][s]["diagnostics"][key]
            d = json.loads(path.read_text())
            d["loss_matrix"]["reference"][1][:2] = real
            d["loss_matrix"]["method"][1][:2] = real
            path.write_text(json.dumps(d))
    # The SGP alpha-25 run (seed 0): same weights at stage 1, same T1 memory, its own diagnostics.
    p = tmp_path / "seq_hetero__sgp_projected_adam__seed0"
    shutil.copytree(tmp_path / "seq_hetero__gpm_projected_adam__seed0" / "method", p / "method")
    (p / "checkpoints").mkdir(parents=True)
    shutil.copy(tmp_path / "seq_hetero__seq_ft__seed0" / "checkpoints" / "stage1.pt",
                p / "checkpoints" / "stage1.pt")
    shutil.copytree(tmp_path / "seq_hetero__gpm_projected_adam__seed0" / "eval", p / "eval")
    OmegaConf.save(OmegaConf.create({
        "seed_namespace_run_id": "seq_hetero__seq_ft__seed0",
        "train": dict(OmegaConf.load(tmp_path / "seq_hetero__seq_ft__seed0" / "config.yaml").train),
        "method": {"name": "sgp", "alpha": 25, "eps": 0.95, "projection": "hard_gradient_scaled_update"}}),
        p / "config.yaml")
    (p / "git_sha").write_text("40cfdda\n")
    d = json.loads((tmp_path / luc["seeds"][0]["diagnostics"]["gpm"]).read_text())
    (tmp_path / "diag_sgp_seed0.json").write_text(json.dumps({**d, "method_run_id": p.name}))
    cfg = copy.deepcopy(ss.load_sensitivity_config())
    cfg["recipe"] = {"steps": 4, "lr": 1e-4}
    cfg["arms"][0]["diagnostics"] = "diag_sgp_seed0.json"
    cfg["validation_seeds"] = [s for s in seeds if s != 0]
    return cfg, luc, build


def common(tmp_path, build, luc):
    return dict(results_root=tmp_path, device="cpu", evaluator=stub_evaluator, bootstrap=BOOT,
                build_datasets=build, low_update_cfg=luc)


def test_bracket_select_validate_summary_end_to_end(spec, tmp_path):
    cfg, luc, build = world(tmp_path, spec)
    kw = common(tmp_path, build, luc)
    rows = ss.run_bracket(cfg, allow_dirty=True, **kw)
    assert [r["run_id"] for r in rows] == ["sens__sgp_a100__seed0", "sens__sgp_a400__seed0",
                                          "sens__sgp_a1600__seed0", "sens__freeze_only__seed0"]
    sel = ss.run_select(cfg, results_root=tmp_path, low_update_cfg=luc, bootstrap=BOOT)
    # The stub gives every arm the same successes: all qualify, and the softest (25) wins.
    assert sel["selected_alpha"] == 25.0 and ss.selection_open(cfg, tmp_path)
    _, sha = ss.load_selection(tmp_path / "sgp_sensitivity")
    v = ss.run_validation(cfg, 1, allow_dirty=True, **kw)
    assert v["selection_sha256"] == sha and v["row"]["method"]["selection_sha256"] == sha
    assert v["row"]["method"]["alpha"] == 25.0
    summary = ss.run_summary(cfg, results_root=tmp_path, device="cpu", build_datasets=build,
                             low_update_cfg=luc, bootstrap=BOOT)
    assert summary["verdict"]["verdict"] == ss.VALIDATED and summary["selection_sha256"] == sha
    b = summary["bracket"]
    assert set(b) == {"sgp_a25", "sgp_a100", "sgp_a400", "sgp_a1600", "gpm", "fine_tuning", "freeze_only"}
    assert b["sgp_a25"]["source"]["kind"] == "reused" and b["sgp_a100"]["source"]["kind"] == "new"
    assert b["freeze_only"]["nominal_attenuation"] is None and b["gpm"]["nominal_attenuation"]["trunk"][
        "energy_weighted_unprotected"] == 0.0
    assert b["freeze_only"]["realised_split"]["trunk"]["parallel_fraction"] > 0  # unprojected
    assert b["sgp_a1600"]["update_size"]["frozen_unchanged"]
    assert "gpm" in b["sgp_a100"]["paired_vs"] and summary["validation"]["1"]["seed"] == 1


def test_no_qualifying_alpha_still_summarises(spec, tmp_path):
    cfg, luc, build = world(tmp_path, spec, seeds=(0,))
    cfg["selection"]["spatial_tolerance"] = -0.5  # an impossible floor: nothing qualifies
    kw = common(tmp_path, build, luc)
    ss.run_bracket(cfg, allow_dirty=True, **kw)
    sel = ss.run_select(cfg, results_root=tmp_path, low_update_cfg=luc, bootstrap=BOOT)
    assert sel["selected_alpha"] is None and not ss.selection_open(cfg, tmp_path)
    with pytest.raises(RuntimeError, match="nothing to validate"):
        ss.run_validation({**cfg, "validation_seeds": [0]}, 0, allow_dirty=True, **kw)
    s = ss.run_summary(cfg, results_root=tmp_path, device="cpu", build_datasets=build,
                       low_update_cfg=luc, bootstrap=BOOT)
    assert s["verdict"]["verdict"] == ss.NO_ALPHA and s["validation"] is None and len(s["bracket"]) == 7


def test_reused_arm_and_pairing_are_verified(spec, tmp_path):
    from flowcl.experiments.low_update import seed_context
    from flowcl.train.checkpoint import load_checkpoint, save_checkpoint

    cfg, luc, build = world(tmp_path, spec, seeds=(0,))
    ctx = seed_context(luc, 0, tmp_path)
    a25 = cfg["arms"][0]
    ss.reused_arm(a25, ctx, tmp_path, cfg, BOOT)
    ss.reused_arm({"name": "fine_tuning", "family": "ft", "source": "reused"}, ctx, tmp_path, cfg, BOOT)
    with pytest.raises(ValueError, match="method"):
        ss.reused_arm({**a25, "alpha": 100}, ctx, tmp_path, cfg, BOOT)
    with pytest.raises(ValueError, match="recipe"):
        ss.reused_arm(a25, ctx, tmp_path, {**cfg, "recipe": {"steps": 30000, "lr": 1e-4}}, BOOT)
    ss.check_start(ctx, tmp_path, cfg)
    g0 = tmp_path / "seq_hetero__gpm_projected_adam__seed0" / "checkpoints" / "stage0.pt"
    loaded = load_checkpoint(g0)
    with torch.no_grad():
        next(p for p in loaded.policy.parameters() if p.requires_grad).add_(1.0)
    save_checkpoint(g0, policy=loaded.policy, policy_config=loaded.payload["policy_config"],
                    spec=loaded.spec, stats=loaded.stats, run_id="g", stage=0, task_key=KEYS[0])
    with pytest.raises(RuntimeError, match="mispaired"):
        ss.check_start(ctx, tmp_path, cfg)


# ---- the queue ---------------------------------------------------------------------------------------

QUEUE = repo_root() / "scripts" / "queue_sgp_sensitivity.sh"
STEPS = ["0_bracket", "1_select", "2_select_gate", "3_validate_s1", "4_validate_s2", "5_summary"]


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
    (logdir,) = list((tmp_path / "logs").glob("queue_*_sgp_sensitivity"))
    return (logdir / "queue.log").read_text()


def ran(log, step):
    return f"START {step}:" in log


def test_queue_order_gate_and_resume(tmp_path):
    log = queue(tmp_path / "a")
    assert [log.index(f"START {s}:") for s in STEPS] == sorted(log.index(f"START {s}:") for s in STEPS)
    log = queue(tmp_path / "b", fail="2_select_gate")
    assert not ran(log, "3_validate_s1") and not ran(log, "4_validate_s2") and ran(log, "5_summary")
    assert "no_qualifying_alpha" in log
    log = queue(tmp_path / "c", fail="0_bracket")
    assert not any(ran(log, s) for s in STEPS[1:]) and "SKIP 1-5" in log
    log = queue(tmp_path / "d", args=("--from-step", "3"),
                present=["sgp_sensitivity/done_bracket_seed0.json",
                         "sgp_sensitivity/selection/selection.done.json"])
    assert "PRIOR 0_bracket ok" in log and "PRIOR 1_select ok" in log and ran(log, "3_validate_s1")
    for bad in (("--from-step", "6"), ("--bogus",)):
        assert subprocess.run(["bash", str(QUEUE), *bad], capture_output=True).returncode == 2
