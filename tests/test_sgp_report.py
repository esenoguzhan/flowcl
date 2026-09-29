"""The pre-registered SGP rule: gates, O1/O2, attribution, replication and claim levels,
the matched-unit bootstraps, and the importance checks on synthetic memory artifacts."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from flowcl.analysis.importance import importance_update
from flowcl.analysis.subspace import SubspaceBasis, extend_basis, save_bases
from flowcl.experiments.sgp_report import (
    ROLES,
    batch_gap_recovery,
    batch_interaction,
    classify_attribution,
    classify_sgp,
    classify_sgp_replication,
    consume_all_blocks,
    episode_interaction,
    gap_recovery,
    importance_check,
    importance_identity,
    limits,
    load_sgp_config,
    objective_checks,
    residual_check,
    resolve_seed,
)

LIM = {"premise": 0.63, "durable": 0.63, "t4": 0.75, "t3": 0.83,
       "min_improvement": 0.20, "min_gap_recovery": 0.5}


def ci(diff, low, high=None):
    return {"diff": diff, "low": low, "high": diff + 0.1 if high is None else high}


def checks(**overrides):
    base = objective_checks(ci(0.60, 0.45), durable=0.80, g=0.7, t4=0.90, t3=0.95,
                            baseline_transition=0.02, lim=LIM)
    return {"identity": True, "implementation": True, **base, **overrides}


# ---- verdicts -------------------------------------------------------------------------


def test_gates_come_first_in_order():
    assert classify_sgp(checks(identity=False, implementation=False))["verdict"] == "invalid_comparison"
    assert classify_sgp(checks(implementation=False, premise=False))["verdict"] == "invalid_implementation"
    assert classify_sgp(checks(premise=False, plasticity_gap=False))["verdict"] == \
        "not_applicable_baseline_retained"
    assert classify_sgp(checks(plasticity_gap=False))["verdict"] == "not_applicable_no_plasticity_gap"


@pytest.mark.parametrize("o1, o2, expected", [
    (True, True, "both_objectives_met"), (True, False, "retention_only"),
    (False, True, "plasticity_only"), (False, False, "neither"),
])
def test_objectives_2x2(o1, o2, expected):
    v = classify_sgp(checks(o1=o1, o2=o2))
    assert v["verdict"] == expected and (v["o1"], v["o2"]) == (o1, o2)


def test_o1_boundaries_and_the_durable_flag():
    ok = objective_checks(ci(0.20, 0.01), 0.63, 0.5, 0.75, 0.83, 0.02, LIM)
    assert ok["o1"] and ok["durable"]
    assert not objective_checks(ci(0.20, 0.0), 0.63, 0.5, 0.75, 0.83, 0.02, LIM)["o1"]  # CI > 0
    assert not objective_checks(ci(0.18, 0.05), 0.63, 0.5, 0.75, 0.83, 0.02, LIM)["o1"]
    v = classify_sgp(checks(durable=False))
    assert v["verdict"] == "both_objectives_met"
    assert v["flags"] == ["durable Object retention below threshold"]


def test_o2_needs_all_three_parts_and_reports_which_failed():
    for kwargs, part in ((dict(g=0.49), "gap_recovery"), (dict(t4=0.74), "t4"),
                         (dict(t3=0.82), "t3")):
        args = dict(transition=ci(0.6, 0.4), durable=0.8, g=0.7, t4=0.9, t3=0.95,
                    baseline_transition=0.02, lim=LIM) | kwargs
        c = objective_checks(**args)
        assert not c["o2"] and not c["o2_parts"][part]
        v = classify_sgp({"identity": True, "implementation": True, **c})
        assert f"O2 part failed: {part}" in v["flags"]
    assert objective_checks(ci(0.6, 0.4), 0.8, 0.5, 0.75, 0.83, 0.02, LIM)["o2"]  # inclusive


def test_premise_and_gap():
    assert not objective_checks(ci(0.6, 0.4), 0.8, 0.7, 0.9, 0.9, 0.63, LIM)["premise"]
    assert objective_checks(ci(0.6, 0.4), 0.8, 0.7, 0.9, 0.9, 0.62, LIM)["premise"]
    assert gap_recovery(0.011, 0.0135, 0.0085) == pytest.approx(0.5)
    assert gap_recovery(0.01, 0.008, 0.0085) is None and gap_recovery(0.01, 0.0085, 0.0085) is None
    assert not objective_checks(ci(0.6, 0.4), 0.8, None, 0.9, 0.9, 0.02, LIM)["plasticity_gap"]


def test_o2_is_not_rollout_recovery():
    """Seed-2-like case: A's T4 76% clears the 75% threshold but sits below adaptive GPM
    (80%); O2 passes on the probe loss, and scaling attribution is loss-level only."""
    c = objective_checks(ci(0.5, 0.35), 0.8, 0.6, 0.76, 0.95, 0.22, LIM)
    assert c["o2"]
    a = classify_attribution(ci(0.1, 0.02), 0.6, ci(-0.04, -0.16, 0.08), 0.5)
    assert a["scaling"] == "loss_level" and a["scaling_rollout"] is False


def test_attribution_levels():
    assert classify_attribution(ci(0.2, 0.05), 0.6, ci(0.1, 0.01), 0.5)["memory"] == "strong"
    assert classify_attribution(ci(0.1, -0.05), 0.6, ci(0.1, 0.0), 0.5)["memory"] == "directional"
    assert classify_attribution(ci(0.0, -0.1), 0.6, ci(0.1, 0.0), 0.5)["memory"] == "none"
    assert classify_attribution(ci(0, 0), 0.5, ci(0, 0), 0.5)["scaling"] == "loss_level"
    assert classify_attribution(ci(0, 0), 0.2, ci(0, 0), 0.5)["scaling"] == "directional"
    assert classify_attribution(ci(0, 0), -0.1, ci(0, 0), 0.5)["scaling"] == "none"
    assert classify_attribution(ci(0, 0), None, ci(0, 0), 0.5)["scaling"] == "not_applicable"
    assert classify_attribution(ci(0, 0), 0.6, ci(0.12, 0.01), 0.5)["scaling_rollout"] is True


def test_o1_can_pass_while_memory_attribution_is_none():
    """A ≈ P: adaptive SGP beats plain GPM on Object, but so does the SGP baseline."""
    v = classify_sgp(checks())
    a = classify_attribution(ci(-0.02, -0.14), 0.7, ci(0.1, 0.01), 0.5)
    assert v["o1"] and a["memory"] == "none"


# ---- replication and claims ---------------------------------------------------------


def seed_row(verdict="both_objectives_met", o1=True, o2=True, memory="strong",
             scaling="loss_level", rollout=False):
    return {"verdict": {"verdict": verdict, "o1": o1, "o2": o2},
            "attribution": {"memory": memory, "scaling": scaling, "scaling_rollout": rollout}}


def rep(*rows):
    return classify_sgp_replication({str(i): r for i, r in enumerate(rows)}, [0, 1, 2])


def test_replication_validity_rows():
    assert rep(seed_row(), seed_row("invalid_implementation", None, None),
               seed_row("not_applicable_baseline_retained", None, None))["validity"] == \
        "inconclusive_invalid"
    r = rep(seed_row(), seed_row(), seed_row("not_applicable_no_plasticity_gap", None, None))
    assert r["validity"] == "not_applicable" and r["claim_level"] == 0
    with pytest.raises(ValueError, match="exactly seeds"):
        classify_sgp_replication({"0": seed_row(), "1": seed_row()}, [0, 1, 2])


def test_replication_per_objective():
    r = rep(seed_row(), seed_row("retention_only", True, False), seed_row())
    assert (r["o1"], r["o2"]) == ("replicated", "not_replicated") and r["claim_level"] == 0
    r = rep(*[seed_row("retention_only", True, False)] * 3)
    assert r["o2"] == "consistently_not_met"


def test_claim_levels():
    r = rep(seed_row(), seed_row(), seed_row())
    assert r["claim_level"] == 2 and "(scaled projection at the loss level)" in r["claim"]
    assert "requires" not in r["claim"]
    r = rep(*[seed_row(rollout=True)] * 3)
    assert r["claim_level"] == 2 and "loss and rollout level" in r["claim"]
    assert r["attribution"]["scaling_rollout"] == "replicated_rollout"
    r = rep(seed_row(), seed_row(memory="directional"), seed_row())
    assert r["claim_level"] == 1 and r["attribution"]["memory"] == "not_replicated"
    r = rep(seed_row(), seed_row(scaling="directional", rollout=True), seed_row(rollout=True))
    assert r["claim_level"] == 1 and r["attribution"]["scaling"] == "not_replicated"


# ---- matched-unit bootstraps ------------------------------------------------------------

BOOT = {"n_resamples": 2000, "seed": 0, "confidence": 0.95}


def test_episode_interaction_is_paired_per_episode():
    rng = np.random.default_rng(0)
    base = rng.random(50) < 0.5
    succ = {"A": base | (rng.random(50) < 0.6), "P": base, "N": base, "G": base}
    seeds = {r: list(range(50)) for r in ROLES}
    out = episode_interaction(succ, seeds, BOOT)
    expected = np.mean(succ["A"].astype(float) - base)
    assert out["diff"] == pytest.approx(expected) and out["low"] <= out["diff"] <= out["high"]
    assert out["unit"] == "matched rollout episode" and out["n"] == 50
    seeds["N"] = list(range(1, 51))
    with pytest.raises(ValueError, match="not paired"):
        episode_interaction(succ, seeds, BOOT)


def probe_cells(losses, weights=None):
    w = [64.0] * len(losses["A"]) if weights is None else weights
    return {r: {"3,3": {"losses": list(losses[r]), "weights": list(w)}} for r in ROLES}


def test_batch_interaction_uses_weighted_matched_batches():
    g = np.random.default_rng(1)
    common = g.random(16) * 0.01
    losses = {"A": common + 0.002, "P": common + 0.001, "N": common + 0.005, "G": common}
    w = list(g.integers(100, 200, 16).astype(float))
    out = batch_interaction(probe_cells(losses, w), "3,3", BOOT)
    assert out["diff"] == pytest.approx(0.002 - 0.001 - 0.005)
    assert out["low"] == pytest.approx(out["diff"]) and out["high"] == pytest.approx(out["diff"])
    bad = probe_cells(losses, w)
    bad["G"]["3,3"]["weights"][0] += 1
    with pytest.raises(ValueError, match="not matched"):
        batch_interaction(bad, "3,3", BOOT)


def test_batch_gap_recovery_matches_the_weighted_means():
    g = np.random.default_rng(2)
    noise = g.normal(0, 1e-4, 16)
    losses = {"G": 0.0085 + noise, "N": 0.0135 + noise, "A": 0.0105 + noise, "P": 0.0120 + noise}
    out = batch_gap_recovery(probe_cells(losses), "3,3", BOOT)
    assert out["g"] == pytest.approx(0.6) and out["low"] <= 0.6 <= out["high"]
    assert out["undefined_resamples"] == 0
    assert batch_gap_recovery(probe_cells(losses), "3,3", BOOT, role="P")["g"] == pytest.approx(0.3)


# ---- importance and residual checks on synthetic artifacts ------------------------------

ALPHA = 25.0
LAYERS = ("trunk.x", "flow_head.y")


def grams(seed, d=8, tasks=3):
    g = torch.Generator().manual_seed(seed)
    out = []
    for t in range(tasks):
        R = torch.randn(d, 40, generator=g, dtype=torch.float64) * torch.linspace(3, 0.05, d).roll(3 * t)[:, None]
        out.append(R @ R.T)
    return out


def write_memories(run_dir, alpha=ALPHA, tasks=3, importance=True):
    """What SGP (or GPM, ``importance=False``) writes: memory_task{t}.pt with the history."""
    state = {name: (None, torch.zeros(0, dtype=torch.float64)) for name in LAYERS}
    per_layer_grams = {name: grams(i) for i, name in enumerate(LAYERS)}
    for t in range(tasks):
        bases = {}
        for name in LAYERS:
            M_old, prev = state[name]
            K = per_layer_grams[name][t]
            M, info = extend_basis(M_old, K, 0.95, name, neg_tol=1e-8)
            spectrum = info["residual_spectrum"]
            sigma = torch.zeros(M.shape[0], dtype=torch.float64)
            sigma[: spectrum.numel()] = spectrum.clamp(min=0).sqrt()
            meta = {"kind": "accumulated_memory"}
            if importance:
                upd = importance_update(M_old, K, spectrum, info["k_added"], prev, alpha)
                meta.update(importance=upd["lambda"], importance_update=upd, importance_update_task=t)
                state[name] = (M, upd["lambda"])
            else:
                state[name] = (M, prev)
            k = M.shape[1]
            bases[name] = SubspaceBasis(name, M.shape[0], 40, k, sigma, (0.95,), {0.95: k},
                                        {0.95: k / M.shape[0]}, M, meta)
        save_bases(run_dir / "method" / f"memory_task{t}.pt", bases,
                   {"kind": "accumulated_memory", "config": {"alpha": alpha if importance else None}})


def test_importance_check_recomputes_every_update(tmp_path):
    write_memories(tmp_path / "sgp")
    ok = importance_check(tmp_path / "sgp", 3, ALPHA, 1e-12)
    assert ok["passed"] and ok["n_checked"] == 6 and ok["n_failures"] == 0
    wrong_alpha = importance_check(tmp_path / "sgp", 3, 10.0, 1e-12)
    assert not wrong_alpha["passed"]
    assert any("alpha" in f["condition"] for f in wrong_alpha["failures"])


def tamper(path, layer, **changes):
    payload = torch.load(path, weights_only=False)
    meta = payload["bases"][layer]["meta"]
    for key, fn in changes.items():
        if key == "importance":
            meta["importance"] = fn(meta["importance"])
        else:
            meta["importance_update"][key] = fn(meta["importance_update"][key])
    torch.save(payload, path)


@pytest.mark.parametrize("changes, condition", [
    ({"importance": lambda x: x * 0.999}, "Eq. 10"),
    ({"lambda_task": lambda x: x * 0.99}, "Eq. 2"),
    ({"lambda_prev": lambda x: x * 0.5}, "lambda_prev"),
    ({"sigma_new": lambda x: x * 1.01}, "sigma-hat"),
])
def test_importance_check_catches_tampering(tmp_path, changes, condition):
    write_memories(tmp_path / "sgp")
    tamper(tmp_path / "sgp" / "method" / "memory_task1.pt", "trunk.x", **changes)
    out = importance_check(tmp_path / "sgp", 3, ALPHA, 1e-12)
    assert not out["passed"]
    assert any(condition in f["condition"] and f["task"] == 1 for f in out["failures"]), out["failures"]


def test_importance_identity_against_the_gpm_spectrum(tmp_path):
    for role, imp in (("G", False), ("P", True), ("A", True)):
        write_memories(tmp_path / role, tasks=1, importance=imp)
    dirs = {r: tmp_path / r for r in ("G", "P", "A")}
    out = importance_identity(dirs, ALPHA, 1e-12)
    assert out["passed"] and out["max_abs_diff"] <= 1e-15
    tamper(tmp_path / "A" / "method" / "memory_task0.pt", "flow_head.y", importance=lambda x: x * 0.9)
    out = importance_identity(dirs, ALPHA, 1e-12)
    assert not out["passed"] and not out["adaptive_equals_baseline_t1"] and out["t1_matches_gpm_spectrum"]
    write_memories(tmp_path / "P", tasks=1, alpha=10.0)
    assert not importance_identity(dirs, ALPHA, 1e-12)["t1_matches_gpm_spectrum"]


def test_residual_check(tmp_path):
    method = tmp_path / "method"
    method.mkdir()
    for t, worst in enumerate([0.0, 0.3, 0.9]):
        (method / f"gpm_logs_task{t}.json").write_text(json.dumps(
            {"residuals": {"a": {"max_residual_over_bound": worst}}}))
    assert residual_check(tmp_path, 3)["passed"]
    assert not residual_check(tmp_path, 4)["passed"]  # a missing log fails
    (method / "gpm_logs_task2.json").write_text(json.dumps(
        {"residuals": {"a": {"max_residual_over_bound": 1.2}}}))
    assert not residual_check(tmp_path, 3)["passed"]


# ---- the committed rule ------------------------------------------------------------------


def tblock(**overrides):
    block = {"mode": "registered", "margin": 0.15, "thresholds": [0.75, 0.63, 0.85, 0.83],
             "reference_run_id": "ref", "reference_result_sha256": "h",
             "seed_namespace_run_id": "ref", "task_keys": ["a", "b", "c", "d"]}
    return {**block, **overrides}


def test_all_four_threshold_blocks_must_agree():
    keys = {r: ["a", "b", "c", "d"] for r in ROLES}
    assert consume_all_blocks({r: tblock() for r in ROLES}, "ref", "h", keys) == tblock()
    blocks = {r: tblock() for r in ROLES}
    blocks["N"] = tblock(thresholds=[0.75, 0.63, 0.85, 0.80])
    with pytest.raises(ValueError, match="P vs N"):
        consume_all_blocks(blocks, "ref", "h", keys)


def test_committed_rule_matches_the_plan():
    from flowcl.experiments.adaptive_report import load_adaptive_config

    cfg = load_sgp_config()
    assert cfg["alpha"] == 25
    assert cfg["retention"] == {"transition_cell": [2, 1], "min_improvement": 0.20, "durable_cell": [3, 1]}
    assert cfg["plasticity"] == {"loss_cell": [3, 3], "min_gap_recovery": 0.5,
                                 "t4_cell": [3, 3], "t3_cell": [2, 2]}
    assert cfg["attribution"] == {"memory_cell": [2, 1], "scaling_rollout_cell": [3, 3]}
    assert cfg["identity"]["baseline_vs_gpm_stages"] == [0]
    assert cfg["identity"]["adaptive_vs_baseline_stages"] == [0, 1]
    assert cfg["replication"]["seeds"] == [0, 1, 2]
    adaptive = load_adaptive_config()
    assert cfg["energy"] == adaptive["energy"]
    for seed in (0, 1, 2):
        r = resolve_seed(cfg, seed)
        a = adaptive["seeds"][seed]
        # G and N are exactly the adaptive-GPM record's runs, reports and diagnostics.
        assert r["runs"]["G"] == a["baseline_run"] and r["runs"]["N"] == a["variant_run"]
        assert r["reference_run"] == a["reference_run"]
        assert r["sequence_reports"]["G"] == a["sequence_reports"]["baseline"]
        assert r["sequence_reports"]["N"] == a["sequence_reports"]["variant"]
        assert r["diagnostics"]["G"] == a["diagnostics"]["baseline"]
        assert r["diagnostics"]["N"] == a["diagnostics"]["variant"]
        assert r["runs"]["P"] == f"seq_hetero__sgp_projected_adam__seed{seed}"
        assert r["runs"]["A"] == f"seq_hetero__sgp_projected_adam_ne90__seed{seed}"
        # The per-seed probe report path must not clobber the shared probe_cells rule.
        assert r["probe_cells"]["cells"] == [[3, 3], [2, 1]]
        assert r["probe_report"] == f"sgp_probe_seed{seed}/report.json"
        assert limits(r, [0.75, 0.63, 0.85, 0.83]) == {
            "premise": 0.63, "durable": 0.63, "t4": 0.83, "t3": 0.85,
            "min_improvement": 0.20, "min_gap_recovery": 0.5}
    with pytest.raises(ValueError, match="not pre-registered"):
        resolve_seed(cfg, 3)


def test_exploratory_alpha_rule_matches_the_registered_rule_and_the_run_names():
    from flowcl.data.config import load_method_config
    from flowcl.methods.base import build_method
    from flowcl.train.continual import continual_run_id
    from flowcl.utils.libero_paths import repo_root

    base = load_sgp_config()
    cfg = load_sgp_config(repo_root() / "configs" / "analysis" / "sgp_a800.yaml")
    assert cfg["alpha"] == 800
    assert {k: v for k, v in cfg.items() if k not in ("alpha", "seeds", "replication")} == \
        {k: v for k, v in base.items() if k not in ("alpha", "seeds", "replication")}
    r, b = resolve_seed(cfg, 0), resolve_seed(base, 0)
    for role, method in (("P", "sgp_a800"), ("A", "sgp_a800_ne90")):
        display = build_method("sgp", **load_method_config(method)[1]).display_name
        assert r["runs"][role] == continual_run_id(display, "seq_hetero", 0)
    for role in ("G", "N"):
        assert r["runs"][role] == b["runs"][role]
        assert r["sequence_reports"][role] == b["sequence_reports"][role]
        assert r["diagnostics"][role] == b["diagnostics"][role]
    assert r["reference_run"] == b["reference_run"]
    assert r["probe_report"] == "sgp_a800_probe_seed0/report.json" and r["out"] == "sgp_a800_seed0/report.json"
    with pytest.raises(ValueError, match="not pre-registered"):
        resolve_seed(cfg, 1)
