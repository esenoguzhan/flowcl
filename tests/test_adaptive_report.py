"""Adaptive-GPM outcome rule: verdict branches, per-layer energy check, boundaries."""

from __future__ import annotations

import pytest

from flowcl.experiments.adaptive_report import (
    REPLICATION,
    VERDICTS,
    classify_adaptive,
    classify_replication,
    energy_check,
    interference_check,
    limits,
    load_adaptive_config,
    resolve_seed,
    rollout_checks,
)

CFG = load_adaptive_config()
ENERGY = CFG["energy"]
LIM0 = limits(resolve_seed(CFG, 0))
LIM1 = limits(resolve_seed(CFG, 1))


def checks(**overrides):
    base = {"identity": True, "energy": True, "premise": True, "interference": True,
            "t3_plasticity": True, "transition_gain": True, "durable_retention": True,
            "t4_plasticity": True}
    return {**base, **overrides}


# ---- verdict ---------------------------------------------------------------------------


def test_gates_come_first():
    assert classify_adaptive(checks(identity=False, energy=False))["verdict"] == "invalid_comparison"
    assert classify_adaptive(checks(energy=False, premise=False))["verdict"] == "invalid_implementation"
    out = classify_adaptive(checks(premise=False, interference=False))
    assert out["verdict"] == "not_applicable_baseline_retained" and out["retention_verdict"] is None


@pytest.mark.parametrize("t3, gain, expected", [
    (True, False, "not_supported"),
    (False, True, "trade_off"),
    (False, False, "inconclusive"),
    (True, True, "durable_support"),
])
def test_t3_plasticity_and_gain_are_judged_jointly(t3, gain, expected):
    out = classify_adaptive(checks(t3_plasticity=t3, transition_gain=gain))
    assert out["verdict"] == expected and out["text"] == VERDICTS[expected]


@pytest.mark.parametrize("durable, t4, flags", [
    (False, True, ["durable Object retention below threshold"]),
    (True, False, ["T4 plasticity below threshold"]),
    (False, False, ["durable Object retention below threshold", "T4 plasticity below threshold"]),
])
def test_durable_failures_leave_transition_support_with_flags(durable, t4, flags):
    out = classify_adaptive(checks(durable_retention=durable, t4_plasticity=t4))
    assert out["verdict"] == "transition_support" and out["flags"] == flags


def test_weak_manipulation_keeps_an_exploratory_retention_verdict():
    out = classify_adaptive(checks(interference=False))
    assert out["verdict"] == "manipulation_weak" and out["exploratory"]
    assert out["retention_verdict"] == "durable_support"
    out = classify_adaptive(checks(interference=False, transition_gain=False))
    assert out["retention_verdict"] == "not_supported"


# ---- per-layer energy check -------------------------------------------------------------


def row(p, captured, target=None):
    t = min(1.0, max(0.95, p + 0.9 * (1 - p))) if target is None else target
    return {"proj_energy_fraction": p, "captured_energy_fraction": captured, "target_fraction": t}


def history(rows_by_task):
    return {str(t): rows for t, rows in rows_by_task.items()}


def test_energy_check_passes_and_marks_layers_without_new_energy():
    h = history({t: {"a": row(0.9, 0.99), "b": row(1 - 1e-13, 1 - 1e-13)} for t in (1, 2, 3)})
    out = energy_check(h, ENERGY)
    assert out["passed"] and out["n_applicable"] == 3 and out["n_not_applicable"] == 3
    assert out["min_new_energy_share"] == pytest.approx(0.9)


def test_energy_check_catches_the_eps_floor_being_missed():
    # p = 1/3: the floor 0.95 binds; 0.94 meets the 90% new-energy share but not the target.
    h = history({1: {"a": row(1 / 3, 0.94)}, 2: {"a": row(0.9, 0.99)}, 3: {"a": row(0.9, 0.99)}})
    out = energy_check(h, ENERGY)
    assert not out["passed"]
    assert [f["condition"] for f in out["failures"]] == ["(a) target not met"]


def test_energy_check_catches_a_share_below_ninety_percent_in_one_layer():
    good = {f"l{i}": row(0.9, 0.99) for i in range(5)}
    h = history({1: {**good, "bad": row(0.9, 0.985)}, 2: good, 3: good})
    out = energy_check(h, ENERGY)
    conditions = {f["condition"] for f in out["failures"]}
    assert not out["passed"] and conditions == {"(a) target not met", "(b) new-energy share below f"}
    assert {f["layer"] for f in out["failures"]} == {"bad"}


def test_energy_check_catches_a_recorded_target_that_disagrees():
    h = history({t: {"a": row(0.9, 0.99, target=0.95)} for t in (1, 2, 3)})
    out = energy_check(h, ENERGY)
    assert not out["passed"]
    assert {f["condition"] for f in out["failures"]} == {"recorded target differs"}


def test_energy_check_tolerance_and_missing_extensions():
    ok = history({t: {"a": row(0.9, 0.99 - 5e-7)} for t in (1, 2, 3)})
    assert energy_check(ok, ENERGY)["passed"]
    bad = history({t: {"a": row(0.9, 0.99 - 2e-6)} for t in (1, 2, 3)})
    assert not energy_check(bad, ENERGY)["passed"]
    missing = history({1: {"a": row(0.9, 0.99)}})
    assert not energy_check(missing, ENERGY)["passed"]


# ---- rollout and interference boundaries ------------------------------------------------


def test_rollout_boundaries():
    at = rollout_checks({"diff": 0.20, "low": 0.02}, 0.0, 0.85, 0.63, 0.83, LIM0)
    assert all(at.values())  # every threshold inclusive
    assert not rollout_checks({"diff": 0.20, "low": 0.0}, 0.0, 0.85, 0.63, 0.83, LIM0)["transition_gain"]
    assert not rollout_checks({"diff": 0.18, "low": 0.02}, 0.0, 0.85, 0.63, 0.83, LIM0)["transition_gain"]
    below = rollout_checks({"diff": 0.5, "low": 0.3}, 0.0, 0.84, 0.62, 0.82, LIM0)
    assert not (below["t3_plasticity"] or below["durable_retention"] or below["t4_plasticity"])
    # Rates are k/50: a difference of 10/50 computed in floats still counts as +20 pp.
    assert rollout_checks({"diff": 0.6 - 0.4, "low": 0.02}, 0.0, 0.85, 0.63, 0.83, LIM0)["transition_gain"]


def test_premise_needs_the_baseline_strictly_below_the_object_threshold():
    ok = {"diff": 0.5, "low": 0.3}
    assert rollout_checks(ok, 0.62, 0.9, 0.9, 0.9, LIM0)["premise"]
    assert not rollout_checks(ok, 0.63, 0.9, 0.9, 0.9, LIM0)["premise"]      # at the threshold
    assert rollout_checks(ok, 0.74, 0.9, 0.9, 0.9, LIM1)["premise"]
    assert not rollout_checks(ok, 0.76, 0.9, 0.9, 0.9, LIM1)["premise"]


def test_seed_resolution_and_per_seed_limits():
    s0, s1 = resolve_seed(CFG, 0), resolve_seed(CFG, 1)
    assert s0["variant_run"] == "seq_hetero__gpm_projected_adam_ne90__seed0"
    assert s1["baseline_run"] == "seq_hetero__gpm_projected_adam__seed1"
    assert s1["diagnostics"] == {"variant": "forgetting_diag_ne90_seed1/report.json",
                                 "baseline": "forgetting_diag_seed1/report.json"}
    assert s0["out"] != s1["out"]
    # Seed 0 keeps exactly the thresholds its run was judged by.
    assert LIM0 == {"min_improvement": 0.20, "premise": 0.63, "durable": 0.63, "t3": 0.85,
                    "t4": 0.83, "goal": 0.85}
    assert (LIM1["premise"], LIM1["durable"], LIM1["t3"], LIM1["t4"]) == (0.75, 0.75, 0.85, 0.83)
    with pytest.raises(ValueError, match="not pre-registered"):
        resolve_seed(CFG, 7)


# ---- replication (symmetric) ------------------------------------------------------------


def v(verdict):
    return {"verdict": verdict}


@pytest.mark.parametrize("verdicts, expected", [
    ({0: v("durable_support"), 1: v("durable_support")}, "durably_replicated"),
    ({0: v("durable_support"), 1: v("transition_support")}, "replicated"),
    ({0: v("durable_support"), 1: v("not_supported")}, "not_replicated"),
    ({0: v("not_supported"), 1: v("durable_support")}, "not_replicated"),       # symmetric
    ({0: v("not_supported"), 1: v("not_supported")}, "consistently_not_supported"),
    ({0: v("durable_support"), 1: v("manipulation_weak")}, "inconclusive"),
    ({0: v("trade_off"), 1: v("durable_support")}, "inconclusive"),
    ({0: v("durable_support"), 1: v("inconclusive")}, "inconclusive"),
    ({0: v("durable_support"), 1: v("not_applicable_baseline_retained")}, "not_applicable"),
    ({0: v("invalid_comparison"), 1: v("durable_support")}, "inconclusive_invalid"),
    # precedence: invalid over not-applicable over inconclusive
    ({0: v("invalid_implementation"), 1: v("not_applicable_baseline_retained")}, "inconclusive_invalid"),
    ({0: v("not_applicable_baseline_retained"), 1: v("trade_off")}, "not_applicable"),
    ({0: v("durable_support"), 1: v("durable_support"), 2: v("transition_support")}, "replicated"),
    ({0: v("durable_support"), 1: v("durable_support"), 2: v("not_supported")}, "not_replicated"),
])
def test_replication_rule(verdicts, expected):
    out = classify_replication(verdicts)
    assert out["replication"] == expected and out["text"] == REPLICATION[expected]


def test_replication_needs_two_seeds():
    with pytest.raises(ValueError, match="two seeds"):
        classify_replication({0: v("durable_support")})


def diag(run, trunk, decoder):
    return {"method_run_id": run, "comparisons": {"primary": {"reported": {"median_r": {
        "target": {"direct": {"trunk": trunk, "decoder": decoder}}}}}}}


def test_interference_ratio_is_inclusive_in_both_halves():
    base = diag("b", 0.10, 0.10)
    assert interference_check(diag("v", 0.07, 0.07), base, CFG["interference"], "v", "b")["passed"]
    assert not interference_check(diag("v", 0.07, 0.08), base, CFG["interference"], "v", "b")["passed"]
    with pytest.raises(ValueError, match="expected"):
        interference_check(diag("x", 0.07, 0.07), base, CFG["interference"], "v", "b")


def test_committed_rule_matches_the_plan():
    assert CFG["interference"]["max_ratio"] == 0.70
    assert CFG["retention"]["min_improvement"] == 0.20
    assert CFG["seeds"][0]["thresholds"] == [0.75, 0.63, 0.85, 0.83]
    assert CFG["seeds"][1]["thresholds"] == [0.81, 0.75, 0.85, 0.83]
    assert CFG["premise"]["baseline_cell"] == CFG["retention"]["transition_cell"] == [2, 1]
