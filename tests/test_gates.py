"""§10.3 gate verdicts and the §4.1 one-time encoder decision."""

from __future__ import annotations

import json

import pytest

from flowcl.analysis.gates import (
    ESCAPE_HATCH_THRESHOLD,
    GATE0_SUCCESS_THRESHOLD,
    active_encoder_decision,
    gate0,
    record_escape_hatch,
)
from flowcl.analysis.metrics import Estimate


def est(value: float, low: float | None = None, high: float | None = None) -> Estimate:
    return Estimate(
        value=value,
        low=low if low is not None else max(0.0, value - 0.12),
        high=high if high is not None else min(1.0, value + 0.12),
        n=50,
    )


def test_thresholds_match_the_spec():
    """§10.3 Gate 0 is 80%; §4.1's encoder escape hatch is ~70%. Not the same number."""
    assert GATE0_SUCCESS_THRESHOLD == 0.80
    assert ESCAPE_HATCH_THRESHOLD == 0.70
    assert ESCAPE_HATCH_THRESHOLD < GATE0_SUCCESS_THRESHOLD


def test_gate0_passes_when_every_task_clears_the_threshold():
    result = gate0({"a": est(0.86), "b": est(0.80)})
    assert result.passed
    assert result.evidence["failing_tasks"] == []
    assert "PASS" in result.describe()


def test_gate0_requires_every_task_not_the_average():
    """A strong task must not carry a failing one: §10.3 says "each chosen task"."""
    result = gate0({"good": est(0.98), "bad": est(0.40)})
    assert not result.passed
    assert result.evidence["failing_tasks"] == ["bad"]


def test_gate0_uses_the_point_estimate_not_the_ci_lower_bound():
    """At n=50 a true 80% rate has a lower bound near 68%.

    Requiring the interval to clear 80% would fail policies that meet the spec, so the
    verdict is on the point estimate and the straddling interval is flagged instead.
    """
    result = gate0({"a": est(0.82, low=0.70, high=0.92)})
    assert result.passed
    assert result.evidence["per_task"]["a"]["borderline"] is True


def test_gate0_recommends_the_encoder_swap_only_below_seventy_percent():
    low = gate0({"a": est(0.55)})
    assert not low.passed
    assert low.evidence["escape_hatch_candidates"] == ["a"]
    assert "escape-hatch threshold" in low.notes


def test_gate0_does_not_recommend_the_swap_in_the_seventy_to_eighty_band():
    """§4.1's swap is a one-time budget; the 70-80% band is a recipe problem."""
    mid = gate0({"a": est(0.74)})
    assert not mid.passed
    assert mid.evidence["escape_hatch_candidates"] == []
    assert "NOT indicated" in mid.notes


def test_gate0_rejects_empty_input():
    with pytest.raises(ValueError, match="no task estimates"):
        gate0({})


def test_gate_result_round_trips_to_disk(tmp_path):
    result = gate0({"a": est(0.9)})
    path = result.save(tmp_path / "gate0.json")
    payload = json.loads(path.read_text())
    assert payload["gate"] == 0
    assert payload["passed"] is True
    assert payload["criterion"].startswith(">= 80%")
    assert "recorded" in payload


# ---- §4.1: decide once, record it ---------------------------------------------


def test_escape_hatch_is_absent_by_default(tmp_path):
    assert active_encoder_decision(results_root=tmp_path) is None


def test_escape_hatch_records_the_decision(tmp_path):
    record_escape_hatch(
        reason="frozen DINOv2-S gave 55% on libero_spatial",
        evidence={"success_rate": 0.55},
        results_root=tmp_path,
    )
    decision = active_encoder_decision(results_root=tmp_path)
    assert decision["encoder"] == "resnet18_scratch"
    assert decision["spec_section"] == "4.1"


def test_escape_hatch_refuses_a_second_decision(tmp_path):
    """§4.1: "Decide this in Month 1, once. Do not silently vary the encoder."."""
    record_escape_hatch("first", {}, results_root=tmp_path)
    with pytest.raises(FileExistsError, match="already recorded"):
        record_escape_hatch("second", {}, results_root=tmp_path)


def test_escape_hatch_lives_outside_any_single_run(tmp_path):
    """The decision applies to every later run, so it must not be run-scoped."""
    path = record_escape_hatch("x", {}, results_root=tmp_path)
    assert path.parent == tmp_path
    assert path.name == "encoder_decision.json"


# ---- Gate 2 ---------------------------------------------------------------------


def gate2_inputs(rhos: list[float]):
    from flowcl.analysis.gates import gate2

    names = [f"layer{i}" for i in range(len(rhos))]
    return gate2, dict(zip(names, rhos)), {n: "trunk_mlp" for n in names}, {
        n: 512 for n in names
    }


def test_gate2_thresholds_are_the_recorded_decision():
    from flowcl.analysis.gates import (
        GATE2_DEFAULT_EPS,
        GATE2_MAX_SATURATED_FRACTION,
        GATE2_SATURATION_RHO,
    )

    assert GATE2_DEFAULT_EPS == 0.95
    assert GATE2_SATURATION_RHO == 0.90
    assert GATE2_MAX_SATURATED_FRACTION == 0.50


def test_gate2_passes_when_few_layers_are_saturated():
    gate2, rhos, groups, d_in = gate2_inputs([0.2, 0.3, 0.95, 0.4])
    result = gate2(rhos, groups, d_in)
    assert result.passed
    assert result.evidence["saturated_layers"] == ["layer2"]
    assert result.evidence["saturated_fraction"] == 0.25


def test_gate2_fails_on_a_strict_majority_of_saturated_layers():
    gate2, rhos, groups, d_in = gate2_inputs([0.91, 0.99, 0.95, 0.4])
    result = gate2(rhos, groups, d_in)
    assert not result.passed
    assert "GPM" in result.notes


def test_gate2_exactly_half_saturated_passes():
    """'Most' is a strict majority."""
    gate2, rhos, groups, d_in = gate2_inputs([0.90, 0.99, 0.2, 0.3])
    result = gate2(rhos, groups, d_in)
    assert result.evidence["saturated_fraction"] == 0.5
    assert result.passed


def test_gate2_reports_views_without_letting_them_decide():
    gate2, rhos, groups, d_in = gate2_inputs([0.2, 0.3])
    result = gate2(
        rhos,
        groups,
        d_in,
        alternative_rhos={
            "valid": {"layer0": 0.2, "layer1": 0.95},
            "all": {"layer0": 0.2, "layer1": 0.3},
        },
    )
    assert result.passed
    assert result.evidence["classification_changes"] == {"valid_vs_all": ["layer1"]}
    assert result.evidence["per_layer"]["layer1"]["rho_valid"] == 0.95


def test_gate2_flags_small_d_groups_and_summarises_per_group():
    from flowcl.analysis.gates import gate2

    result = gate2(
        {"trunk.state_projection": 1.0, "a": 0.2, "b": 0.4},
        {"trunk.state_projection": "trunk_input", "a": "trunk_mlp", "b": "trunk_mlp"},
        {"trunk.state_projection": 8, "a": 512, "b": 512},
    )
    assert result.evidence["per_layer"]["trunk.state_projection"]["small_d"]
    assert not result.evidence["per_layer"]["a"]["small_d"]
    assert result.evidence["per_group"]["trunk_mlp"]["median_rho"] == pytest.approx(0.3)


def test_gate2_rejects_out_of_range_rho():
    gate2, rhos, groups, d_in = gate2_inputs([0.0])
    with pytest.raises(ValueError, match="rho_l"):
        gate2(rhos, groups, d_in)
