"""Four-task sequence report: pre-registered criteria, pairing, capacity, wiring."""

from __future__ import annotations

import json
import math

import pytest
import torch

from flowcl.analysis.metrics import Estimate, success_estimate
from flowcl.analysis.subspace import SubspaceBasis, save_bases
from flowcl.envs.evaluation import EvaluationReport, TaskEvaluation
from flowcl.experiments.sequence_report import (
    RunView,
    build_report,
    c_trajectories,
    capacity_by_stage,
    classify_sequence,
    criteria_thresholds,
    load_report_config,
    paired_cells,
)

KEYS = ["libero_spatial/a", "libero_object/b", "libero_goal/c", "libero_10/d"]
REF_DIAG = [0.90, 0.78, 1.00, 0.98]
BOOT = {"n_resamples": 2000, "confidence": 0.95, "seed": 0}


def est(v, low=None, high=None):
    return Estimate(v, v if low is None else low, v if high is None else high, 50)


def test_thresholds_come_from_the_reference_diagonal():
    criteria = load_report_config()["criteria"]
    assert criteria_thresholds(REF_DIAG, criteria) == [0.75, 0.63, 0.85, 0.83]
    with pytest.raises(ValueError, match="reference run changed"):
        criteria_thresholds([0.90, 0.80, 1.00, 0.98], criteria)


def grid(diag, final_row):
    """Estimates with the given diagonal and final row (everything else 0)."""
    out = {(i, j): est(0.0) for i in range(4) for j in range(4)}
    for j, v in enumerate(diag):
        out[(j, j)] = est(v)
    for j, v in enumerate(final_row):
        out[(3, j)] = est(v)
    return out


THRESH = [0.75, 0.63, 0.85, 0.83]


def test_all_criteria_met():
    o = classify_sequence(grid([0.9, 0.8, 0.9, 0.9], [0.8, 0.7, 0.9, 0.9]), THRESH, [2, 3])
    assert not o["sgp_fallback_triggered"] and not o["plasticity_failures"] and not o["retention_failures"]


def test_plasticity_collapse_on_t3_triggers_the_fallback():
    o = classify_sequence(grid([0.9, 0.8, 0.5, 0.9], [0.8, 0.7, 0.5, 0.9]), THRESH, [2, 3])
    assert o["sgp_fallback_triggered"] and o["sgp_fallback_tasks"] == [2]


def test_a_t2_plasticity_failure_alone_does_not_trigger_the_fallback():
    o = classify_sequence(grid([0.9, 0.5, 0.9, 0.9], [0.8, 0.5, 0.9, 0.9]), THRESH, [2, 3])
    assert o["plasticity_failures"] == [1] and not o["sgp_fallback_triggered"]
    assert o["retention_failures"] == [1]


def test_thresholds_are_inclusive_and_borderline_is_flagged():
    estimates = grid([0.75, 0.63, 0.85, 0.83], [0.75, 0.63, 0.85, 0.83])
    estimates[(3, 0)] = est(0.74, 0.62, 0.86)
    o = classify_sequence(estimates, THRESH, [2, 3])
    assert all(v["ok"] for v in o["plasticity"].values())
    assert o["retention_failures"] == [0] and o["borderline"]["final_retention"] == [0]


def view(succ_by_cell, seeds_by_task=None):
    evals = {}
    for i in range(4):
        tasks = []
        for j, key in enumerate(KEYS):
            s = succ_by_cell[(i, j)]
            seeds = (seeds_by_task or {}).get(j, list(range(len(s))))
            tasks.append(TaskEvaluation(key, s, [100] * len(s), seeds, success_estimate(s)))
        evals[i] = EvaluationReport("ns", i, tasks)
    return RunView(None, {"task_keys": KEYS}, evals)


def test_paired_cells_require_identical_seeds():
    a = view({(i, j): [True, False] for i in range(4) for j in range(4)})
    b = view({(i, j): [False, False] for i in range(4) for j in range(4)})
    diffs = paired_cells(a, b, BOOT)
    assert diffs["0,0"]["diff"] == pytest.approx(0.5)
    c = view({(i, j): [False, False] for i in range(4) for j in range(4)}, seeds_by_task={2: [7, 8]})
    with pytest.raises(ValueError, match="not paired"):
        paired_cells(a, c, BOOT)


def info(rho, proj, exhausted=False, k_added=1):
    return {"rho_after": rho, "proj_energy_fraction": proj, "k_added": k_added,
            "capacity_exhausted": exhausted}


def test_capacity_aggregation_keeps_occupancy_and_energy_apart():
    history = {
        "0": {"trunk.a": info(0.4, 0.0), "flow_head.b": info(0.05, 0.0)},
        "1": {"trunk.a": info(0.6, 0.9), "flow_head.b": info(1.0, 0.97, exhausted=True)},
    }
    cap = capacity_by_stage(history)
    assert cap["1"]["trunk"]["median_rho"] == 0.6
    assert cap["1"]["trunk"]["median_proj_energy_fraction"] == 0.9
    assert cap["1"]["decoder"]["capacity_exhausted"] == 1
    assert cap["1"]["exhausted_layers"] == ["flow_head.b"]
    assert cap["0"]["decoder"]["median_free_fraction"] == pytest.approx(0.95)


def test_c_trajectories_skip_undefined_values():
    logs = {"gradient_c": {"0": {"trunk.a": 0.9, "flow_head.b": math.nan}, "100": {"trunk.a": 0.8, "flow_head.b": 0.7}},
            "update_c": {}}
    out = c_trajectories(logs)
    assert out["gradient_c"]["mean_of_medians"]["trunk"] == pytest.approx(0.85)
    assert out["gradient_c"]["mean_of_medians"]["decoder"] == pytest.approx(0.7)
    assert out["update_c"]["mean_of_medians"]["trunk"] is None


# ---- wiring: build_report on synthetic run directories ---------------------------


def write_run(root, name, diag, final_row, with_memory):
    run_dir = root / name
    (run_dir / "eval").mkdir(parents=True)
    succ = {}
    for i in range(4):
        for j in range(4):
            if i == j:
                p = diag[j]
            elif i == 3:
                p = final_row[j]
            else:
                p = 0.0
            n_true = round(p * 50)
            succ[(i, j)] = [True] * n_true + [False] * (50 - n_true)
    for i in range(4):
        tasks = [TaskEvaluation(k, succ[(i, j)], [100] * 50, list(range(50)), success_estimate(succ[(i, j)]))
                 for j, k in enumerate(KEYS)]
        EvaluationReport("ns", i, tasks, method_run_id=name, seed_namespace_run_id="ref").save(
            run_dir / "eval" / f"stage{i}.json")
    (run_dir / "result.json").write_text(json.dumps({
        "task_keys": KEYS, "method": name, "method_run_id": name, "seed_namespace_run_id": "ref",
        "metrics": {"F_1": 0.5}, "t1_pairing": {"rel_weight_diff": 0.01, "passed": True}}))
    (run_dir / "git_sha").write_text("abc123\n")
    if with_memory:
        history = {}
        for t in range(4):
            history[str(t)] = {"trunk.x": info(0.4 + 0.1 * t, 0.9), "flow_head.y": info(0.05 * (t + 1), 0.95)}
            M = torch.linalg.qr(torch.randn(8, 2 + t, dtype=torch.float64))[0]
            basis = SubspaceBasis("trunk.x", 8, 10, 2 + t, torch.zeros(8, dtype=torch.float64),
                                  (0.95,), {0.95: 2 + t}, {0.95: (2 + t) / 8}, M)
            save_bases(run_dir / "method" / f"memory_task{t}.pt", {"trunk.x": basis},
                       {"memory_history": history})
            (run_dir / "method" / f"gpm_logs_task{t}.json").write_text(json.dumps({
                "projected": t > 0,
                "gradient_c": {"0": {"trunk.x": 0.9, "flow_head.y": 0.8}},
                "update_c": {"0": {"trunk.x": 0.1, "flow_head.y": 0.1}},
                "residuals": {"trunk.x": {"max_residual_over_bound": 0.02}}}))
    return run_dir


def test_build_report_end_to_end_on_synthetic_runs(tmp_path):
    ref = write_run(tmp_path, "ref", REF_DIAG, [0.0, 0.0, 0.0, 0.98], with_memory=False)
    gpm = write_run(tmp_path, "gpm", [0.9, 0.88, 0.5, 0.9], [0.8, 0.7, 0.5, 0.9], with_memory=True)
    report = build_report(gpm, ref, pilot_json=None, verify_checkpoints=False)
    assert report["criteria"]["thresholds"] == [0.75, 0.63, 0.85, 0.83]
    assert report["outcome"]["sgp_fallback_triggered"] and report["outcome"]["sgp_fallback_tasks"] == [2]
    assert report["paired_cells"]["3,0"]["diff"] == pytest.approx(0.8)
    assert report["capacity"]["3"]["trunk"]["median_rho"] == pytest.approx(0.7)
    assert report["dynamics"]["2"]["worst_residual_over_bound"] == 0.02
    checks = report["provenance_checks"]
    assert checks["clean_git_sha"]["passed"] and checks["seed_namespace"]["passed"]
    assert checks["occupancy_non_decreasing"]["passed"] and checks["residuals_within_bound"]["passed"]
    json.dumps(report, default=str)  # serialisable
