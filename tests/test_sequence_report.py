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
    basis_overlap,
    build_report,
    c_trajectories,
    capacity_by_stage,
    classify_sequence,
    criteria_thresholds,
    episode_transitions,
    load_report_config,
    memory_chain_step,
    memory_matrix,
    new_energy_protected,
    paired_cells,
    t1_memory_overlap,
)

KEYS = ["libero_spatial/a", "libero_object/b", "libero_goal/c", "libero_10/d"]
REF_DIAG = [0.90, 0.78, 1.00, 0.98]
BOOT = {"n_resamples": 2000, "confidence": 0.95, "seed": 0}


def est(v, low=None, high=None):
    return Estimate(v, v if low is None else low, v if high is None else high, 50)


def test_thresholds_come_from_the_reference_diagonal():
    criteria = load_report_config()["criteria"]
    assert criteria_thresholds(REF_DIAG, criteria, "seq_hetero__seq_ft__seed0") == [0.75, 0.63, 0.85, 0.83]
    with pytest.raises(ValueError, match="reference run changed"):
        criteria_thresholds([0.90, 0.80, 1.00, 0.98], criteria, "seq_hetero__seq_ft__seed0")


def test_thresholds_are_per_paired_reference_and_must_be_preregistered():
    criteria = load_report_config()["criteria"]
    seed1 = [0.96, 0.90, 1.00, 0.98]
    assert criteria_thresholds(seed1, criteria, "seq_hetero__seq_ft__seed1") == [0.81, 0.75, 0.85, 0.83]
    with pytest.raises(ValueError, match="reference run changed"):
        criteria_thresholds(REF_DIAG, criteria, "seq_hetero__seq_ft__seed1")  # seed 0's diagonal
    with pytest.raises(ValueError, match="pre-registered"):
        criteria_thresholds(seed1, criteria, "seq_hetero__seq_ft__seed2")


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


def info(rho, proj, exhausted=False, k_added=1, captured=None, k_before=0, k_after=None):
    return {"rho_after": rho, "proj_energy_fraction": proj, "k_added": k_added,
            "capacity_exhausted": exhausted,
            "captured_energy_fraction": max(proj, 0.95) if captured is None else captured,
            "k_before": k_before, "k_after": k_before + k_added if k_after is None else k_after}


def test_capacity_aggregation_keeps_occupancy_and_energy_apart():
    history = {
        "0": {"trunk.a": info(0.4, 0.0), "flow_head.b": info(0.05, 0.0)},
        "1": {"trunk.a": info(0.6, 0.9), "flow_head.b": info(1.0, 0.97, exhausted=True, captured=1.0)},
    }
    cap = capacity_by_stage(history)
    assert cap["1"]["trunk"]["median_rho"] == 0.6
    assert cap["1"]["trunk"]["median_proj_energy_fraction"] == 0.9
    assert cap["1"]["decoder"]["capacity_exhausted"] == 1
    assert cap["1"]["exhausted_layers"] == ["flow_head.b"]
    assert cap["0"]["decoder"]["median_free_fraction"] == pytest.approx(0.95)


def test_new_energy_share_separates_first_and_later_tasks_per_group():
    # T1: nothing in memory, 95% captured -> 95% of its (all-new) energy protected.
    assert new_energy_protected(info(0.4, 0.0, captured=0.95)) == pytest.approx(0.95)
    # A later task with 90% already inside memory, extended to 95%: half its new energy.
    assert new_energy_protected(info(0.5, 0.90, captured=0.95)) == pytest.approx(0.5)
    # Everything already in memory: the share is undefined, not 0 or 1.
    assert new_energy_protected(info(0.5, 1.0, captured=1.0)) is None

    history = {"1": {"trunk.a": info(0.5, 0.90, captured=0.95),
                     "trunk.b": info(0.5, 0.80, captured=0.95),
                     "flow_head.c": info(0.1, 1.0, captured=1.0)}}
    groups = {"trunk.a": "trunk_attn", "trunk.b": "trunk_mlp", "flow_head.c": "decoder_mlp"}
    cap = capacity_by_stage(history, groups)
    assert cap["1"]["trunk"]["median_new_energy_protected"] == pytest.approx((0.5 + 0.75) / 2)
    assert cap["1"]["trunk_mlp"]["median_new_energy_protected"] == pytest.approx(0.75)
    assert cap["1"]["trunk"]["median_unprotected_energy"] == pytest.approx(0.05)
    assert cap["1"]["decoder_mlp"]["median_new_energy_protected"] is None


# ---- episode transitions --------------------------------------------------------


def test_episode_transitions_count_paired_episodes_and_timeouts():
    succ = {(i, j): [False] * 4 for i in range(4) for j in range(4)}
    succ[(1, 1)] = [True, True, False, False]
    succ[(2, 1)] = [False, True, True, False]
    out = episode_transitions(view(succ), max_steps=100)  # view() sets n_steps = 100
    row = out[KEYS[1]]["1->2"]
    assert (row["both"], row["lost"], row["gained"]) == (1, 1, 1)
    assert row["lost_episodes"] == [0] and row["gained_episodes"] == [2]
    assert row["shared_horizon_change"] == {"median": 0, "mean": 0.0}
    assert row["failures"]["after"] == {"n": 2, "timeouts": 2}
    assert list(out[KEYS[0]]) == ["0->1", "1->2", "2->3", "0->3"]
    assert list(out[KEYS[2]]) == ["2->3"] and out[KEYS[3]] == {}
    assert out[KEYS[3 - 1]]["2->3"]["shared_horizon_change"] is None


def test_episode_transitions_refuse_unpaired_stages():
    succ = {(i, j): [True, False] for i in range(4) for j in range(4)}
    run = view(succ)
    run.evals[2].tasks[0] = TaskEvaluation(KEYS[0], [True, False], [100, 100], [7, 8],
                                          success_estimate([True, False]))
    with pytest.raises(ValueError, match="not paired"):
        episode_transitions(run)


# ---- memory chaining and basis overlap -------------------------------------------


def orthonormal(d, k, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.linalg.qr(torch.randn(d, k, generator=g, dtype=torch.float64))[0]


def chain_history(k0, k1):
    return ({"L": {"k_before": 0, "k_after": k0}}, {"L": {"k_before": k0, "k_after": k1}})


def test_memory_chain_accepts_an_appended_memory():
    Q = orthonormal(10, 5)
    h0, h1 = chain_history(3, 5)
    step = memory_chain_step({"L": Q[:, :3]}, {"L": Q}, h0, h1)
    assert step["passed"] and step["prefix_identical"]
    assert step["min_containment"] == pytest.approx(1.0)


def test_memory_chain_rejects_an_unrelated_basis_of_the_right_rank():
    h0, h1 = chain_history(3, 5)
    step = memory_chain_step({"L": orthonormal(10, 3, seed=1)}, {"L": orthonormal(10, 5, seed=2)}, h0, h1)
    assert not step["passed"] and step["not_contained"] == ["L"]
    assert step["rank_mismatch"] == []  # ranks alone would have passed


def test_memory_chain_accepts_a_rotation_inside_the_next_memory():
    Q = orthonormal(10, 5)
    R = orthonormal(3, 3, seed=3)  # rotate the old basis within its own span
    h0, h1 = chain_history(3, 5)
    step = memory_chain_step({"L": Q[:, :3] @ R}, {"L": Q}, h0, h1)
    assert step["passed"] and not step["prefix_identical"]


def test_memory_chain_flags_rank_bookkeeping_errors():
    Q = orthonormal(10, 5)
    h0, h1 = chain_history(3, 5)
    h1["L"]["k_before"] = 2
    assert memory_chain_step({"L": Q[:, :3]}, {"L": Q}, h0, h1)["rank_mismatch"] == ["L"]


def gate2_like_basis(U, ranks):
    thresholds = tuple(sorted(ranks))
    return SubspaceBasis("L", U.shape[0], 100, U.shape[1], torch.zeros(U.shape[0], dtype=torch.float64),
                         thresholds, dict(ranks), {e: k / U.shape[0] for e, k in ranks.items()}, U)


def test_memory_matrix_takes_the_eps_prefix_of_a_multi_threshold_basis():
    U = orthonormal(10, 7)  # stored up to eps = 0.99
    basis = gate2_like_basis(U, {0.90: 2, 0.95: 4, 0.99: 7})
    assert memory_matrix(basis, 0.95).shape == (10, 4)
    assert torch.equal(memory_matrix(basis, 0.95), U[:, :4])
    with pytest.raises(ValueError, match="say which eps"):
        memory_matrix(basis)
    with pytest.raises(ValueError, match="no basis at eps"):
        memory_matrix(basis, 0.97)


def test_basis_overlap_is_directional():
    Q = orthonormal(10, 5)
    same = basis_overlap(Q[:, :3], Q[:, :3])
    assert same["s_over_k_a"] == pytest.approx(1.0) and same["s_over_k_b"] == pytest.approx(1.0)
    sub = basis_overlap(Q[:, :3], Q)  # a inside b, b larger
    assert sub["s_over_k_a"] == pytest.approx(1.0) and sub["s_over_k_b"] == pytest.approx(0.6)


def test_t1_overlap_compares_at_the_memory_eps():
    U = orthonormal(10, 7)
    gate2 = {"L": gate2_like_basis(U, {0.95: 4, 0.99: 7})}
    memory = {"L": gate2_like_basis(U[:, :4], {0.95: 4})}
    out = t1_memory_overlap(gate2, memory, 0.95)
    row = out["per_layer"]["L"]
    assert (row["k_a"], row["k_b"]) == (4, 4)  # not Gate 2's full 0.99 U
    assert row["s_over_k_a"] == pytest.approx(1.0) and row["s_over_k_b"] == pytest.approx(1.0)


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
        Q = orthonormal(8, 5)  # every task appends one direction to the last memory
        for t in range(4):
            k_before = 0 if t == 0 else 1 + t
            history[str(t)] = {
                "trunk.x": info(0.4 + 0.1 * t, 0.9, k_before=k_before, k_after=2 + t),
                "flow_head.y": info(0.05 * (t + 1), 0.95),
            }
            basis = SubspaceBasis("trunk.x", 8, 10, 2 + t, torch.zeros(8, dtype=torch.float64),
                                  (0.95,), {0.95: 2 + t}, {0.95: (2 + t) / 8}, Q[:, :2 + t])
            save_bases(run_dir / "method" / f"memory_task{t}.pt", {"trunk.x": basis},
                       {"memory_history": dict(history), "eps": 0.95})
            (run_dir / "method" / f"gpm_logs_task{t}.json").write_text(json.dumps({
                "projected": t > 0,
                "gradient_c": {"0": {"trunk.x": 0.9, "flow_head.y": 0.8}},
                "update_c": {"0": {"trunk.x": 0.1, "flow_head.y": 0.1}},
                "residuals": {"trunk.x": {"max_residual_over_bound": 0.02}}}))
    return run_dir


def config_for(reference_name):
    config = load_report_config()
    config["criteria"]["expected_thresholds_by_reference"] = {reference_name: [0.75, 0.63, 0.85, 0.83]}
    return config


def test_pilot_comparison_is_skipped_for_another_seed_namespace(tmp_path):
    ref = write_run(tmp_path, "ref", REF_DIAG, [0.0, 0.0, 0.0, 0.98], with_memory=False)
    gpm = write_run(tmp_path, "gpm", [0.9, 0.88, 0.5, 0.9], [0.8, 0.7, 0.5, 0.9], with_memory=False)
    pilot = tmp_path / "pilot.json"
    evaluation = {KEYS[0]: {"value": 0.8, "successes": [True] * 40 + [False] * 10}}
    for ns, skipped in (("other_namespace", True), ("ref", False)):
        pilot.write_text(json.dumps({"evaluation_seed_run_id": ns,
                                     "arms": {"gpm_projected_adam": {"evaluation": evaluation}}}))
        report = build_report(gpm, ref, pilot_json=pilot, config=config_for("ref"),
                              verify_checkpoints=False)
        assert ("skipped" in report["pilot_comparison"]) == skipped
        if not skipped:
            assert report["pilot_comparison"][KEYS[0]]["pilot"] == 0.8


def test_build_report_end_to_end_on_synthetic_runs(tmp_path):
    ref = write_run(tmp_path, "ref", REF_DIAG, [0.0, 0.0, 0.0, 0.98], with_memory=False)
    gpm = write_run(tmp_path, "gpm", [0.9, 0.88, 0.5, 0.9], [0.8, 0.7, 0.5, 0.9], with_memory=True)
    report = build_report(gpm, ref, pilot_json=None, config=config_for("ref"),
                          verify_checkpoints=False)
    assert report["criteria"]["thresholds"] == [0.75, 0.63, 0.85, 0.83]
    assert report["outcome"]["sgp_fallback_triggered"] and report["outcome"]["sgp_fallback_tasks"] == [2]
    assert report["paired_cells"]["3,0"]["diff"] == pytest.approx(0.8)
    assert report["capacity"]["3"]["trunk"]["median_rho"] == pytest.approx(0.7)
    assert report["dynamics"]["2"]["worst_residual_over_bound"] == 0.02
    assert report["capacity"]["1"]["trunk"]["median_new_energy_protected"] == pytest.approx(0.5)
    assert report["episode_transitions"][KEYS[0]]["0->3"]["lost"] == 5  # 45 -> 40 successes
    checks = report["provenance_checks"]
    assert checks["clean_git_sha"]["passed"] and checks["seed_namespace"]["passed"]
    assert checks["occupancy_non_decreasing"]["passed"] and checks["residuals_within_bound"]["passed"]
    assert checks["memory_chained"]["passed"] and checks["memory_chained"]["prefix_identical"]
    assert list(checks["memory_chained"]["steps"]) == ["0->1", "1->2", "2->3"]
    json.dumps(report, default=str)  # serialisable
