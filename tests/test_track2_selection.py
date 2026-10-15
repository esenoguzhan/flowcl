"""Track 2 step 3 (flowcl/analysis/predictors.py; behavior_predictors.analyse / run_select /
verify_selection): within-task concordance, the leave-one-root-out curve, K_b, the two-level cluster
bootstrap, the selection rule (ties, undefined and oracle candidates, nothing carried) and the
publication's refusals."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from flowcl.analysis import predictors as pr
from flowcl.experiments import behavior_predictors as bp


def make_cells(task, root, branch, k, n=50):
    m = len(task)
    return pr.Cells(tuple(f"c{i}" for i in range(m)), np.asarray(task, dtype=object), np.asarray(root, dtype=object),
                    np.asarray(branch, dtype=object), np.asarray(k, dtype=np.float64), np.full(m, float(n)))


def sel_cfg(n_boot=200, **over):
    sel = copy.deepcopy(bp.load_config()["selection"])
    sel["bootstrap"] = {**sel["bootstrap"], "n": n_boot}
    sel.update(over)
    return sel


def world(seed=0, n_roots=6, n_branches=3, per_branch=4, tasks=("t0", "t1", "t2")):
    """Success follows one logistic curve of `good` for every task; `m0` adds a per-task offset (cross-task
    miscalibration) and noise; `bad` is pure noise."""
    rng = np.random.default_rng(seed)
    task, root, branch, good = [], [], [], []
    for r in range(n_roots):
        for b in range(n_branches):
            for c in range(per_branch):
                task.append(tasks[(r + b + c) % len(tasks)])
                root.append(f"r{r}")
                branch.append(f"r{r}b{b}")
                good.append(rng.uniform(-3, 1))
    good = np.asarray(good)
    p = 1 / (1 + np.exp(-(-1.0 - 2.0 * good)))
    k = rng.binomial(50, p)
    offset = {"t0": -1.5, "t1": 0.0, "t2": 1.5}
    m0 = good + np.asarray([offset[t] for t in task]) + rng.normal(0, 0.6, len(task))
    cells = make_cells(task, root, branch, k)
    return cells, {"good": good, "m0": m0, "bad": rng.normal(0, 1, len(task)), "oracle": good.copy()}


def test_concordance_pairs_ties_and_no_cross_task_pairs():
    cells = make_cells(["a", "a", "a", "b"], ["r"] * 4, ["x"] * 4, [40, 20, 20, 0])
    pairs = pr.ranking_pairs(cells, 1)
    assert sorted(zip(pairs[0].tolist(), pairs[1].tolist())) == [(0, 1), (0, 2)]   # (1,2) tied counts; "b" alone
    x = np.asarray([0.1, 0.5, 0.1, 9.0])
    conc = pr.pair_concordance(x, pairs)
    assert sorted(conc.tolist()) == [0.5, 1.0] and pr.concordance(conc) == 0.75
    assert len(pr.ranking_pairs(cells, 25)[0]) == 0
    assert np.isnan(pr.concordance(np.zeros(0)))


def test_bootstrap_draws_roots_then_whole_branches_reproducibly():
    cells, _ = world()
    W = pr.bootstrap_weights(cells, 50, 7)
    assert np.array_equal(W, pr.bootstrap_weights(cells, 50, 7))
    assert not np.array_equal(W, pr.bootstrap_weights(cells, 50, 8))
    for row in W:
        for b in set(cells.branch):   # a branch's cells always move together
            assert len(set(row[cells.branch == b])) == 1
        n_draws = sum(row[cells.branch == b][0] for b in set(cells.branch))
        assert n_draws == 6 * 3       # n_roots draws, each drawing n_branches branches
    absent = [(row[cells.root == r] == 0).all() for row in W for r in set(cells.root)]
    assert any(absent)                # some resamples leave a root out


def test_cross_fit_leaves_each_root_out():
    cells, values = world()
    x = values["good"]
    pred = pr.cross_fit(x, cells, 1e-6, 100, 1e-10)
    held = cells.root == "r2"
    beta = pr.fit_logistic(pr.design(x[~held]), cells.k[~held], cells.n[~held])
    assert np.allclose(pred[held], pr.predict(beta, x[held]))
    poisoned = cells.k.copy()
    poisoned[held] = 50 - poisoned[held]      # changing r2's outcomes leaves r2's predictions unchanged
    other = pr.Cells(cells.ids, cells.task, cells.root, cells.branch, poisoned, cells.n)
    assert np.allclose(pr.cross_fit(x, other, 1e-6, 100, 1e-10)[held], pred[held])


def test_calibration_gap_zero_when_tasks_share_the_curve_and_spread_for_a_constant():
    cells = make_cells(["a"] * 4 + ["b"] * 4, ["r"] * 8, ["x"] * 8, [10, 20, 30, 40, 5, 15, 35, 45])
    eligible = pr.eligible_tasks(cells, 4)
    resid = np.asarray([0.1, -0.1, 0.2, -0.2, 0.05, -0.05, 0.3, -0.3])     # per-task means 0
    assert pr.calibration_gap(resid, cells, eligible) == pytest.approx(0.0, abs=1e-12)
    overall = cells.rate.mean()
    gap = pr.calibration_gap(cells.rate - overall, cells, eligible)
    spread = np.mean([abs(cells.rate[cells.task == t].mean() - overall) for t in ("a", "b")])
    assert gap == pytest.approx(spread)
    assert np.isnan(pr.calibration_gap(resid, cells, pr.eligible_tasks(cells, 5), np.ones((2, 8)))).all()


def test_analysis_selects_the_calibrated_measure_and_never_an_oracle():
    cells, values = world()
    sel = sel_cfg(candidates=["m0", "bad", "good"], oracle=["oracle"], reference="m0")
    out = bp.analyse(cells, values, sel)
    cand = out["candidates"]
    assert cand["good"]["k_b"] < cand["m0"]["k_b"]                  # m0's task offsets miscalibrate
    assert cand["good"]["improvement_b"] > 0 and cand["good"]["improvement_b_ci"][0] > 0
    assert cand["good"]["c_a"] > cand["bad"]["c_a"]
    assert cand["good"]["c_a_ci"][0] <= cand["good"]["c_a"] <= cand["good"]["c_a_ci"][1]
    order = sel["candidates"]
    choice_b = pr.select({n: cand[n]["improvement_b"] for n in order + ["oracle"]}, order + ["oracle"], "m0",
                         {"oracle"})
    assert choice_b["carried"] == "good"
    assert pr.select({"oracle": 1.0}, ["m0", "oracle"], "m0", {"oracle"})["carried"] is None


def test_select_ties_undefined_and_nothing_carried():
    order = ["m0", "a", "b", "c"]
    assert pr.select({"a": 0.1, "b": 0.1, "c": None}, order, "m0")["carried"] == "a"     # ties by list order
    assert pr.select({"a": None, "b": float("nan"), "c": 0.2}, order, "m0")["carried"] == "c"
    out = pr.select({"a": -0.1, "b": 0.0, "c": None}, order, "m0")
    assert out["carried"] is None and out["status"] == "not_shown" and out["best"] == "b"
    assert pr.select({}, order, "m0")["best"] is None


def test_dropped_resamples_are_counted_and_shared():
    cells, values = world(n_roots=3)
    sel = sel_cfg(candidates=["m0", "good"], oracle=[], reference="m0")
    W = pr.bootstrap_weights(cells, 20, 1)
    W[0] = 0.0                                   # an empty resample: every score undefined there
    out = bp.analyse(cells, {k: values[k] for k in ("m0", "good")}, sel, weights=W)
    assert out["dropped_resamples"] == {"a": 1, "b": 1}
    assert out["candidates"]["good"]["improvement_a_ci"] is not None


def test_non_convergence_is_undefined(monkeypatch):
    cells, values = world()
    sel = sel_cfg(candidates=["m0", "good"], oracle=[], reference="m0")
    real = pr.cross_fit
    monkeypatch.setattr(pr, "cross_fit", lambda x, *a, **k: None if x is values["good"] else real(x, *a, **k))
    out = bp.analyse(cells, {k: values[k] for k in ("m0", "good")}, sel)
    good = out["candidates"]["good"]
    assert not good["converged"] and good["k_b"] is None and good["improvement_b"] is None
    assert pr.select({n: out["candidates"][n]["improvement_b"] for n in ("m0", "good")}, ["m0", "good"],
                     "m0")["carried"] is None
    assert pr.fit_logistic(np.ones((3, 2)), np.ones(3), np.ones(3) * 2, ridge=0.0, max_iter=1) is None


def test_publication_refusals_and_verify(tmp_path, monkeypatch):
    report = {"inputs": {"x": 1}, "main": {"n_cells": 3, "n_pairs": 1, "n_roots": 1, "n_branches": 1,
                                         "eligible_tasks": [], "dropped_resamples": {}, "candidates": {}},
              "selection": {"a": None, "b": None}, "frozen_coefficients": {}, "sensitivities": {}, "notes": {}}
    monkeypatch.setattr(bp, "selection_report", lambda cfg, root, rule=None: copy.deepcopy(report))
    cfg = {"selection": {"out": "track2/selection"}}
    monkeypatch.setattr(bp, "git_sha", lambda: "abcdef1234-dirty")
    with pytest.raises(RuntimeError, match="clean tree"):
        bp.run_select(cfg, results_root=tmp_path)
    monkeypatch.setattr(bp, "git_sha", lambda: "abcdef1234567")
    with pytest.raises(RuntimeError, match="suite"):
        bp.run_select(cfg, results_root=tmp_path, suite_result=tmp_path / "missing.txt")
    suite = tmp_path / "suite.txt"
    suite.write_text("1 passed\nexit 0\nThu\nabcdef1\n")
    dry = bp.run_select(cfg, results_root=tmp_path, out_root=tmp_path / "dry")
    assert dry["dry_run"] and (tmp_path / "dry" / "selection_dry.json").is_file()
    assert not (tmp_path / "track2" / "selection").exists()
    first = bp.run_select(cfg, results_root=tmp_path, suite_result=suite)
    report["selection"] = {"a": "changed"}
    assert bp.run_select(cfg, results_root=tmp_path, suite_result=suite) == first   # never rewritten
    assert not bp.verify_selection(cfg, results_root=tmp_path)["passed"]            # recomputed differs
    report["selection"] = {"a": None, "b": None}
    assert bp.verify_selection(cfg, results_root=tmp_path)["passed"]
    (tmp_path / "track2" / "selection" / "summary.json").write_text("{}\n")
    with pytest.raises(RuntimeError, match="done-marker"):
        bp.load_selection(tmp_path / "track2" / "selection")


def test_selection_inputs_refuse_a_mismatched_manifest(tmp_path, monkeypatch):
    cfg = bp.load_config()
    monkeypatch.setattr(bp, "load_manifest", lambda d: {"manifest_sha256": "0" * 64, "manifest": {"cells": []}})
    with pytest.raises(ValueError, match="manifest"):
        bp.selection_inputs(cfg, tmp_path)
