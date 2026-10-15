"""The recovery measurement study's pure parts and analysis (flowcl/analysis/recovery.py,
flowcl/experiments/recovery_study.py): the perturbation bank, the controller-relevant effect and no-ops,
query selection, cross-fitted S20 over cells (uniform risk ~0.2 whatever the exposure; phase-only and
component-only worlds pass), fractional ties, fold-wise knots, the bootstrap's fold integrity, usability
and the outcome, the negative control, and the dry-run analysis on synthetic work files."""

from __future__ import annotations

import json

import numpy as np
import pytest

from flowcl.analysis import recovery as rc
from flowcl.experiments import recovery_study as rs

RULE = {"q1": {"top_fraction": 0.2, "min_s20": 0.4, "uniform": 0.2, "halves": [[0, 1], [2, 3]]},
        "q2": {"ridge": 1.0}, "n_phase": 6}
SIGMA = np.asarray([0.3, 0.3, 0.3, 0.1, 0.1, 0.1, 0.9])


def spec(kind, **over):
    return {"kind": kind, "n_queries": 6, "n_seeds": 4, "n_rollouts": 12, "p_success": 0.85, **over}


def test_bank_is_deterministic_per_group_and_isotropic_norms_match():
    a = rc.perturbation_bank(SIGMA, 8, [1.0, 3.0], 2, seed=5)
    b = rc.perturbation_bank(SIGMA, 8, [1.0, 3.0], 2, seed=5)
    assert len(a) == 16 and all(np.array_equal(x["delta"], y["delta"]) for x, y in zip(a, b))
    for x in a:
        if x["kind"] == "structured":
            lo, hi = rc.DIMS[x["group"]]
            outside = np.delete(x["delta"], range(lo, hi), axis=1)
            assert np.all(outside == 0) and np.allclose(x["delta"], x["delta"][0])   # constant over the prefix
    for m in (1.0, 3.0):
        target = np.mean([np.linalg.norm(x["delta"]) for x in a if x["kind"] == "structured" and x["magnitude"] == m])
        assert all(np.isclose(np.linalg.norm(x["delta"]), target) for x in a if x["kind"] == "isotropic"
                   and x["magnitude"] == m)


def test_effect_after_clipping_flips_and_noops():
    prefix = np.zeros((8, 7))
    prefix[:, 0] = 0.95
    prefix[:, 6] = -1.0
    d = np.zeros((8, 7))
    d[:, 0] = 0.2                                    # clipped at 1.0: effective 0.05 per step
    e = rc.effective(prefix, d, SIGMA)
    assert e["size"] == pytest.approx(np.sqrt(8) * 0.05 / 0.3) and e["clipped_fraction"] == 1.0 and not e["noop"]
    g = np.zeros((8, 7))
    g[:, 6] = 0.9                                    # -1 -> -0.1: the same sign, a controller no-op
    e = rc.effective(prefix, g, SIGMA)
    assert e["flips"] == 0 and e["noop"] and e["size"] > 0
    g[:, 6] = 2.7                                    # -1 -> +1: eight flips
    assert rc.effective(prefix, g, SIGMA)["flips"] == 8
    fully_clipped = np.zeros((8, 7))
    fully_clipped[:, 0] = 1.0
    p2 = prefix.copy()
    p2[:, 0] = 1.0
    assert rc.effective(p2, fully_clipped, SIGMA)["noop"]   # no position change and no flip


def test_query_steps_nearest_replan_and_duplicates_dropped():
    replans = list(range(0, 100, 8))
    out = rc.query_steps(replans, 96, [1 / 12, 3 / 12, 5 / 12, 7 / 12, 9 / 12, 11 / 12])
    assert out == [(0, 8), (1, 24), (2, 40), (3, 56), (4, 72), (5, 88)]
    short = rc.query_steps([0, 8, 16], 20, [0.1, 0.2, 0.9])
    assert [s for _, s in short] == [0, 16] and [i for i, _ in short] == [0, 2]


def test_top_share_fractional_ties():
    held = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    assert rc._top_share(np.zeros(5), held, 0.2) == pytest.approx(0.2 * held.sum())
    assert rc._top_share(np.asarray([5, 4, 3, 2, 1.0]), held, 0.2) == pytest.approx(1.0)
    assert rc._top_share(np.asarray([5, 5, 1, 1, 1.0]), held, 0.2) == pytest.approx(0.5 * (1 + 2))


def test_s20_uniform_is_about_02_and_phase_or_component_concentration_passes():
    rng = np.random.default_rng(0)
    uniform = [rc.s20_cells(rc.perturbed(rc.Table.from_rows(rc.simulate_world(spec("null_uniform", base_logit=-1.5),
                                                                               rng)))) for _ in range(30)]
    uniform = [u for u in uniform if u is not None]
    assert abs(np.mean(uniform) - 0.2) < 0.04
    for kind in ("phase_only", "component_only"):
        vals = [rc.s20_cells(rc.perturbed(rc.Table.from_rows(rc.simulate_world(spec(kind, effect=3.0), rng))))
                for _ in range(10)]
        assert np.mean([v for v in vals if v is not None]) > 0.4, kind


def test_s20_uniform_with_unequal_exposure_stays_near_02():
    rng = np.random.default_rng(1)
    vals = []
    for _ in range(300):
        rows = []
        for cell in range(25):
            trials = 1 if cell % 2 else 4                # unequal exposure per cell
            for m in range(4):
                rows.append({"rollout": cell, "query": 0, "seed": m, "kind": 0, "group": -1, "success": True,
                             "size": 0.0, "flips": 0, "noop": False})
                for _ in range(trials):
                    rows.append({"rollout": cell, "query": 0, "seed": m, "kind": 1, "group": 0,
                                 "success": rng.random() > 0.3, "size": 1.0, "flips": 0, "noop": False})
        v = rc.s20_cells(rc.perturbed(rc.Table.from_rows(rows)))
        if v is not None:
            vals.append(v)
    assert abs(np.mean(vals) - 0.2) < 0.02


def test_knots_come_from_the_training_fold_and_fallback(monkeypatch):
    assert rc.rcs_knots(np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 2.0])) is None
    t = rc.Table.from_rows(rc.simulate_world(spec("null_magnitude"), np.random.default_rng(2)))
    p = rc.perturbed(t)
    seen = []
    real = rc.rcs_knots
    monkeypatch.setattr(rc, "rcs_knots", lambda x: (seen.append(np.sort(x)), real(x))[1])
    rc.q2_crossfit(p, p["rollout"], 1.0, 6)
    sel = ~p["noop"]
    folds = np.unique(p["rollout"][sel])
    assert len(seen) == len(folds)
    for f, x in zip(folds, seen):
        train = np.log1p(p["size"][sel][p["rollout"][sel] != f])
        assert np.array_equal(x, np.sort(train[train > 0]))


def test_bootstrap_keeps_copies_of_a_rollout_in_one_fold():
    t = rc.Table.from_rows(rc.simulate_world(spec("null_uniform"), np.random.default_rng(3)))
    tb, folds = rc._resample(t, np.random.default_rng(4))
    for label in np.unique(tb.rollout):
        assert len(set(folds[tb.rollout == label])) == 1          # a copy has one original rollout
    originals = {int(f) for f in folds}
    assert originals <= set(np.unique(t.rollout).tolist())
    b = rc.bootstrap_task(t, RULE, 5, 1)
    assert b["s20"].shape == (5,) and b["q2"].shape == (5,)


def test_usability_and_the_outcome():
    regime = {"min_rollouts": 6, "min_queries": 36, "min_injuries": 10, "min_injury_rollouts": 3}
    t = rc.Table.from_rows(rc.simulate_world(spec("null_uniform", p_success=1.0, base_logit=-1.0),
                                             np.random.default_rng(5)))
    u = rc.usable(t, 12, regime)
    assert u["usable"] and u["n_queries"] == 72
    assert not rc.usable(t, 5, regime)["usable"]
    assert not rc.usable(t, 12, {**regime, "min_injury_rollouts": 99})["usable"]
    good = {"usable": True, "verdict": {"pass": True, "clear_fail": False}}
    bad_clear = {"usable": True, "verdict": {"pass": False, "clear_fail": True}}
    bad_unclear = {"usable": True, "verdict": {"pass": False, "clear_fail": False}}
    unusable = {"usable": False, "verdict": {"pass": False, "clear_fail": False}}
    assert rc.decide({"a": good, "b": good, "c": bad_clear}, True, True) == "GO"
    assert rc.decide({"a": good, "b": good}, False, True) == "INCONCLUSIVE"     # the gate
    assert rc.decide({"a": good, "b": good}, True, False) == "INCONCLUSIVE"     # the negative control
    assert rc.decide({"a": bad_clear, "b": bad_clear, "c": unusable}, True, True) == "NO-GO"
    assert rc.decide({"a": bad_clear, "b": bad_unclear}, True, True) == "INCONCLUSIVE"
    assert rc.decide({"a": bad_clear, "b": unusable}, True, True) == "INCONCLUSIVE"
    v = rc.task_verdict({"s20": 0.5, "q2": 0.01}, {"s20": np.full(50, 0.5), "q2": np.full(50, 0.01)}, RULE, 0.95)
    assert v["pass"] and not v["clear_fail"]
    v = rc.task_verdict({"s20": 0.25, "q2": -0.01}, {"s20": np.full(50, 0.25), "q2": np.full(50, -0.01)}, RULE, 0.95)
    assert not v["pass"] and v["clear_fail"]
    assert rc.clopper_pearson(0, 10)[0] == 0.0 and rc.clopper_pearson(10, 10)[1] == 1.0


def test_negative_control_counts_mismatches_and_injuries():
    nominal = {"rollout": 0, "query": 0, "seed": 0, "kind": 0, "group": -1, "success": True, "n_steps": 100,
               "prefix_ctrl": "a", "suffix": "b", "noop": False}
    same = {**nominal, "kind": 1, "group": 2, "noop": True}
    diff = {**same, "suffix": "c", "success": False}
    out = rs.negative_control([nominal, same, diff])
    assert out["noop_continuations"] == 2 and out["mismatches"] == 1 and out["injuries"] == 1 and out["at_risk"] == 2


def test_small_decision_simulation_runs_end_to_end():
    out = rc.simulate(spec("null_uniform"), RULE, {"min_rollouts": 6, "min_queries": 36, "min_injuries": 10,
                                                   "min_injury_rollouts": 3}, n_worlds=3, n_tasks=4, n_boot=10, seed=1)
    assert set(out["rates"]) == {"0.95", "0.99"}
    assert sum(out["rates"]["0.95"][k] for k in ("GO", "NO-GO", "INCONCLUSIVE")) == 3


def write_world(base, rule, kind, seed):
    rng = np.random.default_rng(seed)
    for task in rule["tasks"]:
        work = base / rule["work"] / rs._slug(task["key"])
        work.mkdir(parents=True)
        rows = rc.simulate_world(spec(kind), rng)
        for r in rows:
            r.update({"n_steps": 100, "prefix_ctrl": "x", "suffix": "y", "magnitude": 1.0})
        ids = sorted({r["rollout"] for r in rows})
        bind = {"rule_sha256": rs.file_sha256(rs.rule_path())}
        (work / "nominal.json").write_text(json.dumps({"binding": bind, "fallback": False, "rollouts": {
            str(i): {"seed": i, "success": True, "n_steps": 100} for i in ids}}))
        (work / "gate.json").write_text(json.dumps({"binding": bind, "passed": True}))
        for i in ids:
            mine = [r for r in rows if r["rollout"] == i]
            (work / f"rollout_{i:02d}.json").write_text(json.dumps({"binding": bind, "queries": [[q, q * 8] for q in range(6)],
                                                                     "dropped_phases": 0, "rows": mine}))


def test_dry_run_analysis_and_publication_refusals(tmp_path, monkeypatch):
    rule = rs.load_rule()
    rule["bootstrap"]["n"] = 20
    write_world(tmp_path, rule, "null_uniform", 7)
    out = rs.analyze(rule, results_root=tmp_path, out_root=tmp_path)
    assert out["outcome"] in ("GO", "NO-GO", "INCONCLUSIVE") and (tmp_path / "analysis_dry.json").is_file()
    assert set(out["tasks"]) == {t["key"] for t in rule["tasks"]}
    monkeypatch.setattr(rs, "git_sha", lambda: "abc1234-dirty")
    with pytest.raises(RuntimeError, match="clean tree"):
        rs.analyze(rule, results_root=tmp_path)
    monkeypatch.setattr(rs, "git_sha", lambda: "abc1234567")
    with pytest.raises(RuntimeError, match="suite"):
        rs.analyze(rule, results_root=tmp_path, suite_result=tmp_path / "missing.txt")
    missing = tmp_path / "other"
    with pytest.raises(RuntimeError, match="work files"):
        rs.analyze(rule, results_root=missing)
