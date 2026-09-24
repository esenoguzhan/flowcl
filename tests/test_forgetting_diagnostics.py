"""Forgetting diagnostics: Gram identities, zero rules, the pre-registered cases, wiring."""

from __future__ import annotations

import itertools
import json
import math

import pytest
import torch

from flowcl.analysis.interference import (
    activation_interference,
    energy_outside,
    relative_interference,
)
from flowcl.experiments.forgetting_diagnostics import (
    CASES,
    _capture_plan,
    capture_seeds,
    classify_forgetting,
    instrument_check,
    jsonable,
    load_diag_config,
    percentile,
    ratio_summary,
)

DECISION = {"loss_ratio_min": 2.0, "selectivity_min": 2.0, "q_large": 2.0}
REPORTING = {"percentiles": [0.75, 0.90], "q_share_threshold": 2.0, "top_k": 2}


def g(seed):
    return torch.Generator().manual_seed(seed)


# ---- Gram identities --------------------------------------------------------------


def test_gram_route_equals_explicit_activations():
    X = torch.randn(6, 40, generator=g(0), dtype=torch.float64)  # inputs as columns
    K = X @ X.T
    W = torch.randn(5, 6, generator=g(1), dtype=torch.float64)
    dW = 0.1 * torch.randn(5, 6, generator=g(2), dtype=torch.float64)
    explicit = torch.linalg.norm(dW @ X) / torch.linalg.norm(W @ X)
    assert activation_interference(dW, W, K) == pytest.approx(float(explicit), rel=1e-12)

    M = torch.linalg.qr(torch.randn(6, 2, generator=g(3), dtype=torch.float64))[0]
    residual = X - M @ (M.T @ X)
    expected = float(torch.linalg.norm(residual) ** 2 / torch.linalg.norm(X) ** 2)
    assert energy_outside(M, K) == pytest.approx(expected, rel=1e-12)


def test_an_update_orthogonal_to_the_inputs_does_not_interfere():
    # Inputs live in span(e1, e2); the update only reads e3..e6.
    X = torch.zeros(6, 10, dtype=torch.float64)
    X[:2] = torch.randn(2, 10, generator=g(4), dtype=torch.float64)
    W = torch.randn(3, 6, generator=g(5), dtype=torch.float64)
    dW = torch.zeros(3, 6, dtype=torch.float64)
    dW[:, 2:] = 1.0
    assert activation_interference(dW, W, X @ X.T) == pytest.approx(0.0, abs=1e-12)


def test_activation_interference_zero_rules_and_orientation():
    K = torch.eye(4, dtype=torch.float64)
    zero = torch.zeros(3, 4, dtype=torch.float64)
    ones = torch.ones(3, 4, dtype=torch.float64)
    assert activation_interference(zero, zero, K) == 0.0            # 0 / 0 -> 0
    assert math.isinf(activation_interference(ones, zero, K))       # x / 0 -> inf
    assert activation_interference(zero, ones, K) == 0.0            # 0 / x -> 0
    assert activation_interference(ones, ones, torch.zeros(4, 4, dtype=torch.float64)) == 0.0
    with pytest.raises(ValueError, match="input dimension"):
        activation_interference(ones.T, ones.T, K)  # (4, 3) weight against a 4x4 Gram


def test_relative_interference_zero_and_inf_rules():
    assert relative_interference(0.0, 0.0) is None
    assert relative_interference(math.inf, math.inf) is None
    assert math.isinf(relative_interference(0.3, 0.0))
    assert math.isinf(relative_interference(math.inf, 0.3))
    assert relative_interference(0.0, 0.3) == 0.0
    assert relative_interference(0.3, math.inf) == 0.0
    assert relative_interference(0.6, 0.3) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        relative_interference(math.nan, 1.0)


# ---- summaries ----------------------------------------------------------------------


def test_percentile_handles_inf_without_nan():
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert percentile([1.0, 3.0, math.inf], 0.5) == 3.0
    assert math.isinf(percentile([1.0, 3.0, math.inf], 0.75))
    assert percentile([], 0.5) is None


def test_ratio_summary_reports_the_tail_and_counts_exclusions():
    q = {"trunk.a": math.inf, "trunk.b": 1.0, "trunk.c": 3.0,
         "flow_head.d": None, "flow_head.e": 0.5}
    groups = {n: ("trunk_mlp" if n.startswith("trunk") else "decoder_mlp") for n in q}
    s = ratio_summary(q, REPORTING, groups)
    assert s["trunk"]["median"] == 3.0 and math.isinf(s["trunk"]["max"])
    assert s["trunk"]["share_ge_threshold"] == pytest.approx(2 / 3)
    assert [t["layer"] for t in s["trunk"]["top"]] == ["trunk.a", "trunk.c"]
    assert s["trunk"]["top"][0]["group"] == "trunk_mlp"
    assert s["decoder"]["n_layers"] == 1 and s["decoder"]["n_excluded"] == 1
    assert s["all"]["n_layers"] == 4
    assert set(s["trunk"]) >= {"p75", "p90"}


# ---- the pre-registered classification ---------------------------------------------


def stats(R_O, R_S, qd=(0.0, 0.0), qr=(0.0, 0.0)):
    return {"R_O": R_O, "R_S": R_S,
            "Q_direct": {"trunk": qd[0], "decoder": qd[1]},
            "Q_drift": {"trunk": qr[0], "decoder": qr[1]}}


@pytest.mark.parametrize("s, case", [
    (stats(1.5, 1.0, qd=(9, 9), qr=(9, 9)), "C"),     # stable loss: C whatever Q is
    (stats(50.0, 40.0, qd=(9, 9)), "E"),              # both losses rise massively: E, not C
    (stats(4.0, 1.0, qd=(2.0, 0.1)), "A"),            # inclusive Q boundary, trunk only
    (stats(4.0, 1.0, qd=(0.1, 3.0)), "A"),            # either half
    (stats(4.0, 1.0, qd=(1.9, 1.9), qr=(0.1, 3.0)), "B"),
    (stats(4.0, 1.0, qd=(1.9, 1.9), qr=(1.9, 1.9)), "D"),
    (stats(2.0, 1.0, qd=(9, 9)), "A"),                # R_O = 2 and R_O/R_S = 2 both inclusive
    (stats(4.0, 2.0, qd=(9, 9)), "A"),                # selectivity exactly 2
    (stats(4.0, 2.01, qd=(9, 9)), "E"),
    (stats(1.99, 0.1, qd=(9, 9)), "C"),               # a falling control cannot manufacture worsening
])
def test_classification_cases(s, case):
    out = classify_forgetting(s, DECISION)
    assert out["case"] == case
    assert out["interpretation"] == CASES[case]


def test_missing_medians_count_as_small():
    s = stats(4.0, 1.0)
    s["Q_direct"] = {"trunk": None, "decoder": None}
    s["Q_drift"] = {"trunk": None, "decoder": 2.5}
    assert classify_forgetting(s, DECISION)["case"] == "B"


def test_exactly_one_case_applies_on_a_grid():
    def conditions(s):
        R_O, R_S = s["R_O"], s["R_S"]
        big = lambda Q: any(v is not None and v >= 2.0 for v in Q.values())  # noqa: E731
        sel = R_O >= 2 and R_O / R_S >= 2
        return {
            "C": R_O < 2,
            "E": R_O >= 2 and R_O / R_S < 2,
            "A": sel and big(s["Q_direct"]),
            "B": sel and not big(s["Q_direct"]) and big(s["Q_drift"]),
            "D": sel and not big(s["Q_direct"]) and not big(s["Q_drift"]),
        }

    values = [0.5, 1.99, 2.0, 5.0]
    for R_O, R_S, qd, qr in itertools.product(values, [0.5, 1.0, 3.0], values, values):
        s = stats(R_O, R_S, qd=(qd, 0.0), qr=(0.0, qr))
        truth = [k for k, v in conditions(s).items() if v]
        assert len(truth) == 1, (s, truth)
        assert classify_forgetting(s, DECISION)["case"] == truth[0]


# ---- config, seeds, instrument --------------------------------------------------------


def test_committed_config_is_the_preregistered_rule():
    cfg = load_diag_config()
    assert cfg["decision"] == DECISION
    assert cfg["comparisons"]["primary"] == {"transition": [1, 2], "target": 1, "control": 0}
    assert cfg["probe"]["seed_tags"]["shuffle"] == "gpm_pilot_probe::shuffle"


def test_config_rejects_a_target_not_yet_trained(tmp_path):
    from omegaconf import OmegaConf

    cfg = load_diag_config()
    cfg["comparisons"]["primary"] = {"transition": [1, 2], "target": 2, "control": 0}
    path = tmp_path / "bad.yaml"
    OmegaConf.save(OmegaConf.create(cfg), path)
    with pytest.raises(ValueError, match="trained by stage"):
        load_diag_config(path)


def test_capture_seeds_do_not_depend_on_the_stage():
    tags = {"probe": "p", "capture": "c"}
    a = capture_seeds("ns", "suite/task", tags)
    assert a == capture_seeds("ns", "suite/task", tags)  # no stage argument exists
    assert a != capture_seeds("ns", "suite/other", tags)
    assert a[0] != a[1]


def test_capture_plan_shares_the_middle_capture():
    plan = _capture_plan({
        "primary": {"transition": [1, 2], "target": 1, "control": 0},
        "secondary": {"transition": [2, 3], "target": 2, "control": 0},
    })
    assert sorted(plan["method"][2]) == [0, 1, 2]
    assert sorted(plan["method"][2][0]) == [("primary", "control", "drift"),
                                           ("secondary", "control", "direct")]
    assert sorted(plan["reference"]) == [1, 2, 3] and sorted(plan["reference"][2]) == [1, 2]


def test_instrument_check_passes_on_a_match_and_raises_on_a_mismatch():
    keys = ["s/a", "o/b"]
    L = [[0.01, 1.62], [0.72, 0.0064]]
    refs = {0: {"s/a": 0.01, "o/b": 1.62}, 1: {"s/a": 0.72, "o/b": 0.0064}}
    assert instrument_check(L, keys, refs, 1e-3)["passed"]
    refs[1]["o/b"] = 0.0070
    with pytest.raises(RuntimeError, match="instrument check failed"):
        instrument_check(L, keys, refs, 1e-3)
    with pytest.raises(ValueError, match="compared nothing"):
        instrument_check(L, keys, {}, 1e-3)


def test_jsonable_writes_inf_as_a_string_and_refuses_nan():
    assert json.loads(json.dumps(jsonable({"q": math.inf, "x": [1.0, -math.inf]}))) == {
        "q": "inf", "x": [1.0, "-inf"]}
    with pytest.raises(ValueError):
        jsonable({"q": math.nan})


# ---- end to end on tiny real runs --------------------------------------------------------


MILK = "libero_object/pick_up_the_milk_and_place_it_in_the_basket"
SAUCE = "libero_object/pick_up_the_tomato_sauce_and_place_it_in_the_basket"
BBQ = "libero_object/pick_up_the_bbq_sauce_and_place_it_in_the_basket"
TINY_POLICY = {"d_model": 384, "n_trunk_layers": 6, "n_heads": 8, "n_decoder_layers": 2,
               "n_context_tokens": 4, "pretrained": False, "euler_steps": 2}


def test_diagnostics_end_to_end_on_tiny_runs(dataset_dir, tmp_path, monkeypatch):
    from omegaconf import OmegaConf

    import flowcl.experiments.gate2 as gate2
    from flowcl.data.config import load_embodiment_spec
    from flowcl.data.curriculum import load_curriculum
    from flowcl.envs.libero_env import EvalConfig
    from flowcl.experiments.forgetting_diagnostics import run_forgetting_diagnostics
    from flowcl.train.continual import run_continual
    from flowcl.train.trainer import TrainConfig
    from flowcl.utils.libero_paths import repo_root

    capture_cfg = OmegaConf.load(repo_root() / "configs" / "analysis" / "subspace.yaml")
    capture_cfg.min_samples_per_dim = 0.01
    capture_cfg.num_workers = 0
    capture_path = tmp_path / "capture.yaml"
    OmegaConf.save(capture_cfg, capture_path)

    trio = load_curriculum({"name": "test_trio", "tasks": [
        {"task_key": MILK, "n_demos": 1}, {"task_key": SAUCE, "n_demos": 1},
        {"task_key": BBQ, "n_demos": 1}]})
    common = dict(spec=load_embodiment_spec("libero_franka"), policy_config=TINY_POLICY,
                  train_cfg=TrainConfig(steps=2, batch_size=2, num_workers=0, device="cpu",
                                        log_every=0, warmup_steps=1),
                  eval_cfg=EvalConfig(n_episodes=1), seed=0, dataset_dir=dataset_dir,
                  results_root=tmp_path, pretrained=False, evaluate=False)
    run_continual(trio, method_name="seq_ft", **common)
    gpm = run_continual(trio, method_name="gpm", method_kwargs={
        "update_memory": True, "capture_config": str(capture_path), "log_interval": 1}, **common)

    seen = []
    real_capture = gate2.capture_task_grams

    def recording_capture(policy, dataset, cfg, device, probe_seed, capture_seed):
        seen.append((dataset.task_ids[0], probe_seed, capture_seed))
        return real_capture(policy, dataset, cfg, device, probe_seed=probe_seed,
                            capture_seed=capture_seed)

    monkeypatch.setattr(gate2, "capture_task_grams", recording_capture)

    cfg = load_diag_config()
    cfg.update({
        "method_run": gpm.run_id, "reference_run": "test_trio__seq_ft__seed0",
        "probe": {**cfg["probe"], "batch_size": 2, "n_batches": 1},
        "instrument_check": None, "capture_config": str(capture_path),
        "comparisons": {"primary": {"transition": [1, 2], "target": 1, "control": 0}},
    })
    report = run_forgetting_diagnostics(cfg, results_root=tmp_path, dataset_dir=dataset_dir,
                                        device="cpu", allow_dirty=True,
                                        out=tmp_path / "diag.json")

    assert len(report["loss_matrix"]["method"]) == 3 and len(report["loss_matrix"]["reference"]) == 3
    primary = report["comparisons"]["primary"]
    assert primary["classification"]["case"] in CASES
    assert report["decision"]["primary_case"] == primary["classification"]["case"]
    assert primary["per_layer"] and all(
        {"q_direct", "q_drift", "r_target_direct", "e_target_drift"} <= set(row)
        for row in primary["per_layer"].values()
    )
    # Paired inputs: every capture of a task used the same seeds at both stages.
    by_task: dict = {}
    for task, p, c in seen:
        by_task.setdefault(task, set()).add((p, c))
    assert all(len(s) == 1 for s in by_task.values()), by_task
    # method: tasks 0, 1 at stages 1 and 2; reference: task 1 at stages 1 and 2.
    assert len(seen) == 6
    written = json.loads((tmp_path / "diag.json").read_text())
    assert written["decision"]["primary_case"] == report["decision"]["primary_case"]
    assert written["allow_dirty"] is True
