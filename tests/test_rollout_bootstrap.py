"""A5 (flowcl/analysis/rollout_bootstrap.py): synthetic cases, orientation, pairing across stages and
runs, and the missing-cell and seed rules (README §11)."""

from __future__ import annotations

import numpy as np
import pytest

from flowcl.analysis.metrics import (
    RetentionMatrix,
    auc_average_success,
    final_average_success,
    negative_backward_transfer,
)
from flowcl.analysis.rollout_bootstrap import (
    RunRollouts,
    paired_rollout_bootstrap,
    required_cells,
    rollout_bootstrap,
)

KEYS = ("t/a", "t/b", "t/c")


def make_run(outcomes: dict[tuple[int, int], list], seeds=None, namespace="ns", label="run") -> RunRollouts:
    cells = {c: np.asarray(v, dtype=np.float64) for c, v in outcomes.items()}
    seeds = seeds or {c: tuple(range(100 * c[1], 100 * c[1] + len(v))) for c, v in outcomes.items()}
    n = 1 + max(max(c) for c in outcomes)
    return RunRollouts(KEYS[:n], cells, seeds, namespace, label)


def lower(n: int, fill) -> dict[tuple[int, int], list]:
    return {(i, j): fill(i, j) for i in range(n) for j in range(i + 1)}


def test_all_success_and_all_fail_give_zero_width():
    for value, expected in ((1, (1.0, 0.0, 1.0)), (0, (0.0, 0.0, 0.0))):
        out = rollout_bootstrap(make_run(lower(3, lambda i, j: [value] * 10)), n_bootstrap=200)
        for name, want in zip(("ACC", "NBT", "AUC"), expected):
            assert out[name].value == out[name].low == out[name].high == want


def test_identical_outcomes_across_stages_give_an_exact_zero_nbt_interval():
    rng = np.random.default_rng(3)
    column = {j: rng.integers(0, 2, size=20).tolist() for j in range(3)}
    run = make_run(lower(3, lambda i, j: column[j]))
    out = rollout_bootstrap(run, n_bootstrap=500)
    assert out["NBT"].value == out["NBT"].low == out["NBT"].high == 0.0  # pairing across stages
    assert out["ACC"].high > out["ACC"].low                              # the rollouts still vary


def test_orientation_on_a_hand_built_two_by_two():
    # R[0][0] = 1 (after T1, T1 solved), R[1][0] = 0 (T1 forgotten), R[1][1] = 0.5
    run = make_run({(0, 0): [1, 1, 1, 1], (1, 0): [0, 0, 0, 0], (1, 1): [1, 0, 1, 0]})
    out = rollout_bootstrap(run, n_bootstrap=100)
    assert out["ACC"].value == pytest.approx(0.25)
    assert out["NBT"].value == pytest.approx(1.0)
    assert out["AUC"].value == pytest.approx(0.625)
    assert rollout_bootstrap(run, metrics=("F_first",), n_bootstrap=100)["F_first"].value == pytest.approx(1.0)


def test_point_values_equal_the_metrics_module():
    rng = np.random.default_rng(0)
    outcomes = {(i, j): rng.integers(0, 2, size=50).tolist() for i in range(3) for j in range(3)}
    run = make_run(outcomes)
    out = rollout_bootstrap(run, n_bootstrap=50)
    m = RetentionMatrix.empty(KEYS)
    for (i, j), v in outcomes.items():
        m.set(i, j, float(np.mean(v)), len(v))
    assert out["ACC"].value == pytest.approx(final_average_success(m))
    assert out["NBT"].value == pytest.approx(negative_backward_transfer(m))
    assert out["AUC"].value == pytest.approx(auc_average_success(m))


def test_identical_runs_have_an_exact_zero_paired_difference():
    rng = np.random.default_rng(1)
    outcomes = lower(3, lambda i, j: rng.integers(0, 2, size=30).tolist())
    a, b = make_run(outcomes, label="a"), make_run(outcomes, label="b")
    for est in paired_rollout_bootstrap(a, b, n_bootstrap=300).values():
        assert est.value == est.low == est.high == 0.0


def test_a_better_run_has_a_positive_paired_acc_difference():
    a = make_run(lower(2, lambda i, j: [1] * 18 + [0] * 2), label="a")
    b = make_run(lower(2, lambda i, j: [1] * 2 + [0] * 18), label="b")
    out = paired_rollout_bootstrap(a, b, metrics=("ACC",), n_bootstrap=500)
    assert out["ACC"].value == pytest.approx(0.8) and out["ACC"].low > 0


def test_absent_upper_cells_are_skipped_but_required_cells_are_not():
    run = make_run(lower(3, lambda i, j: [1, 0, 1]))       # seen-only: no upper triangle
    rollout_bootstrap(run, n_bootstrap=20)
    gappy = lower(3, lambda i, j: [1, 0, 1])
    del gappy[(1, 0)]                                     # AUC needs it; ACC and NBT do not
    rollout_bootstrap(make_run(gappy), metrics=("ACC", "NBT"), n_bootstrap=20)
    with pytest.raises(ValueError, match="not evaluated"):
        rollout_bootstrap(make_run(gappy), metrics=("AUC",), n_bootstrap=20)


def test_required_cells():
    assert required_cells(2, ("ACC",)) == {(1, 0), (1, 1)}
    assert required_cells(2, ("NBT",)) == {(0, 0), (1, 0)}
    assert required_cells(2, ("AUC",)) == {(0, 0), (1, 0), (1, 1)}
    with pytest.raises(ValueError, match="single-task"):
        required_cells(1, ("NBT",))


def test_seed_mismatches_are_refused():
    outcomes = lower(2, lambda i, j: [1, 0, 1])
    seeds = {c: (1, 2, 3) for c in outcomes}
    seeds[(1, 0)] = (1, 2, 4)                              # T1 at stage 1 drew other episodes
    with pytest.raises(ValueError, match="not paired across stages"):
        rollout_bootstrap(make_run(outcomes, seeds=seeds), n_bootstrap=10)
    a = make_run(outcomes, seeds={c: (1, 2, 3) for c in outcomes}, label="a")
    b = make_run(outcomes, seeds={c: (7, 8, 9) for c in outcomes}, label="b")
    with pytest.raises(ValueError, match="different episode seeds"):
        paired_rollout_bootstrap(a, b, n_bootstrap=10)
    c = make_run(outcomes, namespace="other", label="c")
    with pytest.raises(ValueError, match="namespaces"):
        paired_rollout_bootstrap(make_run(outcomes), c, n_bootstrap=10)


def test_a_fixed_seed_is_deterministic():
    rng = np.random.default_rng(5)
    run = make_run(lower(3, lambda i, j: rng.integers(0, 2, size=25).tolist()))
    one, two = rollout_bootstrap(run, n_bootstrap=400, seed=7), rollout_bootstrap(run, n_bootstrap=400, seed=7)
    assert one == two


def test_from_payload_reads_stage_records():
    payload = {"run_id": "r", "seed_namespace_run_id": "ns", "task_keys": ["t/a", "t/b"], "stages": [
        {"stage": 0, "evaluation": {"tasks": [{"task_key": "t/a", "successes": [True, False], "seeds": [1, 2]}]}},
        {"stage": 1, "evaluation": {"tasks": [{"task_key": "t/a", "successes": [False, False], "seeds": [1, 2]},
                                              {"task_key": "t/b", "successes": [True, True], "seeds": [5, 6]}]}},
    ]}
    run = RunRollouts.from_payload(payload)
    assert run.namespace == "ns" and run.label == "r"
    assert set(run.cells) == {(0, 0), (1, 0), (1, 1)}
    assert rollout_bootstrap(run, n_bootstrap=50)["NBT"].value == pytest.approx(0.5)
