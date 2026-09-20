"""§8.2 metrics on hand-built retention matrices with known answers."""

from __future__ import annotations

import numpy as np
import pytest

from flowcl.analysis.metrics import (
    Estimate,
    RetentionMatrix,
    auc_average_success,
    bootstrap_ci,
    final_average_success,
    forward_transfer,
    negative_backward_transfer,
    paired_difference_ci,
    per_task_forgetting,
    success_estimate,
    summarize,
)

TASKS = ("t0", "t1", "t2")


def build(values) -> RetentionMatrix:
    """Retention matrix from a dense list-of-lists; NaN entries stay unevaluated."""
    matrix = RetentionMatrix.empty(TASKS)
    for i, row in enumerate(values):
        for j, value in enumerate(row):
            if value is not None:
                matrix.set(i, j, value, n=50)
    return matrix


# ---- bootstrap CIs ------------------------------------------------------------


def test_bootstrap_ci_brackets_the_mean():
    samples = np.array([1.0] * 40 + [0.0] * 10)  # success rate 0.8
    est = bootstrap_ci(samples, seed=0)
    assert est.value == pytest.approx(0.8)
    assert est.low < 0.8 < est.high
    assert est.n == 50


def test_bootstrap_ci_covers_the_truth_at_the_nominal_rate():
    """Empirical coverage of a Bernoulli mean should be near 95%.

    Percentile bootstrap on a discrete statistic is slightly conservative at n=50, so
    the acceptance band is wide, but a genuinely broken CI (e.g. swapped quantiles or
    a missing /2) falls far outside it.
    """
    truth = 0.7
    rng = np.random.default_rng(1234)
    covered = 0
    trials = 300
    for trial in range(trials):
        sample = rng.binomial(1, truth, size=50).astype(np.float64)
        est = bootstrap_ci(sample, n_bootstrap=2000, seed=trial)
        covered += est.low <= truth <= est.high
    coverage = covered / trials
    assert 0.88 <= coverage <= 0.99, f"empirical coverage {coverage:.3f}"


def test_bootstrap_ci_narrows_as_n_grows():
    rng = np.random.default_rng(0)
    widths = []
    for n in (20, 200, 2000):
        sample = rng.binomial(1, 0.5, size=n).astype(np.float64)
        est = bootstrap_ci(sample, n_bootstrap=2000, seed=0)
        widths.append(est.high - est.low)
    assert widths[0] > widths[1] > widths[2]


def test_bootstrap_ci_is_reproducible():
    sample = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    a = bootstrap_ci(sample, seed=7)
    b = bootstrap_ci(sample, seed=7)
    assert (a.value, a.low, a.high) == (b.value, b.low, b.high)


def test_degenerate_sample_gives_zero_width_interval():
    est = bootstrap_ci(np.ones(30), seed=0)
    assert (est.value, est.low, est.high) == (1.0, 1.0, 1.0)


def test_single_rollout_interval_is_degenerate_not_fake_precise():
    est = bootstrap_ci([1.0], seed=0)
    assert est.n == 1
    assert est.low == est.high == 1.0


def test_bootstrap_rejects_empty_and_bad_confidence():
    with pytest.raises(ValueError, match="empty sample"):
        bootstrap_ci([])
    with pytest.raises(ValueError, match="confidence must be in"):
        bootstrap_ci([1.0, 0.0], confidence=1.5)


def test_success_estimate_from_bools():
    est = success_estimate([True] * 8 + [False] * 2, seed=0)
    assert est.value == pytest.approx(0.8)


def test_paired_difference_is_centred_on_the_paired_mean():
    """§8.1's fixed init set makes two arms paired; the CI must use that."""
    correct = [1.0] * 18 + [0.0] * 2
    swapped = [1.0] * 4 + [0.0] * 16
    est = paired_difference_ci(correct, swapped, seed=0)
    assert est.value == pytest.approx(0.9 - 0.2)
    assert est.low > 0.0, "a large paired difference must exclude zero"


def test_paired_difference_is_tighter_than_unpaired_when_arms_are_correlated():
    """The reason pairing matters: it removes per-initial-state difficulty.

    Here the arms agree on every rollout except a fixed few, so the paired difference
    has almost no spread while the two marginal rates individually do.
    """
    shared = [1.0, 0.0] * 20
    swapped = list(shared)
    for i in range(0, 8, 2):
        swapped[i] = 0.0  # the swapped arm loses four of the shared successes

    paired = paired_difference_ci(shared, swapped, seed=0)
    a = bootstrap_ci(np.asarray(shared), seed=0)
    b = bootstrap_ci(np.asarray(swapped), seed=0)
    unpaired_width = (a.high - a.low) + (b.high - b.low)
    assert (paired.high - paired.low) < unpaired_width


def test_paired_difference_rejects_unequal_lengths():
    with pytest.raises(ValueError, match="equal length"):
        paired_difference_ci([1.0, 0.0], [1.0])


def test_estimate_formats_with_interval():
    est = Estimate(value=0.82, low=0.70, high=0.92, n=50)
    assert est.format_pp() == "82.0 [70.0, 92.0]"


def test_estimate_rejects_value_outside_interval():
    """§11: a number and its CI must be consistent, or the table is misleading."""
    with pytest.raises(ValueError, match="outside its interval"):
        Estimate(value=0.9, low=0.1, high=0.5, n=10)


# ---- retention matrix ---------------------------------------------------------


def test_set_and_get():
    matrix = RetentionMatrix.empty(TASKS)
    matrix.set(0, 0, 0.9, n=50)
    assert matrix.get(0, 0) == 0.9
    assert matrix.n_rollouts[0, 0] == 50


def test_unevaluated_cell_raises_rather_than_returning_zero():
    """A missing evaluation must not silently read as 0% success."""
    matrix = RetentionMatrix.empty(TASKS)
    with pytest.raises(KeyError, match="was never evaluated"):
        matrix.get(1, 0)


def test_rejects_out_of_range_rate_and_index():
    matrix = RetentionMatrix.empty(TASKS)
    with pytest.raises(ValueError, match="outside"):
        matrix.set(0, 0, 1.5, n=1)
    with pytest.raises(IndexError):
        matrix.set(5, 0, 0.5, n=1)


def test_incomplete_row_is_reported():
    matrix = build([[0.9, None, None], [0.5, 0.8, None], [0.4, None, 0.7]])
    with pytest.raises(ValueError, match="missing evaluations for"):
        matrix.assert_complete_through(2)


# ---- §8.2 metrics, known answers ---------------------------------------------


def test_final_average_success_known_answer():
    """Last row is [0.2, 0.5, 0.8] -> F_1 = 0.5."""
    matrix = build([[0.9, 0.1, 0.0], [0.6, 0.9, 0.0], [0.2, 0.5, 0.8]])
    assert final_average_success(matrix) == pytest.approx(0.5)


def test_per_task_forgetting_known_answer():
    """Diagonal [0.9, 0.9, 0.8] minus last row [0.2, 0.5, 0.8]."""
    matrix = build([[0.9, 0.1, 0.0], [0.6, 0.9, 0.0], [0.2, 0.5, 0.8]])
    assert per_task_forgetting(matrix) == {
        "t0": pytest.approx(0.7),
        "t1": pytest.approx(0.4),
        "t2": pytest.approx(0.0),
    }


def test_nbt_excludes_the_final_task():
    """NBT = mean(0.7, 0.4) = 0.55, not mean(0.7, 0.4, 0.0) = 0.3667.

    Including the structurally-zero final task would dilute NBT by 1/T and make
    curricula of different lengths incomparable.
    """
    matrix = build([[0.9, 0.1, 0.0], [0.6, 0.9, 0.0], [0.2, 0.5, 0.8]])
    assert negative_backward_transfer(matrix) == pytest.approx(0.55)
    assert negative_backward_transfer(matrix) != pytest.approx(0.36667, abs=1e-4)


def test_auc_known_answer():
    """Stage means: 0.9; (0.6+0.9)/2 = 0.75; (0.2+0.5+0.8)/3 = 0.5 -> AUC = 0.7167."""
    matrix = build([[0.9, 0.1, 0.0], [0.6, 0.9, 0.0], [0.2, 0.5, 0.8]])
    assert auc_average_success(matrix) == pytest.approx((0.9 + 0.75 + 0.5) / 3)


def test_forward_transfer_known_answer():
    """R[0][1]=0.1 vs baseline 0.05; R[1][2]=0.0 vs baseline 0.10 -> mean(0.05, -0.10)."""
    matrix = build([[0.9, 0.1, 0.0], [0.6, 0.9, 0.0], [0.2, 0.5, 0.8]])
    baseline = {"t1": 0.05, "t2": 0.10}
    assert forward_transfer(matrix, baseline) == pytest.approx(-0.025)


def test_forward_transfer_requires_a_baseline_for_each_task():
    matrix = build([[0.9, 0.1, 0.0], [0.6, 0.9, 0.0], [0.2, 0.5, 0.8]])
    with pytest.raises(KeyError, match="needs a single-task baseline"):
        forward_transfer(matrix, {"t1": 0.05})


def test_forward_transfer_requires_above_diagonal_entries():
    matrix = build([[0.9, None, None], [0.6, 0.9, None], [0.2, 0.5, 0.8]])
    with pytest.raises(ValueError, match="before it was"):
        forward_transfer(matrix, {"t1": 0.0, "t2": 0.0})


def test_perfect_retention_gives_zero_forgetting():
    matrix = build([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    assert final_average_success(matrix) == pytest.approx(1.0)
    assert negative_backward_transfer(matrix) == pytest.approx(0.0)
    assert auc_average_success(matrix) == pytest.approx(1.0)


def test_total_forgetting_gives_nbt_equal_to_the_diagonal():
    matrix = build([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert negative_backward_transfer(matrix) == pytest.approx(1.0)
    assert final_average_success(matrix) == pytest.approx(1 / 3)


def test_summarize_bundles_every_metric():
    matrix = build([[0.9, 0.1, 0.0], [0.6, 0.9, 0.0], [0.2, 0.5, 0.8]])
    summary = summarize(matrix, baseline={"t1": 0.05, "t2": 0.10})
    payload = summary.as_dict()
    assert set(payload) == {"F_1", "NBT", "AUC", "per_task_forgetting", "FWT"}
    assert payload["F_1"] == pytest.approx(0.5)


def test_summarize_omits_fwt_without_a_baseline():
    matrix = build([[0.9, 0.1, 0.0], [0.6, 0.9, 0.0], [0.2, 0.5, 0.8]])
    assert "FWT" not in summarize(matrix).as_dict()


def test_nbt_undefined_for_single_task():
    matrix = RetentionMatrix.empty(("only",))
    matrix.set(0, 0, 0.9, n=10)
    with pytest.raises(ValueError, match="undefined for a single-task"):
        negative_backward_transfer(matrix)
