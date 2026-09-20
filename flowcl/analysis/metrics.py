"""Retention matrix, bootstrap confidence intervals, and CL metrics.

Spec §8.2:

    Retention matrix ``R[i][j]`` = success rate on task ``j`` after training through
    task ``i``. Primary metric: final average success ``F_1 = mean_j R[T][j]``.
    Per-task forgetting ``R[j][j] - R[T][j]``. Also report NBT, FWT and AUC using the
    CLARE definitions. Bootstrap 95% CIs over rollouts.

Spec §11: "no bare percentages, every number carries a CI". :class:`Estimate` exists
so a success rate cannot be passed around as a bare float.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# §8.2: 95% CIs.
DEFAULT_CONFIDENCE = 0.95
DEFAULT_N_BOOTSTRAP = 10000


@dataclass(frozen=True)
class Estimate:
    """A point estimate with a confidence interval.

    Exists to make §11's "no bare percentages" structural: a function returning an
    ``Estimate`` cannot have its uncertainty dropped by accident.
    """

    value: float
    low: float
    high: float
    n: int
    confidence: float = DEFAULT_CONFIDENCE

    def __post_init__(self) -> None:
        if not self.low <= self.value <= self.high:
            raise ValueError(
                f"estimate {self.value} lies outside its interval "
                f"[{self.low}, {self.high}]"
            )
        if self.n <= 0:
            raise ValueError(f"n must be positive, got {self.n}")

    def format_pp(self, decimals: int = 1) -> str:
        """Percentage points with the interval, e.g. ``"82.0 [70.0, 92.0]"``."""
        return (
            f"{100 * self.value:.{decimals}f} "
            f"[{100 * self.low:.{decimals}f}, {100 * self.high:.{decimals}f}]"
        )

    def __str__(self) -> str:
        return self.format_pp()


def bootstrap_ci(
    samples: np.ndarray | list[float],
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 0,
    statistic=np.mean,
) -> Estimate:
    """Percentile bootstrap CI for a statistic of ``samples``.

    Args:
        samples: Per-rollout observations, e.g. 0/1 successes.
        n_bootstrap: Resample count.
        confidence: Coverage, 0.95 by default (§8.2).
        seed: Fixed so a reported interval is reproducible.
        statistic: Applied to each resample; defaults to the mean.

    Returns:
        An :class:`Estimate` whose ``value`` is the statistic on the original sample.
    """
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"expected a 1-D sample, got shape {values.shape}")
    if values.size == 0:
        raise ValueError("bootstrap_ci received an empty sample")
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    point = float(statistic(values))
    if values.size == 1:
        # A single rollout carries no information about spread; say so rather than
        # reporting a zero-width interval that looks precise.
        return Estimate(point, point, point, n=1, confidence=confidence)

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(n_bootstrap, values.size))
    resampled = statistic(values[indices], axis=1)

    alpha = 1.0 - confidence
    low, high = np.quantile(resampled, [alpha / 2, 1.0 - alpha / 2])
    # Clamp so the point estimate is always inside its own interval, which percentile
    # bootstrap can violate for very discrete samples.
    return Estimate(
        value=point,
        low=float(min(low, point)),
        high=float(max(high, point)),
        n=int(values.size),
        confidence=confidence,
    )


def success_estimate(
    successes: list[bool] | np.ndarray, seed: int = 0, **kwargs
) -> Estimate:
    """Bootstrap CI for a success rate over rollouts."""
    return bootstrap_ci(np.asarray(successes, dtype=np.float64), seed=seed, **kwargs)


def paired_difference_ci(
    a: list[bool] | np.ndarray,
    b: list[bool] | np.ndarray,
    seed: int = 0,
    **kwargs,
) -> Estimate:
    """CI for ``mean(a) - mean(b)`` when both arms share the same rollout conditions.

    §8.1 fixes the initial-state set, so two conditions evaluated on it are *paired*:
    rollout ``i`` in each arm starts from the identical state. Resampling the arms
    independently would throw that pairing away and inflate the interval, sometimes
    enough to hide a real difference. Resampling rollout *indices* keeps it.

    Used by the §4.1 language-discriminability check, where the two arms are the
    correct and the swapped instruction on one task's initial states.
    """
    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError(
            f"paired samples must have equal length, got {left.shape} and "
            f"{right.shape}; if the arms were not run on the same initial states, "
            "they are not paired and this function is the wrong tool"
        )
    return bootstrap_ci(left - right, seed=seed, **kwargs)


@dataclass
class RetentionMatrix:
    """``R[i][j]``: success on task ``j`` after training through stage ``i`` (§8.2).

    Both indices are 0-based over the curriculum order. ``R`` is lower-triangular in
    practice for the tasks seen so far, but the full matrix is stored because §8.2's
    FWT needs the above-diagonal entries (performance on a task before it was trained).
    """

    task_keys: tuple[str, ...]
    values: np.ndarray  # (T, T) float, NaN where not evaluated
    n_rollouts: np.ndarray  # (T, T) int

    @classmethod
    def empty(cls, task_keys: tuple[str, ...]) -> "RetentionMatrix":
        n = len(task_keys)
        if n == 0:
            raise ValueError("RetentionMatrix needs at least one task")
        return cls(
            task_keys=tuple(task_keys),
            values=np.full((n, n), np.nan, dtype=np.float64),
            n_rollouts=np.zeros((n, n), dtype=np.int64),
        )

    @property
    def n_tasks(self) -> int:
        return len(self.task_keys)

    def set(self, stage: int, task: int, rate: float, n: int) -> None:
        self._check(stage, task)
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"success rate {rate} outside [0, 1]")
        self.values[stage, task] = rate
        self.n_rollouts[stage, task] = n

    def get(self, stage: int, task: int) -> float:
        self._check(stage, task)
        value = self.values[stage, task]
        if np.isnan(value):
            raise KeyError(
                f"R[{stage}][{task}] ({self.task_keys[task]}) was never evaluated"
            )
        return float(value)

    def _check(self, stage: int, task: int) -> None:
        n = self.n_tasks
        if not 0 <= stage < n or not 0 <= task < n:
            raise IndexError(
                f"({stage}, {task}) out of range for a {n}x{n} retention matrix"
            )

    def assert_complete_through(self, stage: int) -> None:
        """Assert every task up to and including ``stage`` was evaluated at ``stage``."""
        missing = [
            self.task_keys[j]
            for j in range(stage + 1)
            if np.isnan(self.values[stage, j])
        ]
        if missing:
            raise ValueError(
                f"stage {stage} is missing evaluations for {missing}; the retention "
                "matrix row is incomplete so forgetting cannot be computed"
            )


# ---- §8.2 summary metrics ------------------------------------------------------


def final_average_success(matrix: RetentionMatrix) -> float:
    """``F_1 = mean_j R[T][j]`` — the primary metric (§8.2).

    ``T`` is the last stage. Averages over *all* tasks, so a method that keeps the
    last task and drops the rest scores badly, which is the point.
    """
    last = matrix.n_tasks - 1
    matrix.assert_complete_through(last)
    return float(np.mean(matrix.values[last, : matrix.n_tasks]))


def per_task_forgetting(matrix: RetentionMatrix) -> dict[str, float]:
    """``R[j][j] - R[T][j]`` per task (§8.2).

    Positive means the task got worse after later training. The last task's value is 0
    by construction, and is included so the dict covers every task.
    """
    last = matrix.n_tasks - 1
    matrix.assert_complete_through(last)
    out = {}
    for j, key in enumerate(matrix.task_keys):
        if np.isnan(matrix.values[j, j]):
            raise ValueError(
                f"diagonal entry R[{j}][{j}] for {key} was never evaluated; "
                "per-task forgetting needs the just-after-training value"
            )
        out[key] = float(matrix.values[j, j] - matrix.values[last, j])
    return out


def negative_backward_transfer(matrix: RetentionMatrix) -> float:
    """NBT: mean forgetting over the tasks that *can* be forgotten.

    CLARE's negative backward transfer averages ``R[j][j] - R[T][j]`` over
    ``j = 0..T-1``, excluding the final task, which has no subsequent training and so
    contributes a structural zero. Including it would dilute NBT by ``1/T`` purely as
    a function of curriculum length, making runs of different lengths incomparable.

    Reference: CLARE / lifelong-robot-learning convention, as cited in §8.2.
    """
    last = matrix.n_tasks - 1
    if last == 0:
        raise ValueError("NBT is undefined for a single-task curriculum")
    matrix.assert_complete_through(last)
    diffs = [
        matrix.values[j, j] - matrix.values[last, j] for j in range(last)
    ]
    return float(np.mean(diffs))


def forward_transfer(matrix: RetentionMatrix, baseline: dict[str, float]) -> float:
    """FWT: mean of ``R[j-1][j] - baseline_j`` over ``j = 1..T``.

    ``R[j-1][j]`` is performance on task ``j`` *before* it was ever trained, and the
    baseline is an independently trained single-task reference (§10.4's reference
    runs). Positive FWT means earlier tasks helped.

    Args:
        matrix: Retention matrix with above-diagonal entries filled in.
        baseline: ``task_key -> independent single-task success rate``.

    Reference: CLARE / lifelong-robot-learning convention, as cited in §8.2.
    """
    if matrix.n_tasks < 2:
        raise ValueError("FWT is undefined for a single-task curriculum")

    missing = [k for k in matrix.task_keys[1:] if k not in baseline]
    if missing:
        raise KeyError(
            f"FWT needs a single-task baseline for {missing}; run the §10.4 "
            "independent single-task references first"
        )

    diffs = []
    for j in range(1, matrix.n_tasks):
        before = matrix.values[j - 1, j]
        if np.isnan(before):
            raise ValueError(
                f"FWT needs R[{j - 1}][{j}] (task {matrix.task_keys[j]} before it was "
                "trained), which was never evaluated. Evaluate all tasks at every "
                "stage, not just the seen ones."
            )
        diffs.append(before - baseline[matrix.task_keys[j]])
    return float(np.mean(diffs))


def auc_average_success(matrix: RetentionMatrix) -> float:
    """AUC: mean over stages of the average success on tasks seen so far.

    Rewards keeping performance high *throughout* the curriculum rather than only at
    the end, so a method that collapses mid-run and recovers is distinguishable from
    one that never collapses.

    Reference: CLARE / lifelong-robot-learning convention, as cited in §8.2.
    """
    per_stage = []
    for i in range(matrix.n_tasks):
        matrix.assert_complete_through(i)
        per_stage.append(float(np.mean(matrix.values[i, : i + 1])))
    return float(np.mean(per_stage))


@dataclass(frozen=True)
class CLSummary:
    """All §8.2 headline numbers for one run."""

    final_average_success: float
    negative_backward_transfer: float
    auc: float
    per_task_forgetting: dict[str, float]
    forward_transfer: float | None = None

    def as_dict(self) -> dict:
        out = {
            "F_1": self.final_average_success,
            "NBT": self.negative_backward_transfer,
            "AUC": self.auc,
            "per_task_forgetting": self.per_task_forgetting,
        }
        if self.forward_transfer is not None:
            out["FWT"] = self.forward_transfer
        return out


def summarize(
    matrix: RetentionMatrix, baseline: dict[str, float] | None = None
) -> CLSummary:
    """Compute every §8.2 metric. FWT is included only when a baseline is supplied."""
    return CLSummary(
        final_average_success=final_average_success(matrix),
        negative_backward_transfer=negative_backward_transfer(matrix),
        auc=auc_average_success(matrix),
        per_task_forgetting=per_task_forgetting(matrix),
        forward_transfer=forward_transfer(matrix, baseline) if baseline else None,
    )
