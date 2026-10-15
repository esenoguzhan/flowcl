"""Scores, the cluster bootstrap and the selection for Track 2's step 3 (docs/thesis_plan.md, Workstream D;
the rule is the ``selection`` section of ``configs/analysis/behavior_predictors.yaml``). Pure: arrays in,
numbers out; no file access.

* :func:`ranking_pairs` / :func:`concordance` - claim (a): Harrell's concordance over within-task pairs
  whose success counts differ (pair-weighted; ties in the measure count 1/2).
* :func:`fit_logistic` / :func:`cross_fit` - the pooled binomial logistic curve, fitted by Newton's
  method with a small ridge, and its leave-one-root-out predictions.
* :func:`calibration_gap` - claim (b): K_b, the cell-weighted mean over eligible tasks of the absolute
  calibration-in-the-large |mean(observed - predicted)|; :func:`brier` the secondary probability score.
* :func:`bootstrap_weights` - the two-level cluster bootstrap (roots, then branches within each drawn
  root) as per-cell multiplicities; every score accepts such weights, so all candidates share resamples.
* :func:`select` - the rule's choice per claim (best point improvement over the reference; ties by list
  order; carried only if > 0; undefined candidates never selectable).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Cells:
    """The development (or test) cells, aligned: one entry per cell."""

    ids: tuple[str, ...]
    task: np.ndarray        # (n,) object: task keys
    root: np.ndarray        # (n,) object
    branch: np.ndarray      # (n,) object
    k: np.ndarray           # (n,) successes
    n: np.ndarray           # (n,) trials

    @property
    def rate(self) -> np.ndarray:
        return self.k / self.n

    def subset(self, mask: np.ndarray) -> "Cells":
        mask = np.asarray(mask, dtype=bool)
        return Cells(tuple(i for i, m in zip(self.ids, mask) if m), self.task[mask], self.root[mask],
                     self.branch[mask], self.k[mask], self.n[mask])

    def with_branch(self, branch) -> "Cells":
        return Cells(self.ids, self.task, self.root, np.asarray(branch, dtype=object), self.k, self.n)


def log_values(values) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(x)) or np.any(x <= 0):
        raise ValueError("every measure value must be finite and positive before the log transform")
    return np.log(x)


# ---- claim (a): within-task concordance -----------------------------------------------------------


def ranking_pairs(cells: Cells, min_count_difference: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """``(i, j)`` index arrays of the within-task pairs with |k_i - k_j| >= ``min_count_difference``,
    ordered so that cell ``i`` has the higher success."""
    left, right = [], []
    for task in sorted(set(cells.task)):
        idx = np.flatnonzero(cells.task == task)
        for a_pos, a in enumerate(idx):
            for b in idx[a_pos + 1:]:
                d = int(cells.k[a]) - int(cells.k[b])
                if abs(d) >= min_count_difference and d != 0:
                    hi, lo = (a, b) if d > 0 else (b, a)
                    left.append(hi)
                    right.append(lo)
    return np.asarray(left, dtype=np.int64), np.asarray(right, dtype=np.int64)


def pair_concordance(x: np.ndarray, pairs: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Per pair: 1 if the higher-success cell has the lower measure, 1/2 on a tie, else 0."""
    hi, lo = pairs
    return np.where(x[hi] < x[lo], 1.0, np.where(x[hi] == x[lo], 0.5, 0.0))


def pair_weights(weights: np.ndarray, pairs: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """(B, n_pairs): a pair's weight in each resample, the product of its two cells' multiplicities
    (shared by every candidate, so computed once)."""
    hi, lo = pairs
    return weights[:, hi] * weights[:, lo]


def concordance(conc: np.ndarray, pair_w: np.ndarray | None = None):
    """Pair-weighted C; with ``pair_w`` (:func:`pair_weights`) one value per resample, NaN where no pair
    has weight."""
    if pair_w is None:
        return float(conc.mean()) if conc.size else float("nan")
    total = pair_w.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, (pair_w @ conc) / np.where(total > 0, total, 1.0), np.nan)


def task_weighted_concordance(conc: np.ndarray, pairs, cells: Cells) -> float:
    """Each task's C, averaged over tasks with at least one pair (a sensitivity)."""
    hi, _ = pairs
    tasks = cells.task[hi]
    values = [conc[tasks == t].mean() for t in sorted(set(tasks))]
    return float(np.mean(values)) if values else float("nan")


def per_task_concordance(conc: np.ndarray, pairs, cells: Cells) -> dict[str, dict]:
    hi, _ = pairs
    tasks = cells.task[hi]
    return {t: {"c": float(conc[tasks == t].mean()), "pairs": int((tasks == t).sum())}
            for t in sorted(set(tasks))}


# ---- claim (b): the pooled logistic curve and its calibration -------------------------------------


def fit_logistic(X: np.ndarray, k: np.ndarray, n: np.ndarray, ridge: float = 1e-6, max_iter: int = 100,
                 tol: float = 1e-10) -> np.ndarray | None:
    """Binomial logistic regression (``k`` successes of ``n``) on design ``X`` by Newton's method with an
    L2 ridge on every coefficient; None if it does not converge within ``max_iter`` steps."""
    X = np.asarray(X, dtype=np.float64)
    k, n = np.asarray(k, dtype=np.float64), np.asarray(n, dtype=np.float64)
    beta = np.zeros(X.shape[1])
    for _ in range(int(max_iter)):
        eta = np.clip(X @ beta, -50, 50)
        p = 1.0 / (1.0 + np.exp(-eta))
        grad = X.T @ (k - n * p) - ridge * beta
        hess = (X * (n * p * (1 - p))[:, None]).T @ X + ridge * np.eye(X.shape[1])
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            return None
        beta = beta + step
        if not np.all(np.isfinite(beta)):
            return None
        if np.max(np.abs(step)) < tol:
            return beta
    return None


def predict(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(beta[0] + beta[1] * x, -50, 50)))


def design(x: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones_like(x), x])


def cross_fit(x: np.ndarray, cells: Cells, ridge: float, max_iter: int, tol: float) -> np.ndarray | None:
    """Leave-one-root-out predictions of the pooled curve; None if any fold fails to converge."""
    out = np.empty(len(x))
    for r in sorted(set(cells.root)):
        held = cells.root == r
        beta = fit_logistic(design(x[~held]), cells.k[~held], cells.n[~held], ridge, max_iter, tol)
        if beta is None:
            return None
        out[held] = predict(beta, x[held])
    return out


def eligible_tasks(cells: Cells, min_task_cells: int) -> np.ndarray:
    """Per cell: whether its task has at least ``min_task_cells`` cells (fixed on the original data)."""
    counts = {t: int((cells.task == t).sum()) for t in set(cells.task)}
    return np.asarray([counts[t] >= min_task_cells for t in cells.task])


def calibration_gap(residual: np.ndarray, cells: Cells, eligible: np.ndarray, weights: np.ndarray | None = None):
    """K_b = sum_t |sum_{i in t} w_i e_i| / sum_t sum_{i in t} w_i over eligible tasks t (e = observed -
    predicted); one value per resample with ``weights`` (B, n_cells); NaN where no eligible cell is drawn."""
    tasks = sorted(set(cells.task[eligible]))
    if weights is None:
        weights = np.ones((1, len(residual)))
        single = True
    else:
        single = False
    num = np.zeros(weights.shape[0])
    den = np.zeros(weights.shape[0])
    for t in tasks:
        m = (cells.task == t) & eligible
        wt = weights[:, m]
        num += np.abs(wt @ residual[m])
        den += wt.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    return float(out[0]) if single else out


def brier(residual: np.ndarray, weights: np.ndarray | None = None):
    sq = residual ** 2
    if weights is None:
        return float(sq.mean())
    total = weights.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, (weights * sq).sum(axis=1) / np.where(total > 0, total, 1.0), np.nan)


def task_intercept_deviance_gain(x: np.ndarray, cells: Cells, ridge: float, max_iter: int, tol: float):
    """In-sample deviance of the pooled curve minus that of the curve with per-task intercepts (descriptive:
    how much cross-task miscalibration remains); None if either fit fails."""
    tasks = sorted(set(cells.task))
    dummies = np.column_stack([(cells.task == t).astype(float) for t in tasks[1:]]) if len(tasks) > 1 else \
        np.zeros((len(x), 0))
    out = []
    for X in (design(x), np.column_stack([design(x), dummies])):
        beta = fit_logistic(X, cells.k, cells.n, ridge, max_iter, tol)
        if beta is None:
            return None
        p = np.clip(1.0 / (1.0 + np.exp(-np.clip(X @ beta, -50, 50))), 1e-12, 1 - 1e-12)
        out.append(-2.0 * float(np.sum(cells.k * np.log(p) + (cells.n - cells.k) * np.log(1 - p))))
    return out[0] - out[1]


# ---- the two-level cluster bootstrap -------------------------------------------------------------


def bootstrap_weights(cells: Cells, n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, n_cells) multiplicities: draw roots with replacement, then, within each drawn root, its
    branches with replacement; every cell of a drawn branch is counted once per draw."""
    rng = np.random.default_rng(seed)
    roots = sorted(set(cells.root))
    branches = {r: sorted(set(cells.branch[cells.root == r])) for r in roots}
    members = {(r, b): np.flatnonzero((cells.root == r) & (cells.branch == b)) for r in roots for b in branches[r]}
    W = np.zeros((n_boot, len(cells.ids)))
    for i in range(n_boot):
        for r in rng.choice(len(roots), size=len(roots), replace=True):
            root = roots[r]
            bs = branches[root]
            for b in rng.choice(len(bs), size=len(bs), replace=True):
                W[i, members[(root, bs[b])]] += 1.0
    return W


def interval(samples: np.ndarray, confidence: float) -> tuple[float, float] | None:
    s = np.asarray(samples, dtype=np.float64)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return None
    a = (1.0 - confidence) / 2
    lo, hi = np.quantile(s, [a, 1 - a])
    return float(lo), float(hi)


# ---- the selection -------------------------------------------------------------------------------


def select(improvements: dict[str, float | None], order: list[str], reference: str,
           not_selectable: set | frozenset = frozenset()) -> dict:
    """The rule's choice for one claim: the best point improvement over ``reference`` among defined,
    selectable candidates (ties by ``order``); carried only if it is > 0."""
    best, best_value = None, None
    for name in order:
        if name == reference or name in not_selectable:
            continue
        value = improvements.get(name)
        if value is None or not np.isfinite(value):
            continue
        if best_value is None or value > best_value:
            best, best_value = name, value
    carried = best is not None and best_value > 0
    return {"best": best, "improvement": best_value, "carried": best if carried else None,
            "status": "carried" if carried else "not_shown"}
