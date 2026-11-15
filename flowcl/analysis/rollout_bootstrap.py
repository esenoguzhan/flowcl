"""A5: rollout-level paired bootstrap for ACC, NBT and AUC (docs/thesis_plan.md, A5; status.md §5).

The interval is the evaluation uncertainty of **one trained policy**: which 50 initial states the
rollouts happened to draw. It says nothing about variation over training seeds (thesis_plan.md:
"with 3 seeds there is no seed-level CI"), so seeds are shown separately, never pooled into it.

**Pairing.** A rollout's seed depends only on (namespace, task, episode) (``seeding.py``), so
episode ``e`` of task ``j`` starts from the same state and noise at every stage, and in every
run of the same namespace. A bootstrap replicate therefore draws episode indices **once per
task column** and applies them to every stage of that task (NBT and AUC compare stages of the
same task), and to both runs of a paired comparison. Columns are resampled independently.
Before resampling, the episode seeds of each column are checked equal across its stages (and
across the two runs); a mismatch is an error, not a silent unpairing.

**Missing cells.** Upper-triangle cells (unseen tasks) are not used and may be absent (A3). A
cell that a requested metric needs is required: ACC the final row, NBT the diagonal and final
row before the last task, AUC the whole lower triangle.

The metric definitions are :mod:`flowcl.analysis.metrics`' (ACC is ``result.json``'s ``F_1``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flowcl.analysis.metrics import DEFAULT_CONFIDENCE, DEFAULT_N_BOOTSTRAP, Estimate

METRICS = ("ACC", "NBT", "AUC")
# F_first (README §8.2's F₁, first-task forgetting R[0][0] - R[1][0]) on request; not a default.
ALL_METRICS = (*METRICS, "F_first")


@dataclass(frozen=True)
class RunRollouts:
    """Per-cell rollout outcomes and episode seeds of one run, keyed ``(stage, task position)``."""

    task_keys: tuple[str, ...]
    cells: dict[tuple[int, int], np.ndarray]
    seeds: dict[tuple[int, int], tuple]
    namespace: str | None = None
    label: str = ""

    @classmethod
    def from_payload(cls, payload: dict, label: str = "") -> "RunRollouts":
        """From a ``result.json`` payload (its stage records carry every rollout)."""
        keys = tuple(payload["task_keys"])
        position = {k: j for j, k in enumerate(keys)}
        cells, seeds = {}, {}
        for record in payload["stages"]:
            i = int(record["stage"])
            for entry in (record.get("evaluation") or {}).get("tasks", []):
                j = position.get(entry["task_key"])
                if j is None:
                    continue
                cells[(i, j)] = np.asarray(entry["successes"], dtype=np.float64)
                seeds[(i, j)] = tuple(entry.get("seeds") or ())
        return cls(keys, cells, seeds, payload.get("seed_namespace_run_id"),
                   label or payload.get("run_id", ""))

    @property
    def n_tasks(self) -> int:
        return len(self.task_keys)

    def columns(self) -> dict[int, list[int]]:
        """``task position -> stages that evaluated it`` (sorted)."""
        out: dict[int, list[int]] = {}
        for i, j in sorted(self.cells):
            out.setdefault(j, []).append(i)
        return out

    def check_alignment(self) -> None:
        """Every column's cells share one episode-seed list and one rollout count."""
        for j, stages in self.columns().items():
            first = (stages[0], j)
            for i in stages[1:]:
                if self.seeds[(i, j)] != self.seeds[first]:
                    raise ValueError(f"{self.label}: task {self.task_keys[j]} has different episode seeds at "
                                     f"stages {stages[0]} and {i}; its cells are not paired across stages")
                if self.cells[(i, j)].shape != self.cells[first].shape:
                    raise ValueError(f"{self.label}: task {self.task_keys[j]} has different rollout counts "
                                     f"at stages {stages[0]} and {i}")


def required_cells(n_tasks: int, metrics=METRICS) -> set[tuple[int, int]]:
    last = n_tasks - 1
    need: set[tuple[int, int]] = set()
    for name in metrics:
        if name == "ACC":
            need |= {(last, j) for j in range(n_tasks)}
        elif name == "NBT":
            if last == 0:
                raise ValueError("NBT is undefined for a single-task curriculum")
            need |= {(j, j) for j in range(last)} | {(last, j) for j in range(last)}
        elif name == "AUC":
            need |= {(i, j) for i in range(n_tasks) for j in range(i + 1)}
        elif name == "F_first":
            if last == 0:
                raise ValueError("F_first is undefined for a single-task curriculum")
            need |= {(0, 0), (1, 0)}
        else:
            raise ValueError(f"unknown metric {name!r}; expected one of {ALL_METRICS}")
    return need


def _check_required(run: RunRollouts, metrics) -> set[tuple[int, int]]:
    need = required_cells(run.n_tasks, metrics)
    missing = sorted(need - set(run.cells))
    if missing:
        names = [f"R[{i}][{j}] ({run.task_keys[j]})" for i, j in missing[:5]]
        raise ValueError(f"{run.label}: {', '.join(metrics)} need cells that were not evaluated: {names}")
    return need


def metric_values(values: np.ndarray, metrics=METRICS) -> dict[str, np.ndarray]:
    """ACC / NBT / AUC / F_first of retention matrices ``values`` of shape ``(..., T, T)`` (NaN = absent)."""
    n = values.shape[-1]
    last = n - 1
    out = {}
    for name in metrics:
        if name == "ACC":
            out[name] = values[..., last, :].mean(axis=-1)
        elif name == "NBT":
            diag = np.stack([values[..., j, j] for j in range(last)], axis=-1)
            final = values[..., last, :last]
            out[name] = (diag - final).mean(axis=-1)
        elif name == "AUC":
            out[name] = np.stack([values[..., i, : i + 1].mean(axis=-1) for i in range(n)], axis=-1).mean(axis=-1)
        elif name == "F_first":
            out[name] = values[..., 0, 0] - values[..., 1, 0]
    return out


def _point_matrix(run: RunRollouts) -> np.ndarray:
    values = np.full((run.n_tasks, run.n_tasks), np.nan)
    for (i, j), outcomes in run.cells.items():
        values[i, j] = outcomes.mean()
    return values


def _resampled(runs: list[RunRollouts], need: set[tuple[int, int]], n_bootstrap: int, seed: int) -> list[np.ndarray]:
    """``(n_bootstrap, T, T)`` per run, the same episode draw per column for every run."""
    n = runs[0].n_tasks
    rng = np.random.default_rng(seed)
    out = [np.full((n_bootstrap, n, n), np.nan) for _ in runs]
    for j in range(n):
        stages = sorted(i for i, jj in need if jj == j)
        if not stages:
            continue
        size = runs[0].cells[(stages[0], j)].size
        idx = rng.integers(0, size, size=(n_bootstrap, size))
        for r, run in enumerate(runs):
            for i in stages:
                out[r][:, i, j] = run.cells[(i, j)][idx].mean(axis=1)
    return out


def _estimate(point: float, draws: np.ndarray, confidence: float, n: int) -> Estimate:
    alpha = 1.0 - confidence
    low, high = np.quantile(draws, [alpha / 2, 1.0 - alpha / 2])
    return Estimate(value=float(point), low=float(min(low, point)), high=float(max(high, point)), n=n,
                    confidence=confidence)


def rollout_bootstrap(run: RunRollouts, metrics=METRICS, n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
                      confidence: float = DEFAULT_CONFIDENCE, seed: int = 0) -> dict[str, Estimate]:
    """Each metric of one run with its rollout-level, stage-paired percentile CI."""
    metrics = tuple(metrics)
    run.check_alignment()
    need = _check_required(run, metrics)
    point = metric_values(_point_matrix(run), metrics)
    draws = metric_values(_resampled([run], need, n_bootstrap, seed)[0], metrics)
    n = int(sum(run.cells[c].size for c in need))
    return {m: _estimate(point[m], draws[m], confidence, n) for m in metrics}


def paired_rollout_bootstrap(run_a: RunRollouts, run_b: RunRollouts, metrics=METRICS,
                             n_bootstrap: int = DEFAULT_N_BOOTSTRAP, confidence: float = DEFAULT_CONFIDENCE,
                             seed: int = 0) -> dict[str, Estimate]:
    """``metric(a) - metric(b)`` with the same episode draw on both runs (same namespace, same seeds)."""
    metrics = tuple(metrics)
    if run_a.task_keys != run_b.task_keys:
        raise ValueError(f"{run_a.label} and {run_b.label} have different task orders")
    if run_a.namespace and run_b.namespace and run_a.namespace != run_b.namespace:
        raise ValueError(f"{run_a.label} ({run_a.namespace}) and {run_b.label} ({run_b.namespace}) are in "
                         "different seed namespaces, so their rollouts are not paired")
    for run in (run_a, run_b):
        run.check_alignment()
    need = _check_required(run_a, metrics) | _check_required(run_b, metrics)
    for cell in sorted(need):
        if run_a.seeds[cell] != run_b.seeds[cell]:
            i, j = cell
            raise ValueError(f"R[{i}][{j}] ({run_a.task_keys[j]}): {run_a.label} and {run_b.label} used different "
                             "episode seeds, so the cell is not paired")
    point_a = metric_values(_point_matrix(run_a), metrics)
    point_b = metric_values(_point_matrix(run_b), metrics)
    res_a, res_b = _resampled([run_a, run_b], need, n_bootstrap, seed)
    draws_a, draws_b = metric_values(res_a, metrics), metric_values(res_b, metrics)
    n = int(sum(run_a.cells[c].size for c in need))
    return {m: _estimate(point_a[m] - point_b[m], draws_a[m] - draws_b[m], confidence, n) for m in metrics}
