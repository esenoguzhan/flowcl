"""Tables and plots for the results, with §11's "no bare percentages" made structural.

Spec §11: "no bare percentages, every number carries a CI". That is enforced rather
than remembered: :func:`format_cell` accepts an
:class:`~flowcl.analysis.metrics.Estimate` and raises on a bare float, so a table cannot
be produced with an uncertainty-free success rate in it.

Aggregation across seeds pools the *per-rollout* outcomes rather than averaging the
per-seed rates. With three seeds (§5) a bootstrap over three numbers is almost
uninformative, whereas pooling 3 x 50 rollouts gives an interval that reflects both
rollout and seed variation. The cost is that seed variance is folded into the interval
instead of being reported separately, so :func:`seed_spread` reports it alongside.

**The thesis tables use :func:`per_seed_table`** (A5): one row per method and seed, each value with
its rollout-level interval for that one trained policy, seeds never pooled (thesis_plan.md: with 3
seeds there is no seed-level CI). :func:`main_table` keeps ACC pooled over the seeds' rollouts,
labelled so, and shows NBT and AUC per seed with their rollout intervals.

**Names (A6):** ``result.json`` stores final average success under the key ``F_1``; tables call it
**ACC**. README §8.2's first-task forgetting R[0][0] - R[1][0] is **F_first**.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from flowcl.analysis.metrics import (
    DEFAULT_CONFIDENCE,
    Estimate,
    RetentionMatrix,
    bootstrap_ci,
)


def format_cell(value: Estimate, decimals: int = 1) -> str:
    """Format one table cell. Rejects bare numbers (§11).

    Raises:
        TypeError: If handed a float. This is the whole point of the function: a bare
            percentage in a results table is indistinguishable from one with its
            uncertainty dropped, so it must be impossible to write by accident.
    """
    if not isinstance(value, Estimate):
        raise TypeError(
            f"§11 requires every reported number to carry a CI, got "
            f"{type(value).__name__}. Wrap it in an Estimate, or use "
            f"bootstrap_ci()/success_estimate() to produce one."
        )
    return value.format_pp(decimals)


@dataclass
class LoadedRun:
    """One sequential run's ``result.json``, as written by the continual runner."""

    path: Path
    payload: dict

    @property
    def method(self) -> str:
        return self.payload["method"]

    @property
    def curriculum(self) -> str:
        return self.payload["curriculum"]

    @property
    def seed(self) -> int:
        return int(self.payload["seed"])

    @property
    def task_keys(self) -> tuple[str, ...]:
        return tuple(self.payload["task_keys"])

    @property
    def n_stages(self) -> int:
        return len(self.task_keys)

    def matrix(self) -> RetentionMatrix:
        values = np.array(
            [
                [np.nan if v is None else float(v) for v in row]
                for row in self.payload["retention_matrix"]["values"]
            ],
            dtype=np.float64,
        )
        return RetentionMatrix(
            task_keys=self.task_keys,
            values=values,
            n_rollouts=np.array(
                self.payload["retention_matrix"]["n_rollouts"], dtype=np.int64
            ),
        )

    def successes(self, stage: int, task_key: str) -> list[bool]:
        """Per-rollout outcomes for one cell, for pooled bootstrapping."""
        for record in self.payload["stages"]:
            if record["stage"] != stage:
                continue
            for entry in record["evaluation"]["tasks"]:
                if entry["task_key"] == task_key:
                    return list(entry["successes"])
        raise KeyError(
            f"{self.path}: no rollouts recorded for stage {stage}, task {task_key}"
        )

    @property
    def systems(self) -> dict:
        return self.payload.get("systems", {})


def load_runs(results_root: Path | None = None) -> list[LoadedRun]:
    """Load every ``result.json`` under ``results/``."""
    from flowcl.utils.libero_paths import repo_root

    root = Path(results_root) if results_root else (repo_root() / "results")
    if not root.is_dir():
        raise NotADirectoryError(f"results directory not found: {root}")

    runs = [
        LoadedRun(path=path, payload=json.loads(path.read_text()))
        for path in sorted(root.glob("*/result.json"))
    ]
    if not runs:
        raise FileNotFoundError(
            f"no result.json found under {root}; run scripts/run_continual.py first"
        )
    return runs


def group_by_method(runs: list[LoadedRun], curriculum: str) -> dict[str, list[LoadedRun]]:
    """``method -> runs`` for one curriculum, with seeds ordered."""
    grouped: dict[str, list[LoadedRun]] = {}
    for run in runs:
        if run.curriculum == curriculum:
            grouped.setdefault(run.method, []).append(run)
    for method, group in grouped.items():
        group.sort(key=lambda r: r.seed)
        seeds = [r.seed for r in group]
        if len(set(seeds)) != len(seeds):
            raise ValueError(
                f"{curriculum}/{method} has duplicate seeds {seeds}; two runs would "
                "be double-counted in the pooled interval"
            )
    return grouped


# ---- aggregated metrics --------------------------------------------------------


def pooled_final_average_success(
    group: list[LoadedRun],
    bootstrap: dict | None = None,
) -> Estimate:
    """ACC (``result.json``'s ``F_1``) with a CI over rollouts pooled across seeds and tasks.

    ACC is the mean of the final row (§8.2), so pooling every final-stage rollout
    from every task and seed estimates exactly that mean — provided each task
    contributes the same number of rollouts, which §8.1's fixed 50-per-cell guarantees.
    Unequal counts are rejected rather than silently reweighting the tasks.
    """
    if not group:
        raise ValueError("pooled_final_average_success received no runs")
    bootstrap = bootstrap or {}

    per_cell = []
    for run in group:
        last = run.n_stages - 1
        for task_key in run.task_keys:
            per_cell.append(run.successes(last, task_key))

    sizes = {len(cell) for cell in per_cell}
    if len(sizes) != 1:
        raise ValueError(
            f"cells have differing rollout counts {sorted(sizes)}; pooling would "
            "weight tasks unequally. §8.1 fixes 50 rollouts per cell."
        )

    return bootstrap_ci(
        np.concatenate([np.asarray(c, dtype=np.float64) for c in per_cell]),
        seed=bootstrap.get("seed", 0),
        n_bootstrap=bootstrap.get("n_resamples", 10000),
        confidence=bootstrap.get("confidence", DEFAULT_CONFIDENCE),
    )


def pooled_metric_over_seeds(
    group: list[LoadedRun],
    metric,
    bootstrap: dict | None = None,
) -> Estimate:
    """CI for a matrix-level metric (NBT, AUC) by bootstrapping over seeds.

    Unlike ACC, these are nonlinear functions of the whole retention matrix, so they
    cannot be recovered by pooling rollouts. **Not used by the thesis tables:** with three
    seeds a seed-level interval is not defensible (thesis_plan.md, Statistics); use
    :func:`per_seed_table` / :func:`flowcl.analysis.rollout_bootstrap.rollout_bootstrap`.
    """
    if not group:
        raise ValueError("pooled_metric_over_seeds received no runs")
    bootstrap = bootstrap or {}
    values = np.array([metric(run.matrix()) for run in group], dtype=np.float64)
    return bootstrap_ci(
        values,
        seed=bootstrap.get("seed", 0),
        n_bootstrap=bootstrap.get("n_resamples", 10000),
        confidence=bootstrap.get("confidence", DEFAULT_CONFIDENCE),
    )


def seed_spread(group: list[LoadedRun], metric) -> dict:
    """Per-seed values of a metric, so seed variance is visible, not just pooled in."""
    return {
        run.seed: float(metric(run.matrix())) for run in group
    }


# ---- tables --------------------------------------------------------------------


@dataclass
class Table:
    """A rendered table. Cells are already formatted strings with their CIs."""

    title: str
    columns: list[str]
    rows: list[list[str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [f"### {self.title}", ""]
        lines.append("| " + " | ".join(self.columns) + " |")
        lines.append("| " + " | ".join("---" for _ in self.columns) + " |")
        for row in self.rows:
            lines.append("| " + " | ".join(row) + " |")
        if self.notes:
            lines.append("")
            lines.extend(f"- {note}" for note in self.notes)
        return "\n".join(lines) + "\n"

    def to_csv(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(self.columns)
            writer.writerows(self.rows)
        return path


def _rollout_estimates(run: LoadedRun, metrics, bootstrap: dict) -> dict[str, Estimate]:
    from flowcl.analysis.rollout_bootstrap import RunRollouts, rollout_bootstrap

    return rollout_bootstrap(
        RunRollouts.from_payload(run.payload, label=str(run.path)),
        metrics=metrics,
        n_bootstrap=bootstrap.get("n_resamples", 10000),
        confidence=bootstrap.get("confidence", DEFAULT_CONFIDENCE),
        seed=bootstrap.get("seed", 0),
    )


def main_table(
    runs: list[LoadedRun],
    curriculum: str,
    bootstrap: dict | None = None,
) -> Table:
    """The §8.2 headline table for one curriculum: ACC, NBT, AUC per method.

    ACC pools every final-stage rollout of the method's seeds (labelled as pooled; it is
    not a seed-level interval). NBT and AUC are shown per seed, each with its rollout-level
    interval, because with three seeds no seed-level interval is defensible.
    """
    bootstrap = bootstrap or {}
    grouped = group_by_method(runs, curriculum)
    if not grouped:
        raise ValueError(f"no runs found for curriculum {curriculum!r}")

    table = Table(
        title=f"{curriculum}: overall performance (ACC), forgetting (NBT) and AUC",
        columns=[
            "method",
            "seeds",
            "ACC (pp, pooled rollouts, 95% CI)",
            "NBT by seed (pp, rollout 95% CI)",
            "AUC by seed (pp, rollout 95% CI)",
            "exemplar-free",
            "stored (MB)",
        ],
    )

    for method in sorted(grouped):
        group = grouped[method]
        exemplar_free = all(
            run.systems.get("is_exemplar_free", True) for run in group
        )
        stored = max(run.systems.get("method_stored_mb", 0.0) for run in group)
        per_seed = {run.seed: _rollout_estimates(run, ("NBT", "AUC"), bootstrap) for run in group}
        table.rows.append(
            [
                method,
                str([run.seed for run in group]),
                format_cell(pooled_final_average_success(group, bootstrap)),
                "; ".join(f"s{seed} {format_cell(est['NBT'])}" for seed, est in per_seed.items()),
                "; ".join(f"s{seed} {format_cell(est['AUC'])}" for seed, est in per_seed.items()),
                "yes" if exemplar_free else "NO (§6: violates exemplar-free)",
                f"{stored:.2f}",
            ]
        )

    table.notes = [
        "ACC is the final average success (stored in result.json as F_1). Its interval pools "
        "the final-stage rollouts of every seed: rollout uncertainty with the seeds mixed in, "
        "not a seed-level interval.",
        "NBT and AUC: one value per seed (sN), each with a rollout bootstrap that resamples "
        "episodes once per task and applies them to every stage (A5). Three seeds give no "
        "seed-level interval. Seed 0 is the development seed.",
        "Intervals are percentile bootstrap at 95% (§8.2). The thesis tables use per_seed_table.",
    ]
    return table


def per_seed_table(
    runs: list[LoadedRun],
    curriculum: str,
    bootstrap: dict | None = None,
) -> Table:
    """A5's thesis table: one row per method and seed, every value with its rollout interval.

    The interval is the evaluation uncertainty of that one trained policy (which initial
    states the 50 rollouts drew), resampled once per task and applied to every stage, so
    NBT, AUC and F_first are paired across stages. Seeds are separate rows and are never
    pooled: their spread is the training-seed variation, shown, not summarised by an
    interval.
    """
    bootstrap = bootstrap or {}
    grouped = group_by_method(runs, curriculum)
    if not grouped:
        raise ValueError(f"no runs found for curriculum {curriculum!r}")
    table = Table(
        title=f"{curriculum}: per seed, with rollout intervals (one trained policy per row)",
        columns=[
            "method",
            "seed",
            "ACC (pp, rollout 95% CI)",
            "NBT (pp, rollout 95% CI)",
            "AUC (pp, rollout 95% CI)",
            "F_first (pp, rollout 95% CI)",
            "exemplar-free",
        ],
    )
    for method in sorted(grouped):
        for run in grouped[method]:
            est = _rollout_estimates(run, ("ACC", "NBT", "AUC", "F_first"), bootstrap)
            table.rows.append(
                [
                    method,
                    f"{run.seed}{' (dev)' if run.seed == 0 else ''}",
                    format_cell(est["ACC"]),
                    format_cell(est["NBT"]),
                    format_cell(est["AUC"]),
                    format_cell(est["F_first"]),
                    "yes" if run.systems.get("is_exemplar_free", True) else "NO",
                ]
            )
    table.notes = [
        "Uncertainty source 1, rollouts: each interval resamples one policy's evaluation "
        "episodes (once per task, the same draw for every stage), percentile bootstrap at 95%.",
        "Uncertainty source 2, training seeds: the rows of a method. Three seeds give no "
        "seed-level interval; the claim rule is a consistent sign. Seed 0 (dev) is the "
        "development seed; seeds 1-2 are the replication.",
        "ACC = final average success (result.json's F_1). F_first = R[0][0] - R[1][0], the "
        "first task's forgetting after one more task (README §8.2).",
    ]
    return table


def retention_table(run: LoadedRun) -> Table:
    """One run's retention matrix ``R[i][j]``, as percentages with rollout counts."""
    matrix = run.matrix()
    table = Table(
        title=f"{run.curriculum}/{run.method} seed {run.seed}: retention matrix",
        columns=["after stage"] + [key.split("/")[-1][:28] for key in run.task_keys],
    )
    for i in range(matrix.n_tasks):
        row = [f"{i} ({run.task_keys[i].split('/')[-1][:20]})"]
        for j in range(matrix.n_tasks):
            value = matrix.values[i, j]
            row.append(
                "-"
                if np.isnan(value)
                else f"{100 * value:.0f} (n={matrix.n_rollouts[i, j]})"
            )
        table.rows.append(row)
    table.notes = [
        "Entry (i, j) is success on task j after training through stage i (§8.2). "
        "Above-diagonal entries are performance before the task was trained, which FWT "
        "needs.",
    ]
    return table


def systems_table(runs: list[LoadedRun], curriculum: str) -> Table:
    """The §8.2 systems table: parameters, stored memory, wall clock."""
    grouped = group_by_method(runs, curriculum)
    table = Table(
        title=f"{curriculum}: system cost (§8.2)",
        columns=[
            "method",
            "trainable params T1 (M)",
            "trainable params T2+ (M)",
            "frozen params (M)",
            "stored bases/buffers (MB)",
            "train wall clock (min)",
            "total wall clock (min)",
            "peak GPU train (GB, allocated)",
            "peak GPU eval (GB, allocated)",
        ],
    )
    for method in sorted(grouped):
        group = grouped[method]
        systems = [run.systems for run in group]
        table.rows.append(
            [
                method,
                f"{np.mean([_t1_trainable(s) for s in systems]) / 1e6:.1f}",
                _later_trainable(systems),
                f"{np.mean([s.get('frozen_params', 0) for s in systems]) / 1e6:.1f}",
                f"{np.mean([s.get('method_stored_mb', 0.0) for s in systems]):.2f}",
                f"{np.mean([s.get('train_wall_clock_s', 0.0) for s in systems]) / 60:.1f}",
                f"{np.mean([s.get('total_wall_clock_s', 0.0) for s in systems]) / 60:.1f}",
                _peak_gb(systems, "train_allocated_bytes"),
                _peak_gb(systems, "eval_allocated_bytes"),
            ]
        )
    table.notes = [
        "Means over seeds. Inference latency and ODE integration time are measured "
        "separately by scripts/measure_latency.py, since they do not vary by seed.",
        "Trainable params are counted per stage after the method's setup (GPM trains its "
        "allowlist from T2, LoRA its adapters); T2+ is the mean over stages >= 1, with the "
        "range when they differ. Runs recorded before per-stage counts show T1's count only.",
        "Peak GPU: the PyTorch CUDA allocator's maximum over the stages' training and "
        "evaluation windows (not EGL rendering or the CUDA context); the largest over seeds. "
        "'n/a' where it was not measured (runs before Sun 15 Nov, imported stages).",
    ]
    return table


def _peak_gb(systems: list[dict], key: str) -> str:
    """The largest recorded peak over seeds, in GB; "n/a" if any seed did not measure it."""
    values = [(s.get("peak_gpu") or {}).get(key) for s in systems]
    if not values or any(v is None for v in values):
        return "n/a"
    return f"{max(values) / 1e9:.1f}"


def _t1_trainable(systems: dict) -> float:
    by_stage = systems.get("trainable_params_by_stage") or {}
    value = by_stage.get("0")
    return value if value is not None else systems.get("trainable_params", 0)


def _later_trainable(systems: list[dict]) -> str:
    """Mean (and range) of the stages >= 1's trainable counts, in M; "n/a (T1 only)" for old runs."""
    values = []
    for s in systems:
        by_stage = s.get("trainable_params_by_stage") or {}
        later = [v for k, v in by_stage.items() if int(k) >= 1 and v is not None]
        if not later:
            return "n/a (T1 only)"
        values.extend(later)
    if not values:
        return "n/a (T1 only)"
    mean = f"{np.mean(values) / 1e6:.1f}"
    lo, hi = min(values) / 1e6, max(values) / 1e6
    return mean if f"{lo:.1f}" == f"{hi:.1f}" else f"{mean} ({lo:.1f}–{hi:.1f})"


# ---- plots ---------------------------------------------------------------------


def plot_retention_heatmap(run: LoadedRun, path: Path) -> Path:
    """Heatmap of one run's retention matrix."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = run.matrix()
    fig, ax = plt.subplots(figsize=(1.6 * matrix.n_tasks + 2, 1.2 * matrix.n_tasks + 2))
    image = ax.imshow(
        100 * matrix.values, vmin=0, vmax=100, cmap="viridis", aspect="auto"
    )
    labels = [key.split("/")[-1][:24] for key in run.task_keys]
    ax.set_xticks(range(matrix.n_tasks), labels, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(matrix.n_tasks), [f"after {i}" for i in range(matrix.n_tasks)])
    for i in range(matrix.n_tasks):
        for j in range(matrix.n_tasks):
            value = matrix.values[i, j]
            ax.text(
                j,
                i,
                "-" if np.isnan(value) else f"{100 * value:.0f}",
                ha="center",
                va="center",
                color="white" if np.isnan(value) or value < 0.6 else "black",
                fontsize=9,
            )
    ax.set_title(f"{run.curriculum} / {run.method} seed {run.seed}")
    fig.colorbar(image, ax=ax, label="success (%)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_forgetting_curves(
    runs: list[LoadedRun],
    curriculum: str,
    path: Path,
    bootstrap: dict | None = None,
) -> Path:
    """Success on task 1 as later tasks are trained, one line per method.

    The clearest single picture of forgetting: task 1 is trained once, at stage 0, and
    everything after that is pure interference. Error bars are bootstrap CIs over the
    rollouts pooled across seeds.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bootstrap = bootstrap or {}
    grouped = group_by_method(runs, curriculum)
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for method in sorted(grouped):
        group = grouped[method]
        first_task = group[0].task_keys[0]
        stages, centres, lows, highs = [], [], [], []
        for stage in range(group[0].n_stages):
            pooled = np.concatenate(
                [
                    np.asarray(run.successes(stage, first_task), dtype=np.float64)
                    for run in group
                ]
            )
            estimate = bootstrap_ci(
                pooled,
                seed=bootstrap.get("seed", 0),
                n_bootstrap=bootstrap.get("n_resamples", 2000),
                confidence=bootstrap.get("confidence", DEFAULT_CONFIDENCE),
            )
            stages.append(stage)
            centres.append(100 * estimate.value)
            lows.append(100 * (estimate.value - estimate.low))
            highs.append(100 * (estimate.high - estimate.value))
        ax.errorbar(
            stages,
            centres,
            yerr=[lows, highs],
            marker="o",
            capsize=3,
            label=method,
        )

    ax.set_xlabel("stages trained")
    ax.set_ylabel(f"success on task 1 (%)")
    ax.set_title(f"{curriculum}: retention of the first task")
    ax.set_ylim(-2, 102)
    ax.set_xticks(range(grouped[sorted(grouped)[0]][0].n_stages))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def build_all(
    results_root: Path | None = None,
    out_dir: Path | None = None,
    bootstrap: dict | None = None,
) -> Path:
    """Produce every table and plot for every curriculum present in ``results/``."""
    from flowcl.utils.libero_paths import repo_root

    runs = load_runs(results_root)
    out_dir = Path(out_dir) if out_dir else (repo_root() / "results" / "tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    sections = ["# flowcl results", ""]
    for curriculum in sorted({run.curriculum for run in runs}):
        subset = [run for run in runs if run.curriculum == curriculum]
        sections.append(f"## {curriculum}\n")

        main = main_table(subset, curriculum, bootstrap)
        main.to_csv(out_dir / f"{curriculum}_main.csv")
        sections.append(main.to_markdown())

        seeds = per_seed_table(subset, curriculum, bootstrap)
        seeds.to_csv(out_dir / f"{curriculum}_per_seed.csv")
        sections.append(seeds.to_markdown())

        sections.append(systems_table(subset, curriculum).to_markdown())

        for run in subset:
            sections.append(retention_table(run).to_markdown())
            plot_retention_heatmap(
                run,
                out_dir / f"{curriculum}_{run.method}_seed{run.seed}_retention.png",
            )

        plot_forgetting_curves(
            subset, curriculum, out_dir / f"{curriculum}_forgetting.png", bootstrap
        )

    path = out_dir / "results.md"
    path.write_text("\n".join(sections))
    return path
