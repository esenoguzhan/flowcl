"""§11's "no bare percentages" and the aggregation across seeds.

These tests run on synthetic ``result.json`` payloads rather than real runs, so the
reporting layer can be checked without a GPU or a rollout. The payload schema is the
one :meth:`flowcl.train.continual.ContinualResult.as_dict` writes; if that changes,
these tests fail, which is the intent.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from flowcl.analysis.metrics import (
    Estimate,
    auc_average_success,
    negative_backward_transfer,
)
from flowcl.analysis.tables import (
    LoadedRun,
    build_all,
    format_cell,
    group_by_method,
    load_runs,
    main_table,
    plot_forgetting_curves,
    plot_retention_heatmap,
    pooled_final_average_success,
    pooled_metric_over_seeds,
    retention_table,
    seed_spread,
    systems_table,
)

TASKS = ("libero_object/task_a", "libero_object/task_b")
N_ROLLOUTS = 20


def successes(rate: float, n: int = N_ROLLOUTS) -> list[bool]:
    """Exactly ``rate * n`` successes, so pooled means are predictable."""
    k = round(rate * n)
    if not np.isclose(k, rate * n):
        raise AssertionError(f"rate {rate} is not representable with n={n} rollouts")
    return [True] * k + [False] * (n - k)


def make_payload(
    method: str,
    curriculum: str,
    seed: int,
    rates: list[list[float]],
    systems: dict | None = None,
) -> dict:
    """A ``result.json`` payload whose retention matrix is exactly ``rates``."""
    n = len(TASKS)
    stages = []
    for i, row in enumerate(rates):
        stages.append(
            {
                "stage": i,
                "task_key": TASKS[i],
                "evaluation": {
                    "run_id": f"{curriculum}__{method}__seed{seed}",
                    "stage": i,
                    "tasks": [
                        {
                            "task_key": TASKS[j],
                            "successes": successes(row[j]),
                            "n_steps": [10] * N_ROLLOUTS,
                            "seeds": list(range(N_ROLLOUTS)),
                            "success_rate": row[j],
                            "ci_low": max(0.0, row[j] - 0.1),
                            "ci_high": min(1.0, row[j] + 0.1),
                            "n_rollouts": N_ROLLOUTS,
                        }
                        for j in range(n)
                    ],
                },
            }
        )
    return {
        "run_id": f"{curriculum}__{method}__seed{seed}",
        "method": method,
        "curriculum": curriculum,
        "seed": seed,
        "task_keys": list(TASKS),
        "retention_matrix": {
            "task_keys": list(TASKS),
            "values": [[row[j] for j in range(n)] for row in rates],
            "n_rollouts": [[N_ROLLOUTS] * n for _ in range(n)],
        },
        "stages": stages,
        "systems": systems
        or {
            "trainable_params": 30_000_000,
            "frozen_params": 50_000_000,
            "registry_layers": 40,
            "method_stored_mb": 0.0,
            "is_exemplar_free": True,
            "total_wall_clock_s": 600.0,
            "train_wall_clock_s": 540.0,
        },
    }


def write_run(root: Path, payload: dict) -> Path:
    directory = root / payload["run_id"]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "result.json"
    path.write_text(json.dumps(payload))
    return path


def loaded(payload: dict) -> LoadedRun:
    return LoadedRun(path=Path("memory"), payload=payload)


# Retains the first task perfectly; forgets it completely.
KEEP = [[1.0, 0.0], [1.0, 1.0]]
FORGET = [[1.0, 0.0], [0.0, 1.0]]


# ---- §11 enforcement ----------------------------------------------------------


def test_format_cell_renders_an_estimate_with_its_interval():
    assert format_cell(Estimate(0.8, 0.7, 0.9, n=50)) == "80.0 [70.0, 90.0]"


@pytest.mark.parametrize("value", [0.8, 80, "80.0%", np.float64(0.8)])
def test_format_cell_refuses_a_bare_number(value):
    """§11 is enforced by the type system, not by reviewer vigilance."""
    with pytest.raises(TypeError, match="every reported number to carry a CI"):
        format_cell(value)


def test_every_numeric_cell_in_the_main_table_carries_an_interval():
    """The structural claim: no success-rate column can render as a bare percentage."""
    runs = [loaded(make_payload("seq_ft", "seq_hetero", s, FORGET)) for s in (0, 1)]
    table = main_table(runs, "seq_hetero")
    metric_columns = [
        i for i, name in enumerate(table.columns) if "CI)" in name
    ]
    assert metric_columns, "main table has no metric columns to check"
    for row in table.rows:
        for i in metric_columns:
            assert "[" in row[i] and "," in row[i], f"bare number in cell {row[i]!r}"


# ---- loading and grouping -----------------------------------------------------


def test_load_runs_reads_every_result_json(tmp_path):
    write_run(tmp_path, make_payload("seq_ft", "seq_hetero", 0, FORGET))
    write_run(tmp_path, make_payload("replay", "seq_hetero", 0, KEEP))
    runs = load_runs(tmp_path)
    assert {run.method for run in runs} == {"seq_ft", "replay"}
    assert all(run.n_stages == len(TASKS) for run in runs)


def test_load_runs_complains_instead_of_producing_an_empty_table(tmp_path):
    with pytest.raises(FileNotFoundError, match="run_continual"):
        load_runs(tmp_path)
    with pytest.raises(NotADirectoryError):
        load_runs(tmp_path / "absent")


def test_duplicate_seeds_are_rejected_not_double_counted():
    runs = [loaded(make_payload("seq_ft", "seq_hetero", 0, FORGET)) for _ in range(2)]
    with pytest.raises(ValueError, match="duplicate seeds"):
        group_by_method(runs, "seq_hetero")


def test_grouping_ignores_other_curricula():
    runs = [
        loaded(make_payload("seq_ft", "seq_hetero", 0, FORGET)),
        loaded(make_payload("seq_ft", "seq_correlated", 0, KEEP)),
    ]
    grouped = group_by_method(runs, "seq_hetero")
    assert list(grouped) == ["seq_ft"]
    assert grouped["seq_ft"][0].curriculum == "seq_hetero"


def test_matrix_round_trips_through_the_payload():
    run = loaded(make_payload("seq_ft", "seq_hetero", 0, FORGET))
    np.testing.assert_allclose(run.matrix().values, np.array(FORGET))


def test_unevaluated_cells_load_as_nan_not_zero():
    """``None`` in JSON means "not evaluated", which is not the same as 0% success."""
    payload = make_payload("seq_ft", "seq_hetero", 0, FORGET)
    payload["retention_matrix"]["values"][0][1] = None
    assert np.isnan(loaded(payload).matrix().values[0, 1])


def test_missing_rollouts_raise_rather_than_silently_skipping():
    run = loaded(make_payload("seq_ft", "seq_hetero", 0, FORGET))
    with pytest.raises(KeyError, match="no rollouts recorded"):
        run.successes(0, "libero_object/absent")


# ---- aggregation --------------------------------------------------------------


def test_pooled_f1_matches_the_mean_of_the_final_row():
    """F_1 is linear in the final row, so pooling rollouts must reproduce it exactly."""
    run = loaded(make_payload("seq_ft", "seq_hetero", 0, FORGET))
    estimate = pooled_final_average_success([run], {"n_resamples": 500})
    assert estimate.value == pytest.approx(0.5)  # mean(0.0, 1.0)
    assert estimate.n == len(TASKS) * N_ROLLOUTS


def test_pooling_across_seeds_averages_the_seeds():
    runs = [
        loaded(make_payload("seq_ft", "seq_hetero", 0, FORGET)),  # final row mean 0.5
        loaded(make_payload("seq_ft", "seq_hetero", 1, KEEP)),  # final row mean 1.0
    ]
    estimate = pooled_final_average_success(runs, {"n_resamples": 500})
    assert estimate.value == pytest.approx(0.75)
    assert estimate.n == 2 * len(TASKS) * N_ROLLOUTS


def test_pooling_rejects_unequal_rollout_counts():
    """Unequal cells would weight tasks unequally; §8.1 fixes the count per cell."""
    payload = make_payload("seq_ft", "seq_hetero", 0, FORGET)
    payload["stages"][-1]["evaluation"]["tasks"][0]["successes"] = [True] * 5
    with pytest.raises(ValueError, match="differing rollout counts"):
        pooled_final_average_success([loaded(payload)], {"n_resamples": 100})


def test_seed_bootstrap_brackets_the_per_seed_values():
    runs = [
        loaded(make_payload("seq_ft", "seq_hetero", 0, FORGET)),  # NBT 1.0
        loaded(make_payload("seq_ft", "seq_hetero", 1, KEEP)),  # NBT 0.0
    ]
    estimate = pooled_metric_over_seeds(
        runs, negative_backward_transfer, {"n_resamples": 2000}
    )
    assert estimate.value == pytest.approx(0.5)
    assert estimate.low >= 0.0 and estimate.high <= 1.0
    assert estimate.n == 2


def test_seed_spread_exposes_per_seed_values():
    """Pooling folds seed variance into the interval; §8.2 still wants it visible."""
    runs = [
        loaded(make_payload("seq_ft", "seq_hetero", 0, FORGET)),
        loaded(make_payload("seq_ft", "seq_hetero", 1, KEEP)),
    ]
    assert seed_spread(runs, negative_backward_transfer) == {0: 1.0, 1: 0.0}
    assert seed_spread(runs, auc_average_success) == {0: 0.75, 1: 1.0}


def test_aggregators_reject_an_empty_group():
    with pytest.raises(ValueError, match="received no runs"):
        pooled_final_average_success([])
    with pytest.raises(ValueError, match="received no runs"):
        pooled_metric_over_seeds([], negative_backward_transfer)


# ---- table rendering ----------------------------------------------------------


def test_main_table_flags_a_method_that_is_not_exemplar_free():
    """§6: replay stores demos. The table must say so, not quietly win on F_1."""
    systems = {"is_exemplar_free": False, "method_stored_mb": 12.5}
    runs = [
        loaded(make_payload("seq_ft", "seq_hetero", 0, FORGET)),
        loaded(make_payload("replay", "seq_hetero", 0, KEEP, systems=systems)),
    ]
    table = main_table(runs, "seq_hetero", {"n_resamples": 200})
    column = table.columns.index("exemplar-free")
    rendered = {row[0]: row[column] for row in table.rows}
    assert rendered["seq_ft"] == "yes"
    assert "NO" in rendered["replay"]


def test_main_table_rejects_an_unknown_curriculum():
    runs = [loaded(make_payload("seq_ft", "seq_hetero", 0, FORGET))]
    with pytest.raises(ValueError, match="no runs found for curriculum"):
        main_table(runs, "seq_correlated")


def test_retention_table_marks_unevaluated_cells_distinctly_from_zero():
    payload = make_payload("seq_ft", "seq_hetero", 0, FORGET)
    payload["retention_matrix"]["values"][0][1] = None
    table = retention_table(loaded(payload))
    # Row 0: task_a at 100%, task_b never evaluated.
    assert table.rows[0][1].startswith("100")
    assert table.rows[0][2] == "-"
    # Row 1 has a genuine zero, which must not render as "-".
    assert table.rows[1][1].startswith("0 (n=")


def test_systems_table_reports_params_and_wall_clock():
    runs = [loaded(make_payload("seq_ft", "seq_hetero", 0, FORGET))]
    table = systems_table(runs, "seq_hetero")
    row = table.rows[0]
    assert row[0] == "seq_ft"
    assert row[1] == "30.0"  # 30M trainable
    assert row[4] == "9.0"  # 540 s of training -> 9 min


def test_markdown_and_csv_round_trip(tmp_path):
    runs = [loaded(make_payload("seq_ft", "seq_hetero", 0, FORGET))]
    table = main_table(runs, "seq_hetero", {"n_resamples": 200})
    markdown = table.to_markdown()
    assert markdown.startswith("### seq_hetero")
    assert markdown.count("\n|") >= 3  # header, separator, at least one row

    csv_path = table.to_csv(tmp_path / "out" / "main.csv")
    lines = csv_path.read_text().splitlines()
    assert lines[0].split(",")[0] == "method"
    assert lines[1].startswith("seq_ft")


# ---- plots and the end-to-end entry point -------------------------------------


def test_plots_are_written(tmp_path):
    run = loaded(make_payload("seq_ft", "seq_hetero", 0, FORGET))
    heatmap = plot_retention_heatmap(run, tmp_path / "plots" / "heatmap.png")
    curves = plot_forgetting_curves(
        [run], "seq_hetero", tmp_path / "plots" / "curves.png", {"n_resamples": 200}
    )
    for path in (heatmap, curves):
        assert path.is_file() and path.stat().st_size > 0


def test_build_all_covers_every_curriculum_present(tmp_path):
    results = tmp_path / "results"
    for seed in (0, 1):
        write_run(results, make_payload("seq_ft", "seq_hetero", seed, FORGET))
        write_run(results, make_payload("replay", "seq_hetero", seed, KEEP))
    write_run(results, make_payload("seq_ft", "seq_correlated", 0, FORGET))

    path = build_all(results, tmp_path / "tables", {"n_resamples": 200})
    text = path.read_text()
    assert "## seq_hetero" in text and "## seq_correlated" in text
    assert "seq_ft" in text and "replay" in text
    # Every rendered metric carries an interval.
    assert "[" in text
    assert (tmp_path / "tables" / "seq_hetero_main.csv").is_file()
    assert (tmp_path / "tables" / "seq_hetero_forgetting.png").is_file()
