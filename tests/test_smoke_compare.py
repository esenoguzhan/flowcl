"""The smoke comparator (flowcl/experiments/smoke_compare.py): deterministic content decides,
timings and provenance do not."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import torch

from flowcl.experiments.smoke_compare import compare_cells, compare_roots, compare_runs

TASKS = ("suite/a", "suite/b")


def _eval(stage: int, tasks=TASKS, flip: tuple[str, int] | None = None, wall: float = 1.0, seed_shift: int = 0):
    out = []
    for key in tasks:
        successes = [True, False, True]
        if flip and flip[0] == key:
            successes[flip[1]] = not successes[flip[1]]
        out.append({"task_key": key, "seeds": [10 + i + seed_shift for i in range(3)], "successes": successes,
                    "n_steps": [5, 600, 7], "wall_clock_s": wall})
    return {"stage": stage, "tasks": out}


def write_run(root: Path, name: str = "run", *, tensor_delta: float = 0.0, git: str = "abc", timings: float = 1.0,
              eval_kwargs: dict | None = None, artifact_delta: float = 0.0, stages: int = 2,
              eval_tasks: dict[int, tuple] | None = None) -> Path:
    run = root / name
    (run / "checkpoints").mkdir(parents=True)
    (run / "eval").mkdir()
    (run / "method").mkdir()
    for stage in range(stages):
        state = {"w": torch.arange(4, dtype=torch.float32) + stage, "b": torch.zeros(2)}
        if stage == stages - 1:
            state["w"] = state["w"] + tensor_delta
        torch.save({"state_dict": state, "extra": {"git_sha": git, "method_state": {"k": [1, 2], "timings": {"s": timings}}}},
                   run / "checkpoints" / f"stage{stage}.pt")
        tasks = (eval_tasks or {}).get(stage, TASKS)
        (run / "eval" / f"stage{stage}.json").write_text(json.dumps(_eval(stage, tasks, **(eval_kwargs or {}))))
        torch.save({"basis": torch.ones(3) + artifact_delta, "timings": {"x": timings}},
                   run / "method" / f"memory_task{stage}.pt")
    (run / "method" / "gpm_logs_task0.json").write_text(json.dumps({"wall_clock_s": timings, "c": [1.0]}))
    return run


def test_equal_runs_pass_although_timings_and_git_differ(tmp_path):
    a = write_run(tmp_path / "a")
    b = write_run(tmp_path / "b", git="def", timings=99.0, eval_kwargs={"wall": 42.0})
    result = compare_runs(a, b)
    assert result["passed"], result


def test_one_changed_tensor_fails(tmp_path):
    a = write_run(tmp_path / "a")
    b = write_run(tmp_path / "b", tensor_delta=1e-6)
    result = compare_runs(a, b)
    assert not result["passed"]
    assert result["stages"]["1"]["checkpoint"]["different"] == ["w"]
    assert result["stages"]["0"]["checkpoint"]["passed"]


def test_one_changed_success_fails(tmp_path):
    a = write_run(tmp_path / "a")
    b = write_run(tmp_path / "b", eval_kwargs={"flip": ("suite/b", 2)})
    result = compare_runs(a, b)
    assert not result["passed"]
    assert result["stages"]["0"]["evaluation"]["per_task"]["suite/b"]["successes"] is False


def test_changed_rollout_seeds_fail(tmp_path):
    a = write_run(tmp_path / "a")
    b = write_run(tmp_path / "b", eval_kwargs={"seed_shift": 1})
    assert not compare_runs(a, b)["passed"]


def test_changed_artifact_fails(tmp_path):
    a = write_run(tmp_path / "a")
    b = write_run(tmp_path / "b", artifact_delta=0.5)
    result = compare_runs(a, b)
    assert not result["passed"]
    assert not result["method_artifacts"]["memory_task0.pt"]["passed"]


def test_missing_cell_fails_unless_shared_cells_only(tmp_path):
    a = write_run(tmp_path / "a")
    b = write_run(tmp_path / "b", eval_tasks={0: ("suite/a",)})  # seen-only: stage 0 evaluates T1 only
    assert not compare_runs(a, b)["passed"]
    assert compare_runs(a, b, shared_cells_only=True)["passed"]


def test_shared_cells_only_still_requires_the_smaller_runs_cells():
    full = {k: v for k, v in ((t["task_key"], t) for t in _eval(0)["tasks"])}
    other = {"suite/c": copy.deepcopy(full["suite/a"]) | {"task_key": "suite/c"}}
    assert not compare_cells(full, other, shared_cells_only=True)["passed"]


def test_compare_roots_pairs_runs_by_name(tmp_path):
    write_run(tmp_path / "a", "r1")
    write_run(tmp_path / "a", "r2")
    write_run(tmp_path / "b", "r1")
    write_run(tmp_path / "b", "r2", tensor_delta=1.0)
    result = compare_roots(tmp_path / "a", tmp_path / "b")
    assert result["runs"]["r1"]["passed"] and not result["runs"]["r2"]["passed"]
    assert not result["passed"]
