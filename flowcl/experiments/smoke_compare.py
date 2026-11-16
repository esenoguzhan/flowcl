"""Compare two sets of smoke runs for deterministic equality (status.md §5; the to-do plan's step 0).

A code change that must leave training and evaluation unchanged is checked by running the same
smoke before and after it and comparing the runs. Whole files cannot be compared: evaluation files
hold wall-clock times, checkpoint bytes carry provenance (git SHA, timings), and new record fields
are expected to appear. This compares exactly what is deterministic, per run and stage:

- every checkpoint's state-dict tensors (:func:`flowcl.train.continual.stage_identity_check`);
- the checkpoint's ``extra["method_state"]`` without ``timings``;
- every ``method/*.pt`` artifact's content (:func:`flowcl.experiments.resume_check.deep_differences`);
  ``method/*.json`` logs are compared without timing keys and reported, not decisive, as in A1;
- per evaluated cell: the rollout seeds, the per-episode successes and the episode lengths.

Ignored: wall clocks, git SHAs, timings, loss, peak-memory and probe records, and paths.
``shared_cells_only`` compares only the cells both runs evaluated (the seen-only check), and
requires every cell of the run with fewer cells to be present in the other. ``stages`` compares only
those stages (a run resumed later holds fewer checkpoints), and only the method artifacts of those
tasks (``*_task{k}.*``).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch

from flowcl.experiments.resume_check import deep_differences

_TIMING_KEYS = {"timings", "wall_clock_s", "train_wall_clock_s", "total_wall_clock_s"}


def _strip_timing(value):
    if isinstance(value, dict):
        return {k: _strip_timing(v) for k, v in value.items() if k not in _TIMING_KEYS}
    if isinstance(value, list):
        return [_strip_timing(v) for v in value]
    return value


def _stages(run: Path) -> list[int]:
    return sorted(int(p.stem.removeprefix("stage")) for p in (run / "checkpoints").glob("stage*.pt"))


def _eval_cells(run: Path, stage: int) -> dict[str, dict]:
    path = run / "eval" / f"stage{stage}.json"
    if not path.is_file():
        return {}
    return {t["task_key"]: t for t in json.loads(path.read_text())["tasks"]}


def compare_cells(a: dict[str, dict], b: dict[str, dict], shared_cells_only: bool = False) -> dict:
    """Per task: rollout seeds, successes and episode lengths must be identical."""
    if shared_cells_only:
        small, large = (a, b) if len(a) <= len(b) else (b, a)
        missing = sorted(set(small) - set(large))
        keys = sorted(set(a) & set(b))
    else:
        missing = sorted(set(a) ^ set(b))
        keys = sorted(set(a) & set(b))
    per_task = {}
    for key in keys:
        x, y = a[key], b[key]
        checks = {field: x.get(field) == y.get(field) for field in ("seeds", "successes", "n_steps")}
        per_task[key] = {**checks, "passed": all(checks.values())}
    for key in missing:
        per_task[key] = {"passed": False, "reason": "cell missing in one run"}
    return {"passed": all(v["passed"] for v in per_task.values()), "per_task": per_task}


def _artifact_task(name: str) -> int | None:
    match = re.search(r"_task(\d+)\.", name)
    return int(match.group(1)) if match else None


def compare_runs(a: Path, b: Path, shared_cells_only: bool = False, stages: list[int] | None = None) -> dict:
    """Run ``b`` against run ``a`` (see the module docstring)."""
    from flowcl.train.continual import stage_identity_check

    a, b = Path(a), Path(b)
    stages_a, stages_b = _stages(a), _stages(b)
    out: dict = {"a": str(a), "b": str(b), "stages": {}}
    if stages is not None:
        missing = [s for s in stages if s not in stages_a or s not in stages_b]
        if missing:
            out["passed"] = False
            out["reason"] = f"stages {missing} missing in a run ({stages_a} vs {stages_b})"
            return out
        stages_a = stages_b = sorted(stages)
        out["stages_compared"] = stages_a
    if stages_a != stages_b:
        out["passed"] = False
        out["reason"] = f"different checkpoint stages: {stages_a} vs {stages_b}"
        return out
    for stage in stages_a:
        ckpt_b = b / "checkpoints" / f"stage{stage}.pt"
        identity = stage_identity_check(ckpt_b, a, stage)
        extra_a = torch.load(a / "checkpoints" / f"stage{stage}.pt", map_location="cpu",
                             weights_only=False).get("extra") or {}
        extra_b = torch.load(ckpt_b, map_location="cpu", weights_only=False).get("extra") or {}
        state_diffs = deep_differences(_strip_timing(extra_a.get("method_state") or {}),
                                       _strip_timing(extra_b.get("method_state") or {}))
        out["stages"][str(stage)] = {
            "checkpoint": {k: identity[k] for k in ("passed", "n_tensors", "n_different", "different",
                                                    "missing", "extra")},
            "method_state": {"passed": not state_diffs, "differences": state_diffs},
            "evaluation": compare_cells(_eval_cells(a, stage), _eval_cells(b, stage), shared_cells_only),
        }
    artifacts = {}
    names_a = sorted(p.name for p in (a / "method").glob("*")) if (a / "method").is_dir() else []
    names_b = sorted(p.name for p in (b / "method").glob("*")) if (b / "method").is_dir() else []
    for name in sorted(set(names_a) | set(names_b)):
        if stages is not None and _artifact_task(name) not in stages:
            continue
        pa, pb = a / "method" / name, b / "method" / name
        if not (pa.is_file() and pb.is_file()):
            artifacts[name] = {"passed": False, "reason": "missing in one run"}
        elif name.endswith(".pt"):
            diffs = deep_differences(_strip_timing(torch.load(pa, map_location="cpu", weights_only=False)),
                                     _strip_timing(torch.load(pb, map_location="cpu", weights_only=False)))
            artifacts[name] = {"passed": not diffs, "differences": diffs}
        else:
            diffs = deep_differences(_strip_timing(json.loads(pa.read_text())),
                                     _strip_timing(json.loads(pb.read_text())))
            artifacts[name] = {"passed": True, "reported_only": True, "differences": diffs}
    out["method_artifacts"] = artifacts
    out["passed"] = (all(s["checkpoint"]["passed"] and s["method_state"]["passed"] and s["evaluation"]["passed"]
                         for s in out["stages"].values())
                     and all(v["passed"] for v in artifacts.values()))
    return out


def compare_roots(root_a: Path, root_b: Path, runs: list[str] | None = None,
                  shared_cells_only: bool = False) -> dict:
    """Every run directory present under both roots (or the named ``runs``), compared pairwise."""
    root_a, root_b = Path(root_a), Path(root_b)
    if runs is None:
        names_a = {p.parent.parent.name for p in root_a.glob("*/checkpoints/stage0.pt")}
        names_b = {p.parent.parent.name for p in root_b.glob("*/checkpoints/stage0.pt")}
        runs = sorted(names_a & names_b)
        only = sorted(names_a ^ names_b)
    else:
        only = []
    results = {name: compare_runs(root_a / name, root_b / name, shared_cells_only) for name in runs}
    return {"passed": bool(results) and all(r["passed"] for r in results.values()),
            "runs": results, "unmatched_runs": only}
