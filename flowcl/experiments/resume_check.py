"""The stage-boundary resume test (docs/thesis_plan.md, A1; ``configs/analysis/resume_check.yaml``).

A run resumed at stage ``K`` from its own ``stage{K-1}.pt`` must reproduce the source's stage
``K``: the checkpoint's state dict bitwise, the memory artifact's content (every tensor and
metadata value), and every per-episode success and step count of its evaluation.
:func:`compare_resume` checks a resumed run against its source; :func:`deep_differences` is the
content comparison (pickled bytes may differ where object sharing does, so bytes are reported
but do not decide).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from flowcl.utils.run import file_sha256


def deep_differences(a, b, path: str = "", limit: int = 20) -> list[str]:
    """Paths where two nested payloads differ (tensors by dtype, shape and value; mappings by
    keys and order; sequences by length; leaves by type and value)."""
    out: list[str] = []

    def walk(x, y, p):
        if len(out) >= limit:
            return
        if isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor):
            if not (isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor) and x.dtype == y.dtype
                    and x.shape == y.shape and torch.equal(x, y)):
                out.append(f"{p}: tensor")
        elif isinstance(x, dict) and isinstance(y, dict):
            if list(x) != list(y):
                out.append(f"{p}: keys")
            for k in x:
                if k in y:
                    walk(x[k], y[k], f"{p}.{k}")
        elif isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
            if type(x) is not type(y) or len(x) != len(y):
                out.append(f"{p}: sequence")
            for i, (u, v) in enumerate(zip(x, y)):
                walk(u, v, f"{p}[{i}]")
        elif type(x) is not type(y) or x != y:
            out.append(f"{p}: {x!r:.60} != {y!r:.60}")

    walk(a, b, path)
    return out


def compare_evaluations(source: dict, resumed: dict) -> dict:
    """Per task: the per-episode successes and step counts, which must be identical."""
    src = {t["task_key"]: t for t in source["tasks"]}
    new = {t["task_key"]: t for t in resumed["tasks"]}
    per_task = {}
    for key in sorted(set(src) | set(new)):
        a, b = src.get(key), new.get(key)
        if a is None or b is None:
            per_task[key] = {"passed": False, "reason": "task missing"}
            continue
        per_task[key] = {
            "successes_equal": a["successes"] == b["successes"],
            "n_steps_equal": a.get("n_steps") == b.get("n_steps"),
            "success_rate": (sum(a["successes"]) / len(a["successes"]), sum(b["successes"]) / len(b["successes"])),
        }
        per_task[key]["passed"] = per_task[key]["successes_equal"] and per_task[key]["n_steps_equal"]
    return {"passed": all(v["passed"] for v in per_task.values()), "per_task": per_task}


def compare_resume(source_run: Path, resumed_run: Path, stage: int) -> dict:
    """The resumed run's stage ``stage`` against the source's (see the module docstring)."""
    from flowcl.train.continual import stage_identity_check

    source_run, resumed_run = Path(source_run), Path(resumed_run)
    identity = stage_identity_check(resumed_run / "checkpoints" / f"stage{stage}.pt", source_run, stage)
    memories = {}
    for name in sorted(p.name for p in (source_run / "method").glob(f"*_task{stage}.*")):
        a, b = source_run / "method" / name, resumed_run / "method" / name
        if not b.is_file():
            memories[name] = {"passed": False, "reason": "missing in the resumed run"}
            continue
        if name.endswith(".pt"):
            diffs = deep_differences(torch.load(a, map_location="cpu", weights_only=False),
                                     torch.load(b, map_location="cpu", weights_only=False))
            memories[name] = {"passed": not diffs, "differences": diffs, "bytes_equal": file_sha256(a) == file_sha256(b)}
        else:  # per-task logs (c-values, residuals): reported, not decisive
            da, db = json.loads(a.read_text()), json.loads(b.read_text())
            diffs = deep_differences({k: v for k, v in da.items() if k != "method_run_id"},
                                     {k: v for k, v in db.items() if k != "method_run_id"})
            memories[name] = {"passed": True, "reported_only": True, "differences": diffs}
    evaluation = compare_evaluations(json.loads((source_run / "eval" / f"stage{stage}.json").read_text()),
                                     json.loads((resumed_run / "eval" / f"stage{stage}.json").read_text()))
    resume = json.loads((resumed_run / "resume.json").read_text())
    passed = identity["passed"] and all(m["passed"] for m in memories.values()) and evaluation["passed"]
    return {"source": str(source_run), "resumed": str(resumed_run), "stage": stage, "passed": bool(passed),
            "checkpoint": {k: identity[k] for k in ("passed", "n_tensors", "n_different", "different",
                                                    "missing", "extra")},
            "method_artifacts": memories, "evaluation": evaluation,
            "resume": {k: resume[k] for k in ("checkpoint", "sha256", "start_stage", "method_state", "text_cache")}}
