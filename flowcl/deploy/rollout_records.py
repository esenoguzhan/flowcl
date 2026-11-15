"""Read and validate PiperStudio's rollout logs (docs/piper_deploy_spec.md §6.4 and §7).

Each run writes ``<log_root>/<YYYY-MM-DD>/<HHMMSS>_<mode>/meta.json``. From Sun 15 Nov a rollout's
``meta.json`` carries ``record``, the task-agnostic rollout record ``rollout_record/v1``. Older logs
carry the sausage-and-lid ``annotation`` or nothing; they are read as ``legacy`` or
``unannotated``, and nothing is filled in for them. Labels judged by eye after the fact (the
``diag/outcomes_by_eye.json`` files of the Piper records) can be attached as ``by_eye``, kept apart
from the record, never converted into it.

:func:`load_rollouts` gives one row per run; :func:`summarize` counts outcomes and successes over
the rows with a valid record.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

SCHEMA = "rollout_record/v1"
OUTCOMES = ("success", "gripper_never_opens", "reach_miss", "grasp_slip_or_drop", "place_miss",
            "stall_off_path", "safety_stop", "time_limit", "other")
LIGHTS = ("daylight", "lamp", "mixed")
DEFAULT_LOG_ROOT = Path.home() / "piper_rollouts"


def validate_record(record: dict) -> list[str]:
    """The problems with one ``rollout_record/v1`` (empty when valid)."""
    if not isinstance(record, dict):
        return ["record is not an object"]
    problems = []
    if record.get("schema") != SCHEMA:
        problems.append(f"schema {record.get('schema')!r} is not {SCHEMA!r}")
    success, outcome = record.get("success"), record.get("outcome")
    if not isinstance(success, bool):
        problems.append("success must be true or false")
    if outcome not in OUTCOMES:
        problems.append(f"outcome {outcome!r} is not one of {', '.join(OUTCOMES)}")
    elif isinstance(success, bool) and (outcome == "success") != success:
        problems.append("outcome is 'success' exactly when success is true")
    miss = record.get("miss_distance_mm")
    if miss is not None and (isinstance(miss, bool) or not isinstance(miss, (int, float)) or miss < 0):
        problems.append("miss_distance_mm must be a non-negative number or null")
    if record.get("light") not in LIGHTS:
        problems.append(f"light {record.get('light')!r} is not one of {', '.join(LIGHTS)}")
    note = record.get("note", "")
    if not isinstance(note, str):
        problems.append("note must be text")
    elif outcome == "other" and not note.strip():
        problems.append("outcome 'other' needs a note")
    if not isinstance(record.get("annotated_at"), str):
        problems.append("annotated_at must be an ISO time string")
    return problems


@dataclass
class RolloutRow:
    run: str
    date: str
    time: str
    mode: str
    label: str | None
    kind: str | None
    reason: str | None
    duration_s: float | None
    cycle_s: float | None
    checkpoint: str | None
    status: str  # "record", "invalid_record", "legacy", "unannotated"
    record: dict | None = None
    problems: list[str] = field(default_factory=list)
    legacy_annotation: dict | None = None
    by_eye: str | None = None
    start_photo: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def _checkpoint(meta: dict) -> str | None:
    server = meta.get("server_metadata") or {}
    for key in ("checkpoint", "checkpoint_path", "run"):
        if isinstance(server.get(key), str):
            return server[key]
    return None


def read_rollout(run_dir: Path, by_eye: dict[str, str] | None = None) -> RolloutRow:
    """One run's row from its ``meta.json`` (and ``start_front.jpg`` when present)."""
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "meta.json").read_text())
    clock, _, mode = run_dir.name.partition("_")
    record = meta.get("record")
    annotation = meta.get("annotation")
    if record is not None:
        problems = validate_record(record)
        status = "invalid_record" if problems else "record"
    else:
        problems = []
        status = "legacy" if isinstance(annotation, dict) else "unannotated"
    photo = run_dir / "start_front.jpg"
    return RolloutRow(
        run=str(run_dir), date=run_dir.parent.name, time=clock, mode=meta.get("mode") or mode,
        label=meta.get("label"), kind=meta.get("kind"), reason=meta.get("reason"),
        duration_s=meta.get("duration_s"), cycle_s=meta.get("cycle_s"), checkpoint=_checkpoint(meta),
        status=status, record=record, problems=problems,
        legacy_annotation=annotation if isinstance(annotation, dict) else None,
        by_eye=(by_eye or {}).get(clock), start_photo=str(photo) if photo.is_file() else None,
    )


def load_by_eye(path: Path) -> dict[str, str]:
    """``{"label": [HHMMSS, ...], ...}`` (the records' ``outcomes_by_eye.json``) as ``HHMMSS -> label``."""
    data = json.loads(Path(path).read_text())
    out: dict[str, str] = {}
    for label, runs in data.items():
        for run in runs:
            if run in out:
                raise ValueError(f"{path}: run {run} has two labels ({out[run]!r}, {label!r})")
            out[run] = label
    return out


def load_rollouts(log_root: Path = DEFAULT_LOG_ROOT, dates: list[str] | None = None,
                  modes: tuple[str, ...] = ("rollout",), by_eye: dict[str, str] | None = None) -> list[RolloutRow]:
    """Every run under ``log_root`` (optionally only ``dates``) whose mode is in ``modes``, in time order."""
    rows = []
    for meta in sorted(Path(log_root).glob("*/*/meta.json")):
        run_dir = meta.parent
        if dates is not None and run_dir.parent.name not in dates:
            continue
        if run_dir.name.partition("_")[2] not in modes:
            continue
        rows.append(read_rollout(run_dir, by_eye))
    return rows


def summarize(rows: list[RolloutRow]) -> dict:
    """Counts by status; successes and outcomes over the rows with a valid record only."""
    valid = [r for r in rows if r.status == "record"]
    misses = [r.record["miss_distance_mm"] for r in valid if r.record.get("miss_distance_mm") is not None]
    out = {
        "n_runs": len(rows),
        "by_status": dict(Counter(r.status for r in rows)),
        "n_recorded": len(valid),
        "successes": sum(r.record["success"] for r in valid),
        "outcomes": {o: n for o, n in Counter(r.record["outcome"] for r in valid).items()},
        "light": dict(Counter(r.record["light"] for r in valid)),
        "miss_distance_mm": sorted(misses),
        "invalid": {r.run: r.problems for r in rows if r.status == "invalid_record"},
    }
    if valid:
        from flowcl.analysis.metrics import success_estimate

        est = success_estimate([r.record["success"] for r in valid], seed=0)
        out["success_rate"] = {"value": est.value, "low": est.low, "high": est.high}
    return out
