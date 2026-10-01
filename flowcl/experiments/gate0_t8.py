"""Gate 0 for T6-T8 (docs/thesis_plan.md, A2): the pre-registered slots and the backup rule.

The slots, primaries and backups are ``follow_up`` in ``configs/analysis/t5_sweep.yaml``, fixed
before any T5 result. Each primary gets Gate 0 (``scripts/gate0.py``, the registered point-
estimate rule in :mod:`flowcl.analysis.gates`); its backup runs **only** if this invocation's
primary report exists, was written after the primary step started, covers exactly the primary
task, and says ``passed: false``. A crashed primary, a stale or foreign report never triggers a
backup; a failed backup leaves the slot unresolved (no automatic third choice).
"""

from __future__ import annotations

import json
from pathlib import Path

from flowcl.experiments.gate0 import single_task_run_id

RUN_BACKUP, PRIMARY_PASSED = "run_backup", "primary_passed"
NO_REPORT, STALE, WRONG_TASK, MALFORMED = "no_report", "stale_report", "wrong_task", "malformed_report"


def slots(cfg: dict | None = None) -> dict[str, dict[str, str]]:
    """``{"T6": {"primary": key, "backup": key}, ...}`` from the T5 sweep's ``follow_up``."""
    if cfg is None:
        from flowcl.experiments.t5_sweep import load_sweep_config

        cfg = load_sweep_config()
    return {slot: {"primary": v["primary"], "backup": v["backup"]} for slot, v in cfg["follow_up"].items()}


def stale_paths(results_root: Path, slot_dir: Path, task_key: str, seed: int = 0) -> list[str]:
    """Outputs that already exist for this slot: its run directory or its report."""
    found = []
    run = Path(results_root) / single_task_run_id(task_key, seed)
    if run.exists():
        found.append(str(run))
    if (Path(slot_dir) / "gate0.json").exists():
        found.append(str(Path(slot_dir) / "gate0.json"))
    return found


def backup_decision(report: Path, task_key: str, since: float) -> str:
    """Whether the slot's backup must run, from this invocation's primary report."""
    report = Path(report)
    if not report.is_file():
        return NO_REPORT
    if report.stat().st_mtime < since:
        return STALE
    try:
        payload = json.loads(report.read_text())
        per_task = payload["evidence"]["per_task"]
        passed = payload["passed"]
    except (ValueError, KeyError, TypeError):
        return MALFORMED
    if set(per_task) != {task_key}:
        return WRONG_TASK
    if passed is False:
        return RUN_BACKUP
    if passed is True:
        return PRIMARY_PASSED
    return MALFORMED
