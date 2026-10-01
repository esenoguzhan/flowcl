"""Thin CLI for scripts/queue_gate0_t8.sh: the T6-T8 Gate 0 slots and the backup rule.

Spec §1: no logic in scripts/. See :mod:`flowcl.experiments.gate0_t8`.

    gate0_t8.py tasks                                   # one line per slot: SLOT PRIMARY BACKUP
    gate0_t8.py fresh --results-root R --slot-dir D --task KEY     # exit 1 if outputs exist
    gate0_t8.py backup --report P --task KEY --since EPOCH
        # prints the decision; exit 0 run the backup, 1 the primary passed, 2 no valid report
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from flowcl.experiments.gate0_t8 import PRIMARY_PASSED, RUN_BACKUP, backup_decision, slots, stale_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("tasks")
    fresh = sub.add_parser("fresh")
    fresh.add_argument("--results-root", type=Path, required=True)
    fresh.add_argument("--slot-dir", type=Path, required=True)
    fresh.add_argument("--task", required=True)
    backup = sub.add_parser("backup")
    backup.add_argument("--report", type=Path, required=True)
    backup.add_argument("--task", required=True)
    backup.add_argument("--since", type=float, required=True)
    args = parser.parse_args()
    if args.cmd == "tasks":
        for slot, keys in slots().items():
            print(slot, keys["primary"], keys["backup"])
    elif args.cmd == "fresh":
        found = stale_paths(args.results_root, args.slot_dir, args.task)
        if found:
            print("stale: " + " ".join(found))
            sys.exit(1)
    else:
        decision = backup_decision(args.report, args.task, args.since)
        print(decision)
        sys.exit({RUN_BACKUP: 0, PRIMARY_PASSED: 1}.get(decision, 2))


if __name__ == "__main__":
    main()
