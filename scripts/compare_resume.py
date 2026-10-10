"""Thin CLI: a resumed run's stage K against the uninterrupted source's (exit 1 unless equal).

See :func:`flowcl.experiments.resume_check.compare_resume`. ``--all-artifacts`` compares every
artifact the stage checkpoint names (e.g. replay's earlier buffer files), not only ``*_task{K}``;
``--method-state`` also requires the two stage checkpoints' method state to be equal (LoRA's history).

    uv run python scripts/compare_resume.py --source DIR --resumed DIR --stage 3 --all-artifacts
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from flowcl.experiments.resume_check import compare_resume


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--resumed", type=Path, required=True)
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--all-artifacts", action="store_true")
    parser.add_argument("--method-state", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    report = compare_resume(args.source, args.resumed, args.stage, all_artifacts=args.all_artifacts,
                            method_state=args.method_state)
    text = json.dumps(report, indent=2, default=str)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)
    print(f"[flowcl] resume comparison at stage {args.stage}: {'PASS' if report['passed'] else 'FAIL'}")
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
