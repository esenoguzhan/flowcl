"""Compare two smoke roots (or two runs) for deterministic equality; see flowcl/experiments/smoke_compare.py.

Usage:
  uv run python scripts/compare_smokes.py ROOT_A ROOT_B [--runs NAME ...] [--shared-cells-only] [--out FILE]
  uv run python scripts/compare_smokes.py --run-a RUN_A --run-b RUN_B [--shared-cells-only] [--stages K ...]

Exit code 0 when every compared run passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from flowcl.experiments.smoke_compare import compare_roots, compare_runs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root_a", nargs="?", type=Path)
    parser.add_argument("root_b", nargs="?", type=Path)
    parser.add_argument("--runs", nargs="*", default=None)
    parser.add_argument("--run-a", type=Path)
    parser.add_argument("--run-b", type=Path)
    parser.add_argument("--shared-cells-only", action="store_true")
    parser.add_argument("--stages", type=int, nargs="+", default=None,
                        help="with --run-a/--run-b: compare only these stages and their tasks' artifacts")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    if args.run_a or args.run_b:
        if not (args.run_a and args.run_b):
            parser.error("--run-a and --run-b go together")
        result = compare_runs(args.run_a, args.run_b, args.shared_cells_only, stages=args.stages)
        summary = {Path(args.run_b).name: result["passed"]}
    else:
        if not (args.root_a and args.root_b):
            parser.error("give two roots, or --run-a and --run-b")
        if args.stages is not None:
            parser.error("--stages goes with --run-a/--run-b")
        result = compare_roots(args.root_a, args.root_b, args.runs, args.shared_cells_only)
        summary = {name: r["passed"] for name, r in result["runs"].items()}
        if result["unmatched_runs"]:
            summary["(unmatched)"] = result["unmatched_runs"]
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": result["passed"], "runs": summary}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
