"""Thin CLI: the allocation-rules queue's preflight (see :mod:`flowcl.experiments.alloc_queue`).

Exit 0 when every check passed, 1 otherwise; the record is printed as JSON.

Examples::

    uv run python scripts/alloc_queue.py --preflight --seeds 0
    uv run python scripts/alloc_queue.py --preflight --seeds 2 --smoke --results-root DIR --refs REFS
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flowcl.experiments.alloc_queue import preflight


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--smoke", action="store_true", help="smoke references: existence, not pinned hashes")
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--refs", type=Path, default=None, help="where seq_ft and plain GPM live (smoke)")
    args = parser.parse_args()
    out = preflight(args.seeds, results=args.results_root, smoke=args.smoke, refs=args.refs)
    print(json.dumps(out, indent=2, default=str))
    raise SystemExit(0 if out["passed"] else 1)


if __name__ == "__main__":
    main()
