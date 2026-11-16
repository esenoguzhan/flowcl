"""Thin CLI: the f = 0.7 queue's preflight and per-run check (see :mod:`flowcl.experiments.ne70_queue`).

Exit 0 when every check passed, 1 otherwise; the record is printed as JSON.

Examples::

    uv run python scripts/ne70_queue.py --preflight --seeds 0 1 2
    uv run python scripts/ne70_queue.py --check-run results/seq_hetero__gpm_projected_adam_ne70__seed0 \
        --gpm-run results/seq_hetero__gpm_projected_adam__seed0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flowcl.experiments.ne70_queue import check_run, preflight


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--check-run", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--smoke", action="store_true", help="smoke references: existence, not pinned hashes")
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--refs", type=Path, default=None, help="where seq_ft and plain GPM live (smoke)")
    parser.add_argument("--gpm-run", type=Path, help="with --check-run: plain GPM of the same seed")
    args = parser.parse_args()
    if args.preflight:
        out = preflight(args.seeds, results=args.results_root, smoke=args.smoke, refs=args.refs)
    else:
        if args.gpm_run is None:
            parser.error("--check-run needs --gpm-run")
        out = check_run(args.check_run, args.gpm_run)
    print(json.dumps(out, indent=2, default=str))
    raise SystemExit(0 if out["passed"] else 1)


if __name__ == "__main__":
    main()
