"""Thin CLI: the 4-task baseline queue's checks. See :mod:`flowcl.experiments.baseline_queue`.

    uv run python scripts/baseline_queue.py preflight --seed 0 --methods ewc lora ... --out p0.json
    uv run python scripts/baseline_queue.py check-run --run results/seq_hetero__ewc__seed0 --method ewc --seed 0
    uv run python scripts/baseline_queue.py check-report --report results/baselines_seq/seed0/ewc.json

Each prints its JSON (and writes it with --out) and exits 0 if it passed, 1 if not.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from flowcl.experiments import baseline_queue as bq


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--seed", type=int, required=True, choices=sorted(bq.SEQ_FT_STAGE0))
    pre.add_argument("--methods", nargs="+", required=True, choices=bq.METHODS)
    pre.add_argument("--results-root", type=Path, default=None)
    pre.add_argument("--smoke", action="store_true")
    run = sub.add_parser("check-run")
    run.add_argument("--run", type=Path, required=True)
    run.add_argument("--method", required=True, choices=bq.METHODS)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--gpm-memory", type=Path, default=None)
    rep = sub.add_parser("check-report")
    rep.add_argument("--report", type=Path, required=True)
    rep.add_argument("--allow-dirty", action="store_true")
    for p in (pre, run, rep):
        p.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.command == "preflight":
        out = bq.preflight(args.seed, args.methods, results=args.results_root, smoke=args.smoke)
    elif args.command == "check-run":
        out = bq.check_run(args.run, args.method, args.seed, args.gpm_memory)
    else:
        out = bq.check_report(args.report, args.allow_dirty)
    text = json.dumps(out, indent=2, default=str)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)
    sys.exit(0 if out["passed"] else 1)


if __name__ == "__main__":
    main()
