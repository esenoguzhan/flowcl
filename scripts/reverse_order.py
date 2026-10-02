"""Thin CLI: the C2 reverse-order summary for one seed.

Spec §1: no logic in scripts/. See :mod:`flowcl.experiments.reverse_order`; the rule is
``configs/analysis/reverse_order.yaml``. Writes ``<out>/summary.json`` in every case.

Exit codes: 0 a valid verdict (both / position / task / neither / inconclusive), 3 invalid,
4 incomplete. ``--smoke`` (scratch only) relaxes the production settings and excuses a dirty tree.

    uv run python scripts/reverse_order.py --seed 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from flowcl.experiments.reverse_order import EXIT_CODES, load_config, run_summary
from flowcl.utils.libero_paths import repo_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    results = (repo_root() / "results").resolve()
    if args.smoke and (args.results_root is None or args.results_root.resolve() == results
                       or results in args.results_root.resolve().parents):
        parser.error("--smoke needs --results-root outside the repository's results/")
    summary = run_summary(load_config(args.config), args.seed, results_root=args.results_root,
                          out=args.out, smoke=args.smoke)
    sys.exit(EXIT_CODES[summary["verdict"]["verdict"]])


if __name__ == "__main__":
    main()
