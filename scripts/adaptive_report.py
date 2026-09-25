"""Thin CLI: the pre-registered outcome of an adaptive-GPM run, or the replication summary.

Spec §1: no logic in scripts/. See :mod:`flowcl.experiments.adaptive_report`; the rule is
in ``configs/analysis/adaptive_gpm.yaml``. Needs both forgetting-diagnostics reports.

Examples::

    uv run python scripts/adaptive_report.py --seed 1
    uv run python scripts/adaptive_report.py --replication 0 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

from flowcl.experiments.adaptive_report import (
    load_adaptive_config,
    run_adaptive_report,
    run_replication,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--replication", type=int, nargs="+", default=None,
                        help="Seeds whose written reports to compare (e.g. 0 1).")
    args = parser.parse_args()
    cfg = load_adaptive_config(args.config)
    if args.replication:
        run_replication(cfg, args.replication)
    else:
        run_adaptive_report(cfg, seed=args.seed, out=args.out)


if __name__ == "__main__":
    main()
