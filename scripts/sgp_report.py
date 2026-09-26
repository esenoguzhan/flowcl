"""Thin CLI: the pre-registered outcome of the SGP runs for one seed, or the replication.

Spec §1: no logic in scripts/. See :mod:`flowcl.experiments.sgp_report`; the rule is in
``configs/analysis/sgp.yaml``. Needs the four runs' sequence reports and diagnostics, and
the seed's per-batch probe cells (``scripts/probe_cells.py``).

Examples::

    uv run python scripts/sgp_report.py --seed 0
    uv run python scripts/sgp_report.py --replication    # the registered seeds 0 1 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

from flowcl.experiments.sgp_report import load_sgp_config, run_sgp_replication, run_sgp_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--replication", action="store_true",
                        help="Apply the replication rule over the registered seeds' reports.")
    args = parser.parse_args()
    cfg = load_sgp_config(args.config)
    if args.replication:
        run_sgp_replication(cfg)
    else:
        run_sgp_report(cfg, seed=args.seed, out=args.out)


if __name__ == "__main__":
    main()
