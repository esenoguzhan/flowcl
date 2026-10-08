"""Thin CLI: the C1 follow-up report for one seed, or its replication over seeds 0-2.

Spec §1: no logic in scripts/. See :mod:`flowcl.experiments.high_protection_followup`; the rule is
``configs/analysis/high_protection_followup.yaml``. The rollouts themselves come from
``scripts/evaluate.py`` (see ``scripts/queue_high_protection_followup.sh``).

Exit codes: 0 a verdict that is not invalid, 3 ``invalid_followup`` (or an invalid C1 gate); the
replication exits 1 when it refuses its inputs. ``--smoke`` reads the evaluations from
``--results-root`` (scratch) and the runs and C1 reports from the repository's results/.

    uv run python scripts/high_protection_followup.py --seed 0
    uv run python scripts/high_protection_followup.py --replication
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from flowcl.experiments.high_protection_followup import load_config, run_replication, run_report
from flowcl.utils.libero_paths import repo_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--replication", action="store_true")
    parser.add_argument("--results-root", type=Path, default=None,
                        help="Where the follow-up's evaluations and reports live (default results/).")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.replication == (args.seed is not None):
        parser.error("give exactly one of --seed N and --replication")
    results = (repo_root() / "results").resolve()
    if args.smoke and (args.results_root is None or args.results_root.resolve() == results
                       or results in args.results_root.resolve().parents):
        parser.error("--smoke needs --results-root outside the repository's results/")
    cfg = load_config(args.config)
    if args.replication:
        if args.smoke:
            parser.error("the replication never reads smoke reports")
        run_replication(cfg, out_root=args.results_root, rule=args.config)
        return
    report = run_report(cfg, args.seed, out_root=args.results_root, smoke=args.smoke, rule=args.config)
    sys.exit(3 if report["verdict"]["verdict"].startswith("invalid") else 0)


if __name__ == "__main__":
    main()
