"""Thin CLI: four-task sequence report, a method run against the seq_ft reference.

Spec §1: no logic in scripts/. See :mod:`flowcl.experiments.sequence_report`.

Example::

    uv run python scripts/sequence_report.py \
        --method-run results/seq_hetero__gpm_projected_adam__seed0
"""

from __future__ import annotations

import argparse
from pathlib import Path

from flowcl.experiments.sequence_report import run_sequence_report
from flowcl.utils.libero_paths import repo_root


def main() -> None:
    results = repo_root() / "results"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-run", type=Path,
                        default=results / "seq_hetero__gpm_projected_adam__seed0")
    parser.add_argument("--reference-run", type=Path,
                        default=results / "seq_hetero__seq_ft__seed0")
    parser.add_argument("--pilot-json", type=Path, default=results / "gpm_pilot" / "pilot.json")
    parser.add_argument("--out", type=Path, default=results / "gpm_seq" / "report.json")
    args = parser.parse_args()
    run_sequence_report(args.method_run, args.reference_run, args.pilot_json, args.out)


if __name__ == "__main__":
    main()
