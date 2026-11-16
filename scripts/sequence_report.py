"""Thin CLI: sequence report, a method run against the seq_ft reference (4-task, resumed or 8-task).

Spec §1: no logic in scripts/. See :mod:`flowcl.experiments.sequence_report`.

Example::

    uv run python scripts/sequence_report.py \
        --method-run results/seq_hetero__gpm_projected_adam__seed0
"""

from __future__ import annotations

import argparse
from pathlib import Path

from flowcl.experiments.sequence_report import ProvenanceError, run_sequence_report
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
    parser.add_argument("--strict-provenance", action="store_true",
                        help="Exit 3 if any provenance check that applies did not pass (after writing).")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="With --strict-provenance: excuse clean_git_sha (smoke runs only).")
    parser.add_argument("--config", type=Path, default=None,
                        help="Report config (thresholds per reference run); default "
                             "configs/analysis/sequence_report.yaml.")
    args = parser.parse_args()
    try:
        run_sequence_report(args.method_run, args.reference_run, args.pilot_json, args.out,
                            strict=args.strict_provenance, allow_dirty=args.allow_dirty,
                            config=args.config)
    except ProvenanceError as err:
        print(f"[flowcl] STRICT PROVENANCE FAILED: {'; '.join(err.failures)}", flush=True)
        raise SystemExit(3) from err


if __name__ == "__main__":
    main()
