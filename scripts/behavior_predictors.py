"""Thin CLI: Track 2 (docs/thesis_plan.md, Workstream D). See :mod:`flowcl.experiments.behavior_predictors`;
the rule is ``configs/analysis/behavior_predictors.yaml``.

    --manifest            build and publish the development manifest (results/track2/manifest/); an existing
                          one is verified and printed, never rewritten
    --manifest --dry-run  discover, fingerprint (warming results/track2_cache/) and summarise; publish nothing
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flowcl.experiments import behavior_predictors as bp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", action="store_true", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--results-root", type=Path, default=None)
    args = parser.parse_args()
    out = bp.build_manifest(bp.load_config(args.config), results_root=args.results_root, rule=args.config,
                            dry_run=args.dry_run)
    if args.dry_run:
        print(json.dumps(out, indent=2))
    else:
        print(json.dumps({"manifest_sha256": out["manifest_sha256"], "summary": out["summary"]}, indent=2))


if __name__ == "__main__":
    main()
