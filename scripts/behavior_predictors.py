"""Thin CLI: Track 2 (docs/thesis_plan.md, Workstream D). See :mod:`flowcl.experiments.behavior_predictors`;
the rule is ``configs/analysis/behavior_predictors.yaml``.

    --manifest            build and publish the development manifest (results/track2/manifest/); an existing
                          one is verified and printed, never rewritten
    --manifest --dry-run  discover, fingerprint (warming results/track2_cache/) and summarise; publish nothing
    --measures            measure every manifest cell (resumable), run the instrument checks, and publish
                          results/track2/measures/ if they pass and --suite-result reports success here
    --measures --smoke N  the first N cells into --out-root (outside results/); nothing published
    --select              step 3: score and select on the development cells and publish
                          results/track2/selection/ (THE FREEZE), only if --suite-result reports success at
                          this commit on a clean tree; with --out-root, an unpublished dry run
    --verify-selection    recompute the published selection from its inputs; exit 1 if it differs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flowcl.experiments import behavior_predictors as bp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--manifest", action="store_true")
    action.add_argument("--measures", action="store_true")
    action.add_argument("--select", action="store_true")
    action.add_argument("--verify-selection", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--smoke", type=int, default=None, metavar="N")
    parser.add_argument("--suite-result", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.select:
        out = bp.run_select(bp.load_config(args.config), results_root=args.results_root, rule=args.config,
                            suite_result=args.suite_result, out_root=args.out_root)
        print(json.dumps(out.get("summary", out), indent=2))
        return
    if args.verify_selection:
        out = bp.verify_selection(bp.load_config(args.config), results_root=args.results_root, rule=args.config)
        print(json.dumps(out, indent=2))
        raise SystemExit(0 if out["passed"] else 1)
    if args.measures:
        if args.smoke is not None and args.out_root is None:
            parser.error("--smoke needs --out-root outside results/")
        out = bp.run_measures(bp.load_config(args.config), results_root=args.results_root, rule=args.config,
                              out_root=args.out_root, limit=args.smoke, device=args.device,
                              dataset_dir=args.dataset_dir, suite_result=args.suite_result)
        print(json.dumps(out.get("summary", out), indent=2))
        return
    out = bp.build_manifest(bp.load_config(args.config), results_root=args.results_root, rule=args.config,
                            dry_run=args.dry_run)
    if args.dry_run:
        print(json.dumps(out, indent=2))
    else:
        print(json.dumps({"manifest_sha256": out["manifest_sha256"], "summary": out["summary"]}, indent=2))


if __name__ == "__main__":
    main()
