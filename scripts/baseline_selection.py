"""Thin CLI: EWC's λ bracket (calibration, arms, edge extension, selection, verification, smoke).

Spec §1: no logic in scripts/. See :mod:`flowcl.experiments.baseline_selection`; the rule is
``configs/analysis/baseline_selection.yaml``. Exactly one action per call:

    --calibrate                  write results/baseline_selection/ewc_calibration.json (no arm outcome)
    --arm R                      train and publish the grid arm with ratio R
    --select                     apply the rule; exit 10 (nothing published) if the edge extension is owed
    --extension                  train the edge-extension arm (refused unless owed)
    --verify                     every required arm and the bundle under this rule/calibration; exit 1 if not
    --smoke-arm R                one arm at smoke settings into --out-root; exit 1 if its T1 Fisher differs
                                 from the calibration's
    --write-smoke-method PATH    a scratch EWC method YAML with the calibrated λ for r = 1 (smoke runs)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from flowcl.experiments import baseline_selection as bs
from flowcl.utils.libero_paths import repo_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--calibrate", action="store_true")
    action.add_argument("--arm", type=float, metavar="R")
    action.add_argument("--select", action="store_true")
    action.add_argument("--extension", action="store_true")
    action.add_argument("--verify", action="store_true")
    action.add_argument("--smoke-arm", type=float, metavar="R")
    action.add_argument("--write-smoke-method", type=Path, metavar="PATH")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--results-root", type=Path, default=None, help="runs and references (default results/)")
    parser.add_argument("--out-root", type=Path, default=None, help="arm outputs (default: the results root)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--smoke-stop-after", type=int, default=200)
    parser.add_argument("--smoke-episodes", type=int, default=1)
    args = parser.parse_args()
    cfg = bs.load_config(args.config)
    results = (repo_root() / "results").resolve()
    if args.smoke_arm is not None and (args.out_root is None or args.out_root.resolve() == results
                                       or results in args.out_root.resolve().parents):
        parser.error("--smoke-arm needs --out-root outside the repository's results/")
    roots = {"results_root": args.results_root, "out_root": args.out_root, "rule": args.config}
    train = {**roots, "device": args.device, "dataset_dir": args.dataset_dir, "allow_dirty": args.allow_dirty}

    if args.calibrate:
        bs.calibrate(cfg, device=args.device, dataset_dir=args.dataset_dir, **roots)
    elif args.arm is not None:
        bs.run_arm(cfg, args.arm, **train)
    elif args.extension:
        bs.run_extension(cfg, **train)
    elif args.select:
        out = bs.run_select(cfg, **roots)
        sys.exit(bs.EXTENSION_NEEDED if out["status"] == "needs_extension" else 0)
    elif args.verify:
        out = bs.verify(cfg, **roots)
        print(json.dumps(out, indent=2))
        sys.exit(0 if out["passed"] else 1)
    elif args.smoke_arm is not None:
        out = bs.smoke_arm(cfg, args.smoke_arm, stop_after=args.smoke_stop_after, episodes=args.smoke_episodes,
                           **train)
        print(json.dumps(out, indent=2))
        sys.exit(0 if out["passed"] else 1)
    else:
        print(bs.write_smoke_method(cfg, args.write_smoke_method, results_root=args.results_root))


if __name__ == "__main__":
    main()
