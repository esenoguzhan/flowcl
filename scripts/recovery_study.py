"""Thin CLI: the recovery measurement study (configs/analysis/recovery_study.yaml). Spec §1: no logic in
scripts/. See :mod:`flowcl.experiments.recovery_study`. Exactly one action per call:

    --task K                  task K's worker (nominal rollouts, the gate, the continuations; resumable)
    --smoke                   with --task K: 2 nominal rollouts, an early and a late query, 2 seeds, into
                              --out-root (outside results/)
    --benchmark K             per-phase timing at an early and a late query of task K (--out JSON)
    --simulate                the pre-run validation report (--out-root DIR/validation.json)
    --analyze                 Q1-Q4, the negative control, the outcome; publishes results/recovery_study/
                              analysis/ only with --suite-result green at this commit (with --out-root:
                              an unpublished dry run of that root's work files)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from flowcl.experiments import recovery_study as rs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--task", type=int)
    action.add_argument("--benchmark", type=int)
    action.add_argument("--simulate", action="store_true")
    action.add_argument("--analyze", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--fallback", action="store_true", help="the rule's frozen 4-query fallback")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--suite-result", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--n-worlds", type=int, default=None, help="--simulate: override (plumbing checks only)")
    args = parser.parse_args()
    rule = rs.load_rule(args.config)
    results = (rs.repo_root() / "results").resolve()
    if args.smoke and (args.out_root is None or results in args.out_root.resolve().parents
                       or args.out_root.resolve() == results):
        parser.error("--smoke needs --out-root outside results/")
    if args.task is not None:
        smoke = {"n_nominal": 2, "phases": [0.0833333333, 0.9166666667], "n_seeds": 2} if args.smoke else None
        out = rs.run_task(rule, args.task, results_root=args.results_root, out_root=args.out_root, rule_file=args.config,
                          device=args.device, dataset_dir=args.dataset_dir, smoke=smoke, fallback=args.fallback)
        print(json.dumps({k: v for k, v in out.items() if k != "gate"} | {"gate_passed": out["gate"]["passed"]}))
        sys.exit(0 if out["gate"]["passed"] else 1)
    if args.benchmark is not None:
        out = rs.benchmark(rule, args.benchmark, results_root=args.results_root, device=args.device,
                           dataset_dir=args.dataset_dir)
        text = json.dumps(out, indent=2)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text + "\n")
        print(text)
        sys.exit(0 if "error" not in out else 1)
    if args.simulate:
        if args.out_root is None:
            parser.error("--simulate needs --out-root")
        out = rs.validate(rule, out=args.out_root, workers=args.workers, n_worlds=args.n_worlds, rule_file=args.config)
        print(json.dumps(out["decision"], indent=2))
        sys.exit(0 if out["decision"]["production_may_start"] else 1)
    out = rs.analyze(rule, results_root=args.results_root, rule_file=args.config, suite_result=args.suite_result,
                     out_root=args.out_root)
    print(json.dumps({"outcome": out["outcome"], "gate_ok": out["gate_ok"],
                      "negative_control": out["negative_control"]}, indent=2))


if __name__ == "__main__":
    main()
