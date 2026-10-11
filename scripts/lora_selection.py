"""Thin CLI: LoRA's rank × learning-rate bracket (arms, edge extensions, selection, verification,
smoke, benchmark).

Spec §1: no logic in scripts/. See :mod:`flowcl.experiments.lora_selection`; the rule is
``configs/analysis/baseline_selection_lora.yaml``. Exactly one action per call:

    --arm R LR                   train and publish the grid arm (rank R, learning rate LR)
    --select                     apply the rule; exit 10 (nothing published) while an extension is owed
    --extension                  train the next owed edge-extension arm (refused unless one is owed)
    --verify                     inputs, checkpoints, bundle and the recomputed selection; exit 1 if not
    --status                     imported, published and missing cells and the imported rule's verification
                                 (no training); exit 1 unless ready
    --smoke-arm R LR             the bracket path against the runner (into --out-root): an arm from
                                 --reference-run's stage0.pt at --smoke-steps steps must reproduce
                                 --identity-run's stage1.pt under the same TrainConfig; exit 1 if not
    --benchmark                  seconds per T2 step at --bench-ranks (0 = full fine-tuning); writes
                                 --out-root/benchmark.json
    --write-smoke-method PATH    a scratch LoRA method YAML (--rank, --lr) for smoke runs of the runner
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from flowcl.experiments import lora_selection as ls
from flowcl.utils.libero_paths import repo_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--arm", nargs=2, type=float, metavar=("R", "LR"))
    action.add_argument("--select", action="store_true")
    action.add_argument("--extension", action="store_true")
    action.add_argument("--verify", action="store_true")
    action.add_argument("--status", action="store_true")
    action.add_argument("--smoke-arm", nargs=2, type=float, metavar=("R", "LR"))
    action.add_argument("--benchmark", action="store_true")
    action.add_argument("--write-smoke-method", type=Path, metavar="PATH")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--results-root", type=Path, default=None, help="runs and references (default results/)")
    parser.add_argument("--out-root", type=Path, default=None, help="arm outputs (default: the results root)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--reference-run", type=Path, default=None, help="--smoke-arm: a scratch seq_ft run")
    parser.add_argument("--identity-run", type=Path, default=None, help="--smoke-arm: a scratch LoRA run")
    parser.add_argument("--smoke-steps", type=int, default=200)
    parser.add_argument("--smoke-episodes", type=int, default=1)
    parser.add_argument("--bench-ranks", type=int, nargs="+", default=[0, 4, 64, 256])
    parser.add_argument("--bench-steps", type=int, default=300)
    parser.add_argument("--rank", type=int, default=4, help="--write-smoke-method")
    parser.add_argument("--lr", type=float, default=1e-3, help="--write-smoke-method")
    args = parser.parse_args()
    cfg = ls.load_config(args.config)
    results = (repo_root() / "results").resolve()
    scratch = args.smoke_arm is not None or args.benchmark
    if scratch and (args.out_root is None or args.out_root.resolve() == results
                    or results in args.out_root.resolve().parents):
        parser.error("--smoke-arm and --benchmark need --out-root outside the repository's results/")
    if args.smoke_arm is not None and (args.reference_run is None or args.identity_run is None):
        parser.error("--smoke-arm needs --reference-run and --identity-run")
    roots = {"results_root": args.results_root, "out_root": args.out_root, "rule": args.config}
    train = {**roots, "device": args.device, "dataset_dir": args.dataset_dir, "allow_dirty": args.allow_dirty}

    if args.arm is not None:
        ls.run_arm(cfg, int(args.arm[0]), args.arm[1], **train)
    elif args.extension:
        ls.run_extension(cfg, **train)
    elif args.select:
        out = ls.run_select(cfg, allow_dirty=args.allow_dirty, **roots)
        sys.exit(ls.EXTENSION_NEEDED if out["status"] == "needs_extension" else 0)
    elif args.verify:
        out = ls.verify(cfg, allow_dirty=args.allow_dirty, **roots)
        print(json.dumps(out, indent=2))
        sys.exit(0 if out["passed"] else 1)
    elif args.status:
        out = ls.status(cfg, allow_dirty=args.allow_dirty, **roots)
        print(json.dumps(out, indent=2))
        sys.exit(0 if out["ready"] else 1)
    elif args.smoke_arm is not None:
        out = ls.smoke_arm(cfg, int(args.smoke_arm[0]), args.smoke_arm[1], reference_run=args.reference_run,
                           identity_run=args.identity_run, steps=args.smoke_steps, episodes=args.smoke_episodes,
                           **train)
        print(json.dumps(out, indent=2))
        sys.exit(0 if out["passed"] else 1)
    elif args.benchmark:
        out = ls.benchmark(cfg, args.bench_ranks, steps=args.bench_steps, results_root=args.results_root,
                           out_root=args.out_root, device=args.device, dataset_dir=args.dataset_dir)
        args.out_root.mkdir(parents=True, exist_ok=True)
        (args.out_root / "benchmark.json").write_text(json.dumps(out, indent=2) + "\n")
        print(json.dumps(out, indent=2))
    else:
        print(ls.write_smoke_method(args.write_smoke_method, args.rank, args.lr, float(cfg["lora"]["alpha"])))


if __name__ == "__main__":
    main()
