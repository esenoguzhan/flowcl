"""Thin CLI: instruction negatives at seeds 1-2 (stove and cabinet top, GPM + negatives against plain-GPM
controls).

Spec §1: no logic in scripts/. See :mod:`flowcl.experiments.instruction_negatives_seeds`; the rule is
``configs/analysis/instruction_negatives_seeds.yaml``. The eight training runs are
scripts/run_continual.py (queued by scripts/queue_instruction_negatives_seeds.sh).

``--pinned-root`` holds the pinned seed-0 artifacts (default: the repository's ``results/``, also in
smoke). Smoke options need ``--results-root`` outside ``results/``; ``--seeds`` is smoke only.

    uv run python scripts/instruction_negatives_seeds.py --part preflight          # exit 0 ok, 3 refused
    uv run python scripts/instruction_negatives_seeds.py --part panel --seed 1
    uv run python scripts/instruction_negatives_seeds.py --part probe
    uv run python scripts/instruction_negatives_seeds.py --part swaps --seed 1 --b stove
    uv run python scripts/instruction_negatives_seeds.py --part summary            # 0 valid, 3 invalid, 4 incomplete
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from flowcl.experiments.instruction_negatives_seeds import (
    BS, EXIT_CODES, load_config, run_panel, run_preflight, run_probes, run_summary, run_swaps,
)
from flowcl.utils.libero_paths import repo_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--part", required=True, choices=("preflight", "panel", "probe", "swaps", "summary"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--b", choices=BS, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--pinned-root", type=Path, default=None)
    parser.add_argument("--smoke-batches", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    results = (repo_root() / "results").resolve()
    scratch = args.results_root is not None and not (
        args.results_root.resolve() == results or results in args.results_root.resolve().parents)
    if (args.smoke_batches is not None or args.smoke or args.seeds is not None) and not scratch:
        parser.error("smoke options (--smoke, --smoke-batches, --seeds) need --results-root outside results/")
    if args.seeds is not None and not args.smoke:
        parser.error("--seeds needs --smoke")
    common = dict(results_root=args.results_root, allow_dirty=args.allow_dirty, dataset_dir=args.dataset_dir,
                  device=args.device)
    if args.part == "preflight":
        rec = run_preflight(cfg, results_root=args.results_root, pinned_root=args.pinned_root, smoke=args.smoke,
                            seeds=args.seeds)
        if not rec["passed"]:
            print(f"[flowcl] preflight refused: {rec}", flush=True)
        sys.exit(0 if rec["passed"] else 3)
    if args.part == "panel":
        if args.seed is None:
            parser.error("--part panel needs --seed")
        record = run_panel(cfg, args.seed, smoke=args.smoke, **common)
        sys.exit(0 if record["weights_unchanged"] else 3)
    if args.part == "swaps":
        if args.seed is None or args.b is None:
            parser.error("--part swaps needs --seed and --b")
        record = run_swaps(cfg, args.seed, args.b, **common)
        sys.exit(0 if record["weights_unchanged"] else 3)
    if args.part == "probe":
        problems = run_probes(cfg, smoke=args.smoke, seeds=args.seeds, smoke_batches=args.smoke_batches, **common)
        for p in problems:
            print(f"[flowcl] probe problem: {p}", flush=True)
        sys.exit(1 if problems else 0)
    summary = run_summary(cfg, results_root=args.results_root, pinned_root=args.pinned_root, smoke=args.smoke,
                          seeds=args.seeds)
    sys.exit(EXIT_CODES[summary["status"]])


if __name__ == "__main__":
    main()
