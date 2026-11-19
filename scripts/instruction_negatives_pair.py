"""Thin CLI: instruction negatives on one A -> B pair (``--rule wine`` reads
``configs/analysis/instruction_negatives_wine.yaml``), GPM + negatives against plain-GPM controls.

Spec §1: no logic in scripts/. See :mod:`flowcl.experiments.instruction_negatives_pair`. The training
runs are scripts/run_continual.py (queued by scripts/queue_instruction_negatives_pair.sh). Smoke options
need ``--results-root`` outside ``results/``; ``--seeds`` is smoke only.

    uv run python scripts/instruction_negatives_pair.py --rule wine --part info        # shell assignments
    uv run python scripts/instruction_negatives_pair.py --rule wine --part preflight   # 0 ok, 3 refused
    uv run python scripts/instruction_negatives_pair.py --rule wine --part preflight --resume
    uv run python scripts/instruction_negatives_pair.py --rule wine --part hold        # 0 passed, 3 failed
    uv run python scripts/instruction_negatives_pair.py --rule wine --part panel --seed 0
    uv run python scripts/instruction_negatives_pair.py --rule wine --part probe
    uv run python scripts/instruction_negatives_pair.py --rule wine --part swaps --seed 0
    uv run python scripts/instruction_negatives_pair.py --rule wine --part summary [--final]  # 0 valid, 3 invalid, 4 incomplete
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from flowcl.experiments.instruction_negatives_pair import (
    EXIT_CODES, load_config, run_hold, run_name, run_panel, run_preflight, run_probes, run_summary, run_swaps,
)
from flowcl.utils.libero_paths import repo_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rule", required=True)
    parser.add_argument("--part", required=True,
                        choices=("info", "preflight", "hold", "panel", "probe", "swaps", "summary"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--smoke-batches", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--resume", action="store_true", help="preflight: a restarted queue (targets may exist)")
    parser.add_argument("--final", action="store_true", help="summary: the last one (swaps required)")
    args = parser.parse_args()
    cfg = load_config(args.rule)
    if args.part == "info":
        seeds = " ".join(str(s) for s in cfg["seeds"])
        print(f"SEEDS='{seeds}'\nCURRICULUM={cfg['curriculum']}\nOUT={cfg['out']}\n"
              f"NEG_METHOD={cfg['arms']['neg']['cli_method']}\nCTRL_METHOD={cfg['arms']['ctrl']['cli_method']}\n"
              f"NEG_DISPLAY={cfg['arms']['neg']['display']}\nCTRL_DISPLAY={cfg['arms']['ctrl']['display']}")
        assert run_name(cfg, "neg", 0) == f"{cfg['curriculum']}__{cfg['arms']['neg']['display']}__seed0"
        sys.exit(0)
    results = (repo_root() / "results").resolve()
    scratch = args.results_root is not None and not (
        args.results_root.resolve() == results or results in args.results_root.resolve().parents)
    if (args.smoke_batches is not None or args.smoke or args.seeds is not None) and not scratch:
        parser.error("smoke options (--smoke, --smoke-batches, --seeds) need --results-root outside results/")
    if args.seeds is not None and not args.smoke:
        parser.error("--seeds needs --smoke")
    common = dict(results_root=args.results_root, allow_dirty=args.allow_dirty, dataset_dir=args.dataset_dir)
    if args.part == "preflight":
        rec = run_preflight(cfg, results_root=args.results_root, smoke=args.smoke, seeds=args.seeds, resume=args.resume)
        if not rec["passed"]:
            print(f"[flowcl] preflight refused: {rec}", flush=True)
        sys.exit(0 if rec["passed"] else 3)
    if args.part == "hold":
        rec = run_hold(cfg, **common)
        sys.exit(0 if rec["passed"] else 3)
    if args.part == "panel":
        if args.seed is None:
            parser.error("--part panel needs --seed")
        rec = run_panel(cfg, args.seed, smoke=args.smoke, device=args.device, **common)
        sys.exit(0 if rec["weights_unchanged"] else 3)
    if args.part == "swaps":
        if args.seed is None:
            parser.error("--part swaps needs --seed")
        rec = run_swaps(cfg, args.seed, device=args.device, **common)
        sys.exit(0 if rec["weights_unchanged"] else 3)
    if args.part == "probe":
        problems, skipped = run_probes(cfg, smoke=args.smoke, seeds=args.seeds, smoke_batches=args.smoke_batches,
                                       device=args.device, **common)
        for p in skipped:
            print(f"[flowcl] probe skipped (no checkpoint): {p}", flush=True)
        for p in problems:
            print(f"[flowcl] probe problem: {p}", flush=True)
        sys.exit(1 if problems else 0)
    summary = run_summary(cfg, results_root=args.results_root, smoke=args.smoke, seeds=args.seeds, final=args.final)
    sys.exit(EXIT_CODES[summary["status"]])


if __name__ == "__main__":
    main()
