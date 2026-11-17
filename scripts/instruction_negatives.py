"""Thin CLI: the instruction-negatives pilot (hold check, panel and gate, swaps, probes, summary).

Spec §1: no logic in scripts/. See :mod:`flowcl.experiments.instruction_negatives`; the rule is
``configs/analysis/instruction_negatives.yaml``. The training runs are scripts/run_continual.py with
``gpm_neg`` and ``allowlist_ft_neg`` on ``similar_ab_stove`` (queued by
scripts/queue_instruction_negatives.sh). A dirty tree is refused unless ``--allow-dirty``.

Smoke options need ``--results-root`` outside the repository's ``results/``: ``--smoke`` (the smoke
episode counts and recipe) and ``--smoke-batches N`` (probes). The historical negatives-off runs are
read from ``--historical-root`` (default: the repository's ``results/``, read only).

    uv run python scripts/instruction_negatives.py --part hold
    uv run python scripts/instruction_negatives.py --part panel
    uv run python scripts/instruction_negatives.py --part gate      # exit 0 if the gate passed, 1 if not
    uv run python scripts/instruction_negatives.py --part swaps --arm gpm
    uv run python scripts/instruction_negatives.py --part probe
    uv run python scripts/instruction_negatives.py --part summary

Exit codes: hold 0, or 3 if the check failed; panel 0, or 3 if the weights changed; swaps likewise;
probe 0, or 1 if a needed checkpoint is missing or a probe failed; summary 0 valid, 3 invalid,
4 incomplete.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from flowcl.experiments.instruction_negatives import (
    ARMS, EXIT_CODES, gate_passed, load_config, run_hold, run_panel, run_probes, run_summary, run_swaps,
)
from flowcl.utils.libero_paths import repo_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--part", choices=("hold", "panel", "gate", "swaps", "probe", "summary"), required=True)
    parser.add_argument("--arm", choices=ARMS, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--historical-root", type=Path, default=None)
    parser.add_argument("--smoke-batches", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    results = (repo_root() / "results").resolve()
    scratch = args.results_root is not None and not (
        args.results_root.resolve() == results or results in args.results_root.resolve().parents)
    if (args.smoke_batches is not None or args.smoke) and not scratch:
        parser.error("smoke options need --results-root outside the repository's results/")
    common = dict(results_root=args.results_root, allow_dirty=args.allow_dirty, dataset_dir=args.dataset_dir)
    if args.part == "hold":
        record = run_hold(cfg, **common)
        sys.exit(0 if record["passed"] else 3)
    if args.part == "panel":
        record = run_panel(cfg, device=args.device, smoke=args.smoke, **common)
        sys.exit(0 if record["weights_unchanged"] else 3)
    if args.part == "gate":
        sys.exit(0 if gate_passed(cfg, args.results_root) else 1)
    if args.part == "swaps":
        if args.arm is None:
            parser.error("--part swaps needs --arm")
        record = run_swaps(cfg, args.arm, device=args.device, **common)
        sys.exit(0 if record["weights_unchanged"] else 3)
    if args.part == "probe":
        problems = run_probes(cfg, device=args.device, smoke_batches=args.smoke_batches, **common)
        for p in problems:
            print(f"[flowcl] probe problem: {p}", flush=True)
        sys.exit(1 if problems else 0)
    summary = run_summary(cfg, results_root=args.results_root, historical_root=args.historical_root, smoke=args.smoke)
    sys.exit(EXIT_CODES[summary["status"]])


if __name__ == "__main__":
    main()
