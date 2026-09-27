"""Thin CLI: low-update continual fine-tuning — held-out validation and the gated chain.

Spec §1: no logic in scripts/. See :mod:`flowcl.experiments.low_update`; the rule is fixed in
``configs/analysis/low_update.yaml``. A dirty tree is refused unless ``--allow-dirty``.
``--part gate`` / ``--part pilot_gate`` exit 0 only when the gate is open (1 otherwise).

Examples::

    MUJOCO_GL=egl uv run python scripts/low_update.py --part validation_seed --seed 1
    uv run python scripts/low_update.py --part validation
    uv run python scripts/low_update.py --part gate && \\
        MUJOCO_GL=egl uv run python scripts/low_update.py --part chain --seed 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from flowcl.experiments.low_update import (
    gate_open,
    load_low_update_config,
    pilot_gate_open,
    run_chain,
    run_chain_summary,
    run_validation,
    run_validation_seed,
)

PARTS = ("validation_seed", "validation", "gate", "chain", "chain_summary", "pilot_gate")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--part", choices=PARTS, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    cfg = load_low_update_config(args.config)
    if args.part in ("validation_seed", "chain") and args.seed is None:
        parser.error(f"--part {args.part} needs --seed")
    if args.part == "validation_seed":
        run_validation_seed(cfg, args.seed, device=args.device, allow_dirty=args.allow_dirty,
                            dataset_dir=args.dataset_dir)
    elif args.part == "validation":
        run_validation(cfg)
    elif args.part == "gate":
        is_open = gate_open(cfg)
        print(f"[flowcl] validation gate {'open' if is_open else 'closed'}", flush=True)
        sys.exit(0 if is_open else 1)
    elif args.part == "pilot_gate":
        is_open = pilot_gate_open(cfg)
        print(f"[flowcl] pilot gate {'open' if is_open else 'closed'}", flush=True)
        sys.exit(0 if is_open else 1)
    elif args.part == "chain":
        run_chain(cfg, args.seed, device=args.device, allow_dirty=args.allow_dirty,
                  dataset_dir=args.dataset_dir)
    else:
        run_chain_summary(cfg)


if __name__ == "__main__":
    main()
