"""Thin CLI: SGP sensitivity on the standard recipe (bracket, select, validate, summary).

Spec §1: no logic in scripts/. See :mod:`flowcl.experiments.sgp_sensitivity`; the arms and the
selection and validation rules are fixed in ``configs/analysis/sgp_sensitivity.yaml``. A dirty
tree is refused unless ``--allow-dirty``. ``--part select_gate`` exits 0 only if an alpha was
selected (1 otherwise).

Examples::

    MUJOCO_GL=egl uv run python scripts/sgp_sensitivity.py --part bracket
    uv run python scripts/sgp_sensitivity.py --part select
    MUJOCO_GL=egl uv run python scripts/sgp_sensitivity.py --part validate --seed 1
    MUJOCO_GL=egl uv run python scripts/sgp_sensitivity.py --part summary
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from flowcl.experiments.sgp_sensitivity import (
    load_sensitivity_config,
    run_bracket,
    run_select,
    run_summary,
    run_validation,
    selection_open,
)

PARTS = ("bracket", "select", "select_gate", "validate", "summary")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--part", choices=PARTS, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    cfg = load_sensitivity_config(args.config)
    if args.part == "bracket":
        run_bracket(cfg, device=args.device, allow_dirty=args.allow_dirty, dataset_dir=args.dataset_dir)
    elif args.part == "select":
        run_select(cfg)
    elif args.part == "select_gate":
        is_open = selection_open(cfg)
        print(f"[flowcl] selection gate {'open' if is_open else 'closed'}", flush=True)
        sys.exit(0 if is_open else 1)
    elif args.part == "validate":
        if args.seed is None:
            parser.error("--part validate needs --seed")
        run_validation(cfg, args.seed, device=args.device, allow_dirty=args.allow_dirty,
                       dataset_dir=args.dataset_dir)
    else:
        run_summary(cfg, device=args.device, dataset_dir=args.dataset_dir)


if __name__ == "__main__":
    main()
