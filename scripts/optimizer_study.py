"""Thin CLI: the broader optimizer study (fine-tuning stability-plasticity map, T1 -> T2).

Spec §1: no logic in scripts/. See :mod:`flowcl.experiments.optimizer_study`; the grid, the
variants and the reporting rules are fixed in ``configs/analysis/optimizer_study.yaml``.
A dirty tree is refused unless ``--allow-dirty``.

Examples::

    MUJOCO_GL=egl uv run python scripts/optimizer_study.py --part grid --seed 1
    MUJOCO_GL=egl uv run python scripts/optimizer_study.py --summary
"""

from __future__ import annotations

import argparse
from pathlib import Path

from flowcl.experiments.optimizer_study import load_study_config, run_seed, run_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--part", choices=("grid", "variants"))
    group.add_argument("--summary", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    cfg = load_study_config(args.config)
    if args.summary:
        run_summary(cfg, device=args.device, dataset_dir=args.dataset_dir)
    else:
        if args.seed is None:
            parser.error("--part needs --seed")
        run_seed(cfg, args.seed, args.part, device=args.device, allow_dirty=args.allow_dirty,
                 dataset_dir=args.dataset_dir)


if __name__ == "__main__":
    main()
