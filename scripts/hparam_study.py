"""Thin CLI: the exploratory training-budget and learning-rate study (single seed).

Spec §1: no logic in scripts/. See :mod:`flowcl.experiments.hparam_study`; everything
reported, including the budget rule, is fixed in ``configs/analysis/hparam_study.yaml``.
A dirty tree is refused unless ``--allow-dirty`` (recorded).

Examples::

    MUJOCO_GL=egl uv run python scripts/hparam_study.py --part trajectory --task-index 0
    MUJOCO_GL=egl uv run python scripts/hparam_study.py --part cl_budget
    MUJOCO_GL=egl uv run python scripts/hparam_study.py --part cl_lr   # needs cl_budget's report
"""

from __future__ import annotations

import argparse
from pathlib import Path

from flowcl.experiments.hparam_study import PARTS, load_study_config, run_cl_part, run_trajectory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--part", choices=PARTS, required=True)
    parser.add_argument("--task-index", type=int, default=None,
                        help="trajectory only: index into the config's trajectory.tasks")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    cfg = load_study_config(args.config)
    if args.part == "trajectory":
        if args.task_index is None:
            parser.error("--part trajectory needs --task-index")
        run_trajectory(cfg, cfg["trajectory"]["tasks"][args.task_index],
                       dataset_dir=args.dataset_dir, device=args.device,
                       allow_dirty=args.allow_dirty)
    else:
        run_cl_part(cfg, args.part, dataset_dir=args.dataset_dir, device=args.device,
                    allow_dirty=args.allow_dirty)


if __name__ == "__main__":
    main()
