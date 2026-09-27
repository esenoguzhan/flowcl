"""Thin CLI: the exploratory projection pilot on the low-update backbone (seed 0, T1 -> T2).

Spec §1: no logic in scripts/. See :mod:`flowcl.experiments.projection_pilot`; the grid and
the selection rule are fixed in ``configs/analysis/projection_pilot.yaml``. Runs only behind
the pilot gate. A dirty tree is refused unless ``--allow-dirty``.

Examples::

    MUJOCO_GL=egl uv run python scripts/projection_pilot.py --lr 3e-5
    uv run python scripts/projection_pilot.py --summary
"""

from __future__ import annotations

import argparse
from pathlib import Path

from flowcl.experiments.projection_pilot import load_pilot_config, run_pilot, run_pilot_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--lr", type=float, help="run every grid entry at this learning rate")
    group.add_argument("--summary", action="store_true", help="summarise and select")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    cfg = load_pilot_config(args.config)
    if args.summary:
        run_pilot_summary(cfg)
    else:
        run_pilot(cfg, args.lr, device=args.device, allow_dirty=args.allow_dirty,
                  dataset_dir=args.dataset_dir)


if __name__ == "__main__":
    main()
