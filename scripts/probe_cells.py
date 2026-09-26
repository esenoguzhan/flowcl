"""Thin CLI: per-batch probe losses at the SGP rule's cells for one seed (no training).

Spec §1: no logic in scripts/. See :mod:`flowcl.experiments.probe_cells`; the cells and the
tolerance are in ``configs/analysis/sgp.yaml``. Every weighted mean must reproduce the
run's diagnostics ``loss_matrix`` value (fails loudly otherwise). A dirty tree is refused
unless ``--allow-dirty`` (recorded).

Example::

    MUJOCO_GL=egl uv run python scripts/probe_cells.py --seed 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

from flowcl.experiments.probe_cells import run_probe_cells
from flowcl.experiments.sgp_report import load_sgp_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    run_probe_cells(load_sgp_config(args.config), seed=args.seed, dataset_dir=args.dataset_dir,
                    device=args.device, allow_dirty=args.allow_dirty, out=args.out)


if __name__ == "__main__":
    main()
