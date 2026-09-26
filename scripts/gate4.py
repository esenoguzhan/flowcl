"""Thin CLI: Gate 4, the flow-time (s) characterization over three seeds (no training).

Spec §1: no logic in scripts/. See :mod:`flowcl.experiments.gate4`; the rule is
pre-registered in ``configs/analysis/flowtime.yaml`` and must be committed first (a dirty
tree is refused unless ``--allow-dirty``, recorded).

Example::

    MUJOCO_GL=egl uv run python scripts/gate4.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from flowcl.experiments.gate4 import load_flowtime_config, run_gate4


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    run_gate4(load_flowtime_config(args.config), dataset_dir=args.dataset_dir,
              device=args.device, allow_dirty=args.allow_dirty, out=args.out)


if __name__ == "__main__":
    main()
