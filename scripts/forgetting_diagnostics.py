"""Thin CLI: no-training forgetting diagnostics (loss matrix + activation interference).

Spec §1: no logic in scripts/. See :mod:`flowcl.experiments.forgetting_diagnostics`; the
decision rule is pre-registered in ``configs/analysis/forgetting_diagnostics.yaml`` and
must be committed first (a dirty tree is refused unless ``--allow-dirty``, recorded).

Example::

    MUJOCO_GL=egl uv run python scripts/forgetting_diagnostics.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from flowcl.experiments.forgetting_diagnostics import load_diag_config, run_forgetting_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    run_forgetting_diagnostics(
        load_diag_config(args.config), dataset_dir=args.dataset_dir, device=args.device,
        allow_dirty=args.allow_dirty, out=args.out,
    )


if __name__ == "__main__":
    main()
