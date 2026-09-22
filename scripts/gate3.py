"""Thin CLI: run §10.3 Gate 3 -- does the new task need protected directions?

Spec §1: no logic in scripts/. See :mod:`flowcl.experiments.gate3`.

Example::

    uv run python scripts/gate3.py --curriculum seq_hetero --seed 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

from flowcl.data.curriculum import load_curriculum
from flowcl.experiments.gate3 import default_inputs, load_interference_config, run_gate3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curriculum", default="seq_hetero")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-control", action="store_true", help="Skip the T1 self-gradient control."
    )
    parser.add_argument(
        "--no-replicate",
        action="store_true",
        help="Skip the paired single-Spatial replicate.",
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    run_gate3(
        default_inputs(load_curriculum(args.curriculum), seed=args.seed),
        cfg=load_interference_config(args.config),
        device=args.device,
        out_dir=args.out_dir,
        control=not args.no_control,
        replicate=not args.no_replicate,
    )


if __name__ == "__main__":
    main()
