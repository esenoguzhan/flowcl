"""Thin CLI: the GPM feasibility pilot (T1 -> T2 from the Gate 1 stage-0 checkpoint).

Spec §1: no logic in scripts/. See :mod:`flowcl.experiments.gpm_pilot`.

Examples::

    uv run python scripts/gpm_pilot.py --sanity
    uv run python scripts/gpm_pilot.py --arms gpm_projected_adam freeze_only
"""

from __future__ import annotations

import argparse
from pathlib import Path

from flowcl.data.curriculum import load_curriculum
from flowcl.experiments.gpm_pilot import (
    ARMS,
    default_inputs,
    load_method_config,
    load_pilot_config,
    run_gpm_pilot,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curriculum", default="seq_hetero")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--sanity",
        action="store_true",
        help="Short gpm_projected_adam run with asserted checks and timing; no rollouts.",
    )
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--method-config", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    run_gpm_pilot(
        default_inputs(load_curriculum(args.curriculum), seed=args.seed),
        pilot=load_pilot_config(args.config),
        method_cfg=load_method_config(args.method_config),
        device=args.device,
        sanity=args.sanity,
        arms=tuple(args.arms) if args.arms else None,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
