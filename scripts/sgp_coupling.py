"""Thin CLI: the SGP update-rule coupling test at T4 (arms, pilots, selection, summary).

Spec §1: no logic in scripts/. See :mod:`flowcl.experiments.sgp_coupling`; the arms, the gate,
the rate calibration, the pilot rule and the verdict are fixed in
``configs/analysis/sgp_coupling.yaml``. A dirty tree is refused unless ``--allow-dirty``.
``--part repro_gate`` exits 0 only if aw_soft reproduced A800's T4; ``--part rate_gate --rate R``
exits 0 only if rate R was selected (1 otherwise).

``--smoke-stop-after N`` (``--part pilot`` only) replaces the pilot length for a smoke run and
needs ``--results-root`` outside the repository's ``results/``.

Examples::

    MUJOCO_GL=egl uv run python scripts/sgp_coupling.py --part arm --arm aw_soft
    uv run python scripts/sgp_coupling.py --part repro_gate
    uv run python scripts/sgp_coupling.py --part pilot --arm aw_hard
    uv run python scripts/sgp_coupling.py --part pilot --arm sgd_hard --scale 0.5
    uv run python scripts/sgp_coupling.py --part select
    uv run python scripts/sgp_coupling.py --part rate_gate --rate proj
    MUJOCO_GL=egl uv run python scripts/sgp_coupling.py --part summary
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from flowcl.experiments.sgp_coupling import (
    gate_open,
    load_coupling_config,
    rate_open,
    run_arm,
    run_pilot,
    run_select,
    run_summary,
)
from flowcl.utils.libero_paths import repo_root

PARTS = ("arm", "repro_gate", "pilot", "select", "rate_gate", "summary")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--part", choices=PARTS, required=True)
    parser.add_argument("--arm", default=None)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--rate", choices=("free", "proj"), default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--smoke-stop-after", type=int, default=None)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    cfg = load_coupling_config(args.config)
    if args.smoke_stop_after is not None:
        if args.part != "pilot":
            parser.error("--smoke-stop-after applies to --part pilot only")
        if args.results_root is None or args.results_root.resolve() == (repo_root() / "results").resolve():
            parser.error("--smoke-stop-after needs --results-root outside the repository's results/")
    common = {"results_root": args.results_root}
    if args.part in ("arm", "pilot") and args.arm is None:
        parser.error(f"--part {args.part} needs --arm")
    if args.part == "arm":
        run_arm(cfg, args.arm, device=args.device, allow_dirty=args.allow_dirty,
                dataset_dir=args.dataset_dir, **common)
    elif args.part == "repro_gate":
        is_open = gate_open(cfg, args.results_root)
        print(f"[flowcl] reproduction gate {'open' if is_open else 'closed'}", flush=True)
        sys.exit(0 if is_open else 1)
    elif args.part == "pilot":
        run_pilot(cfg, args.arm, args.scale, device=args.device, allow_dirty=args.allow_dirty,
                  dataset_dir=args.dataset_dir, stop_after=args.smoke_stop_after, **common)
    elif args.part == "select":
        run_select(cfg, **common)
    elif args.part == "rate_gate":
        if args.rate is None:
            parser.error("--part rate_gate needs --rate")
        is_open = rate_open(cfg, args.rate, args.results_root)
        print(f"[flowcl] rate {args.rate} gate {'open' if is_open else 'closed'}", flush=True)
        sys.exit(0 if is_open else 1)
    else:
        run_summary(cfg, device=args.device, dataset_dir=args.dataset_dir, **common)


if __name__ == "__main__":
    main()
