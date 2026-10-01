"""Thin CLI: the T5 saturation stress test (replay, gate, arms, summary).

Spec §1: no logic in scripts/. See :mod:`flowcl.experiments.t5_sweep`; the start, the replay
gate, the arms and the verdict are fixed in ``configs/analysis/t5_sweep.yaml``. A dirty tree is
refused unless ``--allow-dirty``. ``--part replay_gate`` exits 0 only if the replay passed.

Smoke options (each needs ``--results-root`` outside the repository's ``results/``):
``--smoke-replay-tasks N`` replays only the first N tasks; ``--smoke-stop-after S`` and
``--smoke-episodes E`` shorten an arm's training and evaluation.

Examples::

    MUJOCO_GL=egl uv run python scripts/t5_sweep.py --part replay
    uv run python scripts/t5_sweep.py --part replay_gate
    MUJOCO_GL=egl uv run python scripts/t5_sweep.py --part arm --arm sgp_a100
    MUJOCO_GL=egl uv run python scripts/t5_sweep.py --part summary
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from flowcl.experiments.t5_sweep import (
    load_sweep_config,
    replay_open,
    run_arm,
    run_replay,
    run_summary,
)
from flowcl.utils.libero_paths import repo_root

PARTS = ("replay", "replay_gate", "arm", "summary")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--part", choices=PARTS, required=True)
    parser.add_argument("--arm", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--smoke-replay-tasks", type=int, default=None)
    parser.add_argument("--smoke-stop-after", type=int, default=None)
    parser.add_argument("--smoke-episodes", type=int, default=None)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    cfg = load_sweep_config(args.config)
    smoke = any(v is not None for v in (args.smoke_replay_tasks, args.smoke_stop_after, args.smoke_episodes))
    if smoke and (args.results_root is None
                  or args.results_root.resolve() == (repo_root() / "results").resolve()):
        parser.error("the smoke options need --results-root outside the repository's results/")
    common = {"results_root": args.results_root}
    if args.part == "replay":
        run_replay(cfg, device=args.device, allow_dirty=args.allow_dirty, dataset_dir=args.dataset_dir,
                   n_tasks=args.smoke_replay_tasks, **common)
    elif args.part == "replay_gate":
        is_open = replay_open(cfg, args.results_root)
        print(f"[flowcl] replay gate {'open' if is_open else 'closed'}", flush=True)
        sys.exit(0 if is_open else 1)
    elif args.part == "arm":
        if args.arm is None:
            parser.error("--part arm needs --arm")
        run_arm(cfg, args.arm, device=args.device, allow_dirty=args.allow_dirty, dataset_dir=args.dataset_dir,
                stop_after=args.smoke_stop_after, n_episodes=args.smoke_episodes, **common)
    else:
        run_summary(cfg, device=args.device, dataset_dir=args.dataset_dir, **common)


if __name__ == "__main__":
    main()
