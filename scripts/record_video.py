"""Thin CLI: record qualitative rollout videos from a checkpoint.

Spec §1: no logic in scripts/. See :mod:`flowcl.envs.video`.

This is not a Gate / retention eval. ``record_video`` stays off in the default
eval config; these artifacts are for looking at the policy.

Example::

    MUJOCO_GL=egl uv run python scripts/record_video.py \\
        --checkpoint results/single__libero_object__pick_up_the_milk_and_place_it_in_the_basket__seed0/checkpoints/final.pt \\
        --episodes 0 6
"""

from __future__ import annotations

import argparse
from pathlib import Path

from flowcl.envs.video import load_video_config, record_episodes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--episodes",
        nargs="+",
        type=int,
        required=True,
        metavar="IDX",
        help="0-based episode indices from the shared init-state set.",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="suite/task_name. Defaults to the checkpoint's recorded task_key.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Defaults to the checkpoint's run_id so seeds match the eval table.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Defaults to configs/eval/video.yaml out_dir.",
    )
    args = parser.parse_args()

    artifacts = record_episodes(
        args.checkpoint,
        args.episodes,
        task_key=args.task,
        run_id=args.run_id,
        device=args.device,
        out_dir=args.out_dir,
        video_cfg=load_video_config(),
    )
    for art in artifacts:
        status = "success" if art.meta["success"] else "fail"
        print(
            f"[flowcl] {art.meta['task_key']} ep {art.meta['episode_idx']}: "
            f"{status}, {art.n_frames} frames -> {art.directory}",
            flush=True,
        )


if __name__ == "__main__":
    main()
