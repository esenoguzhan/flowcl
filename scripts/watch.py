"""Thin CLI: localhost watcher for qualitative rollout videos.

Spec §1: no logic in scripts/. See :mod:`flowcl.envs.viewer`.

Opens a page on loopback. Pick a Gate 0 (or later) checkpoint, choose an episode,
record it, watch agentview + wrist at the control rate.

Example::

    MUJOCO_GL=egl uv run python scripts/watch.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from flowcl.envs.video import load_video_config
from flowcl.envs.viewer import serve_viewer
from flowcl.utils.libero_paths import repo_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=None, help="Defaults to video.yaml host.")
    parser.add_argument("--port", type=int, default=None, help="Defaults to video.yaml port.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help="Defaults to <repo>/results.",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    serve_viewer(
        host=args.host,
        port=args.port,
        results_root=args.results_root or (repo_root() / "results"),
        out_dir=args.out_dir,
        device=args.device,
        video_cfg=load_video_config(),
    )


if __name__ == "__main__":
    main()
