"""Open-loop check of a served checkpoint on recorded episodes (4090 PC, no robot).

For each episode and every 24th tick (one executed chunk), it sends the recorded
frames and state to the running server and compares the predicted 30 Hz prefix with the
recorded actions. Two image paths with the same fixed sampling noise:

* ``jpeg``: the recorded 640x480 video frame -> JPEG q50 -> server decode -> policy frame,
  i.e. the live path, with the video frame standing in for the camera.
* ``cache``: the training cache's policy frame, sent as ``policy_images``.

Their difference measures the image path. The held-out vs training gap is a first look
at generalisation; it is not closed-loop success.

    uv run python scripts/hw/offline_check.py --server ws://127.0.0.1:8000 \
        --task puzzle_red_retrieve --episodes 110-119 0-9
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from flowcl.data.dobot_adapter import read_info, episode_paths, CAMERA_KEYS  # noqa: E402
from flowcl.data.dobot_tasks import TASKS, cache_path, raw_dir  # noqa: E402
from flowcl.deploy.images import encode_jpeg  # noqa: E402
from flowcl.deploy.safety import JOINT_DIMS  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dobot_run import PolicyClient, parse_episodes  # noqa: E402


def video_frames(path: Path) -> list[np.ndarray]:
    import cv2

    cap, frames = cv2.VideoCapture(str(path)), []
    while True:
        ok, f = cap.read()
        if not ok:
            return frames
        frames.append(f)


def check_episode(client: PolicyClient, task: str, ep: int, execute: int, seed: int) -> dict:
    import h5py

    language = TASKS[task].language
    root = raw_dir(task)
    _, videos = episode_paths(root, read_info(root), ep)
    raw = {cam: video_frames(videos[cam]) for cam in CAMERA_KEYS}
    with h5py.File(cache_path(task), "r") as f:
        g = f["episodes"][f"{ep:06d}"]
        actions, qpos = np.asarray(g["actions"]), np.asarray(g["qpos"])
        cached = {cam: np.asarray(g[f"images/{cam}"]) for cam in CAMERA_KEYS}
    errors = {"jpeg": [], "cache": []}
    gap = []
    for t in range(0, len(actions) - execute, execute):
        state = qpos[t].copy()
        if t > 0:
            state[[6, 13]] = actions[t - 1, [6, 13]]
        target = actions[t : t + execute]
        base = {"type": "infer", "prompt": language, "state": state.astype(np.float32), "seed": seed + t}
        r_jpeg = client.call({**base, "images": {c: encode_jpeg(raw[c][t]) for c in CAMERA_KEYS}})
        r_cache = client.call({**base, "policy_images": {c: cached[c][t] for c in CAMERA_KEYS}})
        for name, r in (("jpeg", r_jpeg), ("cache", r_cache)):
            pred = np.asarray(r["actions_30hz"])[:execute]
            errors[name].append(np.abs(pred - target)[:, JOINT_DIMS].mean(axis=0))
        gap.append(np.abs(np.asarray(r_jpeg["actions_30hz"])[:execute] - np.asarray(r_cache["actions_30hz"])[:execute])[:, JOINT_DIMS].max())
    out = {"episode": ep, "chunks": len(gap)}
    for name, errs in errors.items():
        e = np.asarray(errs)
        out[f"{name}_mae_rad"] = float(e.mean())
        out[f"{name}_mae_per_joint"] = e.mean(axis=0).round(4).tolist()
    out["jpeg_vs_cache_max_rad"] = float(np.max(gap))
    out["jpeg_vs_cache_median_rad"] = float(np.median(gap))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server", default="ws://127.0.0.1:8000")
    parser.add_argument("--task", required=True)
    parser.add_argument("--episodes", nargs="+", required=True, help="e.g. 110-119 0-9")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    client = PolicyClient(args.server, timeout_s=30.0)
    execute = int(client.meta["execute_ticks"])
    holdout_start = TASKS[args.task].n_episodes - int(client.meta["holdout"])
    rows = []
    for group in args.episodes:
        for ep in parse_episodes(group):
            row = check_episode(client, args.task, ep, execute, args.seed)
            row["held_out"] = ep >= holdout_start
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if "per_joint" not in k}), flush=True)
    summary = {}
    for held in (True, False):
        sel = [r for r in rows if r["held_out"] == held]
        if sel:
            key = "held_out" if held else "train"
            summary[key] = {
                "episodes": [r["episode"] for r in sel],
                "jpeg_mae_rad": float(np.mean([r["jpeg_mae_rad"] for r in sel])),
                "cache_mae_rad": float(np.mean([r["cache_mae_rad"] for r in sel])),
                "jpeg_vs_cache_median_rad": float(np.median([r["jpeg_vs_cache_median_rad"] for r in sel])),
                "jpeg_vs_cache_max_rad": float(np.max([r["jpeg_vs_cache_max_rad"] for r in sel])),
            }
    result = {"checkpoint": client.meta["checkpoint"], "task": args.task, "execute_ticks": execute,
              "summary": summary, "episodes": rows}
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    client.close()


if __name__ == "__main__":
    main()
