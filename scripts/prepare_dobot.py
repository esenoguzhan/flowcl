"""Download and decode the Dobot X-Trainer puzzle recordings (branch dobot-hw).

For each task in :mod:`flowcl.data.dobot_tasks`:

1. ``snapshot_download`` of the pinned revision into ``datasets/dobot/raw/``;
2. a parallel decode into ``datasets/dobot/cache/<task>.h5`` (30 fps, 128 px policy frames);
3. a report (``datasets/dobot/report_<task>.json``, also printed): per-channel action
   statistics, |z| of every task under the first task's stats with and without the
   embodiment's std floor, and how often a demo's per-tick joint step exceeds the robot
   runner's clip limits;
4. initial-condition photos: the first top/wrist frames (640x480) of the held-out
   episodes (the last ``--holdout``), for setting the scene before a rollout.

    uv run python scripts/prepare_dobot.py --tasks puzzle_red_retrieve
    uv run python scripts/prepare_dobot.py --tasks all --workers 16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from flowcl.data.config import load_embodiment_spec
from flowcl.data.dobot_adapter import (
    CAMERA_KEYS,
    build_cache,
    cache_attrs,
    episode_paths,
    read_info,
)
from flowcl.data.dobot_tasks import TASKS, cache_path, dobot_root, get_task, raw_dir

JOINTS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
CLIPS = (0.03, 0.05)


def download(name: str) -> Path:
    from huggingface_hub import snapshot_download

    task = get_task(name)
    target = raw_dir(name)
    snapshot_download(task.repo_id, repo_type="dataset", revision=task.revision,
                      local_dir=str(target), max_workers=4)
    return target


def load_actions(name: str, episodes: list[int]) -> list[np.ndarray]:
    import h5py

    with h5py.File(cache_path(name), "r") as f:
        return [np.asarray(f["episodes"][f"{i:06d}"]["actions"]) for i in episodes]


def report(name: str, holdout: int, reference: str, floor: float) -> dict:
    task = get_task(name)
    n_train = task.n_episodes - holdout
    acts = load_actions(name, list(range(n_train)))
    stacked = np.concatenate(acts)
    steps = np.concatenate([np.abs(np.diff(a[:, JOINTS], axis=0)).max(axis=1) for a in acts])
    out = {
        "task": name,
        "n_train_episodes": n_train,
        "n_holdout_episodes": holdout,
        "frames_30hz": int(stacked.shape[0]),
        "seconds_median": float(np.median([len(a) for a in acts]) / 30.0),
        "action_mean": stacked.mean(0).round(4).tolist(),
        "action_std": stacked.std(0).round(4).tolist(),
        "action_min": stacked.min(0).round(4).tolist(),
        "action_max": stacked.max(0).round(4).tolist(),
        "joint_step_p99_rad": float(np.percentile(steps, 99)),
        "joint_step_max_rad": float(steps.max()),
        "frac_ticks_over_clip": {str(c): float(np.mean(steps > c)) for c in CLIPS},
    }
    if cache_path(reference).is_file():
        ref = np.concatenate(load_actions(reference, list(range(get_task(reference).n_episodes - holdout))))
        mean, std = ref.mean(0), ref.std(0)
        out["z_under_reference"] = {
            "reference": reference,
            "max_abs_z_no_floor": (np.abs(stacked - mean) / np.maximum(std, 1e-6)).max(0).round(1).tolist(),
            f"max_abs_z_floor_{floor}": (np.abs(stacked - mean) / np.maximum(std, floor)).max(0).round(1).tolist(),
        }
    return out


def photos(name: str, holdout: int) -> list[str]:
    """First 640x480 frame of every camera for each held-out episode."""
    import cv2

    task = get_task(name)
    root = raw_dir(name)
    info = read_info(root)
    out_dir = dobot_root() / "photos" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    tiles, written = [], []
    for ep in range(task.n_episodes - holdout, task.n_episodes):
        _, videos = episode_paths(root, info, ep)
        row = []
        for cam in CAMERA_KEYS:
            ok, frame = cv2.VideoCapture(str(videos[cam])).read()
            if not ok:
                raise IOError(f"cannot read {videos[cam]}")
            path = out_dir / f"ep{ep:03d}_{cam}.png"
            cv2.imwrite(str(path), frame)
            written.append(str(path))
            small = cv2.resize(frame, (320, 240))
            cv2.putText(small, f"ep{ep} {cam}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            row.append(small)
        tiles.append(np.hstack(row))
    cv2.imwrite(str(out_dir / "holdout_sheet.jpg"), np.vstack(tiles))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", nargs="+", default=["all"])
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--holdout", type=int, default=10, help="last N episodes kept out of training")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--force", action="store_true", help="rebuild an existing complete cache")
    parser.add_argument("--embodiment", default="dobot_xtrainer")
    args = parser.parse_args()

    names = list(TASKS) if args.tasks == ["all"] else args.tasks
    reference = next(iter(TASKS))
    floor = load_embodiment_spec(args.embodiment).stats_std_floor
    for name in names:
        task = get_task(name)
        if not args.skip_download:
            print(f"[prepare_dobot] downloading {task.repo_id}@{task.revision[:10]}", flush=True)
            download(name)
        path = cache_path(name)
        if path.is_file() and cache_attrs(path).get("complete") and not args.force:
            print(f"[prepare_dobot] {name}: cache exists at {path}", flush=True)
        else:
            summary = build_cache(raw_dir(name), path, task_name=name, repo_id=task.repo_id,
                                  revision=task.revision, language=task.language, workers=args.workers)
            print(f"[prepare_dobot] {name}: {summary}", flush=True)
        rep = report(name, args.holdout, reference, floor)
        rep["photos"] = photos(name, args.holdout)
        out = dobot_root() / f"report_{name}.json"
        out.write_text(json.dumps(rep, indent=2) + "\n")
        brief = {k: rep[k] for k in ("n_train_episodes", "frames_30hz", "seconds_median",
                                      "joint_step_p99_rad", "frac_ticks_over_clip")}
        print(f"[prepare_dobot] {name}: {brief}", flush=True)
        if "z_under_reference" in rep:
            print(f"[prepare_dobot] {name}: |z| under {reference} stats: {rep['z_under_reference']}", flush=True)


if __name__ == "__main__":
    main()
