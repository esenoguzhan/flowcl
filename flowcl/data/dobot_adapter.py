"""Dobot X-Trainer LeRobot v2.1 recordings -> canonical :class:`Episode` (branch dobot-hw).

Two steps, so the slow video decode runs once:

1. :func:`build_cache` decodes a downloaded dataset revision into one HDF5 file per task,
   at the recordings' full 30 fps, with every frame already reduced by
   :func:`flowcl.deploy.images.to_policy_frame` (the function the policy server applies to
   live frames).
2. :func:`iter_cache_episodes` reads that cache at the embodiment's control rate.

Conventions (all checked against the recordings, see docs/runs/ on this branch):

* ``actions`` (14) are absolute joint targets ``[L_j1..6, L_grip, R_j1..6, R_grip]``;
  ``obs.joint_positions`` (14) are measured. The videos carry two more frames than the
  parquet has rows; frame ``i`` belongs to row ``i`` (LeRobot's timestamp rule), so the
  trailing frames are dropped.
* At stride ``s`` (30 fps / control rate), ``action[i] = actions[s*i]`` and
  ``state[i] = joint_positions[s*i]`` with the two gripper entries replaced by the
  previous 30 Hz gripper *command* ``actions[s*i - 1]`` (``joint_positions[0]`` at
  ``i = 0``). The robot-side runner builds its live state the same way.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np

from flowcl.data.episode import Episode
from flowcl.data.spec import EmbodimentSpec

SOURCE_FPS = 30
D_JOINTS = 14
GRIPPER_DIMS = (6, 13)
CAMERA_KEYS = {
    "top": "obs.camera.top",
    "left_wrist": "obs.wrist_camera.left",
    "right_wrist": "obs.wrist_camera.right",
}
# Video frames beyond the parquet rows that are tolerated (and dropped).
MAX_EXTRA_VIDEO_FRAMES = 3
CACHE_FORMAT = 1


# ---- raw dataset ----------------------------------------------------------------------


def read_info(raw_root: Path) -> dict:
    info = json.loads((Path(raw_root) / "meta" / "info.json").read_text())
    if info.get("codebase_version") != "v2.1":
        raise ValueError(f"{raw_root}: expected LeRobot v2.1, got {info.get('codebase_version')}")
    if int(info["fps"]) != SOURCE_FPS:
        raise ValueError(f"{raw_root}: expected {SOURCE_FPS} fps, got {info['fps']}")
    for key in ("actions", "obs.joint_positions", *CAMERA_KEYS.values()):
        if key not in info["features"]:
            raise KeyError(f"{raw_root}: feature {key!r} missing; have {sorted(info['features'])}")
    for key in ("actions", "obs.joint_positions"):
        if list(info["features"][key]["shape"]) != [D_JOINTS]:
            raise ValueError(f"{raw_root}: {key} shape {info['features'][key]['shape']} != [14]")
    return info


def episode_paths(raw_root: Path, info: dict, episode_index: int) -> tuple[Path, dict[str, Path]]:
    raw_root = Path(raw_root)
    chunk = episode_index // int(info.get("chunks_size", 1000))
    fmt = {"episode_index": episode_index, "episode_chunk": chunk}
    parquet = raw_root / info["data_path"].format(**fmt)
    videos = {
        cam: raw_root / info["video_path"].format(video_key=key, **fmt)
        for cam, key in CAMERA_KEYS.items()
    }
    return parquet, videos


def decode_video(path: Path, n_expected: int) -> np.ndarray:
    """All frames of one camera video as policy frames ``(n_expected, 128, 128, 3)`` RGB."""
    import cv2

    from flowcl.deploy.images import RECORDED_HW, to_policy_frame

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open {path}")
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if bgr.shape[:2] != RECORDED_HW:
            raise ValueError(f"{path}: frame {len(frames)} is {bgr.shape[:2]}, expected {RECORDED_HW}")
        frames.append(to_policy_frame(bgr))
    cap.release()
    extra = len(frames) - n_expected
    if not 0 <= extra <= MAX_EXTRA_VIDEO_FRAMES:
        raise ValueError(
            f"{path}: {len(frames)} video frames for {n_expected} parquet rows "
            f"(tolerated: 0..{MAX_EXTRA_VIDEO_FRAMES} trailing extra frames)"
        )
    return np.stack(frames[:n_expected])


def read_raw_episode(raw_root: Path, info: dict, episode_index: int) -> dict:
    """One recorded episode at full rate: arrays, policy frames and its prompt."""
    import pyarrow.parquet as pq

    parquet, videos = episode_paths(raw_root, info, episode_index)
    table = pq.read_table(parquet, columns=["actions", "obs.joint_positions", "timestamp", "prompt"])
    actions = np.asarray(table.column("actions").to_pylist(), dtype=np.float32)
    qpos = np.asarray(table.column("obs.joint_positions").to_pylist(), dtype=np.float32)
    timestamps = np.asarray(table.column("timestamp").to_pylist(), dtype=np.float64)
    prompts = set(table.column("prompt").to_pylist())
    if len(prompts) != 1:
        raise ValueError(f"{parquet}: expected one prompt, got {sorted(prompts)}")
    n = actions.shape[0]
    if actions.shape != (n, D_JOINTS) or qpos.shape != (n, D_JOINTS):
        raise ValueError(f"{parquet}: actions {actions.shape}, qpos {qpos.shape}")
    if not (np.isfinite(actions).all() and np.isfinite(qpos).all()):
        raise ValueError(f"{parquet}: non-finite actions or joint positions")
    steps = np.diff(timestamps)
    if n > 1 and np.abs(steps - 1.0 / SOURCE_FPS).max() > 1e-3:
        raise ValueError(f"{parquet}: timestamps are not uniform at {SOURCE_FPS} fps")
    images = {cam: decode_video(path, n) for cam, path in videos.items()}
    return {
        "episode_index": episode_index,
        "actions": actions,
        "qpos": qpos,
        "timestamp": timestamps,
        "prompt": prompts.pop(),
        "images": images,
    }


# ---- cache ----------------------------------------------------------------------------


def _decode_worker(args: tuple) -> dict:
    raw_root, info, episode_index = args
    return read_raw_episode(Path(raw_root), info, episode_index)


def build_cache(
    raw_root: Path,
    out_path: Path,
    *,
    task_name: str,
    repo_id: str,
    revision: str,
    language: str,
    workers: int = 8,
) -> dict:
    """Decode a downloaded dataset revision into ``out_path`` (HDF5). Returns a summary."""
    from multiprocessing import get_context

    raw_root = Path(raw_root)
    info = read_info(raw_root)
    n_episodes = int(info["total_episodes"])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".h5.partial")
    lengths = []
    with h5py.File(tmp, "w") as f:
        f.attrs.update(
            {
                "format": CACHE_FORMAT,
                "task_name": task_name,
                "repo_id": repo_id,
                "revision": revision,
                "language": language,
                "source_fps": SOURCE_FPS,
                "image_fn": "flowcl.deploy.images.to_policy_frame",
                "n_episodes": n_episodes,
                "complete": False,
            }
        )
        group = f.create_group("episodes")
        jobs = [(str(raw_root), info, i) for i in range(n_episodes)]
        with get_context("spawn").Pool(workers) as pool:
            for ep in pool.imap(_decode_worker, jobs):
                if ep["prompt"] != language:
                    raise ValueError(
                        f"{task_name} episode {ep['episode_index']}: prompt {ep['prompt']!r} "
                        f"!= registry instruction {language!r}"
                    )
                g = group.create_group(f"{ep['episode_index']:06d}")
                g.create_dataset("actions", data=ep["actions"])
                g.create_dataset("qpos", data=ep["qpos"])
                g.create_dataset("timestamp", data=ep["timestamp"])
                for cam, frames in ep["images"].items():
                    g.create_dataset(f"images/{cam}", data=frames, chunks=(min(32, len(frames)), *frames.shape[1:]))
                lengths.append(int(ep["actions"].shape[0]))
                if len(lengths) % 10 == 0:
                    print(f"[prepare_dobot] {task_name}: {len(lengths)}/{n_episodes} episodes", flush=True)
        f.attrs["complete"] = True
    tmp.replace(out_path)
    return {"task": task_name, "n_episodes": n_episodes, "frames": int(sum(lengths)),
            "length_min": min(lengths), "length_max": max(lengths)}


def cache_attrs(path: Path) -> dict:
    with h5py.File(path, "r") as f:
        return {k: (v.item() if hasattr(v, "item") else v) for k, v in f.attrs.items()}


def verify_cache(path: Path, *, repo_id: str, revision: str, language: str, n_demos: int | None) -> None:
    """Fail unless the cache is complete and built from the registry's revision."""
    attrs = cache_attrs(path)
    if not attrs.get("complete"):
        raise ValueError(f"{path} is incomplete; rerun scripts/prepare_dobot.py")
    if attrs.get("format") != CACHE_FORMAT:
        raise ValueError(f"{path}: cache format {attrs.get('format')} != {CACHE_FORMAT}")
    for key, want in (("repo_id", repo_id), ("revision", revision), ("language", language)):
        if attrs.get(key) != want:
            raise ValueError(f"{path}: cache {key} {attrs.get(key)!r} != registry {want!r}")
    if n_demos is not None and n_demos > int(attrs["n_episodes"]):
        raise ValueError(f"{path}: {n_demos} demos requested, cache has {attrs['n_episodes']}")


# ---- canonical episodes -----------------------------------------------------------------


def stride_for(spec: EmbodimentSpec, source_fps: int = SOURCE_FPS) -> int:
    ratio = source_fps / spec.action.control_rate_hz
    stride = int(round(ratio))
    if stride < 1 or abs(ratio - stride) > 1e-6:
        raise ValueError(
            f"control_rate_hz {spec.action.control_rate_hz} must divide the {source_fps} fps recordings"
        )
    return stride


def subsample(actions: np.ndarray, qpos: np.ndarray, stride: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(index, state, action)`` at ``stride`` under the gripper-state rule above."""
    index = np.arange(0, actions.shape[0], stride)
    action = actions[index].astype(np.float32, copy=True)
    state = qpos[index].astype(np.float32, copy=True)
    grip = list(GRIPPER_DIMS)
    previous = index[1:] - 1
    state[1:, grip] = actions[previous][:, grip]
    return index, state, action


def iter_cache_episodes(
    path: Path,
    spec: EmbodimentSpec,
    n_demos: int | None = None,
    task_id: str | None = None,
    episode_indices: list[int] | None = None,
) -> Iterator[Episode]:
    """Episodes ``0..n_demos-1`` (or ``episode_indices``) of a cache, at the spec's rate."""
    path = Path(path)
    stride = stride_for(spec)
    with h5py.File(path, "r") as f:
        attrs = dict(f.attrs)
        names = sorted(f["episodes"])
        if episode_indices is None:
            selected = names if n_demos is None else names[:n_demos]
            if n_demos is not None and len(selected) < n_demos:
                raise ValueError(f"{path}: {n_demos} demos requested, cache has {len(names)}")
        else:
            selected = [f"{i:06d}" for i in episode_indices]
        for name in selected:
            g = f["episodes"][name]
            actions = np.asarray(g["actions"], dtype=np.float32)
            qpos = np.asarray(g["qpos"], dtype=np.float32)
            index, state, action = subsample(actions, qpos, stride)
            images = {}
            for cam in spec.cameras:
                if f"images/{cam}" not in g:
                    raise KeyError(f"{path}/{name}: no camera {cam!r}; have {list(g['images'])}")
                # A step-sliced h5py read is a per-element hyperslab and takes minutes;
                # read the episode contiguously and subsample in memory.
                frames = np.ascontiguousarray(g[f"images/{cam}"][()][::stride])
                if frames.shape[1:3] != tuple(spec.observation.image_size):
                    raise ValueError(f"{path}/{name}/{cam}: frames {frames.shape[1:3]} != spec {spec.observation.image_size}")
                images[cam] = frames
            spec.assert_compatible_episode(d_state=state.shape[1], d_action=action.shape[1], cameras=tuple(images))
            yield Episode(
                images=images,
                state=state,
                action=action,
                language=str(attrs["language"]),
                task_id=task_id or f"dobot/{attrs['task_name']}",
                embodiment=spec.name,
                meta={
                    "source_file": str(path),
                    "repo_id": str(attrs["repo_id"]),
                    "revision": str(attrs["revision"]),
                    "episode_index": int(name),
                    "length_source": int(actions.shape[0]),
                    "stride": stride,
                },
            )
