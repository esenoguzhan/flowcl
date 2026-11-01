"""AgileX Piper LeRobot v3.0 recordings -> canonical :class:`Episode`.

Two steps, so the slow video decode runs once:

1. :func:`build_cache` decodes a task's source folders into one HDF5 file, with every
   frame already reduced by :func:`flowcl.data.images.policy_frame` (masks included).
2. :func:`iter_cache_episodes` reads that cache.

Conventions (checked against the recordings of 30 Oct 2026, see docs/runs/):

* LeRobot v3.0 concatenates episodes: one parquet holds the rows of several episodes and
  one mp4 per camera holds their frames back to back. ``meta/episodes`` gives each
  episode's rows (``dataset_from_index`` .. ``dataset_to_index``) and its span in the
  video (``from_timestamp`` .. ``to_timestamp``); frame ``round(from_timestamp * fps) + i``
  belongs to the episode's row ``i``.
* ``action`` (7) is the leader arm's absolute joint targets (rad) and gripper target (m);
  ``observation.state`` (7) is the follower's measured joints and gripper opening, which
  trail the action by 1-2 ticks. Both are used as recorded, at the recording rate.
* ``observation.eef`` / ``action.eef`` are not used.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np

from flowcl.data.episode import Episode
from flowcl.data.images import policy_frame
from flowcl.data.spec import EmbodimentSpec

SOURCE_FPS = 15
D_JOINTS = 7
STATE_KEY = "observation.state"
ACTION_KEY = "action"
CAMERA_KEYS = {
    "front": "observation.images.image",
    "wrist": "observation.images.wrist_image",
}
RECORDED_HW = (480, 640)
CACHE_FORMAT = 1


# ---- raw dataset ----------------------------------------------------------------------


def source_fingerprint(source: Path) -> str:
    """sha256 over every file under ``meta/``, ``data/`` and ``videos/`` (paths and bytes).

    LeRobot appends new episodes to the same folder, so a folder recorded into again gets
    a new fingerprint and the registry check fails instead of silently training on more.
    """
    source = Path(source)
    digest = hashlib.sha256()
    files = sorted(
        p for top in ("meta", "data", "videos") for p in (source / top).rglob("*") if p.is_file()
    )
    if not files:
        raise FileNotFoundError(f"{source}: no files under meta/, data/ or videos/")
    for path in files:
        digest.update(str(path.relative_to(source)).encode())
        digest.update(b"\0")
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def read_info(source: Path) -> dict:
    info = json.loads((Path(source) / "meta" / "info.json").read_text())
    if info.get("codebase_version") != "v3.0":
        raise ValueError(f"{source}: expected LeRobot v3.0, got {info.get('codebase_version')}")
    if int(info["fps"]) != SOURCE_FPS:
        raise ValueError(f"{source}: expected {SOURCE_FPS} fps, got {info['fps']}")
    features = info["features"]
    for key in (STATE_KEY, ACTION_KEY, *CAMERA_KEYS.values()):
        if key not in features:
            raise KeyError(f"{source}: feature {key!r} missing; have {sorted(features)}")
    for key in (STATE_KEY, ACTION_KEY):
        if list(features[key]["shape"]) != [D_JOINTS]:
            raise ValueError(f"{source}: {key} shape {features[key]['shape']} != [{D_JOINTS}]")
    for key in CAMERA_KEYS.values():
        if list(features[key]["shape"]) != [*RECORDED_HW, 3]:
            raise ValueError(f"{source}: {key} shape {features[key]['shape']} != {[*RECORDED_HW, 3]}")
    return info


def _read_parquets(paths: list[Path]):
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not paths:
        raise FileNotFoundError("no parquet files")
    return pa.concat_tables([pq.read_table(p) for p in paths])


def read_raw_label(source: Path) -> str:
    """The task string LeRobot recorded (``meta/tasks.parquet``); one task per folder."""
    table = _read_parquets([Path(source) / "meta" / "tasks.parquet"])
    labels = [c for c in table.column_names if c != "task_index"]
    if table.num_rows != 1 or len(labels) != 1:
        raise ValueError(f"{source}: expected one recorded task, got {table.to_pydict()}")
    return str(table.column(labels[0])[0].as_py())


def read_episode_table(source: Path) -> list[dict]:
    """``meta/episodes`` rows (without the per-episode stats), ordered by episode index."""
    paths = sorted((Path(source) / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    table = _read_parquets(paths)
    keep = [c for c in table.column_names if not c.startswith("stats/")]
    rows = table.select(keep).to_pylist()
    rows.sort(key=lambda r: r["episode_index"])
    if [r["episode_index"] for r in rows] != list(range(len(rows))):
        raise ValueError(f"{source}: episode indices are not 0..{len(rows) - 1}")
    return rows


def read_frames_table(source: Path) -> dict[str, np.ndarray]:
    """Every data row of a folder, as arrays, with the row-level checks."""
    paths = sorted((Path(source) / "data").glob("chunk-*/file-*.parquet"))
    table = _read_parquets(paths)

    def stacked(key: str) -> np.ndarray:
        return np.asarray(table.column(key).to_pylist(), dtype=np.float32)

    out = {
        "state": stacked(STATE_KEY),
        "action": stacked(ACTION_KEY),
        "timestamp": np.asarray(table.column("timestamp").to_pylist(), dtype=np.float64),
        "frame_index": np.asarray(table.column("frame_index").to_pylist(), dtype=np.int64),
        "episode_index": np.asarray(table.column("episode_index").to_pylist(), dtype=np.int64),
        "index": np.asarray(table.column("index").to_pylist(), dtype=np.int64),
    }
    n = out["index"].shape[0]
    if not np.array_equal(out["index"], np.arange(n)):
        raise ValueError(f"{source}: the 'index' column is not 0..{n - 1}")
    for key in ("state", "action"):
        if out[key].shape != (n, D_JOINTS):
            raise ValueError(f"{source}: {key} has shape {out[key].shape}")
        if not np.isfinite(out[key]).all():
            raise ValueError(f"{source}: non-finite values in {key}")
    return out


def episode_rows(frames: dict, meta: dict, source: Path) -> slice:
    """The data rows of one episode, checked against its own columns."""
    e, start, stop = meta["episode_index"], meta["dataset_from_index"], meta["dataset_to_index"]
    if stop - start != meta["length"]:
        raise ValueError(f"{source} episode {e}: rows {start}..{stop} but length {meta['length']}")
    rows = slice(start, stop)
    if not (frames["episode_index"][rows] == e).all() or (frames["episode_index"] == e).sum() != meta["length"]:
        raise ValueError(f"{source} episode {e}: rows {start}..{stop} do not hold exactly this episode")
    if not np.array_equal(frames["frame_index"][rows], np.arange(meta["length"])):
        raise ValueError(f"{source} episode {e}: frame_index is not 0..{meta['length'] - 1}")
    steps = np.diff(frames["timestamp"][rows])
    if abs(frames["timestamp"][start]) > 1e-6 or (
        steps.size and np.abs(steps - 1.0 / SOURCE_FPS).max() > 1e-3
    ):
        raise ValueError(f"{source} episode {e}: timestamps are not uniform at {SOURCE_FPS} fps from 0")
    return rows


def video_spans(episodes: list[dict], camera: str, source: Path, info: dict) -> dict[Path, list[tuple[int, int, int]]]:
    """``video file -> [(episode_index, first frame, n frames)]``, contiguous from frame 0."""
    key = f"videos/{CAMERA_KEYS[camera]}"
    spans: dict[Path, list[tuple[int, int, int]]] = {}
    for meta in episodes:
        path = Path(source) / info["video_path"].format(
            video_key=CAMERA_KEYS[camera],
            chunk_index=meta[f"{key}/chunk_index"],
            file_index=meta[f"{key}/file_index"],
        )
        first_f = meta[f"{key}/from_timestamp"] * SOURCE_FPS
        last_f = meta[f"{key}/to_timestamp"] * SOURCE_FPS
        first = int(round(first_f))
        if abs(first_f - first) > 1e-3 or abs(last_f - first_f - meta["length"]) > 1e-3:
            raise ValueError(
                f"{path} episode {meta['episode_index']}: video span {first_f:.4f}..{last_f:.4f} "
                f"frames does not hold its {meta['length']} rows"
            )
        spans.setdefault(path, []).append((meta["episode_index"], first, meta["length"]))
    for path, items in spans.items():
        expected = 0
        for e, first, n in sorted(items, key=lambda s: s[1]):
            if first != expected:
                raise ValueError(f"{path}: episode {e} starts at frame {first}, expected {expected}")
            expected += n
    return spans


def decode_video(path: Path, camera: str, spec: EmbodimentSpec, n_expected: int) -> np.ndarray:
    """All frames of one video as policy frames, ``(n_expected, H, W, 3)`` RGB."""
    import cv2

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
        frames.append(policy_frame(bgr[:, :, ::-1], camera, spec))
    cap.release()
    if len(frames) != n_expected:
        raise ValueError(f"{path}: {len(frames)} video frames, the episodes need exactly {n_expected}")
    return np.stack(frames)


def read_source(source: Path, spec: EmbodimentSpec, exclude: tuple[int, ...] = ()) -> Iterator[dict]:
    """The kept episodes of one folder, in episode order: arrays and policy frames."""
    source = Path(source)
    info = read_info(source)
    episodes = read_episode_table(source)
    if len(episodes) != int(info["total_episodes"]):
        raise ValueError(f"{source}: {len(episodes)} episode rows, info says {info['total_episodes']}")
    unknown = sorted(set(exclude) - {m["episode_index"] for m in episodes})
    if unknown:
        raise ValueError(f"{source}: excluded episodes {unknown} do not exist")
    frames = read_frames_table(source)
    rows = {m["episode_index"]: episode_rows(frames, m, source) for m in episodes}
    images: dict[int, dict[str, np.ndarray]] = {m["episode_index"]: {} for m in episodes}
    for camera in spec.cameras:
        for path, items in video_spans(episodes, camera, source, info).items():
            decoded = decode_video(path, camera, spec, sum(n for _, _, n in items))
            for e, first, n in items:
                images[e][camera] = decoded[first:first + n]
    for meta in episodes:
        e = meta["episode_index"]
        if e in exclude:
            continue
        r = rows[e]
        yield {
            "raw_episode_index": e,
            "action": frames["action"][r],
            "state": frames["state"][r],
            "timestamp": frames["timestamp"][r],
            "images": images[e],
        }


# ---- cache ----------------------------------------------------------------------------


def _masks_json(spec: EmbodimentSpec) -> str:
    obs = spec.to_dict()["observation"]
    return json.dumps({"image_masks": obs.get("image_masks", {}),
                       "mask_source_hw": obs.get("mask_source_hw")}, sort_keys=True)


def _sources_json(task) -> str:
    return json.dumps([{"folder": s.folder, "sha256": s.sha256, "exclude": list(s.exclude)}
                       for s in task.sources], sort_keys=True)


def build_cache(task, raw_root: Path, out_path: Path, spec: EmbodimentSpec) -> dict:
    """Decode a registry task's sources into ``out_path`` (HDF5). Returns a summary.

    Each source folder must match its registry fingerprint first, so the cache can only
    hold the recordings the registry names.
    """
    raw_root = Path(raw_root)
    for src in task.sources:
        actual = source_fingerprint(raw_root / src.folder)
        if actual != src.sha256:
            raise ValueError(
                f"{raw_root / src.folder}: fingerprint {actual} != registry {src.sha256}. "
                "The folder changed since it was registered (new episodes?); check it and "
                "update flowcl/data/piper_tasks.py."
            )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".h5.partial")
    lengths, raw_labels = [], {}
    with h5py.File(tmp, "w") as f:
        f.attrs.update(
            {
                "format": CACHE_FORMAT,
                "task_name": task.name,
                "language": task.language,
                "sources": _sources_json(task),
                "source_fps": SOURCE_FPS,
                "image_size": list(spec.observation.image_size),
                "masks": _masks_json(spec),
                "image_fn": "flowcl.data.images.policy_frame",
                "complete": False,
            }
        )
        group = f.create_group("episodes")
        for src in task.sources:
            folder = raw_root / src.folder
            raw_labels[src.folder] = read_raw_label(folder)
            for ep in read_source(folder, spec, src.exclude):
                g = group.create_group(f"{len(lengths):06d}")
                g.attrs.update({"source": src.folder, "raw_episode_index": ep["raw_episode_index"]})
                g.create_dataset("action", data=ep["action"])
                g.create_dataset("state", data=ep["state"])
                g.create_dataset("timestamp", data=ep["timestamp"])
                for cam, frames in ep["images"].items():
                    g.create_dataset(f"images/{cam}", data=frames,
                                     chunks=(min(32, len(frames)), *frames.shape[1:]))
                lengths.append(int(ep["action"].shape[0]))
        if len(lengths) != task.n_episodes:
            raise ValueError(f"{task.name}: built {len(lengths)} episodes, the registry says {task.n_episodes}")
        f.attrs["raw_labels"] = json.dumps(raw_labels, sort_keys=True)
        f.attrs["n_episodes"] = len(lengths)
        f.attrs["complete"] = True
    tmp.replace(out_path)
    return {"task": task.name, "n_episodes": len(lengths), "frames": int(sum(lengths)),
            "length_min": min(lengths), "length_max": max(lengths), "raw_labels": raw_labels}


def cache_attrs(path: Path) -> dict:
    with h5py.File(path, "r") as f:
        return {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in f.attrs.items()}


def verify_cache(path: Path, task, n_demos: int | None = None) -> None:
    """Fail unless the cache is complete and built from the registry's recordings."""
    attrs = cache_attrs(path)
    if not attrs.get("complete"):
        raise ValueError(f"{path} is incomplete; rerun scripts/prepare_piper.py --build")
    if attrs.get("format") != CACHE_FORMAT:
        raise ValueError(f"{path}: cache format {attrs.get('format')} != {CACHE_FORMAT}")
    for key, want in (("task_name", task.name), ("language", task.language),
                      ("sources", _sources_json(task))):
        if attrs.get(key) != want:
            raise ValueError(f"{path}: cache {key} {attrs.get(key)!r} != registry {want!r}")
    if int(attrs["n_episodes"]) != task.n_episodes:
        raise ValueError(f"{path}: {attrs['n_episodes']} episodes, the registry says {task.n_episodes}")
    if n_demos is not None and n_demos > task.n_train:
        raise ValueError(
            f"{task.name}: {n_demos} demos requested, but only {task.n_train} are for training "
            f"(the last {task.holdout} are held out)"
        )


# ---- canonical episodes -----------------------------------------------------------------


def iter_cache_episodes(
    path: Path,
    spec: EmbodimentSpec,
    n_demos: int | None = None,
    task_id: str | None = None,
    episode_indices: list[int] | None = None,
) -> Iterator[Episode]:
    """Episodes ``0..n_demos-1`` (or ``episode_indices``) of a cache, as recorded."""
    path = Path(path)
    with h5py.File(path, "r") as f:
        attrs = dict(f.attrs)
        if list(attrs["image_size"]) != list(spec.observation.image_size) or attrs["masks"] != _masks_json(spec):
            raise ValueError(
                f"{path} was built for image size {list(attrs['image_size'])} and masks "
                f"{attrs['masks']}, but {spec.name} now asks for "
                f"{list(spec.observation.image_size)} and {_masks_json(spec)}; rerun "
                "scripts/prepare_piper.py --build"
            )
        names = sorted(f["episodes"])
        if episode_indices is None:
            selected = names if n_demos is None else names[:n_demos]
            if n_demos is not None and len(selected) < n_demos:
                raise ValueError(f"{path}: {n_demos} demos requested, cache has {len(names)}")
        else:
            selected = [f"{i:06d}" for i in episode_indices]
            missing = [n for n in selected if n not in f["episodes"]]
            if missing:
                raise ValueError(f"{path}: no episodes {missing}; cache has {len(names)}")
        for name in selected:
            g = f["episodes"][name]
            images = {}
            for cam in spec.cameras:
                if f"images/{cam}" not in g:
                    raise KeyError(f"{path}/{name}: no camera {cam!r}; have {list(g['images'])}")
                images[cam] = np.asarray(g[f"images/{cam}"])
            state = np.asarray(g["state"], dtype=np.float32)
            action = np.asarray(g["action"], dtype=np.float32)
            spec.assert_compatible_episode(d_state=state.shape[1], d_action=action.shape[1], cameras=tuple(images))
            yield Episode(
                images=images,
                state=state,
                action=action,
                language=str(attrs["language"]),
                task_id=task_id or f"piper/{attrs['task_name']}",
                embodiment=spec.name,
                meta={
                    "source_file": str(path),
                    "episode_index": int(name),
                    "source": str(g.attrs["source"]),
                    "raw_episode_index": int(g.attrs["raw_episode_index"]),
                },
            )
