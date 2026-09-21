"""Record qualitative rollout videos from a checkpoint.

This is *not* the §8.1 evaluation protocol. ``record_video`` stays off in
``configs/eval/libero_eval.yaml`` so Gate and retention numbers never pay for, or
depend on, video I/O. This module writes a side artifact: a per-episode directory
of JPEG frames (what the localhost watcher plays), an optional ``.mp4``, and —
when the rollout carried a :class:`~flowcl.envs.libero_env.RolloutTrace` — a
``trace.npz`` / ``trace.json`` for trajectory and action-smoothness plots.

Playback rate is the embodiment control rate, not a guessed 20 fps.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from flowcl.data.tasks import TaskRef
from flowcl.envs.evaluation import eval_config_from_dict
from flowcl.envs.libero_env import EvalConfig, LiberoTaskEnv, RolloutResult, RolloutTrace
from flowcl.train.checkpoint import load_checkpoint
from flowcl.utils.libero_paths import repo_root


def load_video_config(path: Path | None = None) -> dict:
    """Load ``configs/eval/video.yaml``. Unknown keys are an error."""
    cfg_path = path or (repo_root() / "configs" / "eval" / "video.yaml")
    payload = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    if not isinstance(payload, dict):
        raise TypeError(f"{cfg_path} must resolve to a mapping, got {type(payload)}")
    known = {
        "out_dir",
        "host",
        "port",
        "jpeg_quality",
        "display_scale",
        "mp4_codec",
        "clip_abs",
        "clip_rail_tol",
    }
    unknown = sorted(set(payload) - known)
    if unknown:
        raise ValueError(
            f"unknown keys in video config {cfg_path}: {unknown}; known keys are "
            f"{sorted(known)}"
        )
    return payload


def resolve_out_dir(out_dir: str | Path | None = None, video_cfg: dict | None = None) -> Path:
    """Absolute video-artifact root. Relative paths are against the repo root."""
    cfg = video_cfg if video_cfg is not None else load_video_config()
    raw = Path(out_dir) if out_dir is not None else Path(cfg["out_dir"])
    path = raw if raw.is_absolute() else (repo_root() / raw)
    return path.resolve()


def task_slug(task_key: str) -> str:
    """Filesystem-safe task key. ``suite/name`` -> ``suite__name``."""
    if "/" not in task_key:
        raise ValueError(f"task_key {task_key!r} must be '<suite>/<task name>'")
    return task_key.replace("/", "__")


def episode_dir(out_dir: Path, run_id: str, task_key: str, episode_idx: int) -> Path:
    """``<out_dir>/<run_id>/<task_slug>/epXXXX``."""
    if episode_idx < 0:
        raise ValueError(f"episode_idx must be non-negative, got {episode_idx}")
    return out_dir / run_id / task_slug(task_key) / f"ep{episode_idx:04d}"


def assert_under(root: Path, path: Path) -> Path:
    """Resolve ``path`` and refuse anything outside ``root``."""
    root_r = root.resolve()
    path_r = path.resolve()
    if path_r != root_r and root_r not in path_r.parents:
        raise ValueError(f"{path} is outside {root_r}")
    return path_r


@dataclass
class VideoArtifact:
    """One recorded episode on disk."""

    directory: Path
    meta: dict
    cameras: tuple[str, ...] = ()

    @property
    def n_frames(self) -> int:
        return int(self.meta["n_frames"])

    @property
    def frame_relpaths(self) -> dict[str, list[str]]:
        """Camera -> relative JPEG paths from ``directory``."""
        n = self.n_frames
        return {
            camera: [f"{camera}/{i:05d}.jpg" for i in range(n)] for camera in self.cameras
        }

    def as_dict(self) -> dict:
        payload = dict(self.meta)
        payload["directory"] = str(self.directory)
        payload["cameras"] = list(self.cameras)
        return payload


def _require_rgb_uint8(frames: list[np.ndarray], camera: str) -> tuple[int, int]:
    if not frames:
        raise ValueError(f"camera {camera!r} recorded zero frames")
    height = width = None
    for i, frame in enumerate(frames):
        arr = np.asarray(frame)
        if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(
                f"{camera} frame {i} has dtype={arr.dtype} shape={arr.shape}; "
                "expected (H, W, 3) uint8 RGB"
            )
        if height is None:
            height, width = int(arr.shape[0]), int(arr.shape[1])
        elif arr.shape[0] != height or arr.shape[1] != width:
            raise ValueError(
                f"{camera} frame {i} is {arr.shape[:2]}, earlier frames are "
                f"{height}x{width}"
            )
    return height, width


def write_jpeg_sequence(
    frames: list[np.ndarray],
    directory: Path,
    quality: int,
) -> list[Path]:
    """Write ``00000.jpg`` … under ``directory``. Returns the written paths."""
    if not 1 <= quality <= 100:
        raise ValueError(f"jpeg_quality must be in [1, 100], got {quality}")
    _require_rgb_uint8(frames, directory.name)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, frame in enumerate(frames):
        path = directory / f"{i:05d}.jpg"
        _write_jpeg(path, np.asarray(frame), quality)
        paths.append(path)
    return paths


def _write_jpeg(path: Path, frame: np.ndarray, quality: int) -> None:
    try:
        import cv2
    except ImportError:
        from PIL import Image

        Image.fromarray(frame).save(path, format="JPEG", quality=quality)
        return
    ok = cv2.imwrite(
        str(path),
        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed for {path}")


def write_mp4(
    frames: list[np.ndarray],
    path: Path,
    fps: float,
    codec: str,
) -> Path:
    """Write an RGB frame list to ``path``. Requires OpenCV."""
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if len(codec) != 4:
        raise ValueError(f"mp4_codec must be a 4-char fourcc, got {codec!r}")
    height, width = _require_rgb_uint8(frames, path.name)
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "writing mp4 requires opencv-python (already pulled in by the LIBERO "
            "stack). JPEG frames were still written."
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*codec), float(fps), (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(
            f"OpenCV could not open VideoWriter for {path} with codec {codec!r} "
            f"at {width}x{height} @ {fps} fps"
        )
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    if path.stat().st_size == 0:
        raise RuntimeError(f"wrote an empty mp4 at {path}")
    return path


def _named_columns(names: tuple[str, ...] | list[str], prefix: str) -> list[int]:
    """Indices whose name is ``prefix`` or ``prefix_<i>``."""
    return [
        i
        for i, name in enumerate(names)
        if name == prefix or name.startswith(f"{prefix}_")
    ]


def smoothness_summary(
    trace: RolloutTrace,
    *,
    fps: float,
    clip_abs: float,
    clip_rail_tol: float,
) -> dict:
    """Watcher-only scalars. Not a Gate metric.

    Cartesian numbers use the ``ee_pos`` columns from the spec names, not "the
    first three state dims". Action saturation uses the executed (clipped) command.
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if clip_abs <= 0:
        raise ValueError(f"clip_abs must be positive, got {clip_abs}")
    if clip_rail_tol < 0:
        raise ValueError(f"clip_rail_tol must be non-negative, got {clip_rail_tol}")

    dt = 1.0 / float(fps)
    rail = float(clip_abs) - float(clip_rail_tol)
    if rail < 0:
        raise ValueError(
            f"clip_abs {clip_abs} - clip_rail_tol {clip_rail_tol} is negative"
        )

    actions = np.asarray(trace.actions, dtype=np.float64)
    n_action = int(actions.size)
    n_clipped = int(np.count_nonzero(np.abs(actions) >= rail)) if n_action else 0

    summary: dict = {
        "dt_s": dt,
        "n_steps": int(actions.shape[0]),
        "n_replans": int(np.count_nonzero(trace.replan)),
        "action_clip_fraction": (n_clipped / n_action) if n_action else 0.0,
        "action_clip_count": n_clipped,
        "action_clip_n": n_action,
        "clip_abs": float(clip_abs),
        "clip_rail_tol": float(clip_rail_tol),
        "ee_pos_columns": [],
        "max_ee_step": None,
        "rms_ee_jerk": None,
    }

    ee_idx = _named_columns(trace.state_names, "ee_pos")
    summary["ee_pos_columns"] = [trace.state_names[i] for i in ee_idx]
    if not ee_idx:
        return summary

    ee = np.asarray(trace.states, dtype=np.float64)[:, ee_idx]
    if ee.shape[0] >= 2:
        steps = np.linalg.norm(np.diff(ee, axis=0), axis=1)
        summary["max_ee_step"] = float(steps.max()) if steps.size else 0.0
    if ee.shape[0] >= 4:
        vel = np.diff(ee, axis=0) / dt
        acc = np.diff(vel, axis=0) / dt
        jerk = np.diff(acc, axis=0) / dt
        summary["rms_ee_jerk"] = float(np.sqrt(np.mean(jerk * jerk)))
    return summary


def persist_trace(
    trace: RolloutTrace,
    directory: Path,
    *,
    fps: float,
    clip_abs: float,
    clip_rail_tol: float,
) -> dict:
    """Write ``trace.npz`` + ``trace.json``. Returns the JSON payload."""
    if (trace.states.shape[0] - 1) != trace.actions.shape[0]:
        raise ValueError(
            f"trace alignment broken: {trace.states.shape[0]} states vs "
            f"{trace.actions.shape[0]} actions"
        )
    summary = smoothness_summary(
        trace, fps=fps, clip_abs=clip_abs, clip_rail_tol=clip_rail_tol
    )
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(
        directory / "trace.npz",
        states=trace.states,
        actions=trace.actions,
        replan=trace.replan.astype(np.uint8),
        chunks=trace.chunks,
        state_names=np.asarray(trace.state_names),
        action_names=np.asarray(trace.action_names),
    )
    payload = {
        "align": (
            "states[i] is simultaneous with video frame i; "
            "actions[t] produced states[t+1]"
        ),
        "state_names": list(trace.state_names),
        "action_names": list(trace.action_names),
        "states": np.asarray(trace.states, dtype=np.float64).tolist(),
        "actions": np.asarray(trace.actions, dtype=np.float64).tolist(),
        "replan": [bool(x) for x in trace.replan],
        "smoothness": summary,
        "n_states": int(trace.states.shape[0]),
        "n_steps": int(trace.actions.shape[0]),
        "n_chunks": int(trace.chunks.shape[0]) if trace.chunks.size else 0,
    }
    (directory / "trace.json").write_text(json.dumps(payload) + "\n")
    return payload


def persist_rollout(
    result: RolloutResult,
    directory: Path,
    *,
    fps: float,
    jpeg_quality: int,
    mp4_codec: str,
    checkpoint: Path,
    extra: dict | None = None,
    clip_abs: float | None = None,
    clip_rail_tol: float | None = None,
    video_cfg: dict | None = None,
) -> VideoArtifact:
    """Write JPEG sequences, optional mp4s, traces, and ``meta.json`` for one rollout."""
    cameras = tuple(result.camera_frames) or (
        ("agentview",) if result.frames else ()
    )
    if not cameras:
        raise ValueError(
            "rollout has no frames; record_video must be on when calling "
            "persist_rollout"
        )
    sources = result.camera_frames or {"agentview": result.frames}
    n_frames = None
    for camera in cameras:
        frames = sources[camera]
        n = len(frames)
        if n_frames is None:
            n_frames = n
        elif n != n_frames:
            raise ValueError(
                f"camera {camera!r} has {n} frames, expected {n_frames} to match "
                "the other cameras"
            )
        write_jpeg_sequence(frames, directory / camera, jpeg_quality)
        try:
            write_mp4(frames, directory / f"{camera}.mp4", fps=fps, codec=mp4_codec)
        except RuntimeError as exc:
            # JPEG sequence is the watcher source of truth. An mp4 codec the
            # local OpenCV build cannot encode is not a reason to drop the artifact.
            (directory / f"{camera}.mp4.error").write_text(str(exc) + "\n")

    trace = result.trace
    has_trace = trace is not None
    if trace is not None:
        if trace.states.shape[0] != n_frames:
            raise ValueError(
                f"trace has {trace.states.shape[0]} states but the video has "
                f"{n_frames} frames; they must be simultaneous"
            )
        cfg = video_cfg if video_cfg is not None else load_video_config()
        persist_trace(
            trace,
            directory,
            fps=fps,
            clip_abs=float(clip_abs if clip_abs is not None else cfg["clip_abs"]),
            clip_rail_tol=float(
                clip_rail_tol if clip_rail_tol is not None else cfg["clip_rail_tol"]
            ),
        )

    meta = {
        "run_id": extra.get("run_id") if extra else None,
        "task_key": result.task_key,
        "episode_idx": result.episode_idx,
        "success": bool(result.success),
        "n_steps": int(result.n_steps),
        "n_frames": int(n_frames),
        "seed": int(result.seed),
        "n_replans": int(result.n_replans),
        "fps": float(fps),
        "cameras": list(cameras),
        "checkpoint": str(checkpoint),
        "recorded": datetime.now(timezone.utc).isoformat(),
        "has_trace": has_trace,
    }
    if extra:
        for key, value in extra.items():
            if key not in meta or meta[key] is None:
                meta[key] = value
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return VideoArtifact(directory=directory, meta=meta, cameras=cameras)


def load_artifact(directory: Path) -> VideoArtifact:
    """Read a previously written episode directory."""
    meta_path = directory / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"no meta.json in {directory}")
    meta = json.loads(meta_path.read_text())
    cameras = tuple(meta.get("cameras") or [])
    if not cameras:
        raise ValueError(f"{meta_path} has no cameras")
    return VideoArtifact(directory=directory, meta=meta, cameras=cameras)


def discover_artifacts(out_dir: Path) -> list[VideoArtifact]:
    """Every episode directory under ``out_dir`` that has ``meta.json``."""
    if not out_dir.is_dir():
        return []
    found = []
    for meta in sorted(out_dir.rglob("meta.json")):
        found.append(load_artifact(meta.parent))
    return found


@dataclass
class RunListing:
    """A train/eval run the watcher can offer as a replay source."""

    run_id: str
    checkpoint: Path
    task_keys: list[str]
    eval_successes: dict[str, list[bool]] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "checkpoint": str(self.checkpoint),
            "task_keys": list(self.task_keys),
            "eval_successes": self.eval_successes,
        }


def discover_runs(results_root: Path) -> list[RunListing]:
    """Find ``checkpoints/final.pt`` plus sibling ``config.yaml`` / ``eval.json``."""
    if not results_root.is_dir():
        return []
    listings = []
    for ckpt in sorted(results_root.glob("*/checkpoints/final.pt")):
        run_dir = ckpt.parent.parent
        config_path = run_dir / "config.yaml"
        if not config_path.is_file():
            continue
        cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
        if not isinstance(cfg, dict):
            raise TypeError(f"{config_path} must resolve to a mapping")
        run_id = cfg.get("run_id") or run_dir.name
        tasks = cfg.get("tasks") or []
        if isinstance(tasks, str):
            tasks = [tasks]
        task_keys = [str(t) for t in tasks]
        eval_successes: dict[str, list[bool]] = {}
        eval_path = run_dir / "eval.json"
        if eval_path.is_file():
            payload = json.loads(eval_path.read_text())
            for entry in payload.get("tasks", []):
                eval_successes[entry["task_key"]] = [bool(s) for s in entry["successes"]]
        listings.append(
            RunListing(
                run_id=str(run_id),
                checkpoint=ckpt.resolve(),
                task_keys=task_keys,
                eval_successes=eval_successes,
            )
        )
    return listings


def record_episode(
    checkpoint: Path,
    episode_idx: int,
    *,
    task_key: str | None = None,
    run_id: str | None = None,
    device: str = "cuda",
    out_dir: Path | None = None,
    eval_cfg: EvalConfig | None = None,
    video_cfg: dict | None = None,
) -> VideoArtifact:
    """Roll out one episode with video on and persist the artifact.

    ``run_id`` defaults to the checkpoint's recorded id so the initial state matches
    the Gate / eval table for that ``episode_idx`` (§8.3). Changing it watches a
    different evaluation.
    """
    video_cfg = video_cfg if video_cfg is not None else load_video_config()
    dest_root = resolve_out_dir(out_dir, video_cfg)
    loaded = load_checkpoint(checkpoint, device=device)
    resolved_run_id = run_id or loaded.run_id
    if not resolved_run_id:
        raise ValueError(
            f"{checkpoint} records no run_id and none was given; initial-state "
            "seeds derive from it (§8.3)"
        )
    resolved_task = task_key or loaded.task_key
    if not resolved_task:
        raise ValueError(
            f"{checkpoint} records no task_key; pass task_key explicitly"
        )
    ref = TaskRef.from_key(resolved_task)

    if eval_cfg is None:
        eval_payload = OmegaConf.to_container(
            OmegaConf.load(repo_root() / "configs" / "eval" / "libero_eval.yaml"),
            resolve=True,
        )
        eval_payload["record_video"] = True
        eval_cfg = eval_config_from_dict(eval_payload)
    else:
        eval_cfg = EvalConfig(
            n_episodes=eval_cfg.n_episodes,
            max_steps=eval_cfg.max_steps,
            execute_k=eval_cfg.execute_k,
            euler_steps=eval_cfg.euler_steps,
            temporal_ensembling=eval_cfg.temporal_ensembling,
            temporal_ensemble_coef=eval_cfg.temporal_ensemble_coef,
            image_size=eval_cfg.image_size,
            record_video=True,
        )

    with LiberoTaskEnv(
        suite=ref.suite,
        task_idx=ref.task_idx,
        spec=loaded.spec,
        image_size=eval_cfg.image_size,
    ) as env:
        result = env.rollout(
            loaded.policy, episode_idx, loaded.stats, resolved_run_id, eval_cfg
        )

    directory = episode_dir(dest_root, resolved_run_id, result.task_key, episode_idx)
    return persist_rollout(
        result,
        directory,
        fps=float(loaded.spec.action.control_rate_hz),
        jpeg_quality=int(video_cfg["jpeg_quality"]),
        mp4_codec=str(video_cfg["mp4_codec"]),
        checkpoint=Path(checkpoint).resolve(),
        extra={"run_id": resolved_run_id},
        video_cfg=video_cfg,
    )


def record_episodes(
    checkpoint: Path,
    episode_idxs: list[int],
    **kwargs,
) -> list[VideoArtifact]:
    """Record several episodes of one checkpoint. Same kwargs as :func:`record_episode`."""
    if not episode_idxs:
        raise ValueError("record_episodes received no episode indices")
    return [record_episode(checkpoint, idx, **kwargs) for idx in episode_idxs]
