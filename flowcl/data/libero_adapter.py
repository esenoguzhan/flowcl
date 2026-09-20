"""LIBERO HDF5 -> canonical :class:`~flowcl.data.episode.Episode`.

Spec §3.2:

* Suites ``libero_spatial``, ``libero_object``, ``libero_goal``, ``libero_10``;
  10 tasks each, 50 demos each.
* Use LIBERO's own HDF5 demos; do not regenerate. **Verify demo count and action
  stats on load and fail loudly on mismatch.**
* Cameras ``agentview`` + ``robot0_eye_in_hand``, resized to 128x128 (configurable).
* State is end-effector pose + gripper qpos. Actions are 7-dim OSC delta
  (6 pose + 1 gripper), already normalised to ``[-1, 1]`` by LIBERO — do not
  renormalise, just record the stats.

On-disk layout, verified against the real files rather than assumed::

    data/                                   attrs: num_demos, problem_info,
                                                   bddl_file_name, env_name,
                                                   macros_image_convention, total
      demo_<i>/                             attrs: init_state (92,), model_file,
                                                   num_samples
        actions          (T, 7)  float64    already in [-1, 1]; dim 6 is +-1 gripper
        dones            (T,)    uint8
        rewards          (T,)    uint8
        states           (T, 92) float64    flattened MuJoCo states, for replay
        robot_states     (T, 9)  float64
        obs/
          agentview_rgb    (T, 128, 128, 3) uint8
          eye_in_hand_rgb  (T, 128, 128, 3) uint8
          ee_pos           (T, 3)  float64
          ee_ori           (T, 3)  float64
          ee_states        (T, 6)  float64   == concat(ee_pos, ee_ori)
          gripper_states   (T, 2)  float64
          joint_states     (T, 7)  float64

Note the HDF5 camera keys (``agentview_rgb``, ``eye_in_hand_rgb``) differ from the
env's observation keys (``agentview_image``, ``robot0_eye_in_hand_image``). The
canonical episode uses the *env* names so that training data and rollout
observations are keyed identically; :data:`HDF5_TO_CANONICAL_CAMERA` is the single
place that mapping lives.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np

from flowcl.data.episode import Episode
from flowcl.data.spec import EmbodimentSpec

# HDF5 dataset name -> canonical camera name (matching the env's observation keys).
HDF5_TO_CANONICAL_CAMERA = {
    "agentview_rgb": "agentview",
    "eye_in_hand_rgb": "robot0_eye_in_hand",
}

# Proprioception layout: end-effector pose + gripper qpos (§3.2). Order is fixed here
# and mirrored by ObservationSpec.state_keys in configs/embodiment/libero_franka.yaml.
STATE_COMPONENTS: tuple[tuple[str, int], ...] = (
    ("ee_pos", 3),
    ("ee_ori", 3),
    ("gripper_states", 2),
)
D_STATE = sum(width for _, width in STATE_COMPONENTS)  # 8

D_ACTION = 7  # 6 OSC pose deltas + 1 gripper

# §3.2: LIBERO ships 50 demos per task.
EXPECTED_DEMOS_PER_TASK = 50

# LIBERO's OSC deltas are already normalised to [-1, 1]. A tiny tolerance allows for
# float64 round-trips in the recorded data without permitting genuine drift.
ACTION_ABS_MAX = 1.0
ACTION_TOLERANCE = 1e-6

# Images are stored with MuJoCo's bottom-up ("opengl") convention. The live env
# applies whatever robosuite.macros.IMAGE_CONVENTION says, so the two must agree or
# training frames are vertically flipped relative to rollout frames — a mismatch that
# leaves the training loss looking perfectly healthy while evaluation collapses.
EXPECTED_IMAGE_CONVENTION = "opengl"


@dataclass(frozen=True)
class ActionStats:
    """Recorded, *not* applied. §3.2 forbids renormalising LIBERO actions."""

    min: np.ndarray
    max: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    n_steps: int

    def as_dict(self) -> dict:
        return {
            "min": self.min.tolist(),
            "max": self.max.tolist(),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "n_steps": self.n_steps,
        }


@dataclass(frozen=True)
class TaskMetadata:
    """Task-level facts read from the HDF5 ``data`` attributes."""

    path: Path
    language: str
    problem_name: str
    bddl_file_name: str
    n_demos: int
    total_steps: int
    image_convention: str
    demo_lengths: tuple[int, ...]


def _assert_image_convention(convention: str, path: Path) -> None:
    import robosuite.macros as robosuite_macros

    live = robosuite_macros.IMAGE_CONVENTION
    if convention != EXPECTED_IMAGE_CONVENTION or live != EXPECTED_IMAGE_CONVENTION:
        raise ValueError(
            f"Image convention mismatch for {path.name}: dataset was recorded with "
            f"{convention!r}, robosuite is configured for {live!r}, and this code "
            f"assumes {EXPECTED_IMAGE_CONVENTION!r}. If these disagree, training "
            "frames are vertically flipped relative to rollout frames and evaluation "
            "will fail while the training loss looks fine. Fix the convention rather "
            "than flipping images here."
        )


def read_task_metadata(path: str | Path) -> TaskMetadata:
    """Read task-level metadata and per-demo lengths without loading pixels."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"LIBERO demo file not found: {path}")

    with h5py.File(path, "r") as f:
        if "data" not in f:
            raise KeyError(f"{path} has no 'data' group; keys are {list(f.keys())}")
        data = f["data"]
        attrs = data.attrs

        for required in ("num_demos", "problem_info", "macros_image_convention"):
            if required not in attrs:
                raise KeyError(
                    f"{path} is missing the {required!r} attribute; "
                    f"present attributes are {list(attrs.keys())}"
                )

        problem_info = json.loads(attrs["problem_info"])
        language = problem_info["language_instruction"]
        if not language:
            raise ValueError(f"{path} has an empty language_instruction")

        demo_names = _sorted_demo_names(data, path)
        lengths = tuple(int(data[name]["actions"].shape[0]) for name in demo_names)

        return TaskMetadata(
            path=path,
            language=language,
            problem_name=problem_info["problem_name"],
            bddl_file_name=str(attrs.get("bddl_file_name", "")),
            n_demos=int(attrs["num_demos"]),
            total_steps=int(attrs.get("total", sum(lengths))),
            image_convention=str(attrs["macros_image_convention"]),
            demo_lengths=lengths,
        )


def _sorted_demo_names(data: h5py.Group, path: Path) -> list[str]:
    """Demo group names in numeric order.

    HDF5 iterates keys alphabetically, so ``demo_10`` precedes ``demo_2``. Sorting
    numerically makes ``demo_idx`` mean the same thing here as in LIBERO's own
    tooling, which matters because demo indices end up in replay buffers and in the
    normalization-stats provenance.
    """
    names = [k for k in data.keys() if k.startswith("demo_")]
    if not names:
        raise KeyError(f"{path} contains no demo_* groups; keys are {list(data.keys())}")
    try:
        return sorted(names, key=lambda s: int(s.split("_", 1)[1]))
    except ValueError as exc:
        raise ValueError(f"{path} has a malformed demo group name in {names}") from exc


def verify_task_file(
    path: str | Path, expected_demos: int = EXPECTED_DEMOS_PER_TASK
) -> TaskMetadata:
    """Check demo count and action statistics, failing loudly on mismatch (§3.2)."""
    meta = read_task_metadata(path)
    path = Path(path)

    if meta.n_demos != expected_demos:
        raise ValueError(
            f"{path.name}: expected {expected_demos} demos per §3.2, found "
            f"{meta.n_demos}"
        )
    if len(meta.demo_lengths) != expected_demos:
        raise ValueError(
            f"{path.name}: num_demos attribute says {meta.n_demos} but there are "
            f"{len(meta.demo_lengths)} demo_* groups"
        )
    _assert_image_convention(meta.image_convention, path)

    with h5py.File(path, "r") as f:
        data = f["data"]
        for name in _sorted_demo_names(data, path):
            demo = data[name]
            actions = np.asarray(demo["actions"])
            if actions.ndim != 2 or actions.shape[1] != D_ACTION:
                raise ValueError(
                    f"{path.name}/{name}: actions shape {actions.shape}, expected "
                    f"(T, {D_ACTION})"
                )
            peak = float(np.abs(actions).max())
            if peak > ACTION_ABS_MAX + ACTION_TOLERANCE:
                raise ValueError(
                    f"{path.name}/{name}: |action| max is {peak}, exceeding the "
                    f"{ACTION_ABS_MAX} that §3.2 says LIBERO guarantees. Do not "
                    "renormalise to hide this; investigate the source data."
                )
            gripper = np.unique(actions[:, 6])
            if not np.all(np.isin(gripper, (-1.0, 1.0))):
                raise ValueError(
                    f"{path.name}/{name}: gripper dim has non-binary values "
                    f"{gripper[:8]}"
                )
    return meta


def compute_action_stats(path: str | Path) -> ActionStats:
    """Aggregate action statistics over every demo in a task file.

    Recorded for provenance only (§3.2): LIBERO actions are already normalised and
    must not be transformed.
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        data = f["data"]
        chunks = [
            np.asarray(data[name]["actions"], dtype=np.float64)
            for name in _sorted_demo_names(data, path)
        ]
    stacked = np.concatenate(chunks, axis=0)
    return ActionStats(
        min=stacked.min(axis=0),
        max=stacked.max(axis=0),
        mean=stacked.mean(axis=0),
        std=stacked.std(axis=0),
        n_steps=int(stacked.shape[0]),
    )


def _resize_frames(frames: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize ``(T, H, W, 3) uint8`` frames to ``size = (H_out, W_out)``.

    Returns the input untouched when it already matches, so the common LIBERO case
    (native 128x128) costs nothing and introduces no resampling artefacts.
    """
    if frames.shape[1:3] == size:
        return frames

    from PIL import Image

    out_h, out_w = size
    out = np.empty((frames.shape[0], out_h, out_w, 3), dtype=np.uint8)
    for i, frame in enumerate(frames):
        out[i] = np.asarray(
            Image.fromarray(frame).resize((out_w, out_h), Image.BILINEAR)
        )
    return out


def load_episode(
    path: str | Path,
    demo_idx: int,
    spec: EmbodimentSpec,
    task_id: str | None = None,
    with_sim_states: bool = False,
) -> Episode:
    """Convert one LIBERO demo into a canonical :class:`Episode`.

    Args:
        path: Task HDF5 file.
        demo_idx: Index into the numerically sorted demo list.
        spec: Target embodiment spec; shapes are asserted against it.
        task_id: Bookkeeping id. Defaults to ``"<suite>/<task stem>"``.
        with_sim_states: Also stash the flattened MuJoCo ``states`` and the demo's
            ``init_state`` in ``meta``. Needed by the §10.1 replay test; off by
            default because the arrays are large and are never a policy input.

    Returns:
        The episode, already validated.
    """
    path = Path(path)
    meta = read_task_metadata(path)
    _assert_image_convention(meta.image_convention, path)

    if not 0 <= demo_idx < len(meta.demo_lengths):
        raise IndexError(
            f"demo_idx {demo_idx} out of range for {path.name}, which has "
            f"{len(meta.demo_lengths)} demos"
        )

    with h5py.File(path, "r") as f:
        data = f["data"]
        name = _sorted_demo_names(data, path)[demo_idx]
        demo = data[name]
        obs = demo["obs"]

        actions = np.asarray(demo["actions"], dtype=np.float32)

        missing = [k for k, _ in STATE_COMPONENTS if k not in obs]
        if missing:
            raise KeyError(
                f"{path.name}/{name}: obs is missing {missing}; present keys are "
                f"{list(obs.keys())}"
            )
        parts = []
        for key, width in STATE_COMPONENTS:
            component = np.asarray(obs[key], dtype=np.float32)
            if component.ndim != 2 or component.shape[1] != width:
                raise ValueError(
                    f"{path.name}/{name}: obs/{key} has shape {component.shape}, "
                    f"expected (T, {width})"
                )
            parts.append(component)
        state = np.concatenate(parts, axis=1)

        images = {}
        for hdf5_key, canonical in HDF5_TO_CANONICAL_CAMERA.items():
            if canonical not in spec.cameras:
                continue
            if hdf5_key not in obs:
                raise KeyError(
                    f"{path.name}/{name}: obs is missing camera {hdf5_key!r}; "
                    f"present keys are {list(obs.keys())}"
                )
            images[canonical] = _resize_frames(
                np.asarray(obs[hdf5_key]), spec.observation.image_size
            )

        unknown = [c for c in spec.cameras if c not in images]
        if unknown:
            raise KeyError(
                f"Embodiment {spec.name!r} requests cameras {unknown} which LIBERO "
                f"does not provide; known mapping is {HDF5_TO_CANONICAL_CAMERA}"
            )

        dones = np.asarray(demo["dones"])
        episode_meta = {
            "source_file": str(path),
            "demo_name": name,
            "demo_idx": int(demo_idx),
            "success": bool(dones[-1] == 1),
            "image_convention": meta.image_convention,
            "bddl_file_name": meta.bddl_file_name,
            "problem_name": meta.problem_name,
        }
        if with_sim_states:
            episode_meta["sim_states"] = np.asarray(demo["states"])
            episode_meta["init_state"] = np.asarray(demo.attrs["init_state"])
            episode_meta["model_file"] = str(demo.attrs["model_file"])

    spec.assert_compatible_episode(
        d_state=state.shape[1], d_action=actions.shape[1], cameras=tuple(images)
    )

    return Episode(
        images=images,
        state=state,
        action=actions,
        language=meta.language,
        task_id=task_id or default_task_id(path),
        embodiment=spec.name,
        meta=episode_meta,
    )


def default_task_id(path: str | Path) -> str:
    """``"<suite>/<task stem without _demo>"`` — stable and human-readable."""
    path = Path(path)
    stem = path.stem
    if stem.endswith("_demo"):
        stem = stem[: -len("_demo")]
    return f"{path.parent.name}/{stem}"


def iter_episodes(
    path: str | Path,
    spec: EmbodimentSpec,
    n_demos: int | None = None,
    task_id: str | None = None,
    with_sim_states: bool = False,
) -> Iterator[Episode]:
    """Yield episodes from a task file, in numeric demo order.

    Args:
        n_demos: Take only the first ``n_demos`` demos (curricula declare this).
            ``None`` means all of them.
    """
    meta = read_task_metadata(path)
    total = len(meta.demo_lengths)
    take = total if n_demos is None else n_demos
    if take > total:
        raise ValueError(
            f"{Path(path).name} has {total} demos but {take} were requested"
        )
    for demo_idx in range(take):
        yield load_episode(
            path,
            demo_idx,
            spec,
            task_id=task_id,
            with_sim_states=with_sim_states,
        )
