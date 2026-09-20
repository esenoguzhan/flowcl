"""LIBERO rollout wrapper.

Spec §4.4 (execution):

    Predict a chunk of ``H = 16``, execute the first ``k = 8`` open-loop, then replan.
    Optionally implement temporal ensembling but keep it **off by default** and report
    it as an ablation, not as part of the main method.

Spec §8.1 (evaluation protocol):

    Fixed initial-state set: for each task, sample 50 initial states once using
    LIBERO's ``init_files`` and reuse the identical set for every method, seed and
    stage. Seeding derives from ``(run_id, task, episode_idx)`` and never from the
    method name (§8.3).

The single most dangerous detail in this file is the proprioception layout. LIBERO's
``scripts/create_dataset.py`` built the recorded ``ee_states`` as::

    np.hstack((obs["robot0_eef_pos"], T.quat2axisangle(obs["robot0_eef_quat"])))

with ``gripper_states = obs["robot0_gripper_qpos"]``. If a rollout assembled the state
vector any other way — Euler angles, raw quaternion, a different component order — the
policy would receive out-of-distribution proprioception at evaluation while the
training loss stayed perfectly healthy. :func:`observation_to_state` is the one place
that mapping lives, and ``tests/test_libero_env.py`` checks it against the recorded
HDF5 values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from flowcl.data.spec import EmbodimentSpec
from flowcl.data.stats import NormalizationStats
from flowcl.utils.libero_paths import ensure_libero_config
from flowcl.utils.seeding import derive_seed

# Canonical camera name -> the env's observation key.
CAMERA_TO_OBS_KEY = {
    "agentview": "agentview_image",
    "robot0_eye_in_hand": "robot0_eye_in_hand_image",
}

# §8.1: the same 50 initial states for every method, seed and stage.
N_EVAL_EPISODES = 50

# LIBERO's own cap; OSC at 20 Hz, so 600 steps is 30 s of simulated time.
DEFAULT_MAX_STEPS = 600


def observation_to_state(obs: dict) -> np.ndarray:
    """Assemble the 8-dim proprioception vector exactly as the demos recorded it.

    Layout, matching ``flowcl.data.libero_adapter.STATE_COMPONENTS``:
    ``ee_pos`` (3) + ``ee_ori`` (3, axis-angle) + ``gripper_states`` (2).

    Raises:
        KeyError: If the env did not provide a required observation key. Silently
            substituting zeros here would produce a policy that fails for reasons no
            metric could explain.
    """
    from robosuite.utils import transform_utils

    required = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
    missing = [key for key in required if key not in obs]
    if missing:
        raise KeyError(
            f"env observation is missing {missing}; got {sorted(obs)}. The state "
            "vector must match how LIBERO recorded the demos "
            "(scripts/create_dataset.py), or evaluation runs out of distribution."
        )

    state = np.concatenate(
        [
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            transform_utils.quat2axisangle(obs["robot0_eef_quat"]).astype(np.float32),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
        ]
    )
    if state.shape != (8,):
        raise ValueError(
            f"assembled state has shape {state.shape}, expected (8,); the env's "
            "observation widths do not match the recorded demos"
        )
    return state


@dataclass
class RolloutResult:
    """Outcome of one episode."""

    success: bool
    n_steps: int
    task_key: str
    episode_idx: int
    seed: int
    n_replans: int
    frames: list[np.ndarray] = field(default_factory=list)


@dataclass
class EvalConfig:
    """Rollout settings (§4.4, §8.1)."""

    n_episodes: int = N_EVAL_EPISODES
    max_steps: int = DEFAULT_MAX_STEPS
    # k in §4.4: how many of the H predicted actions to execute before replanning.
    execute_k: int | None = None
    euler_steps: int = 10
    # §4.4: implemented but OFF by default, reported as an ablation only.
    temporal_ensembling: bool = False
    temporal_ensemble_coef: float = 0.01
    image_size: int = 128
    record_video: bool = False


class TemporalEnsembler:
    """Exponentially-weighted averaging of overlapping action-chunk predictions.

    Off by default (§4.4). Follows ACT's scheme: for a given absolute timestep,
    predictions made at earlier planning steps get weight ``exp(-coef * age)``, so the
    most recent prediction dominates while older ones smooth it.

    Kept in its own class so the main execution path is readable and so enabling it is
    a config change with an obvious blast radius.
    """

    def __init__(self, horizon: int, d_action: int, coef: float = 0.01) -> None:
        self.horizon = horizon
        self.d_action = d_action
        self.coef = coef
        # absolute timestep -> list of (age_rank, action)
        self._pending: dict[int, list[np.ndarray]] = {}

    def add(self, start_t: int, chunk: np.ndarray) -> None:
        if chunk.shape != (self.horizon, self.d_action):
            raise ValueError(
                f"chunk shape {chunk.shape} != ({self.horizon}, {self.d_action})"
            )
        for offset in range(self.horizon):
            self._pending.setdefault(start_t + offset, []).append(chunk[offset])

    def pop(self, t: int) -> np.ndarray:
        """Weighted average of every prediction made for timestep ``t``."""
        predictions = self._pending.pop(t, None)
        if not predictions:
            raise KeyError(f"no prediction available for timestep {t}")
        # predictions[0] is the oldest, so ages run high -> low.
        n = len(predictions)
        ages = np.arange(n - 1, -1, -1, dtype=np.float64)
        weights = np.exp(-self.coef * ages)
        weights /= weights.sum()
        stacked = np.stack(predictions, axis=0)
        return (stacked * weights[:, None]).sum(axis=0).astype(np.float32)


class LiberoTaskEnv:
    """One LIBERO task, with the fixed initial-state set from ``init_files``.

    Args:
        suite: e.g. ``"libero_object"``.
        task_idx: Index within the suite.
        spec: Embodiment spec, supplying cameras, ``H`` and ``k``.
        image_size: Render size; must match what the policy was trained on.
    """

    def __init__(
        self,
        suite: str,
        task_idx: int,
        spec: EmbodimentSpec,
        image_size: int = 128,
    ) -> None:
        ensure_libero_config()
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        suites = benchmark.get_benchmark_dict()
        if suite not in suites:
            raise ValueError(
                f"Unknown suite {suite!r}; available: {sorted(suites)}"
            )
        self.suite_name = suite
        self.benchmark = suites[suite]()
        n_tasks = self.benchmark.n_tasks
        if not 0 <= task_idx < n_tasks:
            raise IndexError(
                f"task_idx {task_idx} out of range for {suite} ({n_tasks} tasks)"
            )

        self.task_idx = task_idx
        self.task = self.benchmark.get_task(task_idx)
        self.spec = spec
        self.image_size = image_size
        self.language = self.benchmark.get_task_names()[task_idx].replace("_", " ")
        # Prefer the instruction LIBERO ships with the task definition.
        if getattr(self.task, "language", None):
            self.language = self.task.language

        bddl = Path(get_libero_path("bddl_files")) / self.task.problem_folder / self.task.bddl_file
        if not bddl.is_file():
            raise FileNotFoundError(f"BDDL file not found: {bddl}")
        self.bddl_file = bddl

        # §8.1: the fixed initial-state set, from the .pruned_init files.
        self.init_states = self.benchmark.get_task_init_states(task_idx)
        if len(self.init_states) == 0:
            raise RuntimeError(
                f"{suite}/{self.task.name} has no initial states; the LIBERO "
                "submodule's init_files may be incomplete"
            )

        self.env = OffScreenRenderEnv(
            bddl_file_name=str(bddl),
            camera_heights=image_size,
            camera_widths=image_size,
            camera_names=list(spec.cameras),
        )

    @property
    def task_key(self) -> str:
        """``"<suite>/<task name>"`` — matches the adapter's ``task_id`` convention."""
        return f"{self.suite_name}/{self.task.name}"

    @property
    def n_init_states(self) -> int:
        return len(self.init_states)

    def close(self) -> None:
        self.env.close()

    def __enter__(self) -> "LiberoTaskEnv":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- observation plumbing -------------------------------------------------

    def _build_policy_batch(
        self,
        obs: dict,
        stats: NormalizationStats,
        device: torch.device,
        language: str | None = None,
    ) -> dict:
        images = {}
        for camera in self.spec.cameras:
            key = CAMERA_TO_OBS_KEY.get(camera)
            if key is None:
                raise KeyError(
                    f"no env observation key known for camera {camera!r}; known "
                    f"mapping is {CAMERA_TO_OBS_KEY}"
                )
            if key not in obs:
                raise KeyError(
                    f"env observation is missing {key!r}; got {sorted(obs)}"
                )
            frame = np.asarray(obs[key], dtype=np.uint8)
            images[camera] = torch.from_numpy(frame[None]).to(device)

        state = stats.normalize_state(observation_to_state(obs)[None])
        return {
            "images": images,
            "state": torch.from_numpy(state).to(device),
            "language": [language if language is not None else self.language],
        }

    # ---- rollout --------------------------------------------------------------

    def rollout(
        self,
        policy,
        episode_idx: int,
        stats: NormalizationStats,
        run_id: str,
        cfg: EvalConfig | None = None,
        language: str | None = None,
    ) -> RolloutResult:
        """Run one episode from fixed initial state ``episode_idx`` (§4.4, §8.1).

        The initial state is taken from the fixed ``init_files`` set, so it depends on
        ``episode_idx`` alone. ``derive_seed(run_id, task_key, episode_idx)`` seeds the
        env and the policy's sampler, and takes no method argument (§8.3).

        Args:
            language: Override the task's instruction while keeping its scene and its
                success predicate. Used only by the §4.1 language-discriminability
                check, which needs to run this task's env under another task's
                instruction. Leave ``None`` for every real evaluation.
        """
        cfg = cfg or EvalConfig()
        execute_k = cfg.execute_k if cfg.execute_k is not None else self.spec.action.execute_k
        horizon = self.spec.action.chunk_horizon
        if not 1 <= execute_k <= horizon:
            raise ValueError(
                f"execute_k {execute_k} must be in [1, {horizon}]"
            )
        if episode_idx >= self.n_init_states:
            raise IndexError(
                f"episode_idx {episode_idx} >= {self.n_init_states} available initial "
                f"states for {self.task_key}"
            )

        seed = derive_seed(run_id, self.task_key, episode_idx)
        device = next(policy.parameters()).device
        generator = torch.Generator(device="cpu").manual_seed(seed)

        self.env.seed(seed)
        self.env.reset()
        obs = self.env.set_init_state(self.init_states[episode_idx])

        ensembler = (
            TemporalEnsembler(horizon, self.spec.d_action, cfg.temporal_ensemble_coef)
            if cfg.temporal_ensembling
            else None
        )

        policy.eval()
        frames: list[np.ndarray] = []
        success = False
        n_replans = 0
        step = 0

        while step < cfg.max_steps:
            batch = self._build_policy_batch(obs, stats, device, language=language)
            with torch.no_grad():
                chunk = policy.sample(
                    batch, n_steps=cfg.euler_steps, generator=generator
                )
            chunk_np = chunk[0].detach().cpu().numpy().astype(np.float32)
            chunk_np = stats.denormalize_action(chunk_np)
            n_replans += 1

            if ensembler is not None:
                ensembler.add(step, chunk_np)

            for offset in range(execute_k):
                if step >= cfg.max_steps:
                    break
                action = (
                    ensembler.pop(step) if ensembler is not None else chunk_np[offset]
                )
                # LIBERO actions are already in [-1, 1]; clip only to satisfy the
                # controller's bounds, never to rescale (§3.2).
                obs, _reward, _done, _info = self.env.step(
                    np.clip(action, -1.0, 1.0).astype(np.float64)
                )
                step += 1
                if cfg.record_video:
                    frames.append(np.asarray(obs["agentview_image"], dtype=np.uint8))
                if self.env.check_success():
                    success = True
                    break
            if success:
                break

        return RolloutResult(
            success=success,
            n_steps=step,
            task_key=self.task_key,
            episode_idx=episode_idx,
            seed=seed,
            n_replans=n_replans,
            frames=frames,
        )

    def evaluate(
        self,
        policy,
        stats: NormalizationStats,
        run_id: str,
        cfg: EvalConfig | None = None,
    ) -> list[RolloutResult]:
        """Run ``cfg.n_episodes`` rollouts on the shared fixed init-state set."""
        cfg = cfg or EvalConfig()
        if cfg.n_episodes > self.n_init_states:
            raise ValueError(
                f"{self.task_key} has {self.n_init_states} fixed initial states but "
                f"{cfg.n_episodes} episodes were requested; §8.1 requires the shared "
                "fixed set, so do not sample extra states"
            )
        return [
            self.rollout(policy, episode_idx, stats, run_id, cfg)
            for episode_idx in range(cfg.n_episodes)
        ]


def success_rate(results: list[RolloutResult]) -> float:
    """Fraction of successful rollouts. Raises on an empty list."""
    if not results:
        raise ValueError("success_rate received no rollouts")
    return sum(r.success for r in results) / len(results)
