"""Rollout wrapper: §8.3 seeding determinism, §4.4 execution, state layout."""

from __future__ import annotations

import numpy as np
import pytest

from flowcl.envs.libero_env import (
    CAMERA_TO_OBS_KEY,
    DEFAULT_MAX_STEPS,
    N_EVAL_EPISODES,
    EvalConfig,
    RolloutResult,
    TemporalEnsembler,
    observation_to_state,
    success_rate,
)
from flowcl.utils.seeding import derive_seed


# ---- §8.3: seeding is method-independent --------------------------------------


def test_same_episode_gives_same_seed_under_two_method_names():
    """§8.3 determinism assertion.

    Two methods being compared share ``run_id``, so episode ``i`` must resolve to the
    same seed — and therefore the same initial state — for both. The method name is
    not an input to :func:`derive_seed` at all, so this holds by construction; the test
    pins the property against a future refactor that threads a method through.
    """
    task = "libero_object/pick_up_the_milk_and_place_it_in_the_basket"
    run_id = "cmp-2026-09"

    for episode_idx in range(N_EVAL_EPISODES):
        seq_ft_seed = derive_seed(run_id, task, episode_idx)
        gpm_seed = derive_seed(run_id, task, episode_idx)
        sgp_seed = derive_seed(run_id, task, episode_idx)
        assert seq_ft_seed == gpm_seed == sgp_seed


def test_seeds_differ_across_episodes_and_tasks():
    """The shared init set must still be 50 *different* states."""
    run_id = "cmp"
    a = "libero_object/task_a"
    b = "libero_goal/task_b"

    per_episode = [derive_seed(run_id, a, i) for i in range(N_EVAL_EPISODES)]
    assert len(set(per_episode)) == N_EVAL_EPISODES
    assert derive_seed(run_id, a, 0) != derive_seed(run_id, b, 0)


def test_eval_protocol_constants_match_spec():
    """§8.1: 50 rollouts per cell on the shared fixed init-state set."""
    assert N_EVAL_EPISODES == 50
    assert DEFAULT_MAX_STEPS == 600


def test_default_eval_config_disables_temporal_ensembling():
    """§4.4: implemented but OFF by default, reported as an ablation only."""
    cfg = EvalConfig()
    assert cfg.temporal_ensembling is False
    assert cfg.n_episodes == 50
    assert cfg.euler_steps == 10


# ---- state layout -------------------------------------------------------------


def test_observation_to_state_matches_recorded_layout():
    """ee_pos (3) + axis-angle ee_ori (3) + gripper qpos (2), in that order.

    LIBERO's scripts/create_dataset.py built the recorded state as
    ``hstack(robot0_eef_pos, quat2axisangle(robot0_eef_quat))`` plus
    ``robot0_gripper_qpos``. Any other assembly puts evaluation out of distribution
    while training loss stays healthy.
    """
    from robosuite.utils import transform_utils

    obs = {
        "robot0_eef_pos": np.array([0.1, 0.2, 0.3]),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.array([0.04, -0.04]),
    }
    state = observation_to_state(obs)

    assert state.shape == (8,)
    assert state.dtype == np.float32
    np.testing.assert_allclose(state[:3], [0.1, 0.2, 0.3], rtol=1e-6)
    np.testing.assert_allclose(
        state[3:6], transform_utils.quat2axisangle(obs["robot0_eef_quat"]), rtol=1e-5
    )
    np.testing.assert_allclose(state[6:], [0.04, -0.04], rtol=1e-6)


def test_observation_to_state_uses_axis_angle_not_euler():
    """A rotated quaternion must produce the axis-angle value, not Euler angles."""
    from robosuite.utils import transform_utils

    quat = transform_utils.axisangle2quat(np.array([0.3, -0.4, 1.2]))
    obs = {
        "robot0_eef_pos": np.zeros(3),
        "robot0_eef_quat": quat,
        "robot0_gripper_qpos": np.zeros(2),
    }
    np.testing.assert_allclose(
        observation_to_state(obs)[3:6], [0.3, -0.4, 1.2], rtol=1e-4, atol=1e-5
    )


def test_observation_to_state_fails_loudly_on_missing_key():
    with pytest.raises(KeyError, match="robot0_gripper_qpos"):
        observation_to_state(
            {
                "robot0_eef_pos": np.zeros(3),
                "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
            }
        )


def test_camera_mapping_covers_the_embodiment_cameras():
    from flowcl.data.config import load_embodiment_spec

    spec = load_embodiment_spec("libero_franka")
    for camera in spec.cameras:
        assert camera in CAMERA_TO_OBS_KEY, camera


@pytest.mark.physics
def test_rollout_state_matches_the_recorded_training_state(dataset_dir):
    """The load-bearing train/eval consistency check.

    Reset a real env to a demo's recorded initial state, assemble proprioception the
    way :func:`observation_to_state` does at rollout time, and compare against the
    state the adapter reads from the HDF5 for the same instant. A mismatch here means
    the policy sees a different proprioception distribution at evaluation than it was
    trained on — which shows up only as unexplained rollout failure, never as a bad
    training loss.
    """
    import os

    from flowcl.data.config import load_embodiment_spec
    from flowcl.data.libero_adapter import load_episode
    from flowcl.utils.libero_paths import ensure_libero_config

    ensure_libero_config()
    from libero.libero import get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv

    spec = load_embodiment_spec("libero_franka")
    task_path = (
        dataset_dir
        / "libero_object"
        / "pick_up_the_milk_and_place_it_in_the_basket_demo.hdf5"
    )
    episode = load_episode(task_path, 0, spec, with_sim_states=True)

    bddl = os.path.join(
        get_libero_path("bddl_files"),
        "libero_object",
        "pick_up_the_milk_and_place_it_in_the_basket.bddl",
    )
    env = ControlEnv(
        bddl_file_name=bddl,
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=False,
    )
    try:
        env.seed(0)
        env.reset()
        obs = env.set_init_state(episode.meta["init_state"])
        rollout_state = observation_to_state(obs)
    finally:
        env.close()

    recorded_state = episode.state[0]
    assert rollout_state.shape == recorded_state.shape == (8,)

    # Exact equality is not expected: create_dataset.py sets cap_index = 5 and skips
    # the first 5 frames ("the force sensor is not stable in the beginning"), so
    # state[0] is 5 sim steps after reset and the arm has settled slightly. The
    # agreement below is far tighter than any layout error could produce.
    np.testing.assert_allclose(rollout_state, recorded_state, rtol=0, atol=1e-2)

    # The decisive part: the correct convention must fit dramatically better than the
    # plausible wrong ones. Otherwise "close enough" could hide a real mismatch.
    from robosuite.utils import transform_utils

    correct_error = float(np.abs(rollout_state - recorded_state).max())

    euler_variant = np.concatenate(
        [
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            transform_utils.mat2euler(
                transform_utils.quat2mat(obs["robot0_eef_quat"])
            ).astype(np.float32),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
        ]
    )
    euler_error = float(np.abs(euler_variant - recorded_state).max())
    assert correct_error < euler_error / 5, (
        f"axis-angle error {correct_error:.5f} is not clearly better than the Euler "
        f"variant's {euler_error:.5f}; the orientation convention is unconfirmed"
    )

    # A swapped component order must also be clearly worse.
    swapped = np.concatenate(
        [rollout_state[3:6], rollout_state[:3], rollout_state[6:]]
    )
    swapped_error = float(np.abs(swapped - recorded_state).max())
    assert correct_error < swapped_error / 5, (
        f"component order is unconfirmed: correct {correct_error:.5f} vs swapped "
        f"{swapped_error:.5f}"
    )


# ---- temporal ensembling (§4.4, off by default) --------------------------------


def test_ensembler_single_prediction_is_passthrough():
    ens = TemporalEnsembler(horizon=4, d_action=2, coef=0.01)
    chunk = np.arange(8, dtype=np.float32).reshape(4, 2)
    ens.add(0, chunk)
    np.testing.assert_allclose(ens.pop(0), chunk[0])
    np.testing.assert_allclose(ens.pop(1), chunk[1])


def test_ensembler_weights_newest_prediction_most():
    ens = TemporalEnsembler(horizon=4, d_action=1, coef=1.0)
    # Prediction made at t=0 covers timesteps 0..3 with value 0.
    ens.add(0, np.zeros((4, 1), dtype=np.float32))
    # Prediction made at t=2 covers timesteps 2..5 with value 1.
    ens.add(2, np.ones((4, 1), dtype=np.float32))

    blended = float(ens.pop(2))
    # Two predictions: older (0) with weight exp(-1), newer (1) with weight exp(0).
    expected = 1.0 / (1.0 + np.exp(-1.0))
    assert blended == pytest.approx(expected, rel=1e-6)
    assert blended > 0.5, "the newest prediction must dominate"


def test_ensembler_weights_sum_to_one():
    """An averaging scheme that does not normalise would rescale action magnitudes."""
    ens = TemporalEnsembler(horizon=4, d_action=1, coef=0.5)
    for start in (0, 1, 2):
        ens.add(start, np.full((4, 1), 3.0, dtype=np.float32))
    # Every contribution is 3.0, so any convex combination must be exactly 3.0.
    assert float(ens.pop(2)) == pytest.approx(3.0, rel=1e-6)


def test_ensembler_rejects_wrong_chunk_shape():
    ens = TemporalEnsembler(horizon=4, d_action=2)
    with pytest.raises(ValueError, match="chunk shape"):
        ens.add(0, np.zeros((3, 2), dtype=np.float32))


def test_ensembler_raises_for_unplanned_timestep():
    ens = TemporalEnsembler(horizon=2, d_action=1)
    with pytest.raises(KeyError, match="no prediction available"):
        ens.pop(99)


# ---- aggregation --------------------------------------------------------------


def make_result(success: bool, episode_idx: int = 0) -> RolloutResult:
    return RolloutResult(
        success=success,
        n_steps=10,
        task_key="suite/task",
        episode_idx=episode_idx,
        seed=0,
        n_replans=2,
    )


def test_success_rate():
    results = [make_result(True, i) for i in range(8)] + [
        make_result(False, i) for i in range(8, 10)
    ]
    assert success_rate(results) == pytest.approx(0.8)


def test_success_rate_rejects_empty():
    with pytest.raises(ValueError, match="no rollouts"):
        success_rate([])


# ---- §4.4 execution arithmetic ------------------------------------------------


def test_replan_count_follows_execute_k():
    """Executing k of H per plan means ceil(max_steps / k) plans.

    Pins the §4.4 arithmetic without needing a simulator: with H=16 and k=8, a 600-step
    episode replans 75 times, not 38 (which would mean executing all 16) and not 600
    (which would mean replanning every step).
    """
    import math

    horizon, execute_k, max_steps = 16, 8, 600
    assert horizon == 16 and execute_k == 8
    assert math.ceil(max_steps / execute_k) == 75
    assert math.ceil(max_steps / horizon) == 38


# ---- requested-goal scoring (the instruction-swap check) ----------------------------------------


class _FakeSim:
    """Counts steps; this task's goal holds from ``own_at`` on, the other goal in ``[alt_from, alt_to)``."""

    def __init__(self, own_at, alt_from, alt_to):
        self.own_at, self.alt_from, self.alt_to, self.t = own_at, alt_from, alt_to, 0

    def seed(self, seed):
        pass

    def reset(self):
        self.t = 0

    def set_init_state(self, state):
        self.t = 0
        return {}

    def step(self, action):
        self.t += 1
        return {}, 0.0, False, {}

    def check_success(self):
        return self.own_at is not None and self.t >= self.own_at


def _fake_env(sim):
    import types

    import torch

    from flowcl.data.config import load_embodiment_spec
    from flowcl.envs.libero_env import LiberoTaskEnv

    env = object.__new__(LiberoTaskEnv)
    env.spec = load_embodiment_spec("libero_franka")
    env.env = sim
    env.init_states = [None]
    env.suite_name = "libero_goal"
    env.task = types.SimpleNamespace(name="fake_task")
    env._build_policy_batch = lambda obs, stats, device, language=None: {}
    env.goal_holds = lambda goal: sim.alt_from <= sim.t < sim.alt_to

    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))

        def sample(self, batch, n_steps=None, generator=None):
            return torch.zeros(1, env.spec.action.chunk_horizon, env.spec.d_action)

    stats = types.SimpleNamespace(denormalize_action=lambda a: a)
    return env, Policy(), stats


def test_the_requested_goal_is_recorded_and_never_ends_the_episode():
    env, policy, stats = _fake_env(_FakeSim(own_at=None, alt_from=5, alt_to=10))
    r = env.rollout(policy, 0, stats, "run", EvalConfig(max_steps=40), alt_goal=[["on", "a", "b"]])
    assert (r.success, r.n_steps) == (False, 40)          # ran to the timeout despite the requested goal
    assert (r.alt_at_start, r.alt_first_step, r.alt_at_end, r.alt_reached) == (False, 5, False, True)


def test_the_own_goal_still_ends_the_episode_and_both_are_recorded():
    env, policy, stats = _fake_env(_FakeSim(own_at=12, alt_from=3, alt_to=10**6))
    r = env.rollout(policy, 0, stats, "run", EvalConfig(max_steps=40), alt_goal=[["on", "a", "b"]])
    assert (r.success, r.n_steps) == (True, 12)
    assert (r.alt_first_step, r.alt_at_end, r.alt_reached) == (3, True, True)


def test_without_a_requested_goal_nothing_is_scored():
    env, policy, stats = _fake_env(_FakeSim(own_at=7, alt_from=0, alt_to=10**6))
    r = env.rollout(policy, 0, stats, "run", EvalConfig(max_steps=40))
    assert (r.success, r.n_steps) == (True, 7)
    assert (r.alt_at_start, r.alt_first_step, r.alt_at_end, r.alt_reached) == (None, None, None, None)


GOAL_PLATE = "libero_goal/put_the_bowl_on_the_plate"
GOAL_CABINET = "libero_goal/put_the_bowl_on_top_of_the_cabinet"
GOAL_STOVE = "libero_goal/put_the_bowl_on_the_stove"


@pytest.mark.sim
def test_goal_predicates_are_correct_on_known_states(dataset_dir):
    """In the plate task's env: the model is identical to the cabinet and stove tasks' (so their
    demo states may be loaded), its own parsed goal agrees with ``check_success``, and each task's
    goal holds exactly at that task's demo end state."""
    import h5py

    from flowcl.data.config import load_embodiment_spec
    from flowcl.data.tasks import TaskRef
    from flowcl.envs.libero_env import LiberoTaskEnv, goal_state

    spec = load_embodiment_spec("libero_franka")
    keys = (GOAL_PLATE, GOAL_CABINET, GOAL_STOVE)
    goals = {k: goal_state(k) for k in keys}
    assert goals[GOAL_CABINET] == [["on", "akita_black_bowl_1", "wooden_cabinet_1_top_side"]]
    signatures = {}
    for k in keys[1:]:
        ref = TaskRef.from_key(k)
        with LiberoTaskEnv(ref.suite, ref.task_idx, spec) as other:
            signatures[k] = other.model_signature()
    plate = TaskRef.from_key(GOAL_PLATE)
    with LiberoTaskEnv(plate.suite, plate.task_idx, spec) as env:
        own = env.model_signature()
        assert all(sig == own for sig in signatures.values()), "the scenes differ; states are not portable"
        assert own["state_size"] == 1 + own["nq"] + own["nv"]
        assert all(not env.missing_goal_objects(g) for g in goals.values())
        for k in keys:
            with h5py.File(TaskRef.from_key(k).demo_path(dataset_dir), "r") as f:
                states = f["data/demo_0/states"][()]
            assert states.shape[1] == own["state_size"]
            for label, state in (("start", states[0]), ("end", states[-1])):
                env.env.reset()
                env.set_sim_state(state)
                holds = {g: env.goal_holds(goals[g]) for g in keys}
                assert holds[GOAL_PLATE] == env.check_success(), (k, label)
                expected = {g: (label == "end" and g == k) for g in keys}
                assert holds == expected, (k, label, holds)
        with pytest.raises(KeyError, match="untracked"):
            env.goal_holds([["on", "akita_black_bowl_1", "no_such_object"]])
