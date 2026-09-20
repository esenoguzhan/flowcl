"""LIBERO HDF5 -> canonical Episode (§3.2), and the §10.1 replay test.

The pure-metadata tests need the datasets on disk but not MuJoCo, so they are marked
``sim`` only where rendering or stepping is actually required. ``dataset_dir`` skips
when the suites are not downloaded.
"""

from __future__ import annotations

import numpy as np
import pytest

from flowcl.data import libero_adapter
from flowcl.data.config import load_embodiment_spec
from flowcl.data.libero_adapter import (
    D_ACTION,
    D_STATE,
    EXPECTED_DEMOS_PER_TASK,
    compute_action_stats,
    default_task_id,
    iter_episodes,
    load_episode,
    read_task_metadata,
    verify_task_file,
)


@pytest.fixture(scope="module")
def spec():
    return load_embodiment_spec("libero_franka")


@pytest.fixture(scope="module")
def task_path(dataset_dir):
    """A concrete libero_object task file."""
    path = (
        dataset_dir
        / "libero_object"
        / "pick_up_the_milk_and_place_it_in_the_basket_demo.hdf5"
    )
    if not path.is_file():
        pytest.skip(f"{path} not downloaded")
    return path


# ---- layout constants ---------------------------------------------------------


def test_state_layout_matches_spec_section_3_2():
    """§3.2: state is end-effector pose + gripper qpos -> 3 + 3 + 2 = 8."""
    assert libero_adapter.STATE_COMPONENTS == (
        ("ee_pos", 3),
        ("ee_ori", 3),
        ("gripper_states", 2),
    )
    assert D_STATE == 8
    assert D_ACTION == 7
    assert EXPECTED_DEMOS_PER_TASK == 50


def test_camera_key_mapping_is_hdf5_to_env_names():
    """HDF5 uses *_rgb; the env uses *_image. The canonical name is the env's."""
    assert libero_adapter.HDF5_TO_CANONICAL_CAMERA == {
        "agentview_rgb": "agentview",
        "eye_in_hand_rgb": "robot0_eye_in_hand",
    }


def test_default_task_id_strips_demo_suffix(tmp_path):
    path = tmp_path / "libero_object" / "pick_up_the_milk_demo.hdf5"
    assert default_task_id(path) == "libero_object/pick_up_the_milk"


# ---- metadata -----------------------------------------------------------------


def test_read_task_metadata(task_path):
    meta = read_task_metadata(task_path)
    assert meta.n_demos == 50
    assert len(meta.demo_lengths) == 50
    assert meta.language == "pick up the milk and place it in the basket"
    assert meta.image_convention == "opengl"
    assert all(length > 0 for length in meta.demo_lengths)


def test_verify_task_file_passes_on_real_data(task_path):
    """§3.2: verify demo count and action stats on load."""
    meta = verify_task_file(task_path)
    assert meta.n_demos == EXPECTED_DEMOS_PER_TASK


def test_verify_task_file_fails_loudly_on_wrong_demo_count(task_path):
    with pytest.raises(ValueError, match="expected 7 demos"):
        verify_task_file(task_path, expected_demos=7)


def test_read_task_metadata_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        read_task_metadata(tmp_path / "nope.hdf5")


def test_all_required_suites_verify(dataset_dir):
    """Every task file in every curriculum suite passes the §3.2 checks."""
    from flowcl.data.libero_setup import REQUIRED_SUITES

    for suite in REQUIRED_SUITES:
        files = sorted((dataset_dir / suite).glob("*.hdf5"))
        assert len(files) == 10, f"{suite} has {len(files)} files"
        for path in files:
            verify_task_file(path)


# ---- episode conversion -------------------------------------------------------


def test_load_episode_shapes_and_dtypes(task_path, spec):
    episode = load_episode(task_path, 0, spec)

    assert episode.d_state == D_STATE
    assert episode.d_action == D_ACTION
    assert episode.action.dtype == np.float32
    assert episode.state.dtype == np.float32
    assert set(episode.cameras) == set(spec.cameras)

    for camera in spec.cameras:
        frames = episode.images[camera]
        assert frames.shape == (episode.length, 128, 128, 3)
        assert frames.dtype == np.uint8

    assert episode.embodiment == "libero_franka"
    assert episode.task_id == "libero_object/pick_up_the_milk_and_place_it_in_the_basket"
    assert episode.language == "pick up the milk and place it in the basket"
    assert episode.meta["demo_idx"] == 0
    assert episode.meta["demo_name"] == "demo_0"


def test_actions_are_not_renormalized(task_path, spec):
    """§3.2: LIBERO actions are already in [-1, 1]; do not renormalise."""
    import h5py

    episode = load_episode(task_path, 3, spec)
    with h5py.File(task_path, "r") as f:
        raw = np.asarray(f["data"]["demo_3"]["actions"], dtype=np.float32)

    np.testing.assert_array_equal(episode.action, raw)
    assert np.abs(episode.action).max() <= 1.0
    # Gripper dimension stays binary.
    assert set(np.unique(episode.action[:, 6]).tolist()) <= {-1.0, 1.0}


def test_state_is_concatenation_in_declared_order(task_path, spec):
    import h5py

    episode = load_episode(task_path, 1, spec)
    with h5py.File(task_path, "r") as f:
        obs = f["data"]["demo_1"]["obs"]
        expected = np.concatenate(
            [
                np.asarray(obs["ee_pos"], dtype=np.float32),
                np.asarray(obs["ee_ori"], dtype=np.float32),
                np.asarray(obs["gripper_states"], dtype=np.float32),
            ],
            axis=1,
        )
    np.testing.assert_array_equal(episode.state, expected)


def test_demo_indices_are_numeric_not_alphabetical(task_path, spec):
    """demo_10 must be index 10, not index 2.

    HDF5 iterates keys alphabetically, so a naive sort puts demo_10 right after
    demo_1. Demo indices end up in replay buffers and stats provenance, so they must
    mean the same thing here as in LIBERO's own tooling.
    """
    episode = load_episode(task_path, 10, spec)
    assert episode.meta["demo_name"] == "demo_10"


def test_load_episode_rejects_out_of_range_demo(task_path, spec):
    with pytest.raises(IndexError, match="out of range"):
        load_episode(task_path, 50, spec)


def test_iter_episodes_respects_n_demos(task_path, spec):
    episodes = list(iter_episodes(task_path, spec, n_demos=3))
    assert len(episodes) == 3
    assert [e.meta["demo_idx"] for e in episodes] == [0, 1, 2]


def test_iter_episodes_rejects_too_many(task_path, spec):
    with pytest.raises(ValueError, match="has 50 demos but 51 were requested"):
        list(iter_episodes(task_path, spec, n_demos=51))


def test_action_stats_recorded_over_all_demos(task_path):
    stats = compute_action_stats(task_path)
    assert stats.min.shape == (D_ACTION,)
    assert stats.n_steps > 0
    assert stats.min.min() >= -1.0
    assert stats.max.max() <= 1.0


# ---- the §10.1 replay test ----------------------------------------------------


@pytest.mark.physics
def test_replay_demo_actions_reproduces_success(task_path, spec):
    """§10.1: reconstruct a demo, replay its actions in the env, confirm success.

    This is the end-to-end check that our adapter reads the same actions, in the same
    order, with the same initial state that LIBERO recorded. If the action ordering or
    the init-state handling were wrong, the replay simply would not succeed.

    Deliberately runs with rendering disabled: success is a pure-physics property, so
    coupling this check to a working GL context would make the most important data-layer
    test unrunnable anywhere without a GPU.
    """
    import os

    from flowcl.utils.libero_paths import ensure_libero_config

    ensure_libero_config()
    from libero.libero import get_libero_path

    # ControlEnv is defined in env_wrapper but not re-exported by envs/__init__.py,
    # which only surfaces the rendering variants.
    from libero.libero.envs.env_wrapper import ControlEnv

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
        env.set_init_state(episode.meta["init_state"])

        succeeded = False
        for action in episode.action:
            env.step(action.astype(np.float64))
            if env.check_success():
                succeeded = True
                break
        assert succeeded, (
            "replaying the recorded actions from the recorded init state did not "
            "reach the success condition; the adapter's action ordering or init-state "
            "handling is wrong"
        )
    finally:
        env.close()


@pytest.mark.physics
def test_recorded_sim_states_match_replay_start(task_path, spec):
    """The demo's ``init_state`` is the first row of its ``states`` array.

    Confirms that ``meta["init_state"]`` and ``meta["sim_states"]`` describe the same
    starting configuration, so either can be used to seed a replay.
    """
    episode = load_episode(task_path, 0, spec, with_sim_states=True)
    init_state = episode.meta["init_state"]
    sim_states = episode.meta["sim_states"]

    # Width is the flattened MuJoCo state size, which depends on the scene (e.g. 92
    # for a libero_spatial tabletop, 110 for this libero_object scene) — so assert the
    # two agree with each other rather than pinning a magic number.
    assert init_state.ndim == 1
    assert sim_states.shape == (episode.length, init_state.shape[0])
    np.testing.assert_allclose(init_state, sim_states[0], rtol=0, atol=1e-9)
