"""Dobot X-Trainer adapter rules (branch dobot-hw): stride, gripper state, cache checks."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from flowcl.data.config import load_embodiment_spec
from flowcl.data.dobot_adapter import (
    CACHE_FORMAT,
    iter_cache_episodes,
    stride_for,
    subsample,
    verify_cache,
)
from flowcl.data.episode import Episode
from flowcl.data.stats import compute_stats


@pytest.fixture
def spec():
    return load_embodiment_spec("dobot_xtrainer")


def _write_cache(path, lengths, language="do it"):
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        f.attrs.update({"format": CACHE_FORMAT, "task_name": "toy", "repo_id": "org/toy",
                        "revision": "abc", "language": language, "source_fps": 30,
                        "n_episodes": len(lengths), "complete": True})
        for i, n in enumerate(lengths):
            g = f.create_group(f"episodes/{i:06d}")
            g["actions"] = rng.normal(size=(n, 14)).astype(np.float32)
            g["qpos"] = rng.normal(size=(n, 14)).astype(np.float32)
            g["timestamp"] = np.arange(n) / 30.0
            for cam in ("top", "left_wrist", "right_wrist"):
                frames = np.zeros((n, 128, 128, 3), np.uint8)
                frames[:, 0, 0, 0] = np.arange(n) % 256  # frame id in one pixel
                g[f"images/{cam}"] = frames


def test_spec_is_ten_hz_with_floor(spec):
    assert stride_for(spec) == 3
    assert spec.stats_std_floor == 0.2
    assert spec.to_dict()["stats_std_floor"] == 0.2


def test_subsample_uses_previous_gripper_command():
    n = 10
    actions = np.arange(n * 14, dtype=np.float32).reshape(n, 14)
    qpos = -np.arange(n * 14, dtype=np.float32).reshape(n, 14)
    index, state, action = subsample(actions, qpos, 3)
    assert index.tolist() == [0, 3, 6, 9]
    np.testing.assert_array_equal(action, actions[[0, 3, 6, 9]])
    joints = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
    np.testing.assert_array_equal(state[:, joints], qpos[[0, 3, 6, 9]][:, joints])
    # gripper state: measured at i = 0, else the previous 30 Hz command
    np.testing.assert_array_equal(state[0, [6, 13]], qpos[0, [6, 13]])
    np.testing.assert_array_equal(state[1:, [6, 13]], actions[[2, 5, 8]][:, [6, 13]])


def test_iter_cache_episodes_aligns_frames(tmp_path, spec):
    path = tmp_path / "toy.h5"
    _write_cache(path, [31, 17, 20])
    eps = list(iter_cache_episodes(path, spec, n_demos=2))
    assert [e.length for e in eps] == [11, 6]
    for e in eps:
        assert isinstance(e, Episode)
        assert e.language == "do it" and e.task_id == "dobot/toy"
        # frame i of the episode is source frame 3*i
        np.testing.assert_array_equal(e.images["top"][:, 0, 0, 0], (np.arange(e.length) * 3) % 256)
    held_out = list(iter_cache_episodes(path, spec, episode_indices=[2]))
    assert held_out[0].meta["episode_index"] == 2
    with pytest.raises(ValueError):
        list(iter_cache_episodes(path, spec, n_demos=4))


def test_verify_cache_rejects_wrong_revision_or_language(tmp_path):
    path = tmp_path / "toy.h5"
    _write_cache(path, [9])
    verify_cache(path, repo_id="org/toy", revision="abc", language="do it", n_demos=1)
    with pytest.raises(ValueError, match="revision"):
        verify_cache(path, repo_id="org/toy", revision="xyz", language="do it", n_demos=1)
    with pytest.raises(ValueError, match="language"):
        verify_cache(path, repo_id="org/toy", revision="abc", language="other", n_demos=1)
    with pytest.raises(ValueError, match="demos requested"):
        verify_cache(path, repo_id="org/toy", revision="abc", language="do it", n_demos=2)


def test_std_floor_only_raises_small_stds(tmp_path, spec):
    path = tmp_path / "toy.h5"
    _write_cache(path, [30, 30])
    eps = list(iter_cache_episodes(path, spec))
    for e in eps:
        e.action[:, 4] = 0.5 + 0.01 * np.sin(np.arange(e.length))  # nearly constant joint
    plain = compute_stats(eps, embodiment=spec.name, task_id="dobot/toy", normalize_actions=True)
    floored = compute_stats(eps, embodiment=spec.name, task_id="dobot/toy",
                            normalize_actions=True, std_floor=0.1)
    assert plain.action.std[4] < 0.1 and floored.action.std[4] == pytest.approx(0.1)
    others = [i for i in range(14) if i != 4]
    np.testing.assert_allclose(np.asarray(floored.action.std)[others], np.asarray(plain.action.std)[others])
    # no floor = the old computation exactly
    again = compute_stats(eps, embodiment=spec.name, task_id="dobot/toy", normalize_actions=True, std_floor=0.0)
    assert again.fingerprint() == plain.fingerprint()
