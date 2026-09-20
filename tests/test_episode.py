"""Canonical episode validation (§3.1)."""

from __future__ import annotations

import numpy as np
import pytest

from flowcl.data.episode import Episode


def make(**overrides) -> Episode:
    n = 5
    kwargs = dict(
        images={"cam": np.zeros((n, 4, 4, 3), dtype=np.uint8)},
        state=np.zeros((n, 8), dtype=np.float32),
        action=np.zeros((n, 7), dtype=np.float32),
        language="pick up the milk",
        task_id="libero_object/pick_up_the_milk",
        embodiment="libero_franka",
        meta={"source_file": "x.hdf5", "demo_idx": 0},
    )
    kwargs.update(overrides)
    return Episode(**kwargs)


def test_valid_episode_exposes_shapes():
    ep = make()
    assert ep.length == 5
    assert ep.d_state == 8
    assert ep.d_action == 7
    assert ep.cameras == ("cam",)


def test_rejects_length_mismatch_between_state_and_action():
    with pytest.raises(ValueError, match="state has 4 steps but action has 5"):
        make(state=np.zeros((4, 8), dtype=np.float32))


def test_rejects_image_length_mismatch():
    with pytest.raises(ValueError, match="has 3 frames but action has 5"):
        make(images={"cam": np.zeros((3, 4, 4, 3), dtype=np.uint8)})


def test_rejects_float64_action():
    """float64 would silently double memory and change loss precision."""
    with pytest.raises(TypeError, match="action must be float32"):
        make(action=np.zeros((5, 7), dtype=np.float64))


def test_rejects_non_uint8_images():
    with pytest.raises(TypeError, match="must be uint8"):
        make(images={"cam": np.zeros((5, 4, 4, 3), dtype=np.float32)})


def test_rejects_wrong_image_rank():
    with pytest.raises(ValueError, match=r"must be \(T, H, W, 3\)"):
        make(images={"cam": np.zeros((5, 4, 4), dtype=np.uint8)})


def test_rejects_empty_images():
    with pytest.raises(ValueError, match="non-empty dict"):
        make(images={})


def test_rejects_empty_language():
    """Language is the only task signal at inference (§0), so it cannot be blank."""
    with pytest.raises(ValueError, match="empty language string"):
        make(language="")


def test_rejects_nan_action():
    action = np.zeros((5, 7), dtype=np.float32)
    action[2, 3] = np.nan
    with pytest.raises(ValueError, match="action contains non-finite"):
        make(action=action)


def test_rejects_zero_length():
    with pytest.raises(ValueError, match="zero timesteps"):
        make(
            action=np.zeros((0, 7), dtype=np.float32),
            state=np.zeros((0, 8), dtype=np.float32),
            images={"cam": np.zeros((0, 4, 4, 3), dtype=np.uint8)},
        )
