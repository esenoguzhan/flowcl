"""Camera masks in the embodiment spec and the shared recorded-frame -> policy-input path."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from flowcl.data.config import load_embodiment_spec
from flowcl.data.images import policy_frame
from flowcl.data.spec import EmbodimentSpec, ObservationSpec

# fillPoly includes the boundary pixels, so x = 319 is the last masked column.
LEFT_HALF = ((0, 0), (319, 0), (319, 479), (0, 479))


def masked_obs(**kw):
    defaults = dict(cameras=("front", "wrist"), image_size=(128, 128), d_state=7,
                    image_masks=(("front", (LEFT_HALF,)),), mask_source_hw=(480, 640))
    defaults.update(kw)
    return ObservationSpec(**defaults)


def test_unmasked_specs_serialise_as_before():
    """LIBERO specs (and every existing checkpoint) carry no mask keys."""
    spec = load_embodiment_spec("libero_franka")
    obs = spec.to_dict()["observation"]
    assert set(obs) == {"cameras", "image_size", "d_state", "state_keys"}
    assert EmbodimentSpec.from_dict(spec.to_dict()) == spec


def test_piper_spec_round_trips_with_its_mask():
    spec = load_embodiment_spec("agilex_piper")
    assert spec.cameras == ("front", "wrist")
    assert spec.d_state == 7 and spec.d_action == 7
    assert spec.action.control_rate_hz == 15.0 and spec.action.already_normalized is False
    assert spec.observation.mask_source_hw == (480, 640)
    assert len(spec.observation.masks_for("front")) == 1 and spec.observation.masks_for("wrist") == ()
    assert EmbodimentSpec.from_dict(spec.to_dict()) == spec


@pytest.mark.parametrize(
    "kw, match",
    [
        (dict(image_masks=(("top", (LEFT_HALF,)),)), "not one of"),
        (dict(image_masks=(("front", (((0, 0), (5, 5)),)),)), ">= 3 vertices"),
        (dict(image_masks=(("front", (((0, 0), (700, 0), (0, 10)),)),)), "outside"),
        (dict(mask_source_hw=None), "mask_source_hw"),
        (dict(image_masks=()), "image_masks is empty"),
    ],
)
def test_bad_masks_are_rejected(kw, match):
    with pytest.raises(ValueError, match=match):
        masked_obs(**kw)


def test_policy_frame_blacks_out_only_the_polygon():
    obs = masked_obs()
    white = np.full((480, 640, 3), 255, np.uint8)
    front = policy_frame(white, "front", obs)
    assert front.shape == (128, 128, 3) and front.dtype == np.uint8
    assert front[:, :64].max() == 0 and front[:, 64:].min() == 255
    assert policy_frame(white, "wrist", obs).min() == 255
    # the recorded frame itself is untouched
    assert white.min() == 255


def test_policy_frame_rejects_frames_the_mask_does_not_fit():
    obs = masked_obs()
    with pytest.raises(ValueError, match="mask vertices"):
        policy_frame(np.zeros((240, 320, 3), np.uint8), "front", obs)
    with pytest.raises(ValueError, match="unknown camera"):
        policy_frame(np.zeros((480, 640, 3), np.uint8), "top", obs)
    # an unmasked camera accepts any size
    assert policy_frame(np.zeros((240, 320, 3), np.uint8), "wrist", obs).shape == (128, 128, 3)


def test_policy_frame_takes_the_spec_or_its_observation():
    spec = load_embodiment_spec("agilex_piper")
    frame = np.random.default_rng(0).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    np.testing.assert_array_equal(policy_frame(frame, "front", spec),
                                  policy_frame(frame, "front", spec.observation))
    unmasked = dataclasses.replace(spec.observation, image_masks=(), mask_source_hw=None)
    assert not np.array_equal(policy_frame(frame, "front", spec), policy_frame(frame, "front", unmasked))


RIGHT_HALF = ((320, 0), (639, 0), (639, 479), (320, 479))


def cropped_obs(**kw):
    defaults = dict(cameras=("front", "wrist"), image_size=(128, 128), d_state=7,
                    image_crops=(("front", RIGHT_HALF),), mask_source_hw=(480, 640))
    defaults.update(kw)
    return ObservationSpec(**defaults)


def test_policy_frame_keeps_only_the_crop_polygon():
    """A rectangle cuts the frame to it: the right half here."""
    import cv2

    obs = cropped_obs()
    frame = np.zeros((480, 640, 3), np.uint8)
    frame[:, 320:] = 255
    frame[100:140, 400:440] = (255, 0, 0)  # a red square inside the polygon
    front = policy_frame(frame, "front", obs)
    assert front.shape == (128, 128, 3)
    np.testing.assert_array_equal(front, cv2.resize(frame[:, 320:], (128, 128), interpolation=cv2.INTER_AREA))
    assert front[:, :, 1].max() == 255 and front[:, :, 1].min() == 0  # the square stays, at twice the width
    np.testing.assert_array_equal(policy_frame(frame, "wrist", obs),
                                  cv2.resize(frame, (128, 128), interpolation=cv2.INTER_AREA))


def test_outside_the_crop_polygon_is_black_inside_its_box():
    """A triangle: its bounding box is kept, the corner outside the triangle is black."""
    obs = cropped_obs(image_crops=(("front", ((0, 0), (639, 0), (639, 479))),), image_size=(48, 64))
    front = policy_frame(np.full((480, 640, 3), 255, np.uint8), "front", obs)
    assert front[0, -1].min() == 255 and front[-1, 0].max() == 0


def test_masks_apply_in_recorded_pixels_before_the_crop():
    obs = cropped_obs(image_masks=(("front", (((320, 0), (479, 0), (479, 479), (320, 479)),)),))
    front = policy_frame(np.full((480, 640, 3), 255, np.uint8), "front", obs)
    assert front[:, :64].max() == 0 and front[:, 64:].min() == 255


@pytest.mark.parametrize(
    "kw, match",
    [
        (dict(image_crops=(("top", RIGHT_HALF),)), "not one of"),
        (dict(image_crops=(("front", ((0, 0), (700, 0), (0, 10))),)), "outside"),
        (dict(image_crops=(("front", ((5, 0), (5, 100), (5, 200))),)), "empty"),
        (dict(image_crops=(("front", ((0, 0), (5, 5))),)), ">= 3 vertices"),
        (dict(mask_source_hw=None), "mask_source_hw"),
    ],
)
def test_bad_crops_are_rejected(kw, match):
    with pytest.raises(ValueError, match=match):
        cropped_obs(**kw)


def test_crop_round_trips_and_needs_the_recorded_size():
    spec = dataclasses.replace(load_embodiment_spec("agilex_piper"),
                               observation=cropped_obs(image_masks=(("front", (LEFT_HALF,)),)))
    obs = spec.to_dict()["observation"]
    assert obs["image_crops"] == {"front": [[x, y] for x, y in RIGHT_HALF]} and obs["mask_source_hw"] == [480, 640]
    assert EmbodimentSpec.from_dict(spec.to_dict()) == spec
    with pytest.raises(ValueError, match="crop polygon"):
        policy_frame(np.zeros((240, 320, 3), np.uint8), "front", cropped_obs())
    # the plain Piper spec carries no crop key
    assert "image_crops" not in load_embodiment_spec("agilex_piper").to_dict()["observation"]
