"""Deployment pieces for the Dobot X-Trainer try-out (branch dobot-hw): CPU only."""

from __future__ import annotations

import numpy as np
import pytest

from flowcl.deploy import wire
from flowcl.deploy.images import (
    decode_jpeg,
    encode_jpeg,
    recorder_bgr,
    to_policy_frame,
)
from flowcl.deploy.safety import (
    Envelope,
    cartesian_violations,
    check_prefix,
    hard_joint_violations,
    limit_action_step,
    upsample_chunk,
)

# A pose inside the lab's J3/J4 limits (left J3 < 0, right J3 > 0).
HOME = np.array([-1.5, 0.0, -1.6, 0.05, 1.5, 1.6, 1.0, 1.5, 0.0, 1.5, -0.05, -1.5, -1.6, 1.0], np.float32)


def _xtrainer_limit(action, last_action, max_joint_step_rad, max_gripper_step):
    """Verbatim semantics of xtrainer experiments/run_policy_control.py::_limit_action_step."""
    joint_dims = np.array([i for i in range(14) if i not in (6, 13)])
    limited = action.copy()
    joint_delta = limited[joint_dims] - last_action[joint_dims]
    limited[joint_dims] -= (2.0 * np.pi) * np.round(joint_delta / (2.0 * np.pi))
    joint_delta = limited[joint_dims] - last_action[joint_dims]
    clipped = np.clip(joint_delta, -max_joint_step_rad, max_joint_step_rad)
    limited[joint_dims] = last_action[joint_dims] + clipped
    grip = [6, 13]
    gd = limited[grip] - last_action[grip]
    limited[grip] = last_action[grip] + np.clip(gd, -max_gripper_step, max_gripper_step)
    limited[grip] = np.clip(limited[grip], 0.0, 1.0)
    return limited.astype(np.float32)


def test_wire_roundtrip_numpy_and_bytes():
    msg = {"state": np.arange(14, dtype=np.float32), "images": {"top": b"\xff\xd8jpeg"},
           "prompt": "do it", "seq": 3, "nested": [np.ones((2, 3), np.uint8)], "x": np.float32(1.5)}
    back = wire.unpackb(wire.packb(msg))
    np.testing.assert_array_equal(back["state"], msg["state"])
    assert back["state"].dtype == np.float32 and back["state"].flags.writeable
    assert back["images"]["top"] == b"\xff\xd8jpeg"
    assert back["prompt"] == "do it" and back["seq"] == 3
    np.testing.assert_array_equal(back["nested"][0], msg["nested"][0])
    assert back["x"] == np.float32(1.5)


def test_wire_uses_openpi_ndarray_convention():
    import msgpack

    raw = msgpack.unpackb(wire.packb({"a": np.zeros(2, np.float32)}), raw=False)
    assert raw["a"][b"__ndarray__"] is True and raw["a"][b"dtype"] == "<f4"


def test_recorder_bgr_matches_teleop_recording():
    import cv2

    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    top = recorder_bgr(rgb, "top")
    expected = cv2.resize(rgb[150:420, 220:480, ::-1], (640, 480))
    np.testing.assert_array_equal(top, expected)
    np.testing.assert_array_equal(recorder_bgr(rgb, "left_wrist"), rgb[:, :, ::-1])
    with pytest.raises(ValueError):
        recorder_bgr(rgb, "front")


def test_server_and_cache_share_the_policy_frame():
    rng = np.random.default_rng(1)
    bgr = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    frame = to_policy_frame(bgr)
    assert frame.shape == (128, 128, 3) and frame.dtype == np.uint8
    # BGR -> RGB: a pure-blue BGR frame becomes blue in RGB channel 2
    blue = np.zeros((480, 640, 3), np.uint8)
    blue[..., 0] = 200
    assert to_policy_frame(blue)[..., 2].min() == 200 and to_policy_frame(blue)[..., 0].max() == 0
    # the server's JPEG path differs from the frame only by the codec
    via_jpeg = to_policy_frame(decode_jpeg(encode_jpeg(bgr)))
    smooth = np.full((480, 640, 3), 90, np.uint8)
    np.testing.assert_array_equal(to_policy_frame(decode_jpeg(encode_jpeg(smooth))), to_policy_frame(smooth))
    assert via_jpeg.shape == frame.shape


def test_limiter_matches_xtrainer_semantics():
    rng = np.random.default_rng(2)
    last = HOME.copy()
    for _ in range(200):
        action = last + rng.normal(scale=0.08, size=14).astype(np.float32)
        action[[6, 13]] = rng.uniform(-0.2, 1.2, 2)
        if rng.random() < 0.1:
            action[0] += 2 * np.pi  # wrapped joint
        ours, _, _ = limit_action_step(action, last, 0.03, 0.05)
        np.testing.assert_allclose(ours, _xtrainer_limit(action, last, 0.03, 0.05), atol=1e-6)
        last = ours


def test_upsample_hits_policy_targets():
    chunk = np.cumsum(np.ones((16, 14), np.float32), axis=0)
    up = upsample_chunk(chunk, 3)
    assert up.shape == (46, 14)
    np.testing.assert_array_equal(up[::3], chunk)
    np.testing.assert_allclose(up[1], chunk[0] + (chunk[1] - chunk[0]) / 3, rtol=1e-6)


def test_check_prefix_flags_jumps_envelope_and_limits():
    env = Envelope.from_actions(np.stack([HOME - 0.2, HOME + 0.2]), margin_rad=0.1)
    assert env.lo[6] == pytest.approx(-0.1) and env.hi[6] == pytest.approx(1.1)
    ok = np.repeat(HOME[None], 24, axis=0)
    assert check_prefix(ok, HOME, env, 0.15) == []
    jump = ok.copy()
    jump[:, 1] += 0.25
    assert any("from the current pose" in p for p in check_prefix(jump, HOME, env, 0.15))
    out = ok.copy()
    out[-1, 4] += 0.5
    assert any("envelope" in p for p in check_prefix(out, HOME, env, 1.0))
    bad = ok.copy()
    bad[5:, 2] = 0.1  # left J3 must stay negative
    msgs = check_prefix(bad, HOME, None, 10.0)
    assert any("J3/J4" in p and "tick 5" in p for p in msgs)
    nan = ok.copy()
    nan[3, 0] = np.nan
    assert check_prefix(nan, HOME, env, 0.15) == ["prefix contains non-finite values"]


def test_hard_and_cartesian_limits():
    assert hard_joint_violations(HOME) == []
    assert cartesian_violations(np.array([0, -400, 200, 0, 0, 0, 0, -400, 200, 0, 0, 0])) == []
    assert cartesian_violations(np.array([0, -400, 30, 0, 0, 0, 0, -400, 200, 0, 0, 0]))
