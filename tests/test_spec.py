"""EmbodimentSpec / ActionSpec invariants (§0, §3.1)."""

from __future__ import annotations

import pytest

from flowcl.data.config import load_embodiment_spec
from flowcl.data.spec import (
    ActionSpec,
    EmbodimentSpec,
    ObservationSpec,
    assert_no_cross_embodiment_padding,
)


def make_obs(**kw):
    defaults = dict(
        cameras=("agentview", "robot0_eye_in_hand"),
        image_size=(128, 128),
        d_state=8,
        state_keys=(("ee_pos", 3), ("ee_ori", 3), ("gripper_states", 2)),
    )
    defaults.update(kw)
    return ObservationSpec(**defaults)


def make_action(**kw):
    defaults = dict(
        d_action=7,
        control_mode="osc_pose_delta",
        control_rate_hz=20.0,
        chunk_horizon=16,
        execute_k=8,
        already_normalized=True,
    )
    defaults.update(kw)
    return ActionSpec(**defaults)


def test_libero_franka_yaml_matches_spec_section_3_2():
    """The shipped config must encode §3.2's numbers, not drift from them."""
    spec = load_embodiment_spec("libero_franka")
    assert spec.name == "libero_franka"
    assert spec.cameras == ("agentview", "robot0_eye_in_hand")
    assert spec.observation.image_size == (128, 128)
    assert spec.d_state == 8
    assert spec.d_action == 7
    assert spec.action.control_mode == "osc_pose_delta"
    assert spec.action.chunk_horizon == 16  # H in §3.3
    assert spec.action.execute_k == 8  # k in §4.4
    assert spec.action.already_normalized is True
    assert spec.action.control_rate_hz == 20.0


def test_state_keys_must_sum_to_d_state():
    with pytest.raises(ValueError, match="sum to 7 but d_state is 8"):
        make_obs(state_keys=(("ee_pos", 3), ("ee_ori", 3), ("gripper", 1)))


def test_component_keys_must_sum_to_d_action():
    with pytest.raises(ValueError, match="sum to 6 but d_action is 7"):
        make_action(component_keys=(("osc", 6),))


def test_execute_k_cannot_exceed_horizon():
    with pytest.raises(ValueError, match="execute_k must be in"):
        make_action(chunk_horizon=8, execute_k=16)


def test_rejects_duplicate_cameras():
    with pytest.raises(ValueError, match="duplicate camera"):
        make_obs(cameras=("agentview", "agentview"))


def test_rejects_empty_cameras():
    with pytest.raises(ValueError, match="must be non-empty"):
        make_obs(cameras=())


def test_assert_compatible_episode_rejects_wrong_widths():
    spec = EmbodimentSpec("libero_franka", make_obs(), make_action())
    spec.assert_compatible_episode(8, 7, ("agentview", "robot0_eye_in_hand"))

    with pytest.raises(ValueError, match="d_action 14 != spec 7"):
        spec.assert_compatible_episode(8, 14, spec.cameras)
    with pytest.raises(ValueError, match="d_state 9 != spec 8"):
        spec.assert_compatible_episode(9, 7, spec.cameras)
    with pytest.raises(ValueError, match="cameras missing"):
        spec.assert_compatible_episode(8, 7, ("agentview",))


def test_padding_detector_flags_shared_d_action_with_different_control_modes():
    """§0: padding actions across embodiments produces artifactual forgetting.

    Its signature is two embodiments claiming the same d_action while driving
    different controllers.
    """
    franka = EmbodimentSpec("libero_franka", make_obs(), make_action(d_action=14))
    agilex = EmbodimentSpec(
        "agilex_dual",
        make_obs(d_state=8, state_keys=()),
        make_action(d_action=14, control_mode="joint_position"),
    )
    with pytest.raises(ValueError, match="padded cross-embodiment action vector"):
        assert_no_cross_embodiment_padding([franka, agilex])


def test_padding_detector_allows_genuinely_distinct_embodiments():
    franka = EmbodimentSpec("libero_franka", make_obs(), make_action(d_action=7))
    agilex = EmbodimentSpec(
        "agilex_dual",
        make_obs(),
        make_action(d_action=14, control_mode="joint_position"),
    )
    assert_no_cross_embodiment_padding([franka, agilex])
