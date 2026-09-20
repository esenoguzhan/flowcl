"""Checkpoints carry everything a rollout needs, and refuse to be misread."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from flowcl.data.spec import ActionSpec, EmbodimentSpec, ObservationSpec
from flowcl.data.stats import FieldStats, NormalizationStats
from flowcl.models.build import build_policy
from flowcl.train.checkpoint import load_checkpoint, save_checkpoint

# §4.2 constrains d_model to 384-512 and the trunk to 6-8 layers, so there is no
# "tiny" policy to test with; flowpolicy_small is the smallest legal configuration.
POLICY_CONFIG = {
    "vision_backbone": "dinov2_s",
    "text_backbone": "clip_b",
    "d_model": 384,
    "n_trunk_layers": 6,
    "n_heads": 8,
    "n_decoder_layers": 2,
    "n_context_tokens": 8,
    "pretrained": False,
    "euler_steps": 2,
}


@pytest.fixture(scope="module")
def spec() -> EmbodimentSpec:
    return EmbodimentSpec(
        name="libero_franka",
        observation=ObservationSpec(
            cameras=("agentview", "robot0_eye_in_hand"),
            image_size=(128, 128),
            d_state=8,
            state_keys=(("ee_pos", 3), ("ee_ori", 3), ("gripper_states", 2)),
        ),
        action=ActionSpec(
            d_action=7,
            control_mode="osc_pose_delta",
            control_rate_hz=20.0,
            chunk_horizon=16,
            execute_k=8,
            already_normalized=True,
        ),
    )


@pytest.fixture(scope="module")
def stats(spec) -> NormalizationStats:
    return NormalizationStats(
        embodiment=spec.name,
        fitted_on_task_id="libero_spatial/task_one",
        fitted_on_n_demos=50,
        n_steps=1000,
        state=FieldStats(
            mean=[0.1] * 8, std=[0.5] * 8, min=[-1.0] * 8, max=[1.0] * 8, apply=True
        ),
        action=FieldStats(
            mean=[0.0] * 7, std=[1.0] * 7, min=[-1.0] * 7, max=[1.0] * 7, apply=False
        ),
    )


# ---- EmbodimentSpec serialisation ----------------------------------------------


def test_spec_round_trips(spec):
    assert EmbodimentSpec.from_dict(spec.to_dict()) == spec


def test_spec_round_trip_preserves_tuple_types(spec):
    """YAML/JSON turn tuples into lists; the validators depend on the widths matching."""
    restored = EmbodimentSpec.from_dict(spec.to_dict())
    assert isinstance(restored.cameras, tuple)
    assert restored.observation.image_size == (128, 128)
    assert restored.action.execute_k == 8


def test_spec_from_dict_validates(spec):
    """A corrupted payload must fail at load, not produce a subtly wrong policy."""
    payload = spec.to_dict()
    payload["action"]["execute_k"] = 99  # > chunk_horizon
    with pytest.raises(ValueError, match="execute_k must be in"):
        EmbodimentSpec.from_dict(payload)


# ---- checkpoint round trip ------------------------------------------------------


def test_checkpoint_round_trip_restores_identical_weights(tmp_path, spec, stats):
    policy = build_policy(POLICY_CONFIG, spec, pretrained=False)
    path = save_checkpoint(
        tmp_path / "ckpt.pt",
        policy=policy,
        policy_config=POLICY_CONFIG,
        spec=spec,
        stats=stats,
        run_id="test-run",
        stage=2,
        task_key="libero_object/some_task",
    )

    loaded = load_checkpoint(path)
    assert loaded.run_id == "test-run"
    assert loaded.stage == 2
    assert loaded.task_key == "libero_object/some_task"
    assert loaded.spec == spec
    assert loaded.stats.fingerprint() == stats.fingerprint()

    original = policy.state_dict()
    for name, tensor in loaded.policy.state_dict().items():
        torch.testing.assert_close(tensor, original[name])


def test_loaded_policy_reproduces_the_original_sample(tmp_path, spec, stats):
    """The point of a checkpoint: identical weights must give identical actions."""
    policy = build_policy(POLICY_CONFIG, spec, pretrained=False)
    policy.eval()

    batch = {
        "images": {
            camera: torch.from_numpy(
                np.random.default_rng(0).integers(
                    0, 255, size=(2, 128, 128, 3), dtype=np.uint8
                )
            )
            for camera in spec.cameras
        },
        "state": torch.zeros(2, 8),
        "language": ["pick up the milk"] * 2,
    }
    noise = torch.randn(2, spec.action.chunk_horizon, spec.d_action, generator=
                        torch.Generator().manual_seed(0))

    before = policy.sample(batch, noise=noise)
    path = save_checkpoint(
        tmp_path / "ckpt.pt",
        policy=policy,
        policy_config=POLICY_CONFIG,
        spec=spec,
        stats=stats,
        run_id="r",
    )
    after = load_checkpoint(path).policy.sample(batch, noise=noise)
    torch.testing.assert_close(before, after)


def test_checkpoint_rejects_a_wrong_format_version(tmp_path, spec, stats):
    policy = build_policy(POLICY_CONFIG, spec, pretrained=False)
    path = save_checkpoint(
        tmp_path / "ckpt.pt",
        policy=policy,
        policy_config=POLICY_CONFIG,
        spec=spec,
        stats=stats,
        run_id="r",
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["version"] = 999
    torch.save(payload, path)

    with pytest.raises(ValueError, match="format version"):
        load_checkpoint(path)


def test_checkpoint_detects_tampered_stats(tmp_path, spec, stats):
    """Evaluating with different normalisation than training would fail invisibly."""
    policy = build_policy(POLICY_CONFIG, spec, pretrained=False)
    path = save_checkpoint(
        tmp_path / "ckpt.pt",
        policy=policy,
        policy_config=POLICY_CONFIG,
        spec=spec,
        stats=stats,
        run_id="r",
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["stats"]["state"]["mean"] = [9.9] * 8
    torch.save(payload, path)

    with pytest.raises(ValueError, match="do not match their recorded"):
        load_checkpoint(path)


def test_checkpoint_load_is_strict_about_missing_weights(tmp_path, spec, stats):
    """A silently partial load would evaluate a half-random policy."""
    policy = build_policy(POLICY_CONFIG, spec, pretrained=False)
    path = save_checkpoint(
        tmp_path / "ckpt.pt",
        policy=policy,
        policy_config=POLICY_CONFIG,
        spec=spec,
        stats=stats,
        run_id="r",
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    dropped = next(
        k for k in payload["state_dict"] if k.startswith("trunk.blocks.0")
    )
    del payload["state_dict"][dropped]
    torch.save(payload, path)

    with pytest.raises(RuntimeError, match="Missing key"):
        load_checkpoint(path)


def test_missing_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "nope.pt")
