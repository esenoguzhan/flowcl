"""Chunking and masking (§3.3), on synthetic episodes with known answers."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from flowcl.data.dataset import ChunkedActionDataset, collate_chunks
from flowcl.data.episode import Episode
from flowcl.data.spec import ActionSpec, EmbodimentSpec, ObservationSpec
from flowcl.data.stats import compute_stats

D_STATE = 2
D_ACTION = 3


def make_spec(horizon: int = 4, execute_k: int = 2) -> EmbodimentSpec:
    return EmbodimentSpec(
        name="toy",
        observation=ObservationSpec(
            cameras=("cam",), image_size=(2, 2), d_state=D_STATE
        ),
        action=ActionSpec(
            d_action=D_ACTION,
            control_mode="osc_pose_delta",
            control_rate_hz=20.0,
            chunk_horizon=horizon,
            execute_k=execute_k,
        ),
    )


def make_episode(n_steps: int, task_id: str = "toy/t0", offset: float = 0.0) -> Episode:
    """Episode whose action at step t is ``[t, t+0.1, t+0.2] + offset``.

    Encoding the timestep in the values makes the chunk contents checkable by hand.
    """
    action = np.stack(
        [np.arange(n_steps) + offset + d * 0.1 for d in range(D_ACTION)], axis=1
    ).astype(np.float32)
    return Episode(
        images={"cam": np.zeros((n_steps, 2, 2, 3), dtype=np.uint8)},
        state=np.zeros((n_steps, D_STATE), dtype=np.float32),
        action=action,
        language="do the toy task",
        task_id=task_id,
        embodiment="toy",
    )


def make_dataset(n_steps: int, horizon: int = 4) -> ChunkedActionDataset:
    spec = make_spec(horizon=horizon)
    episode = make_episode(n_steps)
    stats = compute_stats([episode], embodiment="toy", task_id="toy/t0")
    return ChunkedActionDataset([episode], spec, stats, horizon=horizon)


def test_known_answer_three_step_episode_horizon_four():
    """A 3-step episode with H=4: every chunk needs padding, masks are known exactly.

    t=0 -> actions 0,1,2,PAD  mask 1,1,1,0
    t=1 -> actions 1,2,PAD,PAD mask 1,1,0,0
    t=2 -> actions 2,PAD,PAD,PAD mask 1,0,0,0
    """
    ds = make_dataset(n_steps=3, horizon=4)
    assert len(ds) == 3

    expected_masks = [
        [1.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ]
    for t in range(3):
        sample = ds[t]
        assert sample["actions"].shape == (4, D_ACTION)
        assert sample["action_mask"].tolist() == expected_masks[t]

        n_valid = int(sum(expected_masks[t]))
        # Valid entries are the true actions, in order, starting at t.
        for j in range(n_valid):
            np.testing.assert_allclose(
                sample["actions"][j].numpy(),
                [(t + j) + d * 0.1 for d in range(D_ACTION)],
                rtol=1e-6,
            )
        # Padded entries are exactly zero.
        assert torch.all(sample["actions"][n_valid:] == 0.0)


def test_padding_count_is_horizon_minus_one_per_episode():
    """Independent of episode length: only the last H-1 samples need padding."""
    for n_steps in (20, 50, 168):
        ds = make_dataset(n_steps=n_steps, horizon=16)
        assert ds.n_padded_samples() == 15


def test_sample_count_equals_total_timesteps():
    spec = make_spec(horizon=16)
    episodes = [make_episode(20), make_episode(31)]
    stats = compute_stats(episodes, embodiment="toy", task_id="toy/t0")
    ds = ChunkedActionDataset(episodes, spec, stats, horizon=16)
    assert len(ds) == 51
    assert ds.n_episodes == 2


def test_masked_steps_contribute_exactly_zero_loss():
    """§3.3: 'masked steps contribute zero loss'.

    Perturbing the prediction only where the mask is zero must leave the masked loss
    bit-for-bit unchanged.
    """
    ds = make_dataset(n_steps=3, horizon=4)
    sample = ds[2]  # mask = [1, 0, 0, 0]
    target = sample["actions"]
    mask = sample["action_mask"]

    prediction = torch.zeros_like(target)

    def masked_mse(pred):
        per_element = (pred - target) ** 2
        return (per_element * mask.unsqueeze(-1)).sum() / (
            mask.sum() * target.shape[-1]
        )

    baseline = masked_mse(prediction)

    perturbed = prediction.clone()
    perturbed[1:] += 1e6  # entirely inside the padded region
    assert masked_mse(perturbed) == baseline

    # Sanity: perturbing a *valid* step does change the loss.
    changed = prediction.clone()
    changed[0] += 1.0
    assert masked_mse(changed) != baseline


def test_state_is_normalized_and_actions_are_not_for_already_normalized_data():
    episode = make_episode(10)
    episode.state[:] = np.linspace(0, 1, 10 * D_STATE).reshape(10, D_STATE)
    spec = make_spec(horizon=4)
    stats = compute_stats(
        [episode], embodiment="toy", task_id="toy/t0", normalize_actions=False
    )
    ds = ChunkedActionDataset([episode], spec, stats, horizon=4)

    sample = ds[0]
    # Actions pass through untouched.
    np.testing.assert_allclose(
        sample["actions"][0].numpy(), episode.action[0], rtol=1e-6
    )
    # State is standardised, so it differs from the raw value.
    assert not np.allclose(sample["state"].numpy(), episode.state[0])


def test_images_stay_uint8():
    ds = make_dataset(n_steps=5, horizon=4)
    sample = ds[0]
    assert sample["images"]["cam"].dtype == torch.uint8
    assert sample["images"]["cam"].shape == (2, 2, 3)


def test_collate_preserves_language_strings_as_list():
    """The language encoder caches per unique string (§4.1), so strings must survive."""
    ds = make_dataset(n_steps=5, horizon=4)
    batch = collate_chunks([ds[0], ds[1], ds[2]])

    assert batch["actions"].shape == (3, 4, D_ACTION)
    assert batch["action_mask"].shape == (3, 4)
    assert batch["state"].shape == (3, D_STATE)
    assert batch["images"]["cam"].shape == (3, 2, 2, 3)
    assert batch["language"] == ["do the toy task"] * 3
    assert batch["t"].tolist() == [0, 1, 2]


def test_collate_rejects_empty_batch():
    with pytest.raises(ValueError, match="empty batch"):
        collate_chunks([])


def test_dataset_rejects_stats_from_other_embodiment():
    episode = make_episode(5)
    stats = compute_stats([episode], embodiment="toy", task_id="toy/t0")
    other = EmbodimentSpec(
        name="other",
        observation=ObservationSpec(cameras=("cam",), image_size=(2, 2), d_state=D_STATE),
        action=ActionSpec(
            d_action=D_ACTION,
            control_mode="osc_pose_delta",
            control_rate_hz=20.0,
            chunk_horizon=4,
            execute_k=2,
        ),
    )
    with pytest.raises(ValueError, match="stats are for embodiment"):
        ChunkedActionDataset([episode], other, stats, horizon=4)


def test_dataset_requires_episodes():
    spec = make_spec()
    episode = make_episode(4)
    stats = compute_stats([episode], embodiment="toy", task_id="toy/t0")
    with pytest.raises(ValueError, match="at least one episode"):
        ChunkedActionDataset([], spec, stats)
