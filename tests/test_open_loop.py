"""The offline open-loop check's metrics, on a toy episode with hand-computed answers."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from flowcl.analysis import open_loop
from flowcl.data.episode import Episode
from flowcl.data.spec import ActionSpec, EmbodimentSpec, ObservationSpec
from flowcl.data.stats import compute_stats

K, H, LENGTH = 2, 4, 7


def make_spec() -> EmbodimentSpec:
    return EmbodimentSpec(
        name="toy",
        observation=ObservationSpec(cameras=("cam",), image_size=(2, 2), d_state=2),
        action=ActionSpec(d_action=7, control_mode="osc_pose_delta", control_rate_hz=20.0,
                          chunk_horizon=H, execute_k=K),
    )


def make_episode() -> Episode:
    """Pose deltas ``t + 1`` on all six dims, gripper 0."""
    action = np.zeros((LENGTH, 7), dtype=np.float32)
    action[:, :6] = np.arange(1, LENGTH + 1, dtype=np.float32)[:, None]
    return Episode(
        images={"cam": np.zeros((LENGTH, 2, 2, 3), dtype=np.uint8)},
        state=np.zeros((LENGTH, 2), dtype=np.float32),
        action=action,
        language="do the toy task",
        task_id="toy/t0",
        embodiment="toy",
    )


class Oracle:
    """Returns the recorded (normalized) chunk."""

    def sample(self, batch, noise):
        return batch["actions"].float()


class Zeros:
    def sample(self, batch, noise):
        return torch.zeros_like(noise)


def check(policy) -> tuple[open_loop.ChunkPredictions, dict]:
    spec, episode = make_spec(), make_episode()
    # LIBERO's actions are recorded, never renormalised, so the stats leave them alone.
    stats = compute_stats([episode], embodiment="toy", task_id="toy/t0")
    chunks = open_loop.predict_chunks(policy, spec, stats, episode, torch.device("cpu"), seed=0, batch_size=3)
    return chunks, open_loop.delta_metrics(chunks, episode.action, K)


def test_chunks_start_every_k_ticks_and_mask_the_episode_end():
    chunks, row = check(Oracle())
    assert chunks.ticks.tolist() == [0, 2, 4, 6] and row["n_chunks"] == 4
    assert chunks.valid.tolist() == [[True] * 4, [True] * 4, [True, True, True, False],
                                     [True, False, False, False]]
    assert chunks.executed(K)[:, K:].sum() == 0


def test_a_perfect_policy_has_zero_error():
    _, row = check(Oracle())
    for key in ("integrated_policy_mae_executed", "policy_mae_executed", "policy_mae_chunk",
                "gripper_mae_executed"):
        assert row[key] == 0.0
    assert row["integrated_policy_by_tick"] == [0.0] * H


def test_predicting_zeros_is_the_zero_motion_baseline():
    _, row = check(Zeros())
    assert row["policy_mae_executed"] == pytest.approx(row["zero_motion_mae_executed"])
    assert row["integrated_policy_mae_executed"] == pytest.approx(row["integrated_standing_still_mae_executed"])
    assert row["integrated_policy_by_tick"] == pytest.approx(row["integrated_standing_still_by_tick"])


def test_baselines_match_hand_computed_values():
    # Executed ticks (k = 2) record deltas 1,2 | 3,4 | 5,6 | 7 (the last chunk has one valid tick).
    _, row = check(Oracle())
    # Standing still misses by the running sum: 1,3 | 3,7 | 5,11 | 7.
    assert row["integrated_standing_still_mae_executed"] == pytest.approx(37 / 7)
    # Repeating the last delta (the first action before tick 0: 1, then 2, 4, 6): 0,1 | 1,3 | 1,3 | 1.
    assert row["integrated_repeat_last_mae_executed"] == pytest.approx(10 / 7)
    assert row["zero_motion_mae_executed"] == pytest.approx(4.0)
    assert row["repeat_last_mae_executed"] == pytest.approx(8 / 7)
    # Over the whole chunk, tick by tick, on the valid chunks only.
    assert row["integrated_standing_still_by_tick"] == pytest.approx([4.0, 7.0, 12.0, 14.0])


def test_summarize_weights_by_chunks_and_averages_curves_tick_by_tick():
    rows = [{"demo_idx": 40, "n_chunks": 1, "e": 1.0, "curve": [0.0, 2.0]},
            {"demo_idx": 41, "n_chunks": 3, "e": 3.0, "curve": [4.0, 6.0]}]
    s = open_loop.summarize(rows, ("e", "curve"), id_key="demo_idx")
    assert s["episodes"] == [40, 41] and s["n_chunks"] == 4
    assert s["e"] == pytest.approx(2.5) and s["e_range"] == [1.0, 3.0]
    assert s["curve"] == pytest.approx([3.0, 5.0]) and "curve_range" not in s


def test_held_out_demos_must_be_unseen():
    open_loop.assert_unseen(40, list(range(40, 50)))
    with pytest.raises(ValueError, match="trained on"):
        open_loop.assert_unseen(41, list(range(40, 50)))
    with pytest.raises(ValueError, match="every demo"):
        open_loop.assert_unseen(None, list(range(40, 50)))


def test_noise_seed_is_keyed_by_the_recording():
    assert open_loop.episode_seed(0, "a/40") == open_loop.episode_seed(0, "a/40")
    assert open_loop.episode_seed(0, "a/40") != open_loop.episode_seed(0, "a/41")
