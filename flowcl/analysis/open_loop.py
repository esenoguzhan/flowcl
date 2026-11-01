"""Offline open-loop check: a policy's predicted chunks against recorded demos.

Every ``execute_k`` ticks of an episode the policy gets the recorded frames and state and
samples a chunk with fixed noise; the chunk is compared with the recorded actions. No
simulator or robot is involved, so this is not closed-loop success: it asks whether the
policy predicts what the demonstrator did next, against do-nothing baselines on the same
ticks.

:func:`predict_chunks` and :func:`summarize` are shared by the Piper check
(``scripts/hw/piper_offline_check.py``, absolute joint targets) and the LIBERO check
(``scripts/libero_offline_check.py``, OSC pose deltas, :func:`delta_metrics`).
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np
import torch

from flowcl.data.dataset import ChunkedActionDataset, collate_chunks
from flowcl.train.trainer import move_batch


@dataclass
class ChunkPredictions:
    """Predicted and recorded chunks of one episode, both denormalized."""

    ticks: np.ndarray      # (N,) chunk start ticks: 0, k, 2k, ...
    predicted: np.ndarray  # (N, H, d_action)
    recorded: np.ndarray   # (N, H, d_action)
    valid: np.ndarray      # (N, H) bool; False past the episode end

    def executed(self, k: int) -> np.ndarray:
        """The valid ticks among the first ``k`` of each chunk (the ones a robot executes)."""
        executed = self.valid.copy()
        executed[:, k:] = False
        return executed


@torch.no_grad()
def predict_chunks(policy, spec, stats, episode, device, seed: int, batch_size: int) -> ChunkPredictions:
    """The policy's chunks at ticks 0, k, 2k, ... of one episode, with noise from ``seed``."""
    k, horizon = spec.action.execute_k, spec.action.chunk_horizon
    dataset = ChunkedActionDataset([episode], spec, stats, horizon=horizon)
    ticks = list(range(0, episode.length, k))
    generator = torch.Generator().manual_seed(seed)
    predicted, recorded, valid = [], [], []
    for i in range(0, len(ticks), batch_size):
        batch = collate_chunks([dataset[t] for t in ticks[i:i + batch_size]])
        size = batch["actions"].shape[0]
        noise = torch.randn((size, horizon, spec.d_action), generator=generator).to(device)
        chunk = policy.sample(move_batch(batch, device), noise=noise).float().cpu().numpy()
        predicted.append(stats.denormalize_action(chunk))
        recorded.append(stats.denormalize_action(batch["actions"].numpy()))
        valid.append(batch["action_mask"].numpy().astype(bool))
    predicted, recorded, valid = map(np.concatenate, (predicted, recorded, valid))
    return ChunkPredictions(np.asarray(ticks), predicted, recorded, valid)


def episode_seed(seed: int, recording: str) -> int:
    """Sampling noise keyed by the recording, so every checkpoint sees the same noise on it."""
    return seed + zlib.crc32(recording.encode())


def summarize(rows: list[dict], keys: tuple[str, ...], id_key: str) -> dict:
    """Chunk-weighted means over episodes, with each scalar's per-episode range.

    A list-valued key (a per-tick curve) is averaged tick by tick, without a range.
    """
    weights = np.array([r["n_chunks"] for r in rows], dtype=float)
    out = {"episodes": [r[id_key] for r in rows], "n_chunks": int(weights.sum())}
    for key in keys:
        values = np.array([r[key] for r in rows])
        if values.ndim == 1:
            out[key] = float((values * weights).sum() / weights.sum())
            out[f"{key}_range"] = [float(values.min()), float(values.max())]
        else:
            out[key] = ((values * weights[:, None]).sum(axis=0) / weights.sum()).tolist()
    return out


def _per_tick(err: np.ndarray, valid: np.ndarray) -> list[float]:
    """Mean over valid chunks of ``err`` (N, H), tick by tick."""
    counts = valid.sum(axis=0)
    sums = np.where(valid, err, 0.0).sum(axis=0)
    return (sums / np.maximum(counts, 1)).tolist()


def delta_metrics(chunks: ChunkPredictions, actions: np.ndarray, k: int,
                  pose: slice = slice(0, 6), gripper: int = 6) -> dict:
    """Errors of delta-action chunks (LIBERO's OSC pose deltas), in action units.

    Two views of the ``pose`` dims, each against do-nothing baselines on the same ticks:

    * per tick, ``|δ̂ − δ|``, against zero motion (``|δ|``) and repeating the last
      recorded delta (constant velocity);
    * integrated from the chunk start, ``|Σ(δ̂ − δ)|``, against standing still
      (``|Σδ|``). This is the form of an absolute-target check: there, repeating the last
      command misses tick t by ``|a_t − a_{t0−1}|``, the sum of the command's increments.

    ``actions`` is the episode's recorded (denormalized) action sequence; the last delta
    before tick 0 is the first action, as in the Piper check.
    """
    executed = chunks.executed(k)
    recorded = chunks.recorded[..., pose]
    predicted = chunks.predicted[..., pose]
    last = actions[np.maximum(chunks.ticks - 1, 0)][:, None, pose]
    # The valid ticks are a prefix of each chunk, so a cumulative sum never adds a
    # padded tick into a valid one.
    integrated = {
        "policy": np.cumsum(predicted - recorded, axis=1),
        "standing_still": np.cumsum(-recorded, axis=1),
        "repeat_last": np.cumsum(last - recorded, axis=1),
    }
    per_tick = {
        "policy": predicted - recorded,
        "zero_motion": -recorded,
        "repeat_last": np.broadcast_to(last, recorded.shape) - recorded,
    }
    row = {"n_chunks": int(len(chunks.ticks))}
    for name, err in integrated.items():
        err = np.abs(err)
        row[f"integrated_{name}_mae_executed"] = float(err[executed].mean())
        row[f"integrated_{name}_by_tick"] = _per_tick(err.mean(axis=-1), chunks.valid)
    for name, err in per_tick.items():
        row[f"{name}_mae_executed"] = float(np.abs(err)[executed].mean())
    row["policy_mae_chunk"] = float(np.abs(per_tick["policy"])[chunks.valid].mean())
    row["gripper_mae_executed"] = float(
        np.abs(chunks.predicted[..., gripper] - chunks.recorded[..., gripper])[executed].mean()
    )
    return row


def assert_unseen(n_trained: int | None, held_out: list[int]) -> None:
    """Refuse held-out demos that the checkpoint trained on (it took demos 0..n_trained-1)."""
    if n_trained is None:
        raise ValueError("the run trained on every demo (n_demos is None); nothing is held out")
    seen = [i for i in held_out if i < n_trained]
    if seen:
        raise ValueError(f"held-out demos {seen} are among the {n_trained} the run trained on")
