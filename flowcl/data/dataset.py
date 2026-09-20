"""Chunked action-prediction dataset.

Spec §3.3:

    Action chunk horizon ``H = 16`` (config). Dataset item =
    ``(obs_t, A_t in R^{H x D_action})``, right-padded at episode end with a validity
    mask; masked steps contribute zero loss.

Indexing: every timestep ``t`` of every episode is a sample, so the final samples of
an episode are the ones that need padding. Images stay ``uint8`` all the way to the
encoder — converting to float here would multiply the memory footprint by four for no
benefit, since the frozen backbone normalises anyway.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from flowcl.data.episode import Episode
from flowcl.data.spec import EmbodimentSpec
from flowcl.data.stats import NormalizationStats


@dataclass(frozen=True)
class SampleIndex:
    """Which episode and timestep a flat dataset index refers to."""

    episode_idx: int
    t: int


class ChunkedActionDataset(Dataset):
    """``(obs_t, A_t, mask_t)`` samples over a set of episodes from one task.

    Args:
        episodes: Episodes to index. All must share the embodiment.
        spec: Embodiment spec; shapes are asserted against it.
        stats: Frozen normalization statistics (§3.3). Required — there is no
            "unnormalised" mode, because a run that forgot to pass stats would
            silently train on a different target distribution.
        horizon: ``H``, the action chunk length.
    """

    def __init__(
        self,
        episodes: Iterable[Episode],
        spec: EmbodimentSpec,
        stats: NormalizationStats,
        horizon: int | None = None,
    ) -> None:
        self.episodes: list[Episode] = list(episodes)
        if not self.episodes:
            raise ValueError("ChunkedActionDataset requires at least one episode")

        self.spec = spec
        self.stats = stats
        self.horizon = int(horizon if horizon is not None else spec.action.chunk_horizon)
        if self.horizon <= 0:
            raise ValueError(f"horizon must be positive, got {self.horizon}")

        if stats.embodiment != spec.name:
            raise ValueError(
                f"stats are for embodiment {stats.embodiment!r} but spec is "
                f"{spec.name!r}"
            )

        for episode in self.episodes:
            spec.assert_compatible_episode(
                d_state=episode.d_state,
                d_action=episode.d_action,
                cameras=episode.cameras,
            )

        self._index: list[SampleIndex] = [
            SampleIndex(episode_idx=i, t=t)
            for i, episode in enumerate(self.episodes)
            for t in range(episode.length)
        ]

    # ---- introspection --------------------------------------------------------

    def __len__(self) -> int:
        return len(self._index)

    @property
    def n_episodes(self) -> int:
        return len(self.episodes)

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(sorted({ep.task_id for ep in self.episodes}))

    def sample_index(self, i: int) -> SampleIndex:
        return self._index[i]

    def n_padded_samples(self) -> int:
        """How many samples need right-padding, i.e. have ``t + H > T``.

        Exposed because it is a useful sanity number: with ``H = 16`` it should be
        ``15`` per episode (``H - 1``), independent of episode length.
        """
        return sum(
            1
            for idx in self._index
            if idx.t + self.horizon > self.episodes[idx.episode_idx].length
        )

    # ---- sample construction --------------------------------------------------

    def __getitem__(self, i: int) -> dict:
        idx = self._index[i]
        episode = self.episodes[idx.episode_idx]
        t = idx.t

        # Right-pad the action chunk and build the validity mask.
        end = min(t + self.horizon, episode.length)
        n_valid = end - t
        chunk = np.zeros((self.horizon, episode.d_action), dtype=np.float32)
        chunk[:n_valid] = self.stats.normalize_action(episode.action[t:end])
        mask = np.zeros((self.horizon,), dtype=np.float32)
        mask[:n_valid] = 1.0

        state = self.stats.normalize_state(episode.state[t])

        images = {
            camera: torch.from_numpy(
                np.ascontiguousarray(episode.images[camera][t])
            )
            for camera in self.spec.cameras
        }

        return {
            "images": images,
            "state": torch.from_numpy(state),
            "actions": torch.from_numpy(chunk),
            "action_mask": torch.from_numpy(mask),
            "language": episode.language,
            # Bookkeeping only; §0 forbids task identity as a policy input.
            "task_id": episode.task_id,
            "episode_idx": idx.episode_idx,
            "t": t,
        }


def collate_chunks(batch: Sequence[dict]) -> dict:
    """Collate :class:`ChunkedActionDataset` samples.

    The default collate cannot handle the nested image dict plus the string fields,
    and strings must survive as a list so the language encoder can hit its per-string
    cache (§4.1).
    """
    if not batch:
        raise ValueError("collate_chunks received an empty batch")

    cameras = list(batch[0]["images"])
    for sample in batch:
        if list(sample["images"]) != cameras:
            raise ValueError(
                f"inconsistent cameras within a batch: {list(sample['images'])} vs "
                f"{cameras}"
            )

    return {
        "images": {
            camera: torch.stack([sample["images"][camera] for sample in batch])
            for camera in cameras
        },
        "state": torch.stack([sample["state"] for sample in batch]),
        "actions": torch.stack([sample["actions"] for sample in batch]),
        "action_mask": torch.stack([sample["action_mask"] for sample in batch]),
        "language": [sample["language"] for sample in batch],
        "task_id": [sample["task_id"] for sample in batch],
        "episode_idx": torch.tensor(
            [sample["episode_idx"] for sample in batch], dtype=torch.long
        ),
        "t": torch.tensor([sample["t"] for sample in batch], dtype=torch.long),
    }
