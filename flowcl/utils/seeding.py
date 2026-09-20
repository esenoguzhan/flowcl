"""Deterministic, method-independent seed derivation.

Spec §8.3: rollout seeding is derived from ``(run_id, task, episode_idx)`` and is
independent of the method, so episode index *i* means the same initial state for
every method being compared. The derivation therefore deliberately has no method
argument — it is impossible to pass one.

We use BLAKE2b over the UTF-8 encoding of the tuple rather than Python's ``hash``,
which is salted per process by PYTHONHASHSEED and would silently break determinism
across runs.
"""

from __future__ import annotations

import hashlib
import os
import random

import numpy as np

# numpy legacy seeds and torch manual_seed accept [0, 2**32) reliably.
_SEED_MODULUS = 2**32


def derive_seed(run_id: str, task: str, episode_idx: int) -> int:
    """Derive a reproducible seed in ``[0, 2**32)``.

    Args:
        run_id: Identifier of the run (shared by all methods in a comparison when
            the comparison is meant to share initial states).
        task: Task key, e.g. ``"libero_object/pick_up_the_milk..."``.
        episode_idx: Index of the rollout episode.

    Returns:
        A seed that depends on exactly these three values and nothing else.

    Raises:
        TypeError: If ``episode_idx`` is not an int (a float index silently
            changing the seed is the kind of bug this module exists to prevent).
        ValueError: If ``episode_idx`` is negative.
    """
    if isinstance(episode_idx, bool) or not isinstance(episode_idx, (int, np.integer)):
        raise TypeError(f"episode_idx must be an int, got {type(episode_idx).__name__}")
    if episode_idx < 0:
        raise ValueError(f"episode_idx must be non-negative, got {episode_idx}")

    payload = "\x1f".join((run_id, task, str(int(episode_idx)))).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") % _SEED_MODULUS


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and Torch RNGs.

    Torch is imported lazily so that pure-data utilities can use this module
    without paying the torch import cost.
    """
    if not 0 <= seed < _SEED_MODULUS:
        raise ValueError(f"seed {seed} out of range [0, 2**32)")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
