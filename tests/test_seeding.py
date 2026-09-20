"""Spec §8.3: rollout seeding is method-independent and reproducible."""

from __future__ import annotations

import pytest

from flowcl.utils.seeding import derive_seed


def test_deterministic_across_calls():
    a = derive_seed("run-a", "libero_object/pick_up_the_milk", 7)
    b = derive_seed("run-a", "libero_object/pick_up_the_milk", 7)
    assert a == b


def test_no_method_argument_exists():
    """The signature cannot express a method, so seeds cannot depend on one.

    This is the structural guarantee behind §8.3: episode index *i* means the same
    initial state for every method under comparison.
    """
    import inspect

    params = list(inspect.signature(derive_seed).parameters)
    assert params == ["run_id", "task", "episode_idx"]


def test_same_episode_same_seed_across_method_run_dirs():
    """Two methods sharing a comparison run_id must agree episode by episode."""
    task = "libero_spatial/pick_up_the_black_bowl_next_to_the_plate"
    for episode in range(50):
        seq_ft = derive_seed("cmp-2026-01", task, episode)
        gpm = derive_seed("cmp-2026-01", task, episode)
        assert seq_ft == gpm


def test_distinct_inputs_give_distinct_seeds():
    base = derive_seed("run", "task", 0)
    assert derive_seed("run2", "task", 0) != base
    assert derive_seed("run", "task2", 0) != base
    assert derive_seed("run", "task", 1) != base


def test_seed_in_uint32_range():
    for episode in range(200):
        seed = derive_seed("r", "t", episode)
        assert 0 <= seed < 2**32


def test_known_answer_is_stable():
    """Pin one value so a future refactor of the hash cannot silently move seeds.

    If this test fails, every previously collected rollout used different initial
    states than a rerun would; that is a result-invalidating change, not a cosmetic
    one.
    """
    assert derive_seed("flowcl-test", "libero_object/t0", 0) == 2892175405


def test_rejects_non_integer_episode_idx():
    with pytest.raises(TypeError, match="episode_idx must be an int"):
        derive_seed("r", "t", 1.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        derive_seed("r", "t", True)  # type: ignore[arg-type]


def test_rejects_negative_episode_idx():
    with pytest.raises(ValueError, match="non-negative"):
        derive_seed("r", "t", -1)
