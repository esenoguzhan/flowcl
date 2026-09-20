"""Normalization statistics are fitted on Task 1 only and frozen (§3.3)."""

from __future__ import annotations

import numpy as np
import pytest

from flowcl.data.episode import Episode
from flowcl.data.stats import (
    NormalizationStats,
    assert_frozen,
    compute_stats,
)


def make_episode(task_id: str, value: float, n_steps: int = 10) -> Episode:
    return Episode(
        images={"cam": np.zeros((n_steps, 2, 2, 3), dtype=np.uint8)},
        state=np.full((n_steps, 4), value, dtype=np.float32),
        action=np.full((n_steps, 7), value * 0.1, dtype=np.float32),
        language="instruction",
        task_id=task_id,
        embodiment="libero_franka",
    )


def fit(task_id: str = "t1", value: float = 1.0) -> NormalizationStats:
    episodes = [
        make_episode(task_id, value),
        make_episode(task_id, value + 1.0),
    ]
    return compute_stats(episodes, embodiment="libero_franka", task_id=task_id)


def test_known_answer_mean_and_std():
    """Two constant episodes at 1.0 and 2.0 -> mean 1.5, std 0.5 in every channel."""
    stats = fit(value=1.0)
    np.testing.assert_allclose(stats.state.mean, [1.5] * 4)
    np.testing.assert_allclose(stats.state.std, [0.5] * 4)
    assert stats.n_steps == 20
    assert stats.fitted_on_n_demos == 2


def test_records_provenance():
    stats = fit(task_id="libero_spatial/task_a")
    assert stats.fitted_on_task_id == "libero_spatial/task_a"
    assert stats.embodiment == "libero_franka"


def test_compute_stats_refuses_mixed_tasks():
    """§3.3 is a claim about *which* data the stats come from, so mixing is an error."""
    episodes = [make_episode("t1", 1.0), make_episode("t2", 2.0)]
    with pytest.raises(ValueError, match="per-task by design"):
        compute_stats(episodes, embodiment="libero_franka", task_id="t1")


def test_compute_stats_refuses_mixed_embodiments():
    a = make_episode("t1", 1.0)
    b = make_episode("t1", 2.0)
    b.embodiment = "agilex_dual"
    with pytest.raises(ValueError, match="span embodiments"):
        compute_stats([a, b], embodiment="libero_franka", task_id="t1")


def test_compute_stats_rejects_empty():
    with pytest.raises(ValueError, match="no episodes"):
        compute_stats([], embodiment="libero_franka", task_id="t1")


# ---- the §3.3 stage-boundary assertion ---------------------------------------


def test_assert_frozen_passes_for_first_task_stats():
    stats = fit(task_id="t1")
    assert_frozen(stats, embodiment="libero_franka", first_task_id="t1")


def test_assert_frozen_fires_when_stats_refitted_on_a_later_task():
    """The exact failure §3.3 warns about: stats recomputed at stage 2."""
    refitted = fit(task_id="t2")
    with pytest.raises(ValueError, match="§3.3 violation"):
        assert_frozen(refitted, embodiment="libero_franka", first_task_id="t1")


def test_assert_frozen_fires_on_embodiment_mismatch():
    stats = fit(task_id="t1")
    with pytest.raises(ValueError, match="belong to embodiment"):
        assert_frozen(stats, embodiment="agilex_dual", first_task_id="t1")


def test_assert_frozen_fires_when_numbers_change_mid_curriculum():
    """Catches mutation of a stats object that still claims the right task."""
    stage1 = fit(task_id="t1", value=1.0)
    fingerprint = stage1.fingerprint()

    drifted = fit(task_id="t1", value=5.0)
    assert drifted.fitted_on_task_id == "t1"  # provenance still looks fine
    with pytest.raises(ValueError, match="changed mid-curriculum"):
        assert_frozen(
            drifted,
            embodiment="libero_franka",
            first_task_id="t1",
            expected_fingerprint=fingerprint,
        )


def test_fingerprint_is_stable_and_sensitive():
    a = fit(task_id="t1", value=1.0)
    b = fit(task_id="t1", value=1.0)
    c = fit(task_id="t1", value=2.0)
    assert a.fingerprint() == b.fingerprint()
    assert a.fingerprint() != c.fingerprint()


# ---- transforms ---------------------------------------------------------------


def test_actions_are_identity_when_already_normalized():
    """§3.2: LIBERO actions must not be renormalised."""
    stats = fit()
    assert stats.action.apply is False
    actions = np.random.default_rng(0).uniform(-1, 1, size=(5, 7)).astype(np.float32)
    np.testing.assert_array_equal(stats.normalize_action(actions), actions)
    np.testing.assert_array_equal(stats.denormalize_action(actions), actions)


def test_action_normalization_round_trips_when_enabled():
    episodes = [make_episode("t1", 1.0), make_episode("t1", 3.0)]
    stats = compute_stats(
        episodes, embodiment="libero_franka", task_id="t1", normalize_actions=True
    )
    actions = episodes[0].action
    restored = stats.denormalize_action(stats.normalize_action(actions))
    np.testing.assert_allclose(restored, actions, rtol=1e-5, atol=1e-6)


def test_state_normalization_produces_zero_mean():
    episodes = [make_episode("t1", 1.0), make_episode("t1", 3.0)]
    stats = compute_stats(episodes, embodiment="libero_franka", task_id="t1")
    all_states = np.concatenate([e.state for e in episodes], axis=0)
    normalized = stats.normalize_state(all_states)
    np.testing.assert_allclose(normalized.mean(axis=0), np.zeros(4), atol=1e-5)
    np.testing.assert_allclose(normalized.std(axis=0), np.ones(4), atol=1e-5)


def test_constant_channel_does_not_divide_by_zero():
    """A proprioception channel constant across Task 1 must not produce inf/nan."""
    episodes = [make_episode("t1", 2.0), make_episode("t1", 2.0)]
    stats = compute_stats(episodes, embodiment="libero_franka", task_id="t1")
    out = stats.normalize_state(episodes[0].state)
    assert np.isfinite(out).all()


def test_normalize_rejects_width_mismatch():
    stats = fit()
    with pytest.raises(ValueError, match="width 3 but stats were fitted for 4"):
        stats.normalize_state(np.zeros((2, 3), dtype=np.float32))


# ---- persistence --------------------------------------------------------------


def test_save_load_round_trip(tmp_path):
    stats = fit(task_id="libero_spatial/task_a")
    path = stats.save(tmp_path / "stats.json")
    loaded = NormalizationStats.load(path)
    assert loaded.fingerprint() == stats.fingerprint()
    assert loaded.fitted_on_task_id == "libero_spatial/task_a"


def test_load_detects_tampering(tmp_path):
    import json

    stats = fit()
    path = stats.save(tmp_path / "stats.json")
    payload = json.loads(path.read_text())
    payload["state"]["mean"][0] += 1.0  # edit numbers, keep the old fingerprint
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="corrupt"):
        NormalizationStats.load(path)


def test_load_rejects_foreign_version(tmp_path):
    import json

    stats = fit()
    path = stats.save(tmp_path / "stats.json")
    payload = json.loads(path.read_text())
    payload.pop("fingerprint")
    payload["version"] = 999
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="version 999"):
        NormalizationStats.load(path)
