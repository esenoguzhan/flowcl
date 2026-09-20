"""Per-embodiment normalization statistics, versioned and frozen.

Spec §3.3, quoted because it is the whole point of this module:

    Normalization stats are computed **once per embodiment over Task 1's data only**
    and frozen for the whole curriculum. Recomputing stats per task silently changes
    the target distribution and corrupts forgetting measurements. Assert this at
    every stage boundary.

So the API is deliberately awkward in one direction: :func:`compute_stats` records
which task it was fitted on, and :func:`assert_frozen` refuses stats whose
provenance does not match the curriculum's first task. There is no "refresh" path.

Actions are *not* normalised for LIBERO (§3.2: already in ``[-1, 1]``). The stats
object still carries action statistics, but with ``apply=False``, so the numbers are
recorded for provenance while the transform is the identity. That distinction is
explicit rather than implied by an absent field.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# Bump when the *meaning* of a field changes, so old artefacts cannot be silently
# reinterpreted by new code.
STATS_FORMAT_VERSION = 1

# Guards against dividing by a degenerate std (e.g. a proprioception channel that is
# constant across Task 1).
MIN_STD = 1e-6


@dataclass(frozen=True)
class FieldStats:
    """Mean/std/min/max for one tensor field."""

    mean: list[float]
    std: list[float]
    min: list[float]
    max: list[float]
    apply: bool

    def __post_init__(self) -> None:
        widths = {len(self.mean), len(self.std), len(self.min), len(self.max)}
        if len(widths) != 1:
            raise ValueError(
                f"FieldStats components have inconsistent widths: mean={len(self.mean)}, "
                f"std={len(self.std)}, min={len(self.min)}, max={len(self.max)}"
            )

    @property
    def dim(self) -> int:
        return len(self.mean)

    def mean_array(self) -> np.ndarray:
        return np.asarray(self.mean, dtype=np.float32)

    def std_array(self) -> np.ndarray:
        return np.maximum(np.asarray(self.std, dtype=np.float32), MIN_STD)


@dataclass(frozen=True)
class NormalizationStats:
    """Frozen normalization statistics with full provenance.

    Attributes:
        embodiment: Embodiment name these stats belong to.
        fitted_on_task_id: The single task the stats were computed from (§3.3).
        fitted_on_n_demos: How many demos of that task were used.
        n_steps: Total timesteps aggregated.
        state: Proprioception statistics; applied.
        action: Action statistics; recorded but not applied for LIBERO.
        version: :data:`STATS_FORMAT_VERSION`.
    """

    embodiment: str
    fitted_on_task_id: str
    fitted_on_n_demos: int
    n_steps: int
    state: FieldStats
    action: FieldStats
    version: int = STATS_FORMAT_VERSION
    extra: dict = field(default_factory=dict)

    # ---- provenance -----------------------------------------------------------

    def fingerprint(self) -> str:
        """Content hash over every number, for cheap equality in logs and asserts."""
        payload = json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        return hashlib.blake2b(payload, digest_size=16).hexdigest()

    def to_dict(self) -> dict:
        return {
            "embodiment": self.embodiment,
            "fitted_on_task_id": self.fitted_on_task_id,
            "fitted_on_n_demos": self.fitted_on_n_demos,
            "n_steps": self.n_steps,
            "state": asdict(self.state),
            "action": asdict(self.action),
            "version": self.version,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "NormalizationStats":
        version = payload.get("version")
        if version != STATS_FORMAT_VERSION:
            raise ValueError(
                f"Stats format version {version} != expected {STATS_FORMAT_VERSION}; "
                "refusing to reinterpret old statistics with new code."
            )
        return cls(
            embodiment=payload["embodiment"],
            fitted_on_task_id=payload["fitted_on_task_id"],
            fitted_on_n_demos=payload["fitted_on_n_demos"],
            n_steps=payload["n_steps"],
            state=FieldStats(**payload["state"]),
            action=FieldStats(**payload["action"]),
            version=version,
            extra=payload.get("extra", {}),
        )

    # ---- persistence ----------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        payload["fingerprint"] = self.fingerprint()
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "NormalizationStats":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Normalization stats not found: {path}")
        payload = json.loads(path.read_text())
        recorded = payload.pop("fingerprint", None)
        stats = cls.from_dict(payload)
        if recorded is not None and recorded != stats.fingerprint():
            raise ValueError(
                f"Stats file {path} is corrupt: recorded fingerprint {recorded} != "
                f"recomputed {stats.fingerprint()}"
            )
        return stats

    # ---- transforms -----------------------------------------------------------

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Standardise proprioception. Raises on a width mismatch."""
        if state.shape[-1] != self.state.dim:
            raise ValueError(
                f"state has width {state.shape[-1]} but stats were fitted for "
                f"{self.state.dim}"
            )
        if not self.state.apply:
            return state.astype(np.float32, copy=False)
        return (
            (state.astype(np.float32) - self.state.mean_array()) / self.state.std_array()
        ).astype(np.float32)

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        """Identity for LIBERO (§3.2); standardises only if ``action.apply``."""
        if action.shape[-1] != self.action.dim:
            raise ValueError(
                f"action has width {action.shape[-1]} but stats were fitted for "
                f"{self.action.dim}"
            )
        if not self.action.apply:
            return action.astype(np.float32, copy=False)
        return (
            (action.astype(np.float32) - self.action.mean_array())
            / self.action.std_array()
        ).astype(np.float32)

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`normalize_action`, for turning predictions into commands."""
        if action.shape[-1] != self.action.dim:
            raise ValueError(
                f"action has width {action.shape[-1]} but stats were fitted for "
                f"{self.action.dim}"
            )
        if not self.action.apply:
            return action.astype(np.float32, copy=False)
        return (
            action.astype(np.float32) * self.action.std_array()
            + self.action.mean_array()
        ).astype(np.float32)


def _field_stats(values: np.ndarray, apply: bool) -> FieldStats:
    values = values.astype(np.float64, copy=False)
    return FieldStats(
        mean=values.mean(axis=0).tolist(),
        std=values.std(axis=0).tolist(),
        min=values.min(axis=0).tolist(),
        max=values.max(axis=0).tolist(),
        apply=apply,
    )


def compute_stats(
    episodes,
    embodiment: str,
    task_id: str,
    normalize_actions: bool = False,
) -> NormalizationStats:
    """Fit statistics over the episodes of **one** task (§3.3).

    Args:
        episodes: Iterable of :class:`~flowcl.data.episode.Episode` from a single
            task. A mixed-task iterable is rejected, because §3.3's guarantee is
            about *which* data the stats come from.
        embodiment: Embodiment name.
        task_id: The task these stats are fitted on. Recorded in provenance and
            checked by :func:`assert_frozen`.
        normalize_actions: Leave False for LIBERO (§3.2).

    Returns:
        Frozen statistics carrying their own provenance.
    """
    episodes = list(episodes)
    if not episodes:
        raise ValueError("compute_stats received no episodes")

    task_ids = {ep.task_id for ep in episodes}
    if task_ids != {task_id}:
        raise ValueError(
            f"compute_stats is per-task by design (§3.3) but received episodes from "
            f"{sorted(task_ids)} while task_id={task_id!r}"
        )
    embodiments = {ep.embodiment for ep in episodes}
    if embodiments != {embodiment}:
        raise ValueError(
            f"episodes span embodiments {sorted(embodiments)} but embodiment="
            f"{embodiment!r}"
        )

    states = np.concatenate([ep.state for ep in episodes], axis=0)
    actions = np.concatenate([ep.action for ep in episodes], axis=0)

    return NormalizationStats(
        embodiment=embodiment,
        fitted_on_task_id=task_id,
        fitted_on_n_demos=len(episodes),
        n_steps=int(states.shape[0]),
        state=_field_stats(states, apply=True),
        action=_field_stats(actions, apply=normalize_actions),
    )


def assert_frozen(
    stats: NormalizationStats,
    embodiment: str,
    first_task_id: str,
    expected_fingerprint: str | None = None,
) -> None:
    """Assert the §3.3 invariant at a curriculum stage boundary.

    Call this at *every* stage boundary, before training on the next task. It is the
    guard that stops a later task from silently refitting the target distribution and
    thereby corrupting every forgetting number in the run.

    Args:
        stats: The stats object the trainer is about to use.
        embodiment: Embodiment expected for this curriculum.
        first_task_id: ``task_key`` of the curriculum's *first* task.
        expected_fingerprint: If given, the fingerprint recorded at stage 1. Catches
            in-place mutation of a stats object that still claims the right task.
    """
    if stats.version != STATS_FORMAT_VERSION:
        raise ValueError(
            f"stats version {stats.version} != {STATS_FORMAT_VERSION}"
        )
    if stats.embodiment != embodiment:
        raise ValueError(
            f"Normalization stats belong to embodiment {stats.embodiment!r} but the "
            f"curriculum runs {embodiment!r}. §3.1 requires re-initialised "
            "projections and fresh stats per embodiment, never reuse across them."
        )
    if stats.fitted_on_task_id != first_task_id:
        raise ValueError(
            f"§3.3 violation: normalization stats were fitted on "
            f"{stats.fitted_on_task_id!r} but the curriculum's first task is "
            f"{first_task_id!r}. Stats must be computed once over Task 1 and frozen "
            "for the whole curriculum; recomputing per task silently changes the "
            "target distribution and corrupts forgetting measurements."
        )
    if expected_fingerprint is not None and stats.fingerprint() != expected_fingerprint:
        raise ValueError(
            f"§3.3 violation: normalization stats changed mid-curriculum. "
            f"Fingerprint is {stats.fingerprint()} but stage 1 recorded "
            f"{expected_fingerprint}."
        )
