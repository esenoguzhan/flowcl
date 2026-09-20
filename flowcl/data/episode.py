"""The canonical episode container.

Spec §3.1: both LIBERO and AgileX are converted to this one format, and the policy
code never sees a raw LIBERO HDF5 or a raw teleop log. The field list is exactly as
specified; validation is added because a silently mis-shaped episode corrupts
normalization statistics and every downstream measurement.

``task_id`` is bookkeeping/analysis only and is **never** a policy input (§0: task
identity is not used at inference).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Episode:
    """One demonstration, in canonical form.

    Attributes:
        images: ``camera_name -> (T, H, W, 3) uint8``.
        state: ``(T, D_state) float32`` proprioception.
        action: ``(T, D_action) float32``.
        language: The natural-language instruction.
        task_id: Bookkeeping/analysis only, never a policy input.
        embodiment: ``"libero_franka"`` or ``"agilex_dual"``.
        meta: Source file, demo index, success flag, and anything else needed to
            trace a sample back to its origin.
    """

    images: dict[str, np.ndarray]
    state: np.ndarray
    action: np.ndarray
    language: str
    task_id: str
    embodiment: str
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    @property
    def length(self) -> int:
        """Number of timesteps ``T``."""
        return int(self.action.shape[0])

    @property
    def d_state(self) -> int:
        return int(self.state.shape[1])

    @property
    def d_action(self) -> int:
        return int(self.action.shape[1])

    @property
    def cameras(self) -> tuple[str, ...]:
        return tuple(self.images)

    def validate(self) -> None:
        """Raise on any shape, dtype or length inconsistency.

        Every message names the offending values, per the §11 "fail loudly" rule.
        """
        if not isinstance(self.images, dict) or not self.images:
            raise ValueError(
                f"Episode.images must be a non-empty dict, got {type(self.images)}"
            )
        if self.action.ndim != 2:
            raise ValueError(f"action must be (T, D_action), got {self.action.shape}")
        if self.state.ndim != 2:
            raise ValueError(f"state must be (T, D_state), got {self.state.shape}")

        n_steps = self.action.shape[0]
        if n_steps == 0:
            raise ValueError("Episode has zero timesteps")
        if self.state.shape[0] != n_steps:
            raise ValueError(
                f"state has {self.state.shape[0]} steps but action has {n_steps}"
            )

        if self.action.dtype != np.float32:
            raise TypeError(f"action must be float32, got {self.action.dtype}")
        if self.state.dtype != np.float32:
            raise TypeError(f"state must be float32, got {self.state.dtype}")

        for name, frames in self.images.items():
            if frames.ndim != 4 or frames.shape[-1] != 3:
                raise ValueError(
                    f"images[{name!r}] must be (T, H, W, 3), got {frames.shape}"
                )
            if frames.shape[0] != n_steps:
                raise ValueError(
                    f"images[{name!r}] has {frames.shape[0]} frames but action has "
                    f"{n_steps} steps"
                )
            if frames.dtype != np.uint8:
                raise TypeError(
                    f"images[{name!r}] must be uint8, got {frames.dtype}"
                )

        if not self.language:
            raise ValueError(f"Episode {self.task_id!r} has an empty language string")
        if not np.isfinite(self.action).all():
            bad = np.argwhere(~np.isfinite(self.action))
            raise ValueError(
                f"action contains non-finite values at indices {bad[:5].tolist()}"
            )
        if not np.isfinite(self.state).all():
            bad = np.argwhere(~np.isfinite(self.state))
            raise ValueError(
                f"state contains non-finite values at indices {bad[:5].tolist()}"
            )

    def summary(self) -> str:
        return (
            f"Episode(task_id={self.task_id!r}, embodiment={self.embodiment!r}, "
            f"T={self.length}, d_state={self.d_state}, d_action={self.d_action}, "
            f"cameras={list(self.cameras)})"
        )
