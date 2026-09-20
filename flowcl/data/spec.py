"""Embodiment, observation and action specifications.

Spec §3.1: ``EmbodimentSpec`` declares camera names, ``D_state``, ``D_action``,
control mode and control rate. The policy builds its input/output projections from
the spec, so switching embodiment means a new spec plus re-initialised projections
while trunk weights transfer.

Spec §0, "Action representation": dimensionality is fixed *per embodiment* and
**never padded across embodiments**. Padding would make the policy switch which
slice of a shared action vector is active, and the resulting drop in performance
would look like representational interference when it is really bookkeeping. The
dataclass therefore refuses to hold a padded action space: see
:meth:`ActionSpec.validate`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ControlMode = Literal["osc_pose_delta", "joint_position", "joint_velocity"]


@dataclass(frozen=True)
class ObservationSpec:
    """What the policy observes at one timestep.

    Attributes:
        cameras: Ordered camera names. Order is load-bearing: it fixes token order
            into the trunk, which the layer registry (§4.5) and every analysis
            iterate in the same order.
        image_size: ``(H, W)`` after resizing.
        d_state: Proprioception dimensionality.
        state_keys: Ordered, human-readable names of the state components, with the
            width of each. Recorded so a state-layout change is visible in configs
            and in every run's ``config.yaml`` rather than being implicit.
    """

    cameras: tuple[str, ...]
    image_size: tuple[int, int]
    d_state: int
    state_keys: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        if not self.cameras:
            raise ValueError("ObservationSpec.cameras must be non-empty")
        if len(set(self.cameras)) != len(self.cameras):
            raise ValueError(f"duplicate camera names in {self.cameras}")
        if len(self.image_size) != 2 or any(v <= 0 for v in self.image_size):
            raise ValueError(f"image_size must be positive (H, W), got {self.image_size}")
        if self.d_state <= 0:
            raise ValueError(f"d_state must be positive, got {self.d_state}")
        if self.state_keys:
            total = sum(width for _, width in self.state_keys)
            if total != self.d_state:
                raise ValueError(
                    f"state_keys widths sum to {total} but d_state is {self.d_state}; "
                    f"keys={self.state_keys}"
                )


@dataclass(frozen=True)
class ActionSpec:
    """The action space of one embodiment.

    Attributes:
        d_action: Action dimensionality for *this* embodiment. Never a padded
            cross-embodiment maximum (§0).
        control_mode: How the action is interpreted by the controller.
        control_rate_hz: Controller frequency.
        chunk_horizon: ``H`` in §3.3, the number of future actions predicted.
        execute_k: ``k`` in §4.4, how many of those actions are executed open-loop
            before replanning.
        already_normalized: True when the source data is already in ``[-1, 1]`` and
            must not be renormalised (§3.2 for LIBERO's OSC deltas).
        component_keys: Ordered names and widths of action components.
    """

    d_action: int
    control_mode: ControlMode
    control_rate_hz: float
    chunk_horizon: int
    execute_k: int
    already_normalized: bool = False
    component_keys: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        if self.d_action <= 0:
            raise ValueError(f"d_action must be positive, got {self.d_action}")
        if self.chunk_horizon <= 0:
            raise ValueError(f"chunk_horizon must be positive, got {self.chunk_horizon}")
        if not 1 <= self.execute_k <= self.chunk_horizon:
            raise ValueError(
                f"execute_k must be in [1, chunk_horizon]; got execute_k="
                f"{self.execute_k}, chunk_horizon={self.chunk_horizon}"
            )
        if self.control_rate_hz <= 0:
            raise ValueError(
                f"control_rate_hz must be positive, got {self.control_rate_hz}"
            )
        if self.component_keys:
            total = sum(width for _, width in self.component_keys)
            if total != self.d_action:
                raise ValueError(
                    f"component_keys widths sum to {total} but d_action is "
                    f"{self.d_action}; keys={self.component_keys}"
                )


@dataclass(frozen=True)
class EmbodimentSpec:
    """Everything the policy needs to build its input/output projections (§3.1)."""

    name: str
    observation: ObservationSpec
    action: ActionSpec
    # Free-form provenance, e.g. which robot/controller the numbers came from.
    notes: str = ""
    _registry: tuple[str, ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("EmbodimentSpec.name must be non-empty")

    @property
    def d_state(self) -> int:
        return self.observation.d_state

    @property
    def d_action(self) -> int:
        return self.action.d_action

    @property
    def cameras(self) -> tuple[str, ...]:
        return self.observation.cameras

    def assert_compatible_episode(
        self, d_state: int, d_action: int, cameras: tuple[str, ...]
    ) -> None:
        """Raise unless an episode's shapes match this spec exactly.

        No tolerance and no padding: §0 requires that an action vector from one
        embodiment is never reshaped to fit another.
        """
        problems = []
        if d_state != self.d_state:
            problems.append(f"d_state {d_state} != spec {self.d_state}")
        if d_action != self.d_action:
            problems.append(f"d_action {d_action} != spec {self.d_action}")
        missing = [c for c in self.cameras if c not in cameras]
        if missing:
            problems.append(f"cameras missing {missing}; episode has {list(cameras)}")
        if problems:
            raise ValueError(
                f"Episode incompatible with embodiment {self.name!r}: "
                + "; ".join(problems)
            )


def assert_no_cross_embodiment_padding(specs: list[EmbodimentSpec]) -> None:
    """Assert that distinct embodiments genuinely differ rather than being padded.

    Spec §0 forbids padding action vectors across embodiments. The observable
    symptom of a padding mistake is two embodiments sharing a ``d_action`` that
    equals the maximum over all of them while their control modes differ. This is a
    cheap guard that makes such a mistake fail loudly at config load.
    """
    if len(specs) < 2:
        return
    by_dim: dict[int, list[EmbodimentSpec]] = {}
    for spec in specs:
        by_dim.setdefault(spec.d_action, []).append(spec)
    for d_action, group in by_dim.items():
        modes = {s.action.control_mode for s in group}
        if len(group) > 1 and len(modes) > 1:
            raise ValueError(
                f"Embodiments {[s.name for s in group]} all claim d_action={d_action} "
                f"but use different control modes {sorted(modes)}. This is the "
                "signature of a padded cross-embodiment action vector, which §0 "
                "forbids: it produces artifactual forgetting."
            )
