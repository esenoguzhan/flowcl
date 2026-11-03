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
        image_masks: ``(camera, polygons)`` pairs. Each polygon is a tuple of
            ``(x, y)`` pixel vertices in the camera's recorded frame, filled black
            before resizing by :func:`flowcl.data.images.policy_frame`. For hardware
            cameras that see the teleoperator: the leader arm's pose is the action,
            and it sits still when the policy runs alone. Part of the spec so every
            checkpoint carries the masks it was trained with.
        mask_source_hw: ``(H, W)`` of the recorded frames the mask vertices and crop
            boxes refer to. Required exactly when ``image_masks`` or ``image_crops``
            is set.
        image_crops: ``(camera, polygon)`` pairs: the working area of the recorded
            frame, ``(x, y)`` pixel vertices. After masking, everything outside the
            polygon is filled black and the frame is cut to the polygon's bounding
            box, then resized. For a fixed scene camera, it spends the policy's pixels
            on the workspace rather than the room.
    """

    cameras: tuple[str, ...]
    image_size: tuple[int, int]
    d_state: int
    state_keys: tuple[tuple[str, int], ...] = ()
    image_masks: tuple[tuple[str, tuple[tuple[tuple[int, int], ...], ...]], ...] = ()
    mask_source_hw: tuple[int, int] | None = None
    image_crops: tuple[tuple[str, tuple[tuple[int, int], ...]], ...] = ()

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
        self._validate_masks()

    def _validate_masks(self) -> None:
        if not self.image_masks and not self.image_crops:
            if self.mask_source_hw is not None:
                raise ValueError("mask_source_hw is set but image_masks is empty")
            return
        if self.mask_source_hw is None or len(self.mask_source_hw) != 2 or any(
            v <= 0 for v in self.mask_source_hw
        ):
            raise ValueError(
                f"image_masks and image_crops need a positive mask_source_hw (H, W), "
                f"got {self.mask_source_hw}"
            )
        self._validate_crops()
        height, width = self.mask_source_hw
        named = [camera for camera, _ in self.image_masks]
        if len(set(named)) != len(named):
            raise ValueError(f"image_masks names a camera twice: {named}")
        for camera, polygons in self.image_masks:
            if camera not in self.cameras:
                raise ValueError(
                    f"image_masks camera {camera!r} is not one of {self.cameras}"
                )
            if not polygons:
                raise ValueError(f"image_masks[{camera!r}] has no polygons")
            for polygon in polygons:
                if len(polygon) < 3:
                    raise ValueError(
                        f"image_masks[{camera!r}]: a polygon needs >= 3 vertices, got {polygon}"
                    )
                for x, y in polygon:
                    if not (0 <= x <= width and 0 <= y <= height):
                        raise ValueError(
                            f"image_masks[{camera!r}]: vertex {(x, y)} lies outside the "
                            f"{width}x{height} recorded frame"
                        )

    def _validate_crops(self) -> None:
        height, width = self.mask_source_hw
        named = [camera for camera, _ in self.image_crops]
        if len(set(named)) != len(named):
            raise ValueError(f"image_crops names a camera twice: {named}")
        for camera, polygon in self.image_crops:
            if camera not in self.cameras:
                raise ValueError(f"image_crops camera {camera!r} is not one of {self.cameras}")
            if len(polygon) < 3:
                raise ValueError(f"image_crops[{camera!r}]: a polygon needs >= 3 vertices, got {polygon}")
            for x, y in polygon:
                if not (0 <= x <= width and 0 <= y <= height):
                    raise ValueError(
                        f"image_crops[{camera!r}]: vertex {(x, y)} lies outside the "
                        f"{width}x{height} recorded frame"
                    )
            xs, ys = [x for x, _ in polygon], [y for _, y in polygon]
            if min(xs) == max(xs) or min(ys) == max(ys):
                raise ValueError(f"image_crops[{camera!r}]: polygon {polygon} is empty")

    def crop_for(self, camera: str) -> tuple[tuple[int, int], ...]:
        """The polygon of ``camera``'s frames kept (empty when uncropped)."""
        return dict(self.image_crops).get(camera, ())

    def masks_for(self, camera: str) -> tuple[tuple[tuple[int, int], ...], ...]:
        """The polygons masked out of ``camera``'s frames (empty when unmasked)."""
        return dict(self.image_masks).get(camera, ())

    def flat_names(self) -> tuple[str, ...]:
        """One name per state dimension, from ``state_keys``."""
        return flatten_named_widths(
            self.state_keys, dim=self.d_state, fallback_prefix="state"
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

    def flat_names(self) -> tuple[str, ...]:
        """One name per action dimension, from ``component_keys``."""
        return flatten_named_widths(
            self.component_keys, dim=self.d_action, fallback_prefix="action"
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

    # ---- serialisation --------------------------------------------------------

    def to_dict(self) -> dict:
        """Plain-data form, for embedding in checkpoints and run artifacts.

        Checkpoints carry the spec rather than a config *name* so that evaluating an
        old checkpoint cannot silently pick up an edited YAML. The image masks appear
        and crops appear only when set, so unmasked specs serialise exactly as before.
        """
        observation = {
            "cameras": list(self.observation.cameras),
            "image_size": list(self.observation.image_size),
            "d_state": self.observation.d_state,
            "state_keys": [[k, w] for k, w in self.observation.state_keys],
        }
        if self.observation.image_masks:
            observation["image_masks"] = {
                camera: [[[x, y] for x, y in polygon] for polygon in polygons]
                for camera, polygons in self.observation.image_masks
            }
        if self.observation.image_crops:
            observation["image_crops"] = {
                camera: [[x, y] for x, y in polygon] for camera, polygon in self.observation.image_crops
            }
        if self.observation.image_masks or self.observation.image_crops:
            observation["mask_source_hw"] = list(self.observation.mask_source_hw)
        return {
            "name": self.name,
            "notes": self.notes,
            "observation": observation,
            "action": {
                "d_action": self.action.d_action,
                "control_mode": self.action.control_mode,
                "control_rate_hz": self.action.control_rate_hz,
                "chunk_horizon": self.action.chunk_horizon,
                "execute_k": self.action.execute_k,
                "already_normalized": self.action.already_normalized,
                "component_keys": [[k, w] for k, w in self.action.component_keys],
            },
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "EmbodimentSpec":
        """Inverse of :meth:`to_dict`. Validation runs via the dataclasses."""
        obs = payload["observation"]
        act = payload["action"]
        return cls(
            name=payload["name"],
            notes=payload.get("notes", ""),
            observation=ObservationSpec(
                cameras=tuple(obs["cameras"]),
                image_size=tuple(obs["image_size"]),
                d_state=int(obs["d_state"]),
                state_keys=tuple((k, int(w)) for k, w in obs.get("state_keys", [])),
                image_masks=parse_image_masks(obs.get("image_masks")),
                mask_source_hw=parse_hw(obs.get("mask_source_hw")),
                image_crops=parse_image_crops(obs.get("image_crops")),
            ),
            action=ActionSpec(
                d_action=int(act["d_action"]),
                control_mode=act["control_mode"],
                control_rate_hz=float(act["control_rate_hz"]),
                chunk_horizon=int(act["chunk_horizon"]),
                execute_k=int(act["execute_k"]),
                already_normalized=bool(act.get("already_normalized", False)),
                component_keys=tuple(
                    (k, int(w)) for k, w in act.get("component_keys", [])
                ),
            ),
        )

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


def parse_image_masks(
    raw: dict | None,
) -> tuple[tuple[str, tuple[tuple[tuple[int, int], ...], ...]], ...]:
    """``{camera: [[[x, y], ...], ...]}`` (YAML or ``to_dict`` form) -> spec tuples."""
    if not raw:
        return ()
    return tuple(
        (
            str(camera),
            tuple(tuple((int(x), int(y)) for x, y in polygon) for polygon in polygons),
        )
        for camera, polygons in raw.items()
    )


def parse_image_crops(raw: dict | None) -> tuple[tuple[str, tuple[tuple[int, int], ...]], ...]:
    """``{camera: [[x, y], ...]}`` (YAML or ``to_dict`` form) -> spec tuples."""
    if not raw:
        return ()
    return tuple((str(camera), tuple((int(x), int(y)) for x, y in polygon)) for camera, polygon in raw.items())


def parse_hw(raw) -> tuple[int, int] | None:
    return None if raw is None else tuple(int(v) for v in raw)


def flatten_named_widths(
    keys: tuple[tuple[str, int], ...],
    *,
    dim: int,
    fallback_prefix: str,
) -> tuple[str, ...]:
    """Expand ``(name, width)`` pairs to ``dim`` unique column names.

    A width-1 key keeps its name; wider keys become ``name_0`` … ``name_{w-1}``.
    Empty ``keys`` falls back to ``{fallback_prefix}_i`` so a spec that has not
    declared names still produces a length-``dim`` listing rather than silently
    inventing an ``ee_pos`` layout.
    """
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")
    if not keys:
        return tuple(f"{fallback_prefix}_{i}" for i in range(dim))
    names: list[str] = []
    for name, width in keys:
        if width <= 0:
            raise ValueError(f"component {name!r} has non-positive width {width}")
        if width == 1:
            names.append(name)
        else:
            names.extend(f"{name}_{i}" for i in range(width))
    if len(names) != dim:
        raise ValueError(
            f"flattened names have length {len(names)} but dim is {dim}; keys={keys}"
        )
    return tuple(names)


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
