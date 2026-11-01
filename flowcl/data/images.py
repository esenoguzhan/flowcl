"""The one path from a recorded camera frame to the policy's image input.

Hardware embodiments record full-resolution frames, and the policy sees
``observation.image_size``. A training cache and the policy server must reduce frames
identically, so both call :func:`policy_frame`. The masks it applies come from the
embodiment spec, which every checkpoint carries, so a server built from a checkpoint
masks exactly what training masked.

LIBERO demos are stored at the policy size with no masks, so they never pass through
here.
"""

from __future__ import annotations

import numpy as np

from flowcl.data.spec import EmbodimentSpec, ObservationSpec


def policy_frame(
    rgb: np.ndarray, camera: str, spec: EmbodimentSpec | ObservationSpec
) -> np.ndarray:
    """A recorded RGB frame -> the policy's ``(H, W, 3)`` uint8 RGB input.

    The camera's mask polygons are filled black in the recorded frame, then the frame
    is area-resampled to ``image_size``.
    """
    import cv2

    obs = spec.observation if isinstance(spec, EmbodimentSpec) else spec
    if camera not in obs.cameras:
        raise ValueError(f"unknown camera {camera!r}; the spec has {obs.cameras}")
    frame = np.asarray(rgb)
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError(
            f"{camera}: expected (H, W, 3) uint8 RGB, got {frame.shape} {frame.dtype}"
        )
    polygons = obs.masks_for(camera)
    if polygons:
        if frame.shape[:2] != tuple(obs.mask_source_hw):
            raise ValueError(
                f"{camera}: frame is {frame.shape[:2]}, but the mask vertices refer to "
                f"{tuple(obs.mask_source_hw)} frames"
            )
        frame = frame.copy()
        cv2.fillPoly(frame, [np.asarray(p, dtype=np.int32) for p in polygons], (0, 0, 0))
    height, width = obs.image_size
    if frame.shape[:2] != (height, width):
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(frame)
