"""The one path from a recorded camera frame to the policy's image input.

Hardware embodiments record full-resolution frames, and the policy sees
``observation.image_size``. A training cache and the policy server must reduce frames
identically, so both call :func:`policy_frame`. The masks and crops it applies come
from the embodiment spec, which every checkpoint carries, so a server built from a
checkpoint masks and crops exactly what training did.

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

    The camera's mask polygons are filled black in the recorded frame; with a crop
    polygon, everything outside it is filled black too and the frame is cut to its
    bounding box; then the frame is area-resampled to ``image_size``.
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
    polygons, keep = obs.masks_for(camera), obs.crop_for(camera)
    if (polygons or keep) and frame.shape[:2] != tuple(obs.mask_source_hw):
        raise ValueError(
            f"{camera}: frame is {frame.shape[:2]}, but the mask vertices and crop polygon "
            f"refer to {tuple(obs.mask_source_hw)} frames"
        )
    if polygons:
        frame = frame.copy()
        cv2.fillPoly(frame, [np.asarray(p, dtype=np.int32) for p in polygons], (0, 0, 0))
    if keep:
        inside = np.zeros(frame.shape[:2], np.uint8)
        cv2.fillPoly(inside, [np.asarray(keep, dtype=np.int32)], 1)
        frame = np.where(inside[..., None] > 0, frame, 0).astype(np.uint8)
        xs, ys = [x for x, _ in keep], [y for _, y in keep]
        frame = frame[min(ys): max(ys) + 1, min(xs): max(xs) + 1]
    height, width = obs.image_size
    if frame.shape[:2] != (height, width):
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(frame)
