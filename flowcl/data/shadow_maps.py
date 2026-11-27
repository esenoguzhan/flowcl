"""Real shadow maps for a fixed camera: how sunlight changes the light on the table, as a ratio.

The front camera of the second Piper table does not move, so a frame of the table in sun divided
by the demos' frame in their own (even) light gives a per-pixel, per-channel light ratio that can
be laid over any demo frame of that camera (flowcl.data.augment.shadow). Built by
scripts/hw/piper_shadow_maps.py:

* the **reference** is the per-pixel median of the training demos' first frames (arm at home, the
  objects in different places, so the median is the empty table);
* the **support** is where that reference is bright (the table and the wall behind it), outside the
  operator mask, eroded so edges of the arm and the table do not count;
* objects in the sunny frame (saturated pixels: the grape, the plate) are left out of the support;
* inside the table's outline (the convex hull of the support), the left-out parts (the arm at home,
  objects) are filled smoothly from the surrounding ratio (normalized convolution: a weighted blur,
  no texture); outside it the ratio is 1, with a feathered
  edge, so neither the arm's silhouette nor the outline leaves a hard edge in a map;
* colour: a brightness ratio, with each channel held within 0.85-1.15 of it (sun is warmer than the
  demos' lamp, but blotches of colour are not light);
* the ratio is clipped to [0.25, 2.0] and area-resampled to the policy's image size, as
  flowcl.data.images.policy_frame resamples frames.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

RATIO_CLIP = (0.25, 2.0)
BRIGHT = 110          # reference grey above this is table or wall
ERODE_PX = 15         # the support shrinks by this much, at 640 x 480
OBJECT_SATURATION = 80
OBJECT_DILATE_PX = 20
BLUR_SIGMA = 1.5      # before dividing: JPEG and video noise
FEATHER_PX = 15       # the table outline fades to ratio 1 over about this
CHROMA_CLIP = (0.85, 1.15)   # a channel's ratio stays within this factor of the brightness ratio
FILL_SIGMAS = (8.0, 25.0, 80.0)  # px; the finest with enough valid neighbours fills a hole


def reference_frame(frames: list[np.ndarray]) -> np.ndarray:
    """Per-pixel median of ``(H, W, 3)`` uint8 frames, float32."""
    return np.median(np.stack(frames).astype(np.float32), axis=0)


def support_mask(reference: np.ndarray, masked: np.ndarray | None = None) -> np.ndarray:
    """Bright table and wall in the reference, outside ``masked`` (True = under the operator mask)."""
    import cv2

    grey = cv2.cvtColor(np.clip(reference, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    support = (grey > BRIGHT).astype(np.uint8)
    if masked is not None:
        support[masked] = 0
    kernel = np.ones((2 * ERODE_PX + 1, 2 * ERODE_PX + 1), np.uint8)
    return cv2.erode(support, kernel) > 0


def object_mask(frame: np.ndarray) -> np.ndarray:
    """Saturated pixels (the grape, the plate), dilated."""
    import cv2

    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    objects = (hsv[..., 1] > OBJECT_SATURATION).astype(np.uint8)
    kernel = np.ones((2 * OBJECT_DILATE_PX + 1, 2 * OBJECT_DILATE_PX + 1), np.uint8)
    return cv2.dilate(objects, kernel) > 0


def table_outline(support: np.ndarray) -> np.ndarray:
    """The convex hull of the support, filled."""
    import cv2

    points = cv2.findNonZero(support.astype(np.uint8))
    outline = np.zeros(support.shape, np.uint8)
    if points is not None:
        cv2.fillPoly(outline, [cv2.convexHull(points)], 1)
    return outline > 0


def ratio_map(frame: np.ndarray, reference: np.ndarray, support: np.ndarray,
              size: tuple[int, int] | None = None) -> np.ndarray:
    """The light ratio of one RGB ``frame`` against ``reference``, ``(H, W, 3)`` float32, at
    ``size`` ``(height, width)`` when given."""
    import cv2

    valid = support & ~object_mask(frame)
    blur = lambda img: cv2.GaussianBlur(img.astype(np.float32), (0, 0), BLUR_SIGMA)
    frame_b, ref_b = np.clip(blur(frame), 1.0, None), np.clip(blur(reference), 1.0, None)
    # brightness ratio, plus a colour shift per channel held within CHROMA_CLIP of it
    luma = np.log(frame_b.mean(-1) / ref_b.mean(-1))
    chroma = np.clip(np.log(frame_b / ref_b) - luma[..., None], *np.log(CHROMA_CLIP))
    log_ratio = np.concatenate([luma[..., None], chroma], axis=-1)
    outline = table_outline(support)
    fill = outline & ~valid
    out = np.where(valid[..., None], log_ratio, 0.0).astype(np.float32)
    if fill.any():
        # normalized convolution: the valid ratio, blurred and divided by the blurred weight, fine where
        # the hole is small and coarse where it is large (the arm at home); smooth, no texture
        filled = np.zeros_like(out)
        weight_left = np.ones(fill.shape, np.float32)
        for sigma in FILL_SIGMAS:
            w = cv2.GaussianBlur(valid.astype(np.float32), (0, 0), sigma)
            v = np.stack([cv2.GaussianBlur(out[..., c] * valid, (0, 0), sigma) for c in range(4)], -1)
            ok = (w > 0.05) & (weight_left > 0)
            filled[ok] = v[ok] / w[ok, None]
            weight_left[ok] = 0
        out[fill] = filled[fill]
    out = out[..., :1] + out[..., 1:]                        # back to per-channel log ratios
    # 1 inside the outline (the blur is >= 0.5 there), fading to 0 over about FEATHER_PX outside it
    weight = np.clip(2.0 * cv2.GaussianBlur(outline.astype(np.float32), (0, 0), FEATHER_PX / 2), 0.0, 1.0)
    ratio = np.exp(out * weight[..., None])
    ratio = np.clip(ratio, *RATIO_CLIP).astype(np.float32)
    if size is not None:
        ratio = cv2.resize(ratio, (size[1], size[0]), interpolation=cv2.INTER_AREA)
    return ratio


def save_bank(path: Path, maps: list[np.ndarray], provenance: dict) -> str:
    """Write ``maps`` (float16) and provenance to ``path`` (.npz); returns the sha256 of the maps."""
    stack = np.stack(maps).astype(np.float16)
    digest = hashlib.sha256(stack.tobytes()).hexdigest()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, maps=stack, provenance=json.dumps({**provenance, "maps_sha256": digest}))
    return digest


def load_bank(path: Path) -> tuple[np.ndarray, dict]:
    with np.load(path) as bank:
        return bank["maps"], json.loads(str(bank["provenance"]))
