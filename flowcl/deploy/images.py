"""The one image path shared by recording, training and serving (Dobot X-Trainer).

The lab's teleop recorder (``experiments/run_control_terminal.py:390-425`` in the
xtrainer repo) stores each camera as:

1. the RealSense frame, RGB, already rotated 180 degrees for the top and right cameras;
2. reversed to BGR;
3. top camera only: cropped to ``[150:420, 220:480]`` and ``cv2.resize``-d to 640x480;
4. JPEG-encoded at quality 50 (the collector then writes h264 video).

The robot-side runner rebuilds exactly that with :func:`recorder_bgr` +
:func:`encode_jpeg`, the server decodes it, and both the server and the training cache
reduce the 640x480 BGR frame with the same :func:`to_policy_frame`. The only remaining
difference between training and deployment inputs is the h264 layer of the recordings.
"""

from __future__ import annotations

import cv2
import numpy as np

CAMERAS = ("top", "left_wrist", "right_wrist")
TOP_CROP_ROWS = (150, 420)
TOP_CROP_COLS = (220, 480)
RECORDED_HW = (480, 640)
JPEG_QUALITY = 50
POLICY_HW = (128, 128)


def recorder_bgr(rgb_frame: np.ndarray, camera: str) -> np.ndarray:
    """A live RGB camera frame -> the 640x480 BGR frame the recorder stored."""
    if camera not in CAMERAS:
        raise ValueError(f"unknown camera {camera!r}; expected one of {CAMERAS}")
    frame = np.asarray(rgb_frame)
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError(f"{camera}: expected (H, W, 3) uint8 RGB, got {frame.shape} {frame.dtype}")
    if camera == "top":
        r0, r1 = TOP_CROP_ROWS
        c0, c1 = TOP_CROP_COLS
        bgr = cv2.resize(frame[r0:r1, c0:c1, ::-1], (RECORDED_HW[1], RECORDED_HW[0]))
    else:
        bgr = frame[:, :, ::-1]
    bgr = np.ascontiguousarray(bgr)
    if bgr.shape[:2] != RECORDED_HW:
        raise ValueError(f"{camera}: recorder frame is {bgr.shape[:2]}, expected {RECORDED_HW}")
    return bgr


def encode_jpeg(bgr: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


def decode_jpeg(data: bytes) -> np.ndarray:
    """JPEG bytes -> BGR uint8 frame."""
    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("could not decode JPEG bytes")
    return frame


def to_policy_frame(bgr: np.ndarray, hw: tuple[int, int] = POLICY_HW) -> np.ndarray:
    """A recorder-format BGR frame -> the policy's RGB input (area-resampled)."""
    frame = np.asarray(bgr)
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError(f"expected (H, W, 3) uint8 BGR, got {frame.shape} {frame.dtype}")
    small = cv2.resize(frame, (hw[1], hw[0]), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(small[:, :, ::-1])
