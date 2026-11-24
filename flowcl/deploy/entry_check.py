"""PiperStudio's first-jump check on a new chunk, ported for offline use (piper_studio
backend/executor.py ``check_chunk`` and ``interpolate``, 24 Nov 2026).

The executor takes a new chunk over at position ``p`` (ticks since its observation, at the
control rate) and refuses it when the target there lies further than a per-joint limit outside
the span between the last command and the arm's measured joints. Joints only (J1..J6); the
gripper is not checked.
"""

from __future__ import annotations

import math

import numpy as np

# PolicyConfig.first_jump_limits_rad in PiperStudio's config.yaml (24 Nov): max_first_jump_rad for
# J1-J4, then J5's and J6's own limits.
FIRST_JUMP_LIMITS_RAD = (0.15, 0.15, 0.15, 0.15, 0.35, 0.5)
N_JOINTS = 6


def interpolate(chunk: np.ndarray, p: float) -> np.ndarray:
    """The target at chunk position ``p``, linear between ticks."""
    i = int(math.floor(p))
    if i >= len(chunk) - 1:
        return np.array(chunk[-1], dtype=np.float64)
    if i < 0:
        return np.array(chunk[0], dtype=np.float64)
    w = p - i
    return chunk[i] + (chunk[i + 1] - chunk[i]) * w


def first_jump(chunk, p: float, last_command, measured=None,
               limits=FIRST_JUMP_LIMITS_RAD) -> dict:
    """The executor's first-jump test: ``{"ok", "joint", "value", "limit"}`` for the joint nearest
    (or furthest over) its limit."""
    arr = np.asarray(chunk, dtype=np.float64)
    entry = interpolate(arr, p)[:N_JOINTS]
    command = np.asarray(last_command, dtype=np.float64)[:N_JOINTS]
    arm = command if measured is None else np.asarray(measured, dtype=np.float64)[:N_JOINTS]
    if not np.isfinite(arm).all():
        arm = command
    lo, hi = np.minimum(command, arm), np.maximum(command, arm)
    jump = np.maximum(0.0, np.maximum(lo - entry, entry - hi))
    limit = np.broadcast_to(np.asarray(limits, dtype=np.float64), jump.shape)
    worst = int(np.argmax(jump - limit))
    return {"ok": bool(jump[worst] <= limit[worst]), "joint": worst + 1, "value": float(jump[worst]),
            "limit": float(limit[worst])}
