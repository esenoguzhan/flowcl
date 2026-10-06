"""Action post-processing and motion checks for the Dobot X-Trainer runner.

Index layout of the 14-dim joint vector: ``[L_j1..L_j6, L_grip, R_j1..R_j6, R_grip]``.

* :func:`limit_action_step` has the semantics of the lab's ``_limit_action_step``
  (``experiments/run_policy_control.py`` in the xtrainer repo): unwrap joints by 2*pi
  against the last command, clip the per-tick joint and gripper deltas, clip grippers to
  [0, 1].
* :func:`upsample_chunk` turns a 10 Hz policy chunk into 30 Hz servo targets.
* :func:`check_prefix` validates every action the runner is about to execute from a chunk
  *before* the first one is sent, so a bad chunk never starts moving the robot.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ACTION_DIM = 14
GRIPPER_DIMS = (6, 13)
JOINT_DIMS = np.array([i for i in range(ACTION_DIM) if i not in GRIPPER_DIMS], dtype=np.int64)


def limit_action_step(
    action: np.ndarray,
    last_action: np.ndarray,
    max_joint_step_rad: float,
    max_gripper_step: float,
) -> tuple[np.ndarray, bool, bool]:
    """Clip one 30 Hz command against the previous one.

    Returns ``(command, joints_clipped, gripper_clipped)``.
    """
    limited = np.asarray(action, dtype=np.float32).copy()
    last = np.asarray(last_action, dtype=np.float32)
    delta = limited[JOINT_DIMS] - last[JOINT_DIMS]
    limited[JOINT_DIMS] -= (2.0 * np.pi) * np.round(delta / (2.0 * np.pi))

    joints_clipped = gripper_clipped = False
    if max_joint_step_rad > 0:
        delta = limited[JOINT_DIMS] - last[JOINT_DIMS]
        bounded = np.clip(delta, -max_joint_step_rad, max_joint_step_rad)
        joints_clipped = bool(np.any(bounded != delta))
        limited[JOINT_DIMS] = last[JOINT_DIMS] + bounded

    grip = list(GRIPPER_DIMS)
    if max_gripper_step > 0:
        delta = limited[grip] - last[grip]
        bounded = np.clip(delta, -max_gripper_step, max_gripper_step)
        gripper_clipped = bool(np.any(bounded != delta))
        limited[grip] = last[grip] + bounded
    limited[grip] = np.clip(limited[grip], 0.0, 1.0)
    return limited.astype(np.float32, copy=False), joints_clipped, gripper_clipped


def upsample_chunk(chunk: np.ndarray, factor: int) -> np.ndarray:
    """Linear interpolation of an ``(H, D)`` chunk to ``(factor * (H - 1) + 1, D)``.

    Row ``factor * i`` equals ``chunk[i]`` exactly, so the 10 Hz targets the policy was
    trained on are hit on the 30 Hz grid they were subsampled from.
    """
    chunk = np.asarray(chunk, dtype=np.float32)
    if chunk.ndim != 2 or chunk.shape[0] < 1:
        raise ValueError(f"chunk must be (H, D) with H >= 1, got {chunk.shape}")
    if factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}")
    if factor == 1 or chunk.shape[0] == 1:
        return chunk.copy()
    n = factor * (chunk.shape[0] - 1) + 1
    t = np.arange(n, dtype=np.float64) / factor
    lo = np.minimum(np.floor(t).astype(np.int64), chunk.shape[0] - 2)
    w = (t - lo)[:, None]
    out = (1.0 - w) * chunk[lo].astype(np.float64) + w * chunk[lo + 1].astype(np.float64)
    return out.astype(np.float32)


@dataclass(frozen=True)
class Envelope:
    """Per-dimension command bounds (rad for joints, [0, 1] for grippers)."""

    lo: np.ndarray
    hi: np.ndarray

    @classmethod
    def from_actions(cls, actions: np.ndarray, margin_rad: float) -> "Envelope":
        """Training actions' min/max, widened by ``margin_rad`` on the joints.

        Grippers get [-0.1, 1.1]: a policy predicts slightly past the open/closed ends,
        and :func:`limit_action_step` clips them to [0, 1] before they are sent.
        """
        actions = np.asarray(actions, dtype=np.float64)
        lo = actions.min(axis=0) - margin_rad
        hi = actions.max(axis=0) + margin_rad
        for g in GRIPPER_DIMS:
            lo[g], hi[g] = -0.1, 1.1
        return cls(lo=lo.astype(np.float32), hi=hi.astype(np.float32))

    def to_dict(self) -> dict:
        return {"lo": self.lo.tolist(), "hi": self.hi.tolist()}

    @classmethod
    def from_dict(cls, payload: dict) -> "Envelope":
        return cls(lo=np.asarray(payload["lo"], np.float32), hi=np.asarray(payload["hi"], np.float32))


def hard_joint_violations(action: np.ndarray) -> list[str]:
    """The lab's fixed J3/J4 limits (``experiments/run_inference.py``)."""
    a = np.asarray(action)
    problems = []
    if not -2.6 < a[2] < 0.0:
        problems.append(f"left J3 {a[2]:.3f} outside (-2.6, 0)")
    if not a[3] > -0.6:
        problems.append(f"left J4 {a[3]:.3f} <= -0.6")
    if not 0.0 < a[9] < 2.6:
        problems.append(f"right J3 {a[9]:.3f} outside (0, 2.6)")
    if not a[10] < 0.6:
        problems.append(f"right J4 {a[10]:.3f} >= 0.6")
    return problems


def cartesian_violations(pose: np.ndarray) -> list[str]:
    """The lab's jaw-tip box (mm), from ``get_XYZrxryrz_state`` (left xyz = 0:3, right = 6:9)."""
    p = np.asarray(pose, dtype=np.float64)
    problems = []
    if not (-410 < p[0] < 300 and -700 < p[1] < -210 and p[2] > 42):
        problems.append(f"left tip {p[0:3].round(1).tolist()} outside the safe box")
    if not (-250 < p[6] < 410 and -700 < p[7] < -210 and p[8] > 42):
        problems.append(f"right tip {p[6:9].round(1).tolist()} outside the safe box")
    return problems


def check_prefix(
    prefix: np.ndarray,
    current: np.ndarray,
    envelope: Envelope | None,
    max_first_jump_rad: float,
) -> list[str]:
    """Everything wrong with the actions about to be executed; empty means go.

    Args:
        prefix: ``(n, 14)`` 30 Hz commands, before per-tick limiting.
        current: The command/state the robot is at now (14,).
        envelope: Allowed per-dimension range, or None to skip that check.
        max_first_jump_rad: Largest allowed joint distance from ``current`` to the
            first command (the per-tick limiter would smear a larger jump out, but a
            large jump means the policy is far off-distribution).
    """
    prefix = np.asarray(prefix, dtype=np.float32)
    problems: list[str] = []
    if prefix.ndim != 2 or prefix.shape[1] != ACTION_DIM or prefix.shape[0] == 0:
        return [f"prefix has shape {prefix.shape}, expected (n, {ACTION_DIM})"]
    if not np.isfinite(prefix).all():
        return ["prefix contains non-finite values"]
    jump = float(np.max(np.abs(prefix[0, JOINT_DIMS] - np.asarray(current)[JOINT_DIMS])))
    if jump > max_first_jump_rad:
        problems.append(f"first command is {jump:.3f} rad from the current pose (max {max_first_jump_rad})")
    if envelope is not None:
        below = prefix < envelope.lo - 1e-6
        above = prefix > envelope.hi + 1e-6
        if below.any() or above.any():
            dims = sorted(set(np.where(below | above)[1].tolist()))
            problems.append(f"dims {dims} leave the training envelope")
    bad_rows = [i for i in range(prefix.shape[0]) if hard_joint_violations(prefix[i])]
    if bad_rows:
        first = bad_rows[0]
        problems.append(
            f"{len(bad_rows)} commands break the J3/J4 limits, first at tick {first}: "
            + "; ".join(hard_joint_violations(prefix[first]))
        )
    return problems
