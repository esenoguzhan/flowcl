"""PiperStudio's first-jump check, ported (flowcl.deploy.entry_check)."""

from __future__ import annotations

import numpy as np

from flowcl.deploy.entry_check import first_jump, interpolate


def test_interpolate_between_ticks_and_at_the_ends():
    chunk = np.arange(16, dtype=float)[:, None] * np.ones((1, 7))
    assert np.allclose(interpolate(chunk, 2.5), 2.5)
    assert np.allclose(interpolate(chunk, -1), 0) and np.allclose(interpolate(chunk, 20), 15)


def test_a_jump_outside_the_span_is_refused_per_joint_limit():
    state = np.zeros(7)
    still = np.zeros((16, 7))
    assert first_jump(still, 1.0, state, state)["ok"]
    j2 = still.copy()
    j2[:, 1] = 0.2
    res = first_jump(j2, 1.0, state, state)
    assert not res["ok"] and res["joint"] == 2 and abs(res["value"] - 0.2) < 1e-9
    j5 = still.copy()
    j5[:, 4] = 0.3                                            # J5 has its own 0.35 rad limit
    assert first_jump(j5, 1.0, state, state)["ok"]


def test_an_entry_between_the_command_and_the_arm_is_no_jump():
    command, arm = np.zeros(7), np.zeros(7)
    command[1] = 0.3                                          # the arm has not reached the last command yet
    chunk = np.zeros((16, 7))
    chunk[:, 1] = 0.25
    assert first_jump(chunk, 1.0, command, arm)["ok"]
    assert not first_jump(chunk, 1.0, arm, arm)["ok"]         # against the arm alone it is a jump
