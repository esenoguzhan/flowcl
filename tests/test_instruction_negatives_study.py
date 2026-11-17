"""The instruction-negatives study's pure parts: the closing-run search, displacement and motion
summaries, the discrimination gate, and the labels (gate_failed, acquisition states, the tiers)."""

from __future__ import annotations

import numpy as np

from flowcl.envs.libero_env import RolloutResult
from flowcl.experiments import instruction_negatives as inn

CFG = inn.load_config()
A, B = CFG["task_a"], CFG["task_b"]
BOOT = {"seed": 0, "n_resamples": 2000, "confidence": 0.95}


def test_close_start_finds_the_step_after_the_run():
    actions = np.zeros((40, 7))
    actions[:, 6] = -1
    actions[10:35, 6] = 1
    assert inn.close_start(actions, 6, 20) == 30
    assert inn.close_start(actions, 6, 26) is None


def test_hold_action_is_the_methods_row():
    actions = np.zeros((5, 7), dtype=np.float32)
    actions[:, 6] = [-1, 1, 1, -1, 1]
    assert inn.hold_action(actions, 0, 6)[6] == -1 and inn.hold_action(actions, 3, 6)[6] == 1
    assert np.all(inn.hold_action(actions, 3, 6)[:6] == 0)


def test_displacement_measures_after_the_settle_step_and_reports_the_stop():
    eef = np.zeros((6, 3))
    eef[1:] = [0.03, 0, 0]          # the stop: 3 cm before settling
    eef[4] = [0.033, 0.004, 0]      # then 5 mm
    fingers = np.zeros((6, 2))
    fingers[1] = [0.01, -0.01]
    fingers[4] = [0.0005, 0]
    d = inn.displacement(eef, fingers, settle_step=2)
    assert abs(d["eef_after_m"] - 0.005) < 1e-12 and abs(d["eef_from_start_m"] - np.hypot(0.033, 0.004)) < 1e-12
    assert abs(d["finger_after_m"] - 0.0005) < 1e-12 and abs(d["finger_before_m"] - 0.01) < 1e-12


def test_motion_summary_and_the_default_rollout_field():
    m = {"eef_pos": [[0, 0, 0], [0.1, 0, 0], [0.1, 0.1, 0]], "gripper_qpos": [[0.02, -0.02]] * 3,
         "gripper_cmd": [-1.0, 1.0]}
    s = inn.motion_summary(m, window=2)
    assert abs(s["path_m"] - 0.2) < 1e-12 and s["gripper_sign_changes"] == 1 and s["full_window"]
    assert inn.motion_summary(None, 2) is None
    assert RolloutResult(success=True, n_steps=1, task_key="x", episode_idx=0, seed=0, n_replans=1).motion is None


def panel(rates):
    return {"conditions": {f"held_out::{i}": {"kind": "held_out_task", "own_goal_rate": r}
                           for i, r in enumerate(rates)}}


def test_the_gate_needs_learning_and_suppression_under_every_held_out_instruction():
    assert inn.gate(CFG, [True] * 45 + [False] * 5, panel([0.1, 0.2]))["passed"]
    assert not inn.gate(CFG, [True] * 30 + [False] * 20, panel([0.1, 0.2]))["passed"]
    assert not inn.gate(CFG, [True] * 50, panel([0.1, 0.5]))["passed"]


def cells(r0a_gpm=0.9, r1a_gpm=0.8, r1a_allow=0.2, r1b_gpm=0.9, r1b_allow=0.95, r1a_hist=0.0, r1a_hist_allow=0.0):
    def s(p):
        k = int(round(p * 50))
        return [True] * k + [False] * (50 - k)
    return {"gpm": {0: {A: s(r0a_gpm)}, 1: {A: s(r1a_gpm), B: s(r1b_gpm)}},
            "allow": {0: {A: s(0.9)}, 1: {A: s(r1a_allow), B: s(r1b_allow)}},
            "hist_gpm": {1: {A: s(r1a_hist)}}, "hist_allow": {1: {A: s(r1a_hist_allow)}}}


def test_a_failed_gate_assesses_nothing():
    j = inn.judge(CFG, cells(), {"passed": False}, BOOT)
    assert j["outcome"] == "gate_failed" and set(j["labels"].values()) == {"not_assessed"}


def test_clean_acquisition_gives_unqualified_labels():
    j = inn.judge(CFG, cells(r1a_gpm=0.86, r1a_allow=0.2), {"passed": True}, BOOT)
    assert j["acquisition"]["state"] == "clean"
    assert j["labels"]["effect_present"] == "effect_present"
    assert j["labels"]["retains"] == "retains"
    assert j["labels"]["projection_contributes"] == "projection_contributes"
    assert j["labels"]["allowlist_without_projection_retains"] == "not_shown"
    assert j["interaction"]["flag"] == "against_historical_control"


def test_partial_retention_is_an_effect_without_retaining():
    j = inn.judge(CFG, cells(r1a_gpm=0.5), {"passed": True}, BOOT)
    assert j["labels"]["effect_present"] == "effect_present" and j["labels"]["retains"] == "not_shown"


def test_acquisition_cost_qualifies_and_failure_suppresses_every_label():
    cost = inn.judge(CFG, cells(r1a_gpm=0.86, r1b_gpm=0.70), {"passed": True}, BOOT)
    assert cost["acquisition"]["state"] == "at_acquisition_cost"
    assert cost["labels"]["retains"] == "retains_at_acquisition_cost"
    none = inn.judge(CFG, cells(r1b_gpm=0.30), {"passed": True}, BOOT)
    assert none["acquisition"]["state"] == "inconclusive_acquisition"
    assert set(none["labels"].values()) == {"not_assessed"}
