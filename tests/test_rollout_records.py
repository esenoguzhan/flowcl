"""The rollout record (docs/piper_deploy_spec.md §6.4): validation, legacy logs and the summary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from flowcl.deploy.rollout_records import (
    OUTCOMES,
    load_by_eye,
    load_rollouts,
    read_rollout,
    summarize,
    validate_record,
)


def record(**over):
    base = {"schema": "rollout_record/v1", "success": False, "outcome": "reach_miss", "miss_distance_mm": 31.0,
            "light": "daylight", "note": "", "annotated_at": "2026-11-16T10:00:00"}
    return base | over


def write_log(root: Path, date: str, name: str, *, rec=None, annotation=None, photo=False, mode="rollout") -> Path:
    run = root / date / name
    run.mkdir(parents=True)
    meta = {"mode": mode, "label": "pick up the grape", "kind": "complete", "reason": "stop",
            "duration_s": 12.0, "cycle_s": 30.0, "annotation": annotation,
            "server_metadata": {"checkpoint": "results/x/final.pt"}}
    if rec is not None:
        meta["record"] = rec
    (run / "meta.json").write_text(json.dumps(meta))
    if photo:
        (run / "start_front.jpg").write_bytes(b"\xff\xd8")
    return run


def test_a_complete_record_is_valid():
    assert validate_record(record()) == []
    assert validate_record(record(success=True, outcome="success", miss_distance_mm=None)) == []


@pytest.mark.parametrize("change, fragment", [
    ({"schema": "v0"}, "schema"),
    ({"success": "yes"}, "success must be"),
    ({"outcome": "dropped"}, "outcome 'dropped'"),
    ({"success": True}, "exactly when"),
    ({"outcome": "success"}, "exactly when"),
    ({"miss_distance_mm": -1}, "miss_distance_mm"),
    ({"miss_distance_mm": True}, "miss_distance_mm"),
    ({"light": "sun"}, "light"),
    ({"outcome": "other", "note": " "}, "needs a note"),
    ({"annotated_at": None}, "annotated_at"),
])
def test_each_rule_is_enforced(change, fragment):
    problems = validate_record(record(**change))
    assert any(fragment in p for p in problems), problems


def test_every_outcome_but_success_is_a_failure_category():
    for outcome in OUTCOMES:
        success = outcome == "success"
        note = "x" if outcome == "other" else ""
        assert validate_record(record(outcome=outcome, success=success, note=note)) == []


def test_legacy_and_unannotated_logs_are_read_without_inventing_a_record(tmp_path):
    legacy = write_log(tmp_path, "2026-11-02", "120107_rollout",
                       annotation={"sausage_in_pot": True, "lid_on_pot": False, "success": False})
    bare = write_log(tmp_path, "2026-11-02", "120500_rollout")
    a, b = read_rollout(legacy), read_rollout(bare)
    assert a.status == "legacy" and a.record is None and a.legacy_annotation["sausage_in_pot"] is True
    assert b.status == "unannotated" and b.record is None and b.legacy_annotation is None


def test_load_filters_modes_and_dates_and_attaches_by_eye_labels(tmp_path):
    write_log(tmp_path, "2026-11-16", "100000_rollout", rec=record(), photo=True)
    write_log(tmp_path, "2026-11-16", "100500_rollout", rec=record(success=True, outcome="success",
                                                                   miss_distance_mm=None))
    write_log(tmp_path, "2026-11-16", "101000_probe", mode="probe")
    write_log(tmp_path, "2026-11-17", "090000_rollout", rec=record(light="sun"))
    eye = tmp_path / "eye.json"
    eye.write_text(json.dumps({"placed": ["100500"], "unclear": ["100000"]}))
    rows = load_rollouts(tmp_path, dates=["2026-11-16"], by_eye=load_by_eye(eye))
    assert [r.time for r in rows] == ["100000", "100500"]
    assert rows[0].start_photo and rows[1].start_photo is None
    assert [r.by_eye for r in rows] == ["unclear", "placed"]
    assert rows[0].checkpoint == "results/x/final.pt"


def test_summary_counts_only_valid_records(tmp_path):
    write_log(tmp_path, "d", "100000_rollout", rec=record())
    write_log(tmp_path, "d", "100100_rollout", rec=record(success=True, outcome="success", miss_distance_mm=None))
    write_log(tmp_path, "d", "100200_rollout", rec=record(light="sun"))
    write_log(tmp_path, "d", "100300_rollout")
    out = summarize(load_rollouts(tmp_path))
    assert out["n_runs"] == 4 and out["n_recorded"] == 2 and out["successes"] == 1
    assert out["by_status"] == {"record": 2, "invalid_record": 1, "unannotated": 1}
    assert out["outcomes"] == {"reach_miss": 1, "success": 1}
    assert out["miss_distance_mm"] == [31.0]
    assert len(out["invalid"]) == 1
    assert out["success_rate"]["low"] <= 0.5 <= out["success_rate"]["high"]


def test_by_eye_refuses_a_run_with_two_labels(tmp_path):
    eye = tmp_path / "eye.json"
    eye.write_text(json.dumps({"placed": ["1"], "unclear": ["1"]}))
    with pytest.raises(ValueError, match="two labels"):
        load_by_eye(eye)


def test_the_real_logs_read_cleanly_when_present():
    root = Path.home() / "piper_rollouts"
    if not root.is_dir():
        pytest.skip("no rollout logs on this machine")
    rows = load_rollouts(root)
    assert rows and all(r.status in {"record", "invalid_record", "legacy", "unannotated"} for r in rows)
