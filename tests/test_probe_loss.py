"""Old-task probe loss (status.md §5 item 5): the records, training left bitwise unchanged, module
modes restored, and a resumed run rebuilding the same probe batches."""

from __future__ import annotations

import json

import pytest
import torch

import flowcl.train.continual as continual_module
from flowcl.data.config import load_embodiment_spec
from flowcl.data.curriculum import load_curriculum
from flowcl.envs.libero_env import EvalConfig
from flowcl.train.continual import run_continual
from flowcl.train.probe_loss import probe_losses
from test_continual import BBQ, MILK, SAUCE, TINY_POLICY, fake_evaluation, tiny_train_cfg  # noqa: F401


def trio():
    return load_curriculum({"name": "test_trio", "tasks": [
        {"task_key": MILK, "n_demos": 1}, {"task_key": SAUCE, "n_demos": 1}, {"task_key": BBQ, "n_demos": 1}]})


def _run(dataset_dir, root, cfg, **kw):
    return run_continual(trio(), method_name="seq_ft", spec=load_embodiment_spec("libero_franka"),
                         policy_config=TINY_POLICY, train_cfg=cfg, eval_cfg=EvalConfig(n_episodes=2), seed=0,
                         dataset_dir=dataset_dir, results_root=root, pretrained=False, **kw)


def _state(root, run_id, stage):
    return torch.load(root / run_id / "checkpoints" / f"stage{stage}.pt", weights_only=False)["state_dict"]


def test_probes_are_recorded_and_leave_training_unchanged(dataset_dir, tiny_train_cfg, tmp_path,  # noqa: F811
                                                          monkeypatch):
    monkeypatch.setattr(continual_module, "evaluate_tasks", fake_evaluation([]))
    plain = _run(dataset_dir, tmp_path / "off", tiny_train_cfg)
    probed = _run(dataset_dir, tmp_path / "on", tiny_train_cfg, probe_every=1)
    for stage in range(3):
        a, b = _state(tmp_path / "off", plain.run_id, stage), _state(tmp_path / "on", probed.run_id, stage)
        assert all(torch.equal(a[k], b[k]) for k in a)
    run = tmp_path / "on" / probed.run_id
    saved = json.loads((run / "result.json").read_text())
    assert saved["probe_every"] == 1 and "probe_file" not in saved["stages"][0]   # nothing earlier to probe
    for stage, tasks in ((1, [MILK]), (2, [MILK, SAUCE])):
        data = json.loads((run / saved["stages"][stage]["probe_file"]["path"]).read_text())
        assert data["tasks"] == tasks
        assert [r["step"] for r in data["records"]] == [0, 1, 2]                   # stage start, then every step
        assert all(set(r["losses"]) == set(tasks) and all(v > 0 for v in r["losses"].values())
                   for r in data["records"])
    assert "probe_every" not in json.loads((tmp_path / "off" / plain.run_id / "result.json").read_text())


def test_probe_losses_restore_every_module_mode(dataset_dir, tiny_train_cfg, tmp_path, monkeypatch):  # noqa: F811
    from flowcl.train import probe_loss

    seen = {}
    real = probe_loss.probe_losses

    def spy(policy, probes, device, amp):
        before = [m.training for m in policy.modules()]
        out = real(policy, probes, device, amp)
        seen.setdefault("restored", []).append(before == [m.training for m in policy.modules()])
        seen.setdefault("mixed", []).append(len(set(before)) > 1)   # the text backbone stays in eval
        return out

    monkeypatch.setattr(probe_loss, "probe_losses", spy)
    monkeypatch.setattr(continual_module, "evaluate_tasks", fake_evaluation([]))
    _run(dataset_dir, tmp_path, tiny_train_cfg, probe_every=1)
    assert seen["restored"] and all(seen["restored"]) and any(seen["mixed"])


def test_a_resumed_run_rebuilds_the_same_probe_batches(dataset_dir, tiny_train_cfg, tmp_path,  # noqa: F811
                                                       monkeypatch):
    monkeypatch.setattr(continual_module, "evaluate_tasks", fake_evaluation([]))
    full = _run(dataset_dir, tmp_path / "a", tiny_train_cfg, probe_every=1)
    resumed = _run(dataset_dir, tmp_path / "b", tiny_train_cfg, probe_every=1,
                   resume_run=tmp_path / "a" / full.run_id, start_stage=2)
    a = json.loads((tmp_path / "a" / full.run_id / "train" / "stage2_probes.json").read_text())
    b = json.loads((tmp_path / "b" / resumed.run_id / "train" / "stage2_probes.json").read_text())
    assert a["tasks"] == b["tasks"] and a["records"] == b["records"]


def test_probe_every_must_be_positive(dataset_dir, tiny_train_cfg, tmp_path):  # noqa: F811
    with pytest.raises(ValueError, match="probe_every"):
        _run(dataset_dir, tmp_path, tiny_train_cfg, probe_every=0)


def test_no_probes_means_no_losses():
    assert probe_losses(torch.nn.Linear(1, 1), [], "cpu", False) == {}
