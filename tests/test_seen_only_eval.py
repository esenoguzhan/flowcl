"""A3, seen-only evaluation (eval_unseen=False): which tasks are evaluated, what is recorded, the
text cache it must reproduce, the FWT refusal, and resume (docs/thesis_plan.md A3; status.md §5)."""

from __future__ import annotations

import json

import pytest
import torch
from omegaconf import OmegaConf

import flowcl.train.continual as continual_module
from flowcl.data.config import load_embodiment_spec
from flowcl.data.curriculum import load_curriculum
from flowcl.data.libero_adapter import read_task_metadata
from flowcl.envs.libero_env import EvalConfig
from flowcl.train.continual import run_continual, seen_eval_refs
from test_continual import BBQ, MILK, SAUCE, TINY_POLICY, fake_evaluation, tiny_train_cfg  # noqa: F401


def trio():
    return load_curriculum({"name": "test_trio", "tasks": [
        {"task_key": MILK, "n_demos": 1}, {"task_key": SAUCE, "n_demos": 1}, {"task_key": BBQ, "n_demos": 1}]})


def encoding_eval(dataset_dir, evaluated):
    """A fake evaluation that, like the real one, encodes each task's instruction in fp32."""
    fake = fake_evaluation([])

    def evaluate(policy, refs, *args, **kwargs):
        evaluated.append([r.task_key for r in refs])
        for r in refs:
            with torch.no_grad():
                policy.text_encoder([read_task_metadata(r.demo_path(dataset_dir)).language])
        return fake(policy, refs, *args, **kwargs)
    return evaluate


def _run(curriculum, dataset_dir, root, cfg, **kw):
    return run_continual(curriculum, method_name="seq_ft", spec=load_embodiment_spec("libero_franka"),
                         policy_config=TINY_POLICY, train_cfg=cfg, eval_cfg=EvalConfig(n_episodes=2), seed=0,
                         dataset_dir=dataset_dir, results_root=root, pretrained=False, **kw)


def test_the_seen_set_includes_co_trained_tasks():
    curriculum = load_curriculum({"name": "test_joint", "tasks": [
        {"task_key": MILK, "n_demos": 1, "co_train": [SAUCE]}, {"task_key": BBQ, "n_demos": 1}]})
    assert [r.task_key for r in seen_eval_refs(curriculum, 0)] == [MILK, SAUCE]
    assert [r.task_key for r in seen_eval_refs(curriculum, 1)] == [MILK, BBQ, SAUCE]


def test_seen_only_evaluates_the_trained_tasks_and_records_it(dataset_dir, tiny_train_cfg, tmp_path,  # noqa: F811
                                                              monkeypatch):
    evaluated = []
    monkeypatch.setattr(continual_module, "evaluate_tasks", encoding_eval(dataset_dir, evaluated))
    result = _run(trio(), dataset_dir, tmp_path / "seen", tiny_train_cfg, eval_unseen=False)
    assert evaluated == [[MILK], [MILK, SAUCE], [MILK, SAUCE, BBQ]]
    run = tmp_path / "seen" / result.run_id
    assert OmegaConf.load(run / "config.yaml")["eval"]["eval_unseen"] is False
    saved = json.loads((run / "result.json").read_text())
    assert saved["eval_unseen"] is False
    assert saved["retention_matrix"]["values"][0][1] is None and saved["retention_matrix"]["values"][2][2] is not None
    assert {v["encoded"] for v in saved["seen_only_text_cache"].values()} == {"fp32", "already cached"}

    evaluated.clear()
    default = _run(trio(), dataset_dir, tmp_path / "all", tiny_train_cfg)
    assert evaluated == [[MILK, SAUCE, BBQ]] * 3
    run = tmp_path / "all" / default.run_id
    assert "eval_unseen" not in OmegaConf.to_container(OmegaConf.load(run / "config.yaml"))["eval"]
    assert "eval_unseen" not in json.loads((run / "result.json").read_text())


def test_the_cache_after_stage_zero_and_training_match_the_default(dataset_dir, tiny_train_cfg, tmp_path,  # noqa: F811
                                                                   monkeypatch):
    caches = {}
    real_train = continual_module.train_one_task

    def train_spy(policy, dataset, cfg, **kwargs):
        if kwargs.get("task_idx") == 1:
            caches[mode] = {k: v.clone() for k, v in policy.text_encoder._cache.items()}
        return real_train(policy, dataset, cfg, **kwargs)

    monkeypatch.setattr(continual_module, "train_one_task", train_spy)
    results = {}
    for mode, unseen in (("default", True), ("seen", False)):
        monkeypatch.setattr(continual_module, "evaluate_tasks", encoding_eval(dataset_dir, []))
        results[mode] = _run(trio(), dataset_dir, tmp_path / mode, tiny_train_cfg, eval_unseen=unseen)
    assert sorted(caches["default"]) == sorted(caches["seen"]) and len(caches["seen"]) == 3
    assert all(torch.equal(caches["default"][k], caches["seen"][k]) for k in caches["seen"])
    for stage in range(3):
        a = torch.load(tmp_path / "default" / results["default"].run_id / "checkpoints" / f"stage{stage}.pt",
                       weights_only=False)["state_dict"]
        b = torch.load(tmp_path / "seen" / results["seen"].run_id / "checkpoints" / f"stage{stage}.pt",
                       weights_only=False)["state_dict"]
        assert all(torch.equal(a[k], b[k]) for k in a)


def test_fwt_is_refused_with_seen_only(dataset_dir, tiny_train_cfg, tmp_path):  # noqa: F811
    with pytest.raises(ValueError, match="FWT"):
        _run(trio(), dataset_dir, tmp_path, tiny_train_cfg, eval_unseen=False, single_task_baseline=True)


def test_a_seen_only_run_resumes_seen_only(dataset_dir, tiny_train_cfg, tmp_path, monkeypatch):  # noqa: F811
    evaluated = []
    monkeypatch.setattr(continual_module, "evaluate_tasks", encoding_eval(dataset_dir, evaluated))
    source = _run(trio(), dataset_dir, tmp_path / "a", tiny_train_cfg, eval_unseen=False)
    evaluated.clear()
    resumed = _run(trio(), dataset_dir, tmp_path / "b", tiny_train_cfg, eval_unseen=False,
                   resume_run=tmp_path / "a" / source.run_id, start_stage=1)
    assert evaluated == [[MILK, SAUCE], [MILK, SAUCE, BBQ]]
    saved = json.loads((tmp_path / "b" / resumed.run_id / "result.json").read_text())
    assert saved["eval_unseen"] is False and saved["retention_matrix"]["values"][0][1] is None
    for stage in (1, 2):
        a = torch.load(tmp_path / "a" / source.run_id / "checkpoints" / f"stage{stage}.pt",
                       weights_only=False)["state_dict"]
        b = torch.load(tmp_path / "b" / resumed.run_id / "checkpoints" / f"stage{stage}.pt",
                       weights_only=False)["state_dict"]
        assert all(torch.equal(a[k], b[k]) for k in a)
