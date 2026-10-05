"""Stage-boundary resume (docs/thesis_plan.md, A1): a resumed GPM run reproduces its source
bitwise (checkpoint, memory artifact, imported evaluations), and every mismatch is refused."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

import flowcl.train.continual as continual_module
from flowcl.data.config import load_embodiment_spec
from flowcl.data.curriculum import load_curriculum
from flowcl.envs.libero_env import EvalConfig
from flowcl.methods.base import BaseMethod
from flowcl.train.continual import load_resume_source, run_continual, warm_runner_text_cache
from flowcl.train.trainer import TrainConfig
from flowcl.utils.libero_paths import repo_root
from flowcl.utils.run import file_sha256
from test_continual import BBQ, MILK, SAUCE, TINY_POLICY, fake_evaluation


def tiny_cfg():
    return TrainConfig(steps=2, batch_size=2, num_workers=0, device="cpu", log_every=0, warmup_steps=1)


def trio():
    return load_curriculum({"name": "test_trio", "tasks": [
        {"task_key": MILK, "n_demos": 1}, {"task_key": SAUCE, "n_demos": 1},
        {"task_key": BBQ, "n_demos": 1}]})


@pytest.fixture(scope="module")
def gpm_source(tmp_path_factory, dataset_dir):
    """A three-stage GPM run with (fake) evaluations and its result.json."""
    root = tmp_path_factory.mktemp("source")
    cfg = OmegaConf.load(repo_root() / "configs" / "analysis" / "subspace.yaml")
    cfg.min_samples_per_dim = 0.01
    cfg.num_workers = 0
    capture = root / "capture.yaml"
    OmegaConf.save(cfg, capture)
    kwargs = {"update_memory": True, "capture_config": str(capture), "log_interval": 1}
    mp = pytest.MonkeyPatch()
    mp.setattr(continual_module, "evaluate_tasks", fake_evaluation([]))
    try:
        result = run_continual(trio(), method_name="gpm", method_kwargs=kwargs,
                               spec=load_embodiment_spec("libero_franka"), policy_config=TINY_POLICY,
                               train_cfg=tiny_cfg(), eval_cfg=EvalConfig(n_episodes=2), seed=0,
                               dataset_dir=dataset_dir, results_root=root / "a", pretrained=False)
    finally:
        mp.undo()
    return {"root": root, "run": root / "a" / result.run_id, "kwargs": kwargs, "result": result,
            "dataset_dir": dataset_dir}


def resume(src, out, start=2, **overrides):
    kw = dict(method_name="gpm", method_kwargs=src["kwargs"], spec=load_embodiment_spec("libero_franka"),
              policy_config=TINY_POLICY, train_cfg=tiny_cfg(), eval_cfg=EvalConfig(n_episodes=2), seed=0,
              dataset_dir=src["dataset_dir"], results_root=out, pretrained=False,
              resume_run=src["run"], start_stage=start)
    kw.update(overrides)
    curriculum = kw.pop("curriculum", trio())
    return run_continual(curriculum, **kw)


def test_resumed_gpm_run_reproduces_its_source_bitwise(gpm_source, tmp_path, monkeypatch):
    monkeypatch.setattr(continual_module, "evaluate_tasks", fake_evaluation([]))
    src = gpm_source["run"]
    result = resume(gpm_source, tmp_path / "b", identity_reference_run=src, identity_stages=(2,))
    run = tmp_path / "b" / result.run_id
    assert result.identity_checks["2"]["passed"] and result.identity_checks["2"]["n_tensors"] > 0
    # the resumed stage against the source: checkpoint bitwise, memory content, evaluation episodes
    from flowcl.experiments.resume_check import compare_resume, deep_differences
    report = compare_resume(src, run, 2)
    assert report["passed"], report
    assert report["checkpoint"]["n_different"] == 0 and report["evaluation"]["passed"]
    assert report["method_artifacts"]["memory_task2.pt"]["differences"] == []
    assert deep_differences({"a": torch.ones(2)}, {"a": torch.zeros(2)}) == [".a: tensor"]
    assert deep_differences({"a": 1}, {"a": 1.0}) and deep_differences([1], (1,))
    assert deep_differences({"c": float("nan")}, {"c": float("nan")}) == []
    assert deep_differences({"c": float("nan")}, {"c": 0.5}) == [".c: nan != 0.5"]
    mine = torch.load(run / "checkpoints" / "stage2.pt", weights_only=False)["extra"]
    theirs = torch.load(src / "checkpoints" / "stage2.pt", weights_only=False)["extra"]
    assert mine["method_state"]["memory_extended"] == theirs["method_state"]["memory_extended"]
    assert mine["seed_namespace_run_id"] == theirs["seed_namespace_run_id"] == "test_trio__seq_ft__seed0"
    # imported stages: evaluations copied verbatim, result.json complete and in order
    for k in (0, 1):
        assert (run / "eval" / f"stage{k}.json").read_bytes() == (src / "eval" / f"stage{k}.json").read_bytes()
    saved, source = (json.loads((d / "result.json").read_text()) for d in (run, src))
    assert [s["stage"] for s in saved["stages"]] == [0, 1, 2]
    assert [s.get("imported_from") for s in saved["stages"]] == [str(src), str(src), None]
    assert saved["retention_matrix"] == source["retention_matrix"]
    assert saved["resume"]["start_stage"] == 2 and saved["systems"]["resumed_at_stage"] == 2
    record = json.loads((run / "resume.json").read_text())
    assert record["sha256"] == file_sha256(src / "checkpoints" / "stage1.pt")
    assert record["method_state"]["memory"] == "method/memory_task1.pt"
    assert {v["precision"] for v in record["text_cache"].values()} == {"fp32"}   # CPU, no AMP
    assert OmegaConf.load(run / "config.yaml").provenance.resume.start_stage == 2


def test_resume_refuses_every_mismatch(gpm_source, tmp_path):
    src = gpm_source["run"]
    args = dict(curriculum=trio(), method_name="gpm", method_kwargs=gpm_source["kwargs"],
                raw_policy_cfg=dict(TINY_POLICY), train_cfg=tiny_cfg(), seed=0)

    def refused(match, **changes):
        kw = {**args, **changes}
        start = kw.pop("start_stage", 2)
        with pytest.raises(ValueError, match=match):
            load_resume_source(src, start, **kw)

    refused("start_stage must lie", start_stage=0)
    refused("start_stage must lie", start_stage=3)
    refused("'method'", method_kwargs={**gpm_source["kwargs"], "eps": 0.9})
    refused("'method'", method_name="seq_ft", method_kwargs={})
    swapped = load_curriculum({"name": "test_trio", "tasks": [
        {"task_key": SAUCE, "n_demos": 1}, {"task_key": MILK, "n_demos": 1}, {"task_key": BBQ, "n_demos": 1}]})
    refused("'curriculum'", curriculum=swapped)
    refused("'train'", train_cfg=TrainConfig(**{**vars(tiny_cfg()), "lr": 3e-5}))
    refused("'seed'", seed=1)
    refused("'policy_config'", raw_policy_cfg={**TINY_POLICY, "euler_steps": 3})
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        load_resume_source(src, 2, expected_sha256="0" * 64, **args)
    # exempt fields do not block
    load_resume_source(src, 2, **{**args, "train_cfg": TrainConfig(**{**vars(tiny_cfg()), "num_workers": 2})})
    with pytest.raises(ValueError, match="must not write into its source"):
        resume(gpm_source, gpm_source["root"] / "a")


def test_a_tampered_memory_artifact_is_refused(gpm_source, tmp_path):
    import shutil

    copy = tmp_path / gpm_source["run"].name
    shutil.copytree(gpm_source["run"], copy)
    with open(copy / "method" / "memory_task1.pt", "ab") as f:
        f.write(b"x")
    with pytest.raises(ValueError, match="method_artifacts"):
        load_resume_source(copy, 2, trio(), "gpm", gpm_source["kwargs"], dict(TINY_POLICY), tiny_cfg(), 0)


def test_a_stateful_method_without_restore_refuses():
    class Stateful(BaseMethod):
        name = "stateful_test"

    with pytest.raises(NotImplementedError, match="restore_state"):
        Stateful().restore_state({"method_artifacts": [{"path": "method/x.pt", "sha256": "0"}]}, Path("."), 0)
    assert Stateful().restore_state({"method_artifacts": []}, Path("."), 0) == {}


def test_text_cache_follows_the_runner(dataset_dir):
    class Encoder:
        def __init__(self):
            self.calls = []

        def __call__(self, texts):
            self.calls.append((texts[0], torch.is_autocast_enabled()))

    policy = type("P", (), {"text_encoder": Encoder()})()
    state = warm_runner_text_cache(policy, trio(), dataset_dir, "cpu", amp=True)
    assert list(state) == [MILK, SAUCE, BBQ]
    assert all(v["precision"] == "fp32" for v in state.values())   # AMP is CUDA only
    assert [c[0] for c in policy.text_encoder.calls] == [v["text"] for v in state.values()]
    assert all(t and t[0].islower() for t, _ in policy.text_encoder.calls)


def test_cli_pairs_resume_flags():
    import subprocess

    done = subprocess.run(["uv", "run", "python", str(repo_root() / "scripts" / "run_continual.py"),
                           "--curriculum", "seq_hetero", "--start-stage", "2"],
                          capture_output=True, text=True, timeout=300, cwd=repo_root())
    assert done.returncode == 2 and "go together" in done.stderr


def test_resume_queue_dry_run(tmp_path):
    import os
    import subprocess

    queue = repo_root() / "scripts" / "queue_resume_check.sh"
    env = {**os.environ, "QUEUE_DRY_RUN": "1", "QUEUE_LOG_ROOT": str(tmp_path)}
    done = subprocess.run(["bash", str(queue)], env=env, capture_output=True, text=True, timeout=300)
    (logdir,) = list(tmp_path.glob("queue_*_resume_check"))
    log = (logdir / "queue.log").read_text()
    assert done.returncode == 0 and log.rstrip().endswith("QUEUE DONE ok")
    cmd = (logdir / "0_resume.log").read_text()
    cfg = OmegaConf.load(repo_root() / "configs" / "analysis" / "resume_check.yaml")
    assert f"--resume-sha256 {cfg.source.sha256.stage2}" in cmd and "--start-stage 3" in cmd
    assert "--identity-stages 3" in cmd and "--results-root results/resume_check" in cmd
    done = subprocess.run(["bash", str(queue)], env={**env, "QUEUE_FAIL_STEP": "0_resume"},
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 1


def test_resume_queue_takes_the_confirmatory_config(tmp_path):
    import os
    import subprocess

    queue = repo_root() / "scripts" / "queue_resume_check.sh"
    cfg_path = repo_root() / "configs" / "analysis" / "resume_check_ne90.yaml"
    env = {**os.environ, "QUEUE_DRY_RUN": "1", "QUEUE_LOG_ROOT": str(tmp_path)}
    done = subprocess.run(["bash", str(queue), "--config", str(cfg_path)], env=env, capture_output=True,
                          text=True, timeout=300)
    assert done.returncode == 0, done.stderr
    (logdir,) = list(tmp_path.glob("queue_*_resume_check_ne90"))
    cmd = (logdir / "0_resume.log").read_text()
    cfg = OmegaConf.load(cfg_path)
    assert "--method gpm_ne90" in cmd and f"--resume-sha256 {cfg.source.sha256.stage2}" in cmd
    assert "--resume-run results/seq_hetero__gpm_projected_adam_ne90__seed0" in cmd
    assert "--results-root results/resume_check_ne90" in cmd
    assert f"--config {cfg_path}" in (logdir / "1_check.log").read_text()
    base = OmegaConf.load(repo_root() / "configs" / "analysis" / "resume_check.yaml")
    assert cfg.source.stage == base.source.stage and set(cfg.source.sha256) == set(base.source.sha256)


def test_resume_refuses_a_different_joint_stage(gpm_source):
    """Co-trained tasks and the steps factor are part of a stage's identity; configs written before
    they existed compare as no co-training and factor 1."""
    from flowcl.train.continual import _stage_identity

    assert _stage_identity({"task_key": MILK, "n_demos": 1}) == (MILK, 1, (), 1)
    args = dict(method_name="gpm", method_kwargs=gpm_source["kwargs"], raw_policy_cfg=dict(TINY_POLICY),
                train_cfg=tiny_cfg(), seed=0)
    ketchup = "libero_object/pick_up_the_ketchup_and_place_it_in_the_basket"
    for first in ({"task_key": MILK, "n_demos": 1, "co_train": [ketchup]},
                  {"task_key": MILK, "n_demos": 1, "steps_factor": 2}):
        cur = load_curriculum({"name": "test_trio", "tasks": [
            first, {"task_key": SAUCE, "n_demos": 1}, {"task_key": BBQ, "n_demos": 1}]})
        with pytest.raises(ValueError, match="'curriculum'"):
            load_resume_source(gpm_source["run"], 2, cur, **args)
    load_resume_source(gpm_source["run"], 2, trio(), **args)   # the unchanged curriculum still resumes
