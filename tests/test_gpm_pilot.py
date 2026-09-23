"""GPM feasibility pilot: pre-registered criteria, provenance, and both arms end to end."""

from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from flowcl.analysis.metrics import Estimate, success_estimate
from flowcl.data.config import load_embodiment_spec
from flowcl.data.dataset import ChunkedActionDataset
from flowcl.data.episode import Episode
from flowcl.data.stats import compute_stats
from flowcl.envs.evaluation import EvaluationReport, TaskEvaluation
from flowcl.experiments import gpm_pilot
from flowcl.experiments.gate2 import collect_bases, load_subspace_config, save_checkpoint_bases
from flowcl.experiments.gpm_pilot import (
    PilotInputs,
    classify,
    compare_arms,
    criteria_thresholds,
    load_method_config,
    load_pilot_config,
    run_arm,
)
from flowcl.models.build import build_policy, load_policy_config
from flowcl.train.checkpoint import load_checkpoint, save_checkpoint
from flowcl.train.trainer import TrainConfig

T1, T2 = "toy_suite/spatial_like", "toy_suite/object_like"
BOOT = {"n_resamples": 2000, "confidence": 0.95, "seed": 0}


# ---- pre-registered criteria ----------------------------------------------------


def est(value, low=None, high=None):
    return Estimate(value, value - 0.1 if low is None else low, value + 0.1 if high is None else high, 50)


def test_thresholds_come_from_the_references_and_are_checked():
    cfg = load_pilot_config().criteria
    t = criteria_thresholds(0.78, 0.90, cfg)
    assert (t["object_min"], t["spatial_min"]) == (0.63, 0.75)
    with pytest.raises(ValueError, match="references changed"):
        criteria_thresholds(0.80, 0.90, cfg)


@pytest.mark.parametrize(
    "obj, spa, outcome",
    [
        (0.70, 0.80, "hard projection is viable"),
        (0.40, 0.80, "protection works, plasticity fails"),
        (0.70, 0.20, "plasticity works, protection fails"),
        (0.10, 0.10, "method or optimization failure"),
        (0.63, 0.75, "hard projection is viable"),  # thresholds are inclusive
        (0.62, 0.75, "protection works, plasticity fails"),
    ],
)
def test_four_way_outcome(obj, spa, outcome):
    t = {"object_min": 0.63, "spatial_min": 0.75}
    assert classify(est(obj, obj, obj), est(spa, spa, spa), t)["outcome"] == outcome


def test_borderline_is_flagged_but_the_point_estimate_decides():
    t = {"object_min": 0.63, "spatial_min": 0.75}
    c = classify(est(0.60, 0.46, 0.74), est(0.90, 0.80, 0.98), t)
    assert c["object_borderline"] and not c["object_acceptable"]
    assert not c["spatial_borderline"] and c["spatial_acceptable"]


# ---- end to end on a tiny policy -----------------------------------------------


@pytest.fixture(scope="module")
def spec():
    return load_embodiment_spec("libero_franka")


def make_episode(spec, task, length, seed):
    rng = np.random.default_rng(seed)
    h, w = spec.observation.image_size
    return Episode(
        images={c: rng.integers(0, 255, (length, h, w, 3), dtype=np.uint8) for c in spec.cameras},
        state=rng.normal(size=(length, spec.d_state)).astype(np.float32),
        action=rng.uniform(-1, 1, size=(length, spec.d_action)).astype(np.float32),
        language=f"do {task}",
        task_id=task,
        embodiment=spec.name,
    )


def build_world(spec, root):
    t1 = [make_episode(spec, T1, 18, s) for s in range(2)]
    t2 = [make_episode(spec, T2, 18, 10 + s) for s in range(2)]
    stats = compute_stats(t1, embodiment=spec.name, task_id=T1)
    datasets = {T1: ChunkedActionDataset(t1, spec, stats), T2: ChunkedActionDataset(t2, spec, stats)}

    torch.manual_seed(0)
    policy = build_policy("flowpolicy_small", spec, pretrained=False)
    for p in policy.parameters():
        if p.requires_grad:
            nn.init.normal_(p, std=0.05)
    stage0 = save_checkpoint(
        root / "toy_seq" / "checkpoints" / "stage0.pt", policy=policy,
        policy_config=load_policy_config("flowpolicy_small"), spec=spec, stats=stats,
        run_id="toy_seq", stage=0, task_key=T1,
    )
    sub_cfg = dataclasses.replace(
        load_subspace_config(), min_samples_per_dim=0.01, num_workers=0, batch_size=16
    )
    sub = collect_bases(load_checkpoint(stage0), datasets[T1], sub_cfg, device="cpu",
                        checkpoint_path=stage0)
    bases = save_checkpoint_bases(sub, sub_cfg, results_root=root)
    inputs = PilotInputs(
        curriculum=SimpleNamespace(name="toy", task_keys=(T1, T2), refs=("r_spatial", "r_object")),
        seq_run_dir=root / "toy_seq", stage0=stage0, stage1=stage0, t1_bases=bases, seed=0,
    )
    return root, inputs, datasets


@pytest.fixture(scope="module")
def world(spec, tmp_path_factory):
    return build_world(spec, tmp_path_factory.mktemp("results"))


def fake_evaluate(record, successes_by_task):
    def evaluate(policy, refs, spec, stats, run_id, cfg, bootstrap=None, stage=None):
        record.append(run_id)
        return EvaluationReport(
            run_id=run_id, stage=stage,
            tasks=[
                TaskEvaluation(key, s, [100] * len(s), [0] * len(s), success_estimate(s))
                for key, s in successes_by_task.items()
            ],
        )
    return evaluate


def run(arm, world, monkeypatch, successes):
    root, inputs, datasets = world
    seen: list[str] = []
    monkeypatch.setattr(gpm_pilot, "evaluate_tasks", fake_evaluate(seen, successes))
    result = run_arm(
        arm, inputs, datasets, load_method_config(),
        TrainConfig(steps=4, batch_size=4, device="cpu", amp=False, num_workers=0,
                    warmup_steps=1, log_every=0),
        eval_cfg=None, bootstrap=BOOT,
        pilot=dataclasses.replace(load_pilot_config(), timing_warmup_steps=1),
        device="cpu", steps=4, evaluate=True, sanity=False, results_root=root,
    )
    return result, seen


GPM_SUCC = {T1: [True] * 40 + [False] * 10, T2: [True] * 30 + [False] * 20}
FREEZE_SUCC = {T1: [False] * 50, T2: [True] * 45 + [False] * 5}


def test_both_arms_end_to_end(world, monkeypatch):
    root, inputs, _ = world
    gpm, seen = run("gpm_projected_adam", world, monkeypatch, GPM_SUCC)
    freeze, _ = run("freeze_only", world, monkeypatch, FREEZE_SUCC)

    # Rollouts use seq_ft's run id as the seed namespace; provenance keeps both ids.
    assert seen == ["toy_seq"]
    for r in (gpm, freeze):
        run_dir = root / r.method_run_id
        eval_file = json.loads((run_dir / "eval" / "stage1.json").read_text())
        assert eval_file["method_run_id"] == r.method_run_id
        assert eval_file["evaluation_seed_run_id"] == "toy_seq"
        summary = json.loads((run_dir / "pilot.json").read_text())
        assert summary["method_run_id"] == r.method_run_id
        assert summary["evaluation_seed_run_id"] == "toy_seq"
        assert r.frozen_changed == []
        assert r.freeze_report["trainable_tensors"] == len(r.displacement_norm)
        assert (run_dir / "checkpoints" / "stage1.pt").is_file()
        assert set(r.probes) == {T1, T2}

    assert gpm.method_run_id == "toy__gpm_projected_adam_pilot__seed0"
    # Projection: the total displacement stays out of the T1 subspace; without it, it does not.
    assert gpm.update["c_global"]["0.95"]["all"] < 1e-3
    assert freeze.update["c_global"]["0.95"]["all"] > 0.05
    assert gpm.displacement_norm["flow_head.action_in"] == 0.0
    assert gpm.method_state["display_name"] == "gpm_projected_adam"

    eval0 = {T1: TaskEvaluation(T1, [True] * 45 + [False] * 5, [], [], success_estimate([1] * 45 + [0] * 5))}
    eval1 = {
        T1: TaskEvaluation(T1, [False] * 50, [], [], success_estimate([0] * 50)),
        T2: TaskEvaluation(T2, [True] * 39 + [False] * 11, [], [], success_estimate([1] * 39 + [0] * 11)),
    }
    thresholds = criteria_thresholds(0.78, 0.90, load_pilot_config().criteria)
    out = compare_arms({"gpm_projected_adam": gpm, "freeze_only": freeze}, eval0, eval1, T1, T2,
                       thresholds, BOOT)
    diff = out["paired_differences"]["gpm_projected_adam_minus_freeze_only"]
    assert diff[T1]["diff"] == pytest.approx(0.8 - 0.0)
    assert diff[T2]["diff"] == pytest.approx(0.6 - 0.9)
    assert out["outcomes"]["gpm_projected_adam"]["decides"] is True
    assert out["outcomes"]["gpm_projected_adam"]["outcome"] == "protection works, plasticity fails"
    assert out["outcomes"]["freeze_only"]["decides"] is False


# ---- the orchestration (references, timing, sanity checks, report) --------------


def write_seq_run(root, spec_train_steps):
    """The Gate 1 artifacts the pilot reads: config.yaml, result.json, eval/stage{0,1}.json."""
    from omegaconf import OmegaConf

    run_dir = root / "toy_seq"
    OmegaConf.save(
        OmegaConf.create({
            "train": {"steps": spec_train_steps, "batch_size": 4, "lr": 1e-4, "weight_decay": 1e-4,
                      "grad_clip": 1.0, "warmup_steps": 1, "log_every": 0, "num_workers": 0,
                      "device": "cpu", "accumulation_steps": 1, "amp": False},
            "eval": {"n_episodes": 50, "max_steps": 600, "execute_k": None, "euler_steps": 10,
                     "temporal_ensembling": False, "temporal_ensemble_coef": 0.01,
                     "image_size": 128, "record_video": False},
        }),
        run_dir / "config.yaml",
    )
    (run_dir / "result.json").write_text(json.dumps({"stages": [
        {}, {"final_loss": 0.0076, "mean_last_50_loss": 0.0075, "train_wall_clock_s": 2558.0,
             "steps": 30000}]}))

    def task(key, succ):
        return TaskEvaluation(key, succ, [100] * 50, [0] * 50, success_estimate(succ))

    EvaluationReport("toy_seq", 0, [task(T1, [True] * 45 + [False] * 5)]).save(
        run_dir / "eval" / "stage0.json")
    EvaluationReport("toy_seq", 1, [task(T1, [False] * 50),
                                    task(T2, [True] * 39 + [False] * 11)]).save(
        run_dir / "eval" / "stage1.json")


@pytest.fixture()
def pilot_world(spec, tmp_path, monkeypatch):
    import flowcl.data.tasks as tasks
    import flowcl.train.pipeline as pipeline

    root, inputs, datasets = build_world(spec, tmp_path / "results")
    write_seq_run(root, spec_train_steps=4)
    monkeypatch.setattr(tasks.TaskRef, "from_key", staticmethod(lambda key: key))
    monkeypatch.setattr(pipeline, "build_dataset", lambda refs, *a, **k: datasets[refs[0]])
    cfg = dataclasses.replace(load_pilot_config(), sanity_steps=6, freeze_only_timing_steps=3,
                              timing_warmup_steps=1)
    return root, inputs, cfg


def test_sanity_mode_runs_the_whole_orchestration(pilot_world):
    root, inputs, cfg = pilot_world
    out_dir = root / "gpm_pilot"
    try:
        gpm_pilot.run_gpm_pilot(inputs, cfg, load_method_config(), device="cpu", sanity=True,
                                results_root=root, out_dir=out_dir)
    except RuntimeError as exc:  # 6 steps from random weights may not reduce the loss
        assert "object_loss_decreases" in str(exc) and "updates_orthogonal" not in str(exc)
    report = json.loads((out_dir / "sanity.json").read_text())
    checks = report["sanity_checks"]
    for name in ("updates_orthogonal", "parameters_move", "action_in_frozen",
                 "frozen_untouched", "finite"):
        assert checks[name]["passed"], (name, checks[name])
    assert report["criteria"]["object_min"] == 0.63
    assert set(report["timing"]["extrapolated_30000_steps_min"]) == {"gpm_projected_adam", "freeze_only"}
    assert report["evaluation_seed_run_id"] == "toy_seq"


def test_full_mode_writes_outcomes_and_paired_differences(pilot_world, monkeypatch):
    root, inputs, cfg = pilot_world
    seen: list[str] = []
    monkeypatch.setattr(gpm_pilot, "evaluate_tasks", fake_evaluate(seen, GPM_SUCC))
    report = gpm_pilot.run_gpm_pilot(inputs, cfg, load_method_config(), device="cpu",
                                     results_root=root, out_dir=root / "gpm_pilot")
    assert seen == ["toy_seq", "toy_seq"]
    assert set(report["arms"]) == {"gpm_projected_adam", "freeze_only"}
    assert "gpm_projected_adam_minus_freeze_only" in report["paired_differences"]
    assert report["outcomes"]["gpm_projected_adam"]["decides"]
    saved = json.loads((root / "gpm_pilot" / "pilot.json").read_text())
    assert saved["outcomes"] == report["outcomes"]
