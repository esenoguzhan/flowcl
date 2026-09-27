"""The exploratory budget/learning-rate study: the split, the budget rule, checkpointing,
probe banks, the CL pairing, both parts end to end on the tiny policy, and the queue."""

from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from flowcl.analysis.metrics import success_estimate
from flowcl.data.config import load_embodiment_spec
from flowcl.data.dataset import ChunkedActionDataset
from flowcl.data.episode import Episode
from flowcl.data.stats import compute_stats
from flowcl.envs.evaluation import EvaluationReport, TaskEvaluation
from flowcl.experiments.hparam_study import (
    CheckpointAt,
    ProbeBank,
    choose_budget,
    load_study_config,
    run_cl_part,
    run_trajectory,
    split_episodes,
    stage_seed,
)
from flowcl.models.build import build_policy
from flowcl.utils.libero_paths import repo_root
from flowcl.utils.seeding import derive_seed

OBJECT = "libero_object/pick_up_the_milk_and_place_it_in_the_basket"
SPATIAL = "libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate"
TINY_POLICY = {"d_model": 384, "n_trunk_layers": 6, "n_heads": 8, "n_decoder_layers": 2,
               "n_context_tokens": 4, "pretrained": False, "euler_steps": 2}
TINY_TRAIN = {"steps": 4, "batch_size": 2, "lr": 1e-4, "weight_decay": 1e-4, "grad_clip": 1.0,
              "warmup_steps": 1, "log_every": 0, "num_workers": 0, "device": "cpu",
              "accumulation_steps": 1, "amp": False}


@pytest.fixture(scope="module")
def spec():
    return load_embodiment_spec("libero_franka")


def episode(spec, task, seed, length=18):
    rng = np.random.default_rng(seed)
    h, w = spec.observation.image_size
    return Episode(
        images={c: rng.integers(0, 255, (length, h, w, 3), dtype=np.uint8) for c in spec.cameras},
        state=rng.normal(size=(length, spec.d_state)).astype(np.float32),
        action=rng.uniform(-1, 1, size=(length, spec.d_action)).astype(np.float32),
        language=f"do {task}", task_id=task, embodiment=spec.name,
    )


def episodes(spec, task, n=5, offset=0):
    return [episode(spec, task, offset + i) for i in range(n)]


def tiny_policy(spec):
    torch.manual_seed(0)
    policy = build_policy(TINY_POLICY, spec, pretrained=False)
    for p in policy.parameters():
        if p.requires_grad:
            nn.init.normal_(p, std=0.05)
    return policy


def stub_evaluator(bootstrap_seed=0):
    """Deterministic 'rollouts': episode seeds 0..9, success from a hash of the run id."""
    def evaluate(policy, refs, spec, stats, run_id, stage):
        tasks = []
        for ref in refs:
            seeds = list(range(10))
            successes = [bool((derive_seed(run_id, ref.task_key, i) >> 3) % 2) for i in seeds]
            tasks.append(TaskEvaluation(ref.task_key, successes, [100] * 10, seeds,
                                        success_estimate(successes, seed=bootstrap_seed)))
        return EvaluationReport(run_id=run_id, stage=stage, tasks=tasks)
    return evaluate


BOOT = {"seed": 0, "n_resamples": 200, "confidence": 0.95}


# ---- pure pieces --------------------------------------------------------------------------


def test_split_holds_out_the_last_demos():
    eps = list(range(50))
    train, held = split_episodes(eps, 5)
    assert train == list(range(45)) and held == list(range(45, 50))
    assert split_episodes(eps, 5) == (train, held)
    for bad in (0, 50, -1):
        with pytest.raises(ValueError, match="n_heldout"):
            split_episodes(eps, bad)


def rows(*pairs):
    return [{"budget": b, "new_task": {"success": s}} for b, s in pairs]


def test_budget_rule_boundaries_and_fallback():
    r = rows((2500, 0.55), (5000, 0.72), (10000, 0.68), (15000, 0.80), (20000, 0.90))
    assert choose_budget(r, 0.78, 0.10, 30000)["b_star"] == 5000       # 0.72 >= 0.68
    assert choose_budget(r, 0.83, 0.10, 30000)["b_star"] == 15000      # 0.72 < 0.73
    assert choose_budget(r, 0.70, 0.10, 30000)["b_star"] == 5000       # 0.55 < 0.60 <= 0.72
    edge = choose_budget(rows((2500, 0.68), (5000, 0.9)), 0.78, 0.10, 30000)
    assert edge["b_star"] == 2500 and edge["floor"] == pytest.approx(0.68)  # inclusive
    none = choose_budget(r, 1.2, 0.10, 30000)
    assert none["b_star"] == 30000 and none["fallback_used"] and none["qualifying"] == []


def test_stage_seed_is_the_runners_stream_seed():
    from flowcl.train.continual import seed_namespace_run_id

    ns = seed_namespace_run_id("seq_hetero", 0)
    assert ns == "seq_hetero__seq_ft__seed0"
    assert stage_seed(ns, OBJECT, 1) == derive_seed(ns, OBJECT, 1)
    source = (repo_root() / "flowcl" / "train" / "continual.py").read_text()
    assert "derive_seed(seed_ns, stage.task_key, stage_idx)" in source  # the runner's call


# ---- checkpointing and probe banks -------------------------------------------------------------


def test_checkpoints_are_saved_at_exactly_the_registered_steps(spec, tmp_path):
    from flowcl.train.checkpoint import load_checkpoint
    from flowcl.train.trainer import TrainConfig, train_one_task

    eps = episodes(spec, OBJECT)
    stats = compute_stats(eps, embodiment=spec.name, task_id=OBJECT)
    data = ChunkedActionDataset(eps, spec, stats)
    policy = tiny_policy(spec)
    saver = CheckpointAt([3, 1], tmp_path, policy, TINY_POLICY, spec, stats, "r", OBJECT)
    train_one_task(policy, data, TrainConfig(**TINY_TRAIN), generator=torch.Generator().manual_seed(0),
                   on_step=saver)
    assert sorted(saver.saved) == [1, 3] and sorted(p.name for p in tmp_path.iterdir()) == \
        ["step1.pt", "step3.pt"]
    reloaded = load_checkpoint(saver.saved[3]).policy.state_dict()
    step1 = load_checkpoint(saver.saved[1]).policy.state_dict()
    live = {k: v for k, v in policy.state_dict().items()}
    trained = [k for k, p in policy.named_parameters() if p.requires_grad]
    # 4 steps ran: step 3's checkpoint is not the final weights, step 1's is different again.
    assert any(not torch.equal(reloaded[k], live[k]) for k in trained)
    assert any(not torch.equal(reloaded[k], step1[k]) for k in trained)
    with pytest.raises(ValueError, match="positive"):
        CheckpointAt([0], tmp_path, policy, TINY_POLICY, spec, stats, "r", OBJECT)


def test_probe_bank_is_fixed_and_its_loss_is_the_element_weighted_mean(spec):
    from flowcl.data.dataset import collate_chunks
    from flowcl.models.losses import valid_element_count

    eps = episodes(spec, OBJECT, n=2)
    stats = compute_stats(eps, embodiment=spec.name, task_id=OBJECT)
    data = ChunkedActionDataset(eps, spec, stats)
    policy = tiny_policy(spec)
    idx = [0, 3, 5]
    a = ProbeBank.build("heldout", data, idx, 3, policy.s_sampler, seed=7)
    b = ProbeBank.build("heldout", data, idx, 3, policy.s_sampler, seed=7)
    assert torch.equal(a.s, b.s) and torch.equal(a.noise, b.noise) and a.indices.tolist() == idx
    assert not torch.equal(a.s[:, 0], a.s[:, 1]) and not torch.equal(a.noise[:, 0], a.noise[:, 2])
    first = a.loss(policy, data, batch_size=2, device="cpu")
    assert a.loss(policy, data, batch_size=2, device="cpu") == first  # reusable, unchanged
    total = weight = 0.0
    with torch.no_grad():
        for row, i in enumerate(idx):
            batch = collate_chunks([data[i]])
            n = float(valid_element_count(batch["action_mask"], policy.d_action))
            for k in range(3):
                loss = float(policy(batch, s=a.s[row:row + 1, k], noise=a.noise[row:row + 1, k])["loss"])
                total += loss * n
                weight += n
    assert first["loss"] == pytest.approx(total / weight, rel=1e-5)
    assert first["n_chunks"] == 3 and first["k_draws"] == 3 and len(first["per_draw"]) == 3


# ---- end to end on the tiny policy -------------------------------------------------------------


def study_config(**overrides):
    cfg = copy.deepcopy(load_study_config())
    cfg["trajectory"].update(n_heldout=2, checkpoint_steps=[1, 2, 4])
    cfg["trajectory"]["bank"].update(k_draws=2, batch_size=4)
    cfg["cl"].update(budgets=[2, 3], learning_rates=[3e-4])
    for k, v in overrides.items():
        cfg[k] = v
    return cfg


class Ref:
    def __init__(self, task_key):
        self.task_key = task_key


def write_gate0(root, spec, task):
    from flowcl.experiments.gate0 import single_task_run_id

    run_dir = root / single_task_run_id(task, 0)
    run_dir.mkdir(parents=True)
    OmegaConf.save(OmegaConf.create({"seed": 0, "embodiment": spec.to_dict(), "policy": TINY_POLICY,
                                     "train": TINY_TRAIN, "data": {"n_demos": 5}}),
                   run_dir / "config.yaml")
    stub_evaluator()(None, [Ref(task)], spec, None, run_dir.name, 0).save(run_dir / "eval.json")
    return run_dir


def test_trajectory_end_to_end(spec, tmp_path):
    write_gate0(tmp_path, spec, OBJECT)
    out = run_trajectory(study_config(), OBJECT, results_root=tmp_path, device="cpu",
                         allow_dirty=True, evaluator=stub_evaluator(), bootstrap=BOOT,
                         load_episodes=lambda ref, s: episodes(spec, OBJECT), pretrained=False)
    assert [r["step"] for r in out["rows"]] == [1, 2, 4]
    assert out["demos"] == {"train": 3, "heldout": 2}
    assert out["banks"]["heldout"]["n_chunks"] == out["banks"]["train"]["n_chunks"] > 0
    for row in out["rows"]:
        assert row["paired_vs_gate0_30k"]["diff"] == pytest.approx(0.0)  # same stub, same seeds
        assert row["heldout_bank"]["loss"] > 0 and row["train_bank"]["loss"] > 0
    run_dir = tmp_path / "hparam_study" / "trajectory" / "traj__libero_object__seed0"
    assert json.loads((run_dir / "report.json").read_text())["rows"] == json.loads(json.dumps(out["rows"]))
    banks = torch.load(run_dir / "banks.pt", weights_only=False)
    held, train = set(banks["heldout"]["indices"].tolist()), banks["train"]["indices"].tolist()
    assert len(train) == len(held) and len(set(train)) == len(train)
    assert len(json.loads((run_dir / "losses.json").read_text())) == 4
    assert sorted(p.name for p in (run_dir / "checkpoints").iterdir()) == \
        ["step1.pt", "step2.pt", "step4.pt"]
    with pytest.raises(ValueError, match="not a registered"):
        run_trajectory(study_config(), SPATIAL, results_root=tmp_path, allow_dirty=True)


def write_reference(root, spec):
    """A tiny seq_ft-like reference: stage0.pt, config, result, stage-1 eval, diagnostics."""
    from flowcl.train.checkpoint import save_checkpoint

    run_id = "seq_hetero__seq_ft__seed0"
    run_dir = root / run_id
    stats = compute_stats(episodes(spec, SPATIAL), embodiment=spec.name, task_id=SPATIAL)
    save_checkpoint(run_dir / "checkpoints" / "stage0.pt", policy=tiny_policy(spec),
                    policy_config=TINY_POLICY, spec=spec, stats=stats, run_id=run_id, stage=0,
                    task_key=SPATIAL)
    curriculum = {"name": "seq_hetero", "tasks": [{"task_key": SPATIAL, "n_demos": 5},
                                                  {"task_key": OBJECT, "n_demos": 5}]}
    OmegaConf.save(OmegaConf.create({"run_id": run_id, "seed": 0, "curriculum": curriculum,
                                     "train": TINY_TRAIN}), run_dir / "config.yaml")
    # As Gate 1 wrote it: no seed_namespace_run_id field (it predates the field).
    (run_dir / "result.json").write_text(json.dumps({"run_id": run_id}))
    refs = [Ref(SPATIAL), Ref(OBJECT)]
    stub_evaluator()(None, refs, spec, None, run_id, 1).save(run_dir / "eval" / "stage1.json")
    diag = root / "forgetting_diag" / "report.json"
    diag.parent.mkdir(parents=True)
    diag.write_text(json.dumps({
        "reference_run_id": run_id,
        "config": {"probe": {"batch_size": 2, "n_batches": 2, "seed_tags": {
            "shuffle": "t::shuffle", "flow_time": "t::flow_time", "noise": "t::noise"}}},
        "loss_matrix": {"reference": [[0.01, 1.0], [0.7, 0.006]]},
    }))
    return run_dir


def test_cl_parts_end_to_end(spec, tmp_path):
    write_reference(tmp_path, spec)
    make = lambda stage: ChunkedActionDataset(  # noqa: E731
        episodes(spec, stage.task_key, offset=0 if stage.task_key == SPATIAL else 10), spec,
        compute_stats(episodes(spec, SPATIAL), embodiment=spec.name, task_id=SPATIAL))
    common = dict(results_root=tmp_path, device="cpu", allow_dirty=True,
                  evaluator=stub_evaluator(), bootstrap=BOOT, build_datasets=make)
    budget = run_cl_part(study_config(), "cl_budget", **common)
    assert [(r["budget"], r["lr"]) for r in budget["rows"]] == [(2, 1e-4), (3, 1e-4)]
    assert budget["reference"]["budget"] == 4 and budget["reference"]["probe_loss"]["retained"] == 0.7
    for row in budget["rows"]:
        # The stub makes success a function of the rollout namespace: seq_ft's, so paired = 0.
        assert row["new_task"]["paired_vs_reference"]["diff"] == pytest.approx(0.0)
        assert row["probe_loss"]["retained"] > 0 and row["probe_loss"]["new"] > 0
        cfg = OmegaConf.to_container(OmegaConf.load(
            tmp_path / "hparam_study" / "cl_budget" / row["run_id"] / "config.yaml"))
        assert cfg["stream_seed"] == {"namespace": "seq_hetero__seq_ft__seed0", "stage": 1}
        assert cfg["train"]["steps"] == row["budget"]
    lr = run_cl_part(study_config(), "cl_lr", **common)
    choice = lr["budget_choice"]
    assert choice["b_star"] == 2  # the stub's new-task success equals the reference's
    assert [(r["budget"], r["lr"]) for r in lr["rows"]] == [(2, 3e-4)]
    assert lr["base_lr_row"]["budget"] == 2
    with pytest.raises(ValueError, match="unknown CL part"):
        run_cl_part(study_config(), "trajectory", **common)


def test_reference_namespace_follows_the_runner_rule_and_must_agree(spec, tmp_path):
    from flowcl.experiments.hparam_study import _reference

    run_dir = write_reference(tmp_path, spec)
    assert _reference(study_config(), tmp_path).seed_namespace == "seq_hetero__seq_ft__seed0"
    (run_dir / "result.json").write_text(json.dumps({"seed_namespace_run_id": "other__seed0"}))
    with pytest.raises(ValueError, match="disagrees"):
        _reference(study_config(), tmp_path)
    (run_dir / "result.json").write_text(json.dumps({"run_id": run_dir.name}))
    stub_evaluator()(None, [Ref(SPATIAL), Ref(OBJECT)], spec, None, "elsewhere", 1).save(
        run_dir / "eval" / "stage1.json")
    with pytest.raises(ValueError, match="disagrees"):
        _reference(study_config(), tmp_path)


# ---- config and queue ---------------------------------------------------------------------------


def test_committed_config_matches_the_plan():
    cfg = load_study_config()
    t, c = cfg["trajectory"], cfg["cl"]
    assert cfg["seed"] == 0
    assert [k.split("/")[0] for k in t["tasks"]] == ["libero_object", "libero_10"]
    assert t["n_heldout"] == 5 and t["bank"]["k_draws"] == 8
    assert t["checkpoint_steps"] == [1000, 2500, 5000, 10000, 15000, 20000, 30000]
    assert c["budgets"] == [2500, 5000, 10000, 15000, 20000]
    assert c["learning_rates"] == [3e-5, 3e-4]
    assert c["budget_rule"] == {"tolerance": 0.10, "fallback": 30000}
    assert (c["reference_run"], c["task_index"], c["retained_index"]) == \
        ("seq_hetero__seq_ft__seed0", 1, 0)


QUEUE = repo_root() / "scripts" / "queue_hparam_study.sh"
STEPS = ["0_trajectory_libero_object", "1_trajectory_libero_10", "2_cl_budget", "3_cl_lr"]


def queue(tmp_path, args=(), fail="", present=()):
    results = tmp_path / "results"
    for rel in present:
        path = results / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    env = {**os.environ, "QUEUE_DRY_RUN": "1", "QUEUE_FAIL_STEP": fail,
           "QUEUE_LOG_ROOT": str(tmp_path / "logs"), "QUEUE_RESULTS_ROOT": str(results)}
    done = subprocess.run(["bash", str(QUEUE), *args], env=env, capture_output=True, text=True,
                          timeout=60)
    assert done.returncode == 0, done.stderr
    (logdir,) = list((tmp_path / "logs").glob("queue_*_hparam"))
    return (logdir / "queue.log").read_text(), logdir


def test_queue_runs_every_part_in_order_with_its_flags(tmp_path):
    log, logdir = queue(tmp_path)
    positions = [log.index(f"START {s}") for s in STEPS]
    assert positions == sorted(positions) and log.rstrip().endswith("QUEUE DONE")
    assert "--part trajectory --task-index 1" in (logdir / "1_trajectory_libero_10.log").read_text()
    assert "--part cl_lr" in (logdir / "3_cl_lr.log").read_text()


def test_queue_skips_cl_lr_without_cl_budget(tmp_path):
    log, _ = queue(tmp_path, fail="2_cl_budget")
    assert "START 3_cl_lr" not in log and "SKIP 3_cl_lr" in log
    log, _ = queue(tmp_path / "b", fail="0_trajectory_libero_object")
    assert "START 2_cl_budget" in log  # the parts are independent otherwise


def test_queue_resume_and_its_mirror_of_the_config(tmp_path):
    log, _ = queue(tmp_path, args=("--from-step", "3"),
                   present=["hparam_study/cl_budget/report.json"])
    assert "PRIOR 2_cl_budget ok" in log and "START 3_cl_lr" in log
    assert "START 2_cl_budget" not in log
    log, _ = queue(tmp_path / "b", args=("--from-step", "3"))
    assert "PRIOR 2_cl_budget missing" in log and "START 3_cl_lr" not in log
    bad = subprocess.run(["bash", str(QUEUE), "--from-step", "4"], capture_output=True, text=True)
    assert bad.returncode == 2
    suites = re.search(r'TRAJ_SUITES="([^"]+)"', QUEUE.read_text()).group(1).split()
    assert suites == [k.split("/")[0] for k in load_study_config()["trajectory"]["tasks"]]
    done = subprocess.run([sys.executable, str(repo_root() / "scripts" / "hparam_study.py"), "--help"],
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0 and "--task-index" in done.stdout and "--allow-dirty" in done.stdout
