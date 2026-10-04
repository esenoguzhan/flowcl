"""Similar-vs-diverse continual study: the allowlist_ft control, the stage-0 and stage-1 probes on the
tiny policy (text-cache precision, weights unchanged, both statistics), the summary on fake run
directories (paired differences, P1's judgments and basis, assessability flags, every validity
condition), the queue, the curricula and the config."""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from flowcl.analysis.metrics import success_estimate
from flowcl.envs.evaluation import EvaluationReport, TaskEvaluation
from flowcl.experiments import similar_cl as scl
from flowcl.utils.libero_paths import repo_root
from flowcl.utils.run import file_sha256
from test_low_update import TINY_POLICY, episodes, spec, tiny_policy  # noqa: F401

A = "libero_goal/put_the_bowl_on_the_plate"
STOVE = "libero_goal/put_the_bowl_on_the_stove"
MILK = "libero_object/pick_up_the_milk_and_place_it_in_the_basket"
CAB = "libero_goal/put_the_bowl_on_top_of_the_cabinet"
CUR = {"similar_ab_stove": STOVE, "similar_ab_milk": MILK, "similar_ab_cabinet": CAB}
SEED_BASE = {A: 0, STOVE: 1000, MILK: 2000, CAB: 3000}
N = 50


# ---- allowlist_ft ----------------------------------------------------------------------------------------


def test_allowlist_ft_freezes_from_task_two(spec):
    from flowcl.data.dataset import ChunkedActionDataset
    from flowcl.data.stats import compute_stats
    from flowcl.methods.base import build_method
    from flowcl.methods.gpm import allowlist
    from flowcl.train.trainer import TrainConfig, build_dataloader, build_optimizer

    policy = tiny_policy(spec)
    method = build_method("allowlist_ft")
    with pytest.raises(TypeError):
        build_method("allowlist_ft", eps=0.9)
    before = {n for n, p in policy.named_parameters() if p.requires_grad}
    method.on_task_start(policy, 0, context=None)
    assert {n for n, p in policy.named_parameters() if p.requires_grad} == before
    method.on_task_start(policy, 1, context=None)
    allowed = set(allowlist(policy))
    assert {n for n, p in policy.named_parameters() if p.requires_grad} == allowed and allowed < before
    rep = method.state_dict()["freeze_reports"]["1"]
    assert rep["trainable_tensors"] == len(allowed) and rep["newly_frozen"]
    opt = build_optimizer(policy, TrainConfig(steps=1))
    in_opt = {id(p) for g in opt.param_groups for p in g["params"]}
    named = dict(policy.named_parameters())
    assert in_opt == {id(named[n]) for n in allowed}
    eps = episodes(spec, A)
    ds = ChunkedActionDataset(eps, spec, compute_stats(eps, embodiment=spec.name, task_id=A))
    batch = next(iter(build_dataloader(ds, batch_size=4, shuffle=False)))
    state = {k: v.clone() for k, v in policy.state_dict().items()}
    policy.train()
    for _ in range(2):
        opt.zero_grad()
        policy(batch)["loss"].backward()
        opt.step()
    after = policy.state_dict()
    assert all(torch.equal(state[k], after[k]) for k in state if k not in allowed)
    assert any(not torch.equal(state[k], after[k]) for k in allowed)
    assert method.restore_state({"method_state": method.state_dict()}, Path("."), 0) == {}


# ---- probes on the tiny policy ------------------------------------------------------------------------------


def probe_world(tmp_path, spec):
    from flowcl.data.dataset import ChunkedActionDataset
    from flowcl.data.stats import compute_stats
    from flowcl.train.checkpoint import save_checkpoint

    stats = compute_stats(episodes(spec, A), embodiment=spec.name, task_id=A)
    sub = OmegaConf.load(repo_root() / "configs" / "analysis" / "subspace.yaml")
    sub.min_samples_per_dim, sub.num_workers, sub.batch_size, sub.probe_batch_size = 0.01, 0, 4, 4
    OmegaConf.save(sub, tmp_path / "tiny_subspace.yaml")
    cfg = copy.deepcopy(scl.load_config())
    cfg["swap"].update(capture_config=str(tmp_path / "tiny_subspace.yaml"), n_batches=2)
    cfg["probe"].update(batch_size=4, n_batches=2)
    cfg["chunks"].update(batch_size=4, n_batches=2)
    conf = {"seed": 0, "train": {"amp": True}, "eval": {"n_episodes": 2, "euler_steps": 2}}
    keys = [A, STOVE, MILK, CAB]
    src = tmp_path / scl.run_name(cfg, "similar_ab_stove", "seq_ft")
    src.mkdir(parents=True)
    OmegaConf.save(OmegaConf.create(conf), src / "config.yaml")
    save_checkpoint(src / "checkpoints" / "stage0.pt", policy=tiny_policy(spec), policy_config=TINY_POLICY,
                    spec=spec, stats=stats, run_id=src.name, stage=0, task_key=A)
    run = tmp_path / scl.run_name(cfg, "similar_ab_milk", "gpm")
    run.mkdir()
    OmegaConf.save(OmegaConf.create(conf), run / "config.yaml")
    moved = tiny_policy(spec)
    with torch.no_grad():
        for p in moved.parameters():
            if p.requires_grad:
                p.add_(0.01)
    save_checkpoint(run / "checkpoints" / "stage1.pt", policy=moved, policy_config=TINY_POLICY,
                    spec=spec, stats=stats, run_id=run.name, stage=1, task_key=MILK)

    def build(key):
        return ChunkedActionDataset(episodes(spec, key, offset=10 * keys.index(key)), spec, stats)

    def evaluator(policy, refs, spec_, stats_, run_id, ecfg, bootstrap=None, stage=None, progress=True):
        report = EvaluationReport(run_id=run_id, stage=stage)
        for ref in refs:
            succ = [True] * ecfg.n_episodes
            report.tasks.append(TaskEvaluation(task_key=ref.task_key, successes=succ, n_steps=[1] * len(succ),
                                               seeds=list(range(SEED_BASE[ref.task_key], SEED_BASE[ref.task_key] + len(succ))),
                                               estimate=success_estimate(succ)))
        return report

    return cfg, dict(results_root=tmp_path, device="cpu", allow_dirty=True, build_datasets=build), evaluator


def test_stage0_and_stage1_probes(spec, tmp_path):
    cfg, kw, evaluator = probe_world(tmp_path, spec)
    rec = scl.run_stage0(cfg, **kw, evaluator=evaluator)
    assert rec["weights_unchanged"] and rec["device_type"] == "cpu" and rec["amp"]
    assert [rec["text_cache"][k]["precision"] for k in (A, STOVE, MILK, CAB)] == ["fp32"] * 4  # no AMP on CPU
    assert rec["text_cache"][A]["text"] == f"do {A}"
    assert set(rec["probe_loss"]) == {A, STOVE, MILK, CAB} and all(v > 0 for v in rec["probe_loss"].values())
    assert set(rec["instruction"]) == {STOVE, MILK, CAB}
    for b, pr in rec["instruction"].items():
        assert set(pr) == {"A_data_B_instruction", "B_data_A_instruction"}
        assert pr["A_data_B_instruction"]["reference"] == f"do {A}" and pr["A_data_B_instruction"]["swap"] == f"do {b}"
        for d in pr.values():
            assert d["velocity"]["sensitivity"] > 0 and d["chunks"]["relative"] > 0 and d["chunks"]["n_samples"] == 8
    zero = EvaluationReport.load(tmp_path / "similar_cl" / "stage0" / "zero_shot.json")
    assert [t.task_key for t in zero.tasks] == [MILK, CAB] and zero.run_id == "similar_ab_stove__seq_ft__seed0"
    assert rec["zero_shot"]["sha256"] == file_sha256(tmp_path / "similar_cl" / "stage0" / "zero_shot.json")
    assert scl.run_stage0(cfg, **kw, evaluator=evaluator) == rec                       # published: reused
    ckpt0 = tmp_path / "similar_ab_stove__seq_ft__seed0" / "checkpoints" / "stage0.pt"
    keys0 = [A, STOVE, MILK, CAB]
    assert scl.probe_artifact(scl.stage0_path(cfg, tmp_path), ckpt0, keys0, smoke=True)["status"] == "valid"
    assert scl.probe_artifact(scl.stage0_path(cfg, tmp_path), ckpt0, keys0, smoke=False)["problems"] == ["device cpu"]

    p = scl.run_probe(cfg, "similar_ab_milk", "gpm", **kw)
    assert p["weights_unchanged"] and set(p["probe_loss"]) == {A, MILK} and set(p["instruction"]) == {MILK}
    # identical probe inputs at both checkpoints: a different model gives different numbers
    assert p["probe_loss"][A] != rec["probe_loss"][A]
    ppath = scl.probe_path(cfg, tmp_path, "similar_ab_milk", "gpm")
    ckpt1 = tmp_path / "similar_ab_milk__gpm_projected_adam__seed0" / "checkpoints" / "stage1.pt"
    assert scl.probe_artifact(ppath, ckpt1, [A, MILK], smoke=True)["status"] == "valid"
    with open(ckpt1, "ab") as f:
        f.write(b"x")
    assert scl.probe_artifact(ppath, ckpt1, [A, MILK], smoke=True)["problems"] == ["checkpoint_sha256"]
    # the queue's probe step: a missing stage1.pt is a problem, the others are still probed (or reused)
    problems = scl.run_probes(cfg, ["similar_ab_milk"], **kw)
    assert len(problems) == 2 and all("missing" in x for x in problems)


def test_warm_cache_precision_and_freshness(spec):
    policy = tiny_policy(spec)
    assert scl.warm_cache(policy, ["a", "b"], "cpu", amp=True) == ["fp32", "fp32"]
    with pytest.raises(RuntimeError, match="not fresh"):
        scl.warm_cache(policy, ["a"], "cpu", amp=True)
    assert scl.expected_precision(3, True, "cuda") == ["amp", "fp32", "fp32"]
    assert scl.expected_precision(2, False, "cuda") == ["fp32", "fp32"]


# ---- the summary on fake run directories -------------------------------------------------------------------

MATRIX = {  # arm -> (R00, R10, R11) per curriculum
    "similar_ab_stove": {"seq_ft": (0.9, 0.2, 0.9), "allowlist_ft": (0.9, 0.5, 0.8), "gpm": (0.9, 0.8, 0.5)},
    "similar_ab_milk": {"seq_ft": (0.9, 0.3, 0.9), "allowlist_ft": (0.9, 0.5, 0.8), "gpm": (0.9, 0.8, 0.76)},
    "similar_ab_cabinet": {"seq_ft": (0.9, 0.2, 0.9), "allowlist_ft": (0.9, 0.5, 0.8), "gpm": (0.9, 0.8, 0.6)},
}


def _task(key, rate, n=N, seed_shift=0):
    k = round(rate * n)
    succ = [True] * k + [False] * (n - k)
    e = success_estimate(succ)
    base = SEED_BASE[key] + seed_shift
    return {"task_key": key, "successes": succ, "n_steps": [100] * n, "seeds": list(range(base, base + n)),
            "success_rate": e.value, "ci_low": e.low, "ci_high": e.high, "confidence": 0.95, "n_rollouts": n,
            "wall_clock_s": 0.0}


def _eval(path: Path, stage: int, tasks: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"run_id": "similar_ab_stove__seq_ft__seed0", "stage": stage, "tasks": tasks}))


def write_run(root, cfg, cur, arm, *, ns=None, dirty=False, frozen_moves=False, allowed_moves=True,
              residual=0.5, seed_shift=0, r00=None, r11=None):
    name = scl.run_name(cfg, cur, arm)
    d = root / name
    src = root / scl.run_name(cfg, "similar_ab_stove", arm)
    m_name, kwargs, label = scl.method_spec(cfg, arm)
    b = CUR[cur]
    d.mkdir(parents=True)
    OmegaConf.save(OmegaConf.create({
        "run_id": name, "seed": 0, "seed_namespace_run_id": ns or "similar_ab_stove__seq_ft__seed0",
        "method": {"name": m_name, "display_name": label, **kwargs}, "curriculum": {"name": cur},
        "train": {"steps": 30000, "batch_size": 64, "lr": 1e-4, "amp": True},
        "eval": {"n_episodes": N, "euler_steps": 10}}), d / "config.yaml")
    (d / "checkpoints").mkdir()
    (d / "git_sha").write_text("abc" + ("-dirty" if dirty else "") + "\n")
    (d / "stats.json").write_text(json.dumps({"fingerprint": "fp1", "fitted_on_task_id": A}))
    R00, R10, R11 = MATRIX[cur][arm]
    R00 = r00 if r00 is not None else R00
    R11 = r11 if r11 is not None else R11
    s0 = {"w_allowed": torch.zeros(3), "w_frozen": torch.ones(3)}
    if cur == "similar_ab_stove":
        torch.save({"stats_fingerprint": "fp1", "state_dict": s0, "extra": {}}, d / "checkpoints" / "stage0.pt")
        _eval(d / "eval" / "stage0.json", 0, [_task(A, R00), _task(STOVE, 0.0)])
        if arm != "seq_ft":
            (d / "identity_stage0.json").write_text(json.dumps({"passed": True}))
    else:
        (d / "eval").mkdir()
        shutil.copy2(src / "eval" / "stage0.json", d / "eval" / "stage0.json")
        (d / "resume.json").write_text(json.dumps({"run": str(src), "start_stage": 1,
                                                   "sha256": file_sha256(src / "checkpoints" / "stage0.pt")}))
    s1 = {"w_allowed": torch.zeros(3) + (1.0 if (allowed_moves or arm == "seq_ft") else 0.0),
          "w_frozen": torch.ones(3) + (1.0 if (arm == "seq_ft" or frozen_moves) else 0.0)}
    extra = {}
    if arm == "allowlist_ft":
        extra["method_state"] = {"freeze_reports": {"1": {"trainable_tensors": 1}}}
    if arm == "gpm":
        (d / "method").mkdir()
        torch.save({"M": 1}, d / "method" / "memory_task1.pt")
        (d / "method" / "gpm_logs_task1.json").write_text(json.dumps(
            {"residuals": {"L": {"max_residual_over_bound": residual}}}))
        extra["method_artifacts"] = [{"path": "method/memory_task1.pt",
                                      "sha256": file_sha256(d / "method" / "memory_task1.pt")}]
    torch.save({"stats_fingerprint": "fp1", "state_dict": s1, "extra": extra}, d / "checkpoints" / "stage1.pt")
    _eval(d / "eval" / "stage1.json", 1, [_task(A, R10), _task(b, R11, seed_shift=seed_shift)])
    (d / "result.json").write_text(json.dumps({"task_keys": [A, b], "t1_pairing": {"passed": True}}))
    return d


def _probe_record(ckpt: Path, keys: list[str], pairs: list[str], value=0.01):
    instr = {b: {d: {"velocity": {"sensitivity": value}, "chunks": {"relative": value * 2},
                     "size_halves": {"trunk": {"median": value * 3}}}
                 for d in ("A_data_B_instruction", "B_data_A_instruction")} for b in pairs}
    return {"checkpoint_sha256": file_sha256(ckpt), "weights_unchanged": True, "amp": True, "device_type": "cuda",
            "text_cache": {k: {"text": f"do {k}", "precision": "amp" if i == 0 else "fp32"} for i, k in enumerate(keys)},
            "probe_loss": {k: 0.1 for k in keys}, "instruction": instr}


def summary_world(root, overrides=None, *, skip=(), stage0=True, probes=True):
    """Nine valid runs, the stage-0 file and every probe; ``overrides[(cur, arm)]`` overrides write_run."""
    per_run = overrides or {}
    cfg = copy.deepcopy(scl.load_config())
    for cur in CUR:
        for arm in scl.ARMS:
            if (cur, arm) in skip or cur in skip:
                continue
            write_run(root, cfg, cur, arm, **per_run.get((cur, arm), {}))
    ckpt0 = root / scl.run_name(cfg, "similar_ab_stove", "seq_ft") / "checkpoints" / "stage0.pt"
    if stage0:
        d0 = root / "similar_cl" / "stage0"
        _eval(d0 / "zero_shot.json", 0, [_task(MILK, 0.0), _task(CAB, 0.1)])
        rec = _probe_record(ckpt0, [A, STOVE, MILK, CAB], [STOVE, MILK, CAB], value=0.001)
        rec["zero_shot"] = {"path": "zero_shot.json", "sha256": file_sha256(d0 / "zero_shot.json")}
        (d0 / "stage0.json").write_text(json.dumps(rec))
    if probes:
        for cur, b in CUR.items():
            for arm in scl.ARMS:
                ckpt = root / scl.run_name(cfg, cur, arm) / "checkpoints" / "stage1.pt"
                if ckpt.is_file():
                    p = scl.probe_path(cfg, root, cur, arm)
                    p.parent.mkdir(parents=True)
                    p.write_text(json.dumps(_probe_record(ckpt, [A, b], [b])))
    return cfg


def summarize(root, cfg):
    return scl.run_summary(cfg, results_root=root, out=root / "out", allowlist_names={"w_allowed"})


def test_valid_study_differences_and_predictions(tmp_path):
    cfg = summary_world(tmp_path)
    s = summarize(tmp_path, cfg)
    assert s["status"] == "valid" and s["exit_code"] == 0, {n: b["problems"] for n, b in s["curricula"].items()}
    S = s["curricula"]["similar_ab_stove"]
    assert S["differences"]["dP"]["proj"]["diff"] == pytest.approx(-0.3)
    assert S["differences"]["dP"]["freeze"]["diff"] == pytest.approx(-0.1)
    assert S["differences"]["dP"]["tot"]["diff"] == pytest.approx(-0.4)
    assert S["differences"]["dR"]["tot"]["diff"] == pytest.approx(0.6) and S["differences"]["dR"]["tot"]["low"] > 0
    assert S["differences"]["forgetting"]["seq_ft"]["diff"] == pytest.approx(0.7)
    assert S["cells"]["Z"]["value"] == 0.0 and s["curricula"]["similar_ab_cabinet"]["cells"]["Z"]["value"] == pytest.approx(0.1)
    assert S["checks"]["frozen_gpm"]["n_unmoved_allowlisted"] == 0
    p1 = s["predictions"]["P1"]
    assert p1["judgment"] == "supported" and p1["basis"].startswith("both similar curricula (S, S2)")
    assert p1["contrasts"] == {"S-D": pytest.approx(-0.26), "S2-D": pytest.approx(-0.16)}
    assert p1["mean_contrast"] == pytest.approx(-0.21)
    assert s["predictions"]["P2"]["overall"] == "holds_in_every_curriculum"
    assert s["predictions"]["P4"]["D"]["gpm"]["A_data_B_instruction"]["chunk_relative"] == pytest.approx(0.02)
    assert s["predictions"]["P4"]["D"]["stage0"]["B_data_A_instruction"]["velocity_sensitivity"] == pytest.approx(0.001)
    assert json.loads((tmp_path / "out" / "summary.json").read_text())["status"] == "valid"


@pytest.mark.parametrize("override,cur,check", [
    ({("similar_ab_milk", "seq_ft"): {"ns": "similar_ab_milk__seq_ft__seed0"}}, "similar_ab_milk", "production_seq_ft"),
    ({("similar_ab_milk", "gpm"): {"dirty": True}}, "similar_ab_milk", "production_gpm"),
    ({("similar_ab_milk", "allowlist_ft"): {"frozen_moves": True}}, "similar_ab_milk", "frozen_allowlist_ft"),
    ({("similar_ab_cabinet", "gpm"): {"residual": 1.5}}, "similar_ab_cabinet", "gpm_memory"),
    ({("similar_ab_milk", "gpm"): {"seed_shift": 7}}, "similar_ab_milk", "episode_pairing"),
])
def test_each_validity_condition_gives_invalid(tmp_path, override, cur, check):
    cfg = summary_world(tmp_path, override)
    s = summarize(tmp_path, cfg)
    assert s["status"] == "invalid" and s["exit_code"] == 3
    assert s["curricula"][cur]["status"] == "invalid" and check in s["curricula"][cur]["problems"]
    assert s["curricula"]["similar_ab_stove"]["status"] == "valid"


def test_resume_identity_t1_and_zero_shot_sabotage(tmp_path):
    cfg = summary_world(tmp_path)
    d = tmp_path / scl.run_name(cfg, "similar_ab_milk", "allowlist_ft")
    rec = json.loads((d / "resume.json").read_text())
    (d / "resume.json").write_text(json.dumps({**rec, "sha256": "0" * 64}))
    g = tmp_path / scl.run_name(cfg, "similar_ab_stove", "gpm")
    (g / "identity_stage0.json").write_text(json.dumps({"passed": False}))
    (g / "result.json").write_text(json.dumps({"task_keys": [A, STOVE], "t1_pairing": {"passed": False}}))
    s = summarize(tmp_path, cfg)
    assert "resume_allowlist_ft" in s["curricula"]["similar_ab_milk"]["problems"]
    assert {"identity_gpm", "t1_pairing"} <= set(s["curricula"]["similar_ab_stove"]["problems"])
    assert s["predictions"]["P1"]["judgment"] == "inconclusive"
    # a zero-shot file that names another checkpoint invalidates D and S2 (and the stage-0 artifact)
    cfg = summary_world(tmp_path / "z")
    p = scl.stage0_path(cfg, tmp_path / "z")
    p.write_text(json.dumps({**json.loads(p.read_text()), "checkpoint_sha256": "f" * 64}))
    s = summarize(tmp_path / "z", cfg)
    assert s["stage0"]["artifact"]["status"] == "invalid" and s["stage0"]["zero_shot_sha_ok"] is False
    assert "zero_shot" in s["curricula"]["similar_ab_milk"]["problems"]
    assert "zero_shot" in s["curricula"]["similar_ab_cabinet"]["problems"]
    assert s["curricula"]["similar_ab_stove"]["status"] == "valid" and "stage0" not in s["predictions"]["P4"].get("D", {})


def test_unmoved_allowlisted_tensor_is_reported_not_failed(tmp_path):
    cfg = summary_world(tmp_path, {("similar_ab_milk", "gpm"): {"allowed_moves": False}})
    s = summarize(tmp_path, cfg)
    assert s["status"] == "valid"
    assert s["curricula"]["similar_ab_milk"]["checks"]["frozen_gpm"]["unmoved_allowlisted"] == ["w_allowed"]


def test_probe_artifacts_missing_and_invalid(tmp_path):
    cfg = summary_world(tmp_path)
    p = scl.probe_path(cfg, tmp_path, "similar_ab_milk", "gpm")
    rec = json.loads(p.read_text())
    p.write_text(json.dumps({**rec, "checkpoint_sha256": "0" * 64}))
    s = summarize(tmp_path, cfg)
    assert s["status"] == "invalid" and s["probes"]["similar_ab_milk__gpm_projected_adam__seed0"]["status"] == "invalid"
    assert all(b["status"] == "valid" for b in s["curricula"].values())             # the success comparisons stand
    assert "gpm" not in s["predictions"]["P4"]["D"] and "seq_ft" in s["predictions"]["P4"]["D"]
    p.write_text(json.dumps({**rec, "text_cache": {A: {"precision": "fp32"}, MILK: {"precision": "fp32"}}}))
    assert summarize(tmp_path, cfg)["probes"]["similar_ab_milk__gpm_projected_adam__seed0"]["status"] == "invalid"
    p.unlink()
    s = summarize(tmp_path, cfg)
    assert s["status"] == "incomplete" and s["exit_code"] == 4


def test_s2_stopped_is_incomplete_and_p1_labelled_one_similar_task(tmp_path):
    cfg = summary_world(tmp_path, skip=("similar_ab_cabinet",))
    s = summarize(tmp_path, cfg)
    assert s["status"] == "incomplete" and s["exit_code"] == 4
    assert s["curricula"]["similar_ab_cabinet"]["status"] == "incomplete"
    assert s["curricula"]["similar_ab_stove"]["status"] == "valid" and s["curricula"]["similar_ab_milk"]["status"] == "valid"
    p1 = s["predictions"]["P1"]
    assert p1["basis"] == "S versus D, one similar task" and set(p1["similar_missing"]) == {"S2"}
    assert p1["judgment"] == "supported" and p1["mean_contrast"] == pytest.approx(-0.26)
    assert s["predictions"]["P2"]["per_curriculum"]["S2"] == "not_judged"
    # nothing at all: incomplete, P1 not judged
    s = summarize(tmp_path / "empty", copy.deepcopy(scl.load_config()))
    assert s["status"] == "incomplete" and s["predictions"]["P1"]["judgment"] == "not_judged"


def test_assessability_flags(tmp_path):
    over = {(c, "seq_ft"): {"r00": 0.7} for c in CUR}
    over[("similar_ab_milk", "allowlist_ft")] = {"r11": 0.4}
    cfg = summary_world(tmp_path, over)
    s = summarize(tmp_path, cfg)
    assert s["status"] == "valid"
    D = s["curricula"]["similar_ab_milk"]
    assert D["flags"]["A_weak"] and D["assess"]["retention"] == "not_assessable"
    assert D["assess"]["dP_proj"] == "inconclusive" and D["assess"]["dP_tot"] == "ok"
    assert s["predictions"]["P1"]["judgment"] == "inconclusive"
    assert set(s["predictions"]["P2"]["per_curriculum"].values()) == {"not_assessable"}


@pytest.mark.parametrize("sim,div,judgment", [
    (-0.30, -0.19, "supported"), (-0.30, -0.20, "direction_only"), (-0.10, -0.10, "not_supported"),
    (-0.05, -0.10, "not_supported"),
])
def test_p1_boundaries(sim, div, judgment):
    cfg = scl.load_config()

    def block(v):
        return {"status": "valid", "assess": {"dP_proj": "ok"}, "differences": {"dP": {"proj": {"diff": v}}}}

    blocks = {"similar_ab_stove": block(sim), "similar_ab_milk": block(div), "similar_ab_cabinet": block(sim)}
    assert scl.judge_p1(cfg, blocks)["judgment"] == judgment


# ---- the queue ----------------------------------------------------------------------------------------------

QUEUE = repo_root() / "scripts" / "queue_similar_cl.sh"


def run_queue(tmp_path, args=(), fail=""):
    env = {**os.environ, "QUEUE_DRY_RUN": "1", "QUEUE_FAIL_STEP": fail, "QUEUE_LOG_ROOT": str(tmp_path / "logs"),
           "QUEUE_RESULTS_ROOT": str(tmp_path / "results")}
    done = subprocess.run(["bash", str(QUEUE), *args], env=env, capture_output=True, text=True, timeout=180)
    (logdir,) = list((tmp_path / "logs").glob("queue_*"))
    return done.returncode, (logdir / "queue.log").read_text(), logdir


def _starts(log):
    return [l.split("START ")[1].split(":")[0] for l in log.splitlines() if " START " in l and "QUEUE START" not in l]


def test_queue_order_and_commands(tmp_path):
    assert subprocess.run(["bash", "-n", str(QUEUE)]).returncode == 0
    rc, log, logdir = run_queue(tmp_path)
    assert rc == 0 and log.rstrip().endswith("QUEUE DONE ok")
    assert _starts(log) == ["0_S_seqft", "1_stage0", "2_S_allow", "3_S_gpm", "4_D_seqft", "5_D_allow", "6_D_gpm",
                            "7_probes_SD", "8_S2_seqft", "9_S2_allow", "10_S2_gpm", "11_probes_S2", "12_summary"]
    res = tmp_path / "results"
    first = (logdir / "0_S_seqft.log").read_text()
    assert "--curriculum similar_ab_stove --seed 0 --amp --method seq_ft" in first and "--steps-per-task" not in first
    for name, method in (("2_S_allow", "allowlist_ft"), ("3_S_gpm", "gpm")):
        text = (logdir / f"{name}.log").read_text()
        assert f"--method {method}" in text
        assert f"--t1-reference-run {res}/similar_ab_stove__seq_ft__seed0" in text
        assert f"--identity-reference-run {res}/similar_ab_stove__seq_ft__seed0 --identity-stages 0" in text
    for name, cur, arm, label in (("5_D_allow", "similar_ab_milk", "allowlist_ft", "allowlist_ft"),
                                  ("10_S2_gpm", "similar_ab_cabinet", "gpm", "gpm_projected_adam")):
        text = (logdir / f"{name}.log").read_text()
        assert f"--curriculum {cur}" in text and f"--method {arm}" in text
        assert f"--resume-run {res}/similar_ab_stove__{label}__seed0 --start-stage 1 --resume-sha256 DRYRUN" in text
        assert "--identity" not in text
    assert "--part probe --curricula similar_ab_stove similar_ab_milk" in (logdir / "7_probes_SD.log").read_text()
    assert "--part summary" in (logdir / "12_summary.log").read_text()
    assert "RESUME 4_D_seqft from" in log


def test_queue_failures_skip_dependents(tmp_path):
    rc, log, _ = run_queue(tmp_path / "a", fail="2_S_allow")
    starts = _starts(log)
    assert rc == 1 and "FAILED: 2_S_allow" in log
    assert "5_D_allow" not in starts and "9_S2_allow" not in starts
    assert {"4_D_seqft", "6_D_gpm", "8_S2_seqft", "10_S2_gpm", "12_summary"} <= set(starts)
    rc, log, _ = run_queue(tmp_path / "b", fail="0_S_seqft")
    assert rc == 1 and _starts(log) == ["0_S_seqft", "12_summary"]
    rc, log, _ = run_queue(tmp_path / "c", ["--from-step", "8"])   # nothing on disk: priors missing
    assert rc == 1 and _starts(log) == ["12_summary"] and "PRIOR 0_S_seqft missing" in log


def test_queue_smoke_mode(tmp_path):
    scratch = tmp_path / "scratch"
    rc, log, logdir = run_queue(tmp_path, ["--smoke", str(scratch)])
    assert rc == 0
    first = (logdir / "0_S_seqft.log").read_text()
    assert f"--results-root {scratch} --steps-per-task 200 --n-episodes 1 --allow-dirty" in first
    assert f"--results-root {scratch} --smoke-batches 3 --allow-dirty" in (logdir / "1_stage0.log").read_text()
    assert f"--results-root {scratch} --smoke" in (logdir / "12_summary.log").read_text()
    assert f"--resume-run {scratch}/similar_ab_stove__seq_ft__seed0" in (logdir / "4_D_seqft.log").read_text()
    assert subprocess.run(["bash", str(QUEUE), "--smoke", "rel"], capture_output=True).returncode == 2
    assert subprocess.run(["bash", str(QUEUE), "--smoke", str(repo_root() / "results" / "x")],
                          capture_output=True).returncode == 2


# ---- curricula and config ----------------------------------------------------------------------------------


def test_curricula_and_config_are_consistent():
    from flowcl.data.config import load_method_config
    from flowcl.experiments import similar_tasks as st

    cfg = scl.load_config()
    assert cfg["task_a"] == A and cfg["source_curriculum"] == "similar_ab_stove"
    assert [c["name"] for c in cfg["curricula"]] == list(CUR)
    diagnostic = {t["key"]: t["category"] for t in st.load_config()["tasks_b"]}
    assert st.load_config()["task_a"] == A
    for c in cfg["curricula"]:
        conf = OmegaConf.load(repo_root() / "configs" / "curriculum" / f"{c['name']}.yaml")
        assert conf.name == c["name"] and len(conf.tasks) == 2
        assert conf.tasks[0].task_key == A and conf.tasks[0].n_demos == conf.tasks[1].n_demos == 50
        assert scl.b_key(c["name"]) == CUR[c["name"]] and diagnostic[CUR[c["name"]]] == c["category"]
    assert len(set(CUR.values())) == 3
    assert load_method_config("allowlist_ft")[0] == "allowlist_ft"
    assert [scl.method_spec(cfg, a)[2] for a in scl.ARMS] == ["seq_ft", "allowlist_ft", "gpm_projected_adam"]
    assert scl.namespace(cfg) == "similar_ab_stove__seq_ft__seed0"
    assert cfg["production"]["n_episodes"] == 50 and cfg["p1_supported_below"] == -0.10
