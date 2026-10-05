"""Language-conditioned base study: the L1a / L1b / L3 judgments on synthetic cells, the summary on
fake run directories (valid, each validity condition, missing artifacts), the text-cache rule, memory
ranks, the queue, and the config's consistency with the curriculum and with similar_cl."""

from __future__ import annotations

import copy
import json
import os
import subprocess
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from flowcl.analysis.metrics import success_estimate
from flowcl.experiments import language_cl as lcl
from flowcl.utils.libero_paths import repo_root
from flowcl.utils.run import file_sha256

A = "libero_goal/put_the_bowl_on_the_plate"
S2 = "libero_goal/put_the_bowl_on_top_of_the_cabinet"
S = "libero_goal/put_the_bowl_on_the_stove"
N = 50
BOOT = {"seed": 0, "n_resamples": 2000, "confidence": 0.95}


def succ(rate, n=N):
    k = round(rate * n)
    return [True] * k + [False] * (n - k)


def cfg():
    return copy.deepcopy(lcl.load_config())


def swap_rec(drop, low, requested):
    return {"drop_estimate": {"value": drop, "low": low, "high": drop + 0.1},
            "requested_estimate": None if requested is None else {"value": requested, "low": 0.0, "high": 1.0}}


# ---- judgments -------------------------------------------------------------------------------------------


@pytest.mark.parametrize("ra, rs2, swaps, verdict", [
    (0.9, 0.9, [(0.8, 0.6), (0.7, 0.5)], "instruction_dependence_demonstrated"),
    (0.78, 0.9, [(0.8, 0.6), (0.7, 0.5)], "constituent_not_learned"),
    (0.9, 0.9, [(0.8, 0.6), (0.14, 0.02)], "instruction_dependence_not_demonstrated"),
    (0.9, 0.9, [(0.8, 0.6), (0.3, 0.0)], "instruction_dependence_not_demonstrated"),   # CI touches 0
    (0.8, 0.8, [(0.15, 0.01), (0.15, 0.01)], "instruction_dependence_demonstrated"),   # boundaries inclusive
])
def test_l1a(ra, rs2, swaps, verdict):
    sw = {f"{A}->{S2}": swap_rec(*swaps[0], 0.5), f"{S2}->{A}": swap_rec(*swaps[1], 0.5)}
    assert lcl.judge_l1a({A: succ(ra), S2: succ(rs2)}, sw, cfg())["verdict"] == verdict


@pytest.mark.parametrize("req, verdict", [
    ((0.5, 0.5), "task_selection_threshold_met"),
    ((0.9, 0.48), "task_selection_not_demonstrated"),
    ((0.0, 0.0), "task_selection_not_demonstrated"),
])
def test_l1b(req, verdict):
    sw = {f"{A}->{S2}": swap_rec(0.8, 0.6, req[0]), f"{S2}->{A}": swap_rec(0.8, 0.6, req[1])}
    l1a = lcl.judge_l1a({A: succ(0.9), S2: succ(0.9)}, sw, cfg())
    out = lcl.judge_l1b(sw, cfg(), l1a)
    assert out["verdict"] == verdict
    if verdict == "task_selection_not_demonstrated":
        assert out["wording"] == ("instruction dependence demonstrated; task selection not demonstrated under "
                                  "the chosen criterion")


def test_l1b_is_not_judged_without_l1a():
    sw = {f"{A}->{S2}": swap_rec(0.0, -0.1, 0.9), f"{S2}->{A}": swap_rec(0.0, -0.1, 0.9)}
    l1a = lcl.judge_l1a({A: succ(0.9), S2: succ(0.9)}, sw, cfg())
    assert lcl.judge_l1b(sw, cfg(), l1a)["verdict"] == "not_judged"


@pytest.mark.parametrize("a0, a1_seq, a1_gpm, s_seq, s_gpm, outcome", [
    (0.9, 0.0, 0.9, 0.96, 0.94, "protects"),
    (0.9, 0.0, 0.86, 0.96, 0.80, "protects_at_plasticity_cost"),
    (0.9, 0.0, 0.5, 0.96, 0.96, "partial"),
    (0.9, 0.0, 0.04, 0.96, 0.96, "no_demonstrated_advantage"),
    (0.9, 0.84, 0.9, 0.96, 0.96, "no_interference"),
    (0.9, 0.0, 0.9, 0.96, 0.48, "inconclusive_acquisition"),
    (0.9, 0.9, 0.9, 0.40, 0.96, "inconclusive_acquisition"),   # seq_ft retains A only because it did not learn S
])
def test_l3_outcomes(a0, a1_seq, a1_gpm, s_seq, s_gpm, outcome):
    l1a = {"verdict": "instruction_dependence_demonstrated"}
    out = lcl.judge_l3("A", succ(a0), {"seq_ft": succ(a1_seq), "gpm": succ(a1_gpm)}, succ(a0),
                       {"seq_ft": succ(s_seq), "gpm": succ(s_gpm)}, cfg(), BOOT, l1a)
    assert out["outcome"] == outcome and out["interpretable"]
    q = out["quantities"]
    assert {"diff", "low", "high"} <= set(q["forgetting"]["gpm"]) and {"diff", "low", "high"} <= set(q["dP"])


def test_l3_flags_an_uninterpretable_base():
    out = lcl.judge_l3("A", succ(0.9), {"seq_ft": succ(0.0), "gpm": succ(0.9)}, succ(0.9),
                       {"seq_ft": succ(0.9), "gpm": succ(0.9)}, cfg(), BOOT,
                       {"verdict": "instruction_dependence_not_demonstrated"})
    assert out["outcome"] == "protects" and out["interpretable"] is False


# ---- small pieces -----------------------------------------------------------------------------------------


def test_expected_precision_follows_the_runner():
    assert lcl.expected_precision("langbase_cabinet_stove", [A, S2, S], True, "cuda") == \
        {A: "amp", S2: "amp", S: "fp32"}
    assert lcl.expected_precision("similar_ab_stove", [A, S2, S], True, "cuda") == {A: "amp", S2: "fp32", S: "fp32"}
    assert set(lcl.expected_precision("langbase_cabinet_stove", [A, S2], True, "cpu").values()) == {"fp32"}


def test_memory_ranks():
    M = {"trunk.blocks.0.mlp.fc1": torch.zeros(4, 2), "trunk.blocks.0.mlp.fc2": torch.zeros(8, 8),
         "flow_head.action_out": torch.zeros(4, 1)}
    dims = {"trunk.blocks.0.mlp.fc1": (8, 4), "trunk.blocks.0.mlp.fc2": (4, 8), "flow_head.action_out": (7, 4)}
    out = lcl.memory_ranks(M, dims)
    assert out["layers"]["trunk.blocks.0.mlp.fc1"] == {"k": 2, "d_in": 4, "fraction": 0.5}
    assert out["halves"]["trunk"]["parameter_weighted"] == pytest.approx((8 * 2 + 4 * 8) / (8 * 4 + 4 * 8))
    assert out["halves"]["trunk"]["total_k"] == 10 and out["halves"]["decoder"]["total_k"] == 1


def test_swaps_are_registered_only_where_the_rule_says(tmp_path):
    for stage, arm in ((0, "gpm"), (2, "seq_ft")):
        with pytest.raises(ValueError, match="registered"):
            lcl.run_swaps(cfg(), stage, arm, results_root=tmp_path, allow_dirty=True)
    assert lcl.swap_pairs(cfg(), 0) == [(A, S2), (S2, A)] and lcl.swap_pairs(cfg(), 1) == [(A, S), (S, A)]


# ---- the summary on fake run directories -------------------------------------------------------------------

RATES = {  # arm -> stage -> task -> rate
    "seq_ft": {0: {A: 0.9, S2: 0.9, S: 0.0}, 1: {A: 0.0, S2: 0.0, S: 0.96}},
    "gpm": {0: {A: 0.9, S2: 0.9, S: 0.0}, 1: {A: 0.86, S2: 0.84, S: 0.94}},
}
SEED_BASE = {A: 0, S2: 3000, S: 1000}


def _task(key, rate, seed_shift=0):
    s = succ(rate)
    e = success_estimate(s)
    base = SEED_BASE[key] + seed_shift
    return {"task_key": key, "successes": s, "n_steps": [100] * N, "seeds": list(range(base, base + N)),
            "success_rate": e.value, "ci_low": e.low, "ci_high": e.high, "confidence": 0.95, "n_rollouts": N,
            "wall_clock_s": 0.0}


def write_run(root, c, arm, *, dirty=False, steps=(60000, 30000), joint_keys=(A, S2), residual=0.5,
              identity=True, seed_shift=0):
    name = lcl.run_name(c, arm)
    d = root / name
    m_name, kwargs, label = lcl.method_spec(c, arm)
    d.mkdir(parents=True)
    OmegaConf.save(OmegaConf.create({
        "run_id": name, "seed": 0, "seed_namespace_run_id": lcl.namespace(c),
        "method": {"name": m_name, "display_name": label, **kwargs},
        "curriculum": {"name": c["curriculum"], "tasks": [
            {"task_key": A, "n_demos": 50, "co_train": [S2], "steps_factor": 2}, {"task_key": S, "n_demos": 50}]},
        "train": {"steps": 30000, "batch_size": 64, "lr": 1e-4, "amp": True},
        "eval": {"n_episodes": N, "euler_steps": 10}}), d / "config.yaml")
    (d / "checkpoints").mkdir()
    (d / "git_sha").write_text("abc" + ("-dirty" if dirty else "") + "\n")
    (d / "stats.json").write_text(json.dumps({"fingerprint": "fp1", "fitted_on_task_id": A}))
    joint = {"train_task_keys": list(joint_keys), "steps_factor": 2, "task_samples": {A: 4669, S2: 5094}}
    for stage in (0, 1):
        extra = {"joint": joint} if stage == 0 else {}
        if arm == "gpm":
            (d / "method").mkdir(exist_ok=True)
            mem = d / "method" / f"memory_task{stage}.pt"
            torch.save({"M": stage}, mem)
            (d / "method" / f"gpm_logs_task{stage}.json").write_text(json.dumps(
                {"residuals": {"L": {"max_residual_over_bound": residual}}}))
            extra["method_artifacts"] = [{"path": f"method/memory_task{stage}.pt", "sha256": file_sha256(mem)}]
        torch.save({"stats_fingerprint": "fp1", "state_dict": {"w": torch.zeros(2)}, "extra": extra},
                   d / "checkpoints" / f"stage{stage}.pt")
        tasks = [_task(k, RATES[arm][stage][k], seed_shift if k == S else 0) for k in (A, S, S2)]
        (d / "eval").mkdir(exist_ok=True)
        (d / "eval" / f"stage{stage}.json").write_text(json.dumps({"run_id": lcl.namespace(c), "stage": stage,
                                                                    "tasks": tasks}))
    if arm == "gpm":
        (d / "identity_stage0.json").write_text(json.dumps({"passed": identity}))
    (d / "result.json").write_text(json.dumps({
        "task_keys": [A, S], "t1_pairing": {"passed": True} if arm == "gpm" else None,
        "stages": [{"stage": 0, "steps": steps[0], "joint": joint}, {"stage": 1, "steps": steps[1]}]}))
    return d


def write_swaps(root, c, stage, arm, *, requested=(0.8, 0.7), drop=(0.8, 0.8), seeds_ok=True, signatures=True):
    d = root / lcl.run_name(c, arm)
    pairs = lcl.swap_pairs(c, stage)
    swaps = {}
    for (task, req), r, dr in zip(pairs, requested, drop):
        swaps[f"{task}->{req}"] = {
            "swapped_successes": [False] * N, "requested_reached": succ(r), "requested_at_start": [False] * N,
            "correct_estimate": {"value": 0.9, "low": 0.8, "high": 1.0},
            "swapped_estimate": {"value": 0.9 - dr, "low": 0.0, "high": 0.2},
            "drop_estimate": {"value": dr, "low": dr - 0.1, "high": dr + 0.1},
            "requested_estimate": {"value": r, "low": r - 0.1, "high": r + 0.1},
            "determinism": None if stage else {"equal": True, "n_different": 0}, "correct_reused": stage == 1,
            "eval_seeds_match": seeds_ok, "own_instruction_matches_env": True}
    keys = sorted({k for p in pairs for k in p})
    rec = {"checkpoint_sha256": file_sha256(d / "checkpoints" / f"stage{stage}.pt"), "weights_unchanged": True,
           "model_signatures_equal": signatures, "amp": True, "device_type": "cuda",
           "text_cache": {k: {"precision": lcl.expected_precision(c["curriculum"], keys, True, "cuda")[k]}
                          for k in keys}, "swaps": swaps}
    p = lcl.swap_path(c, root, stage, arm)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec))


def _probe(ckpt, curriculum, c, value=0.01):
    t = lcl.tasks(c)
    keys = [t["A"], t["S2"], t["S"]]
    instr = {b: {d: {"velocity": {"sensitivity": value}, "chunks": {"relative": value * 2},
                     "size_halves": {"trunk": {"median": value * 3}}}
                 for d in ("A_data_B_instruction", "B_data_A_instruction")} for b in (t["S2"], t["S"])}
    return {"checkpoint_sha256": file_sha256(ckpt), "weights_unchanged": True, "amp": True, "device_type": "cuda",
            "text_cache": {k: {"precision": v} for k, v in
                           lcl.expected_precision(curriculum, keys, True, "cuda").items()},
            "probe_loss": {k: 0.1 for k in keys}, "instruction": instr, "token_contrast": {},
            "token_projection_gradient": {}, "memory": None, "memory_coverage": None}


def world(root, *, runs=None, swaps=True, probes=True, reference_value=0.001):
    c = cfg()
    for arm in lcl.ARMS:
        write_run(root, c, arm, **(runs or {}).get(arm, {}))
    a = c["a_only"]
    for name in (a["seq_ft_run"], a["gpm_run"]):   # the A-only lineage's stage-0 checkpoints
        d = root / name
        (d / "checkpoints").mkdir(parents=True)
        (d / "method").mkdir()
        mem = d / "method" / "memory_task0.pt"
        torch.save({"M": 0}, mem)
        torch.save({"stats_fingerprint": "fp0", "state_dict": {"w": torch.ones(2)},
                    "extra": {"method_artifacts": [{"path": "method/memory_task0.pt", "sha256": file_sha256(mem)}]}},
                   d / "checkpoints" / "stage0.pt")
        OmegaConf.save(OmegaConf.create({"train": {"amp": True}, "eval": {"euler_steps": 10}}), d / "config.yaml")
    if swaps:
        for stage, arm in ((0, "seq_ft"), (1, "seq_ft"), (1, "gpm")):
            write_swaps(root, c, stage, arm)
    if probes:
        for label in lcl.PROBE_LABELS:
            target = lcl.probe_target(c, root, label)
            p = lcl.probe_path(c, root, label)
            p.parent.mkdir(parents=True)
            p.write_text(json.dumps(_probe(target["checkpoint"], target["curriculum"], c,
                                           value=reference_value if label == "a_only_stage0" else 0.01)))
        ref = _probe(root / a["seq_ft_run"] / "checkpoints" / "stage0.pt", "similar_ab_stove", c,
                     value=reference_value)
        (root / a["stage0_probe"]).parent.mkdir(parents=True)
        (root / a["stage0_probe"]).write_text(json.dumps(ref))
    return c


def summarize(root, c):
    return lcl.run_summary(c, results_root=root, out=root / "out")


def test_a_valid_study_and_its_verdicts(tmp_path):
    c = world(tmp_path)
    s = summarize(tmp_path, c)
    assert s["status"] == "valid", (s["problems"], {k: v for k, v in s["checks"].items() if not v["passed"]},
                                    s["swap_artifacts"], s["probe_artifacts"])
    r = s["rules"]
    assert r["L1a"]["verdict"] == "instruction_dependence_demonstrated"
    assert r["L1b"]["verdict"] == "task_selection_threshold_met"
    assert r["L3"]["A"]["outcome"] == "protects" and r["L3"]["S2"]["outcome"] == "protects"
    assert r["L4"]["exposure"] == {A: 4669, S2: 5094} and r["L4"]["S_zero_shot"]["value"] == 0.0
    assert set(r["L4"]["stage1_swaps"]) == {"seq_ft", "gpm"}
    assert s["reproduction"]["passed"] is True and set(s["mechanism"]) == set(lcl.PROBE_LABELS)
    assert json.loads((tmp_path / "out" / "summary.json").read_text())["exit_code"] == 0


@pytest.mark.parametrize("runs, check", [
    ({"seq_ft": {"dirty": True}}, "production_seq_ft"),
    ({"gpm": {"steps": (30000, 30000)}}, "production_gpm"),
    ({"seq_ft": {"joint_keys": (A,)}}, "production_seq_ft"),
    ({"gpm": {"identity": False}}, "gpm"),
    ({"gpm": {"residual": 1.5}}, "gpm"),
    ({"gpm": {"seed_shift": 7}}, "episode_pairing"),
])
def test_each_validity_condition_gives_invalid(tmp_path, runs, check):
    s = summarize(tmp_path, world(tmp_path, runs=runs))
    assert s["status"] == "invalid" and not s["checks"][check]["passed"]


def test_artifact_conditions(tmp_path):
    c = world(tmp_path / "a")
    p = lcl.swap_path(c, tmp_path / "a", 1, "gpm")
    rec = json.loads(p.read_text())
    rec["model_signatures_equal"] = False
    p.write_text(json.dumps(rec))
    s = summarize(tmp_path / "a", c)
    assert s["status"] == "invalid" and "model_signatures_equal" in s["swap_artifacts"]["stage1_gpm"]["problems"]

    c = world(tmp_path / "b", reference_value=0.001)
    ref = tmp_path / "b" / c["a_only"]["stage0_probe"]
    rec = json.loads(ref.read_text())
    rec["probe_loss"][A] = 0.2   # (a) no longer reproduces similar_cl
    ref.write_text(json.dumps(rec))
    s = summarize(tmp_path / "b", c)
    assert s["status"] == "invalid" and s["reproduction"]["passed"] is False

    s = summarize(tmp_path / "c", world(tmp_path / "c", swaps=False))
    assert s["status"] == "incomplete" and s["rules"]["L1a"]["verdict"] == "not_judged"


def test_missing_runs_are_incomplete(tmp_path):
    c = world(tmp_path)
    (tmp_path / lcl.run_name(c, "gpm") / "eval" / "stage1.json").unlink()
    s = summarize(tmp_path, c)
    assert s["status"] == "incomplete" and "rules" not in s


# ---- the queue ------------------------------------------------------------------------------------------------

QUEUE = repo_root() / "scripts" / "queue_language_cl.sh"


def run_queue(tmp_path, args=(), fail=""):
    env = {**os.environ, "QUEUE_DRY_RUN": "1", "QUEUE_FAIL_STEP": fail, "QUEUE_LOG_ROOT": str(tmp_path / "logs"),
           "QUEUE_RESULTS_ROOT": str(tmp_path / "results")}
    done = subprocess.run(["bash", str(QUEUE), *args], env=env, capture_output=True, text=True, timeout=180)
    (logdir,) = list((tmp_path / "logs").glob("queue_*"))
    return done.returncode, (logdir / "queue.log").read_text(), logdir


def _starts(log):
    return [line.split("START ")[1].split(":")[0] for line in log.splitlines()
            if " START " in line and "QUEUE START" not in line]


def test_queue_order_and_commands(tmp_path):
    assert subprocess.run(["bash", "-n", str(QUEUE)]).returncode == 0
    rc, log, logdir = run_queue(tmp_path)
    assert rc == 0 and log.rstrip().endswith("QUEUE DONE ok")
    assert _starts(log) == ["0_seqft", "1_swaps0", "2_gpm", "3_swaps1_seqft", "4_swaps1_gpm", "5_probes",
                            "6_summary"]
    res = tmp_path / "results"
    first = (logdir / "0_seqft.log").read_text()
    assert "--curriculum langbase_cabinet_stove --seed 0 --amp --method seq_ft" in first
    assert "--steps-per-task" not in first
    gpm = (logdir / "2_gpm.log").read_text()
    assert "--method gpm" in gpm and f"--t1-reference-run {res}/langbase_cabinet_stove__seq_ft__seed0" in gpm
    assert f"--identity-reference-run {res}/langbase_cabinet_stove__seq_ft__seed0 --identity-stages 0" in gpm
    assert "--part swaps --stage 0 --arm seq_ft" in (logdir / "1_swaps0.log").read_text()
    assert "--part swaps --stage 1 --arm gpm" in (logdir / "4_swaps1_gpm.log").read_text()
    assert "--part probe" in (logdir / "5_probes.log").read_text()
    assert "--part summary" in (logdir / "6_summary.log").read_text()


def test_queue_failures_skip_dependents(tmp_path):
    rc, log, _ = run_queue(tmp_path / "a", fail="2_gpm")
    assert rc == 1 and "FAILED: 2_gpm" in log
    assert "4_swaps1_gpm" not in _starts(log) and {"3_swaps1_seqft", "5_probes", "6_summary"} <= set(_starts(log))
    rc, log, _ = run_queue(tmp_path / "b", fail="0_seqft")
    assert rc == 1 and _starts(log) == ["0_seqft", "6_summary"]
    rc, log, _ = run_queue(tmp_path / "c", ["--from-step", "5"])   # nothing on disk: priors missing
    assert rc == 1 and _starts(log) == ["6_summary"] and "PRIOR 0_seqft missing" in log


def test_queue_smoke_mode(tmp_path):
    scratch = tmp_path / "scratch"
    rc, log, logdir = run_queue(tmp_path, ["--smoke", str(scratch)])
    assert rc == 0
    first = (logdir / "0_seqft.log").read_text()
    assert f"--results-root {scratch} --steps-per-task 200 --n-episodes 1 --allow-dirty" in first
    probe = (logdir / "5_probes.log").read_text()
    assert f"--results-root {scratch} --allow-dirty --a-only-root {repo_root()}/results --smoke-batches 3" in probe
    assert f"--a-only-root {repo_root()}/results --smoke" in (logdir / "6_summary.log").read_text()
    assert subprocess.run(["bash", str(QUEUE), "--smoke", "rel"], capture_output=True).returncode == 2
    assert subprocess.run(["bash", str(QUEUE), "--smoke", str(repo_root() / "results" / "x")],
                          capture_output=True).returncode == 2


# ---- config ---------------------------------------------------------------------------------------------------


def test_config_is_consistent_with_the_curriculum_and_with_similar_cl():
    from flowcl.analysis.language_check import MIN_SWAP_SUCCESS_DROP
    from flowcl.data.curriculum import load_curriculum
    from flowcl.experiments import similar_cl as scl

    c = cfg()
    cur = load_curriculum(c["curriculum"])
    assert cur.task_keys == (c["task_a"], c["task_b"])
    assert [r.task_key for r in cur.stages[0].co_train] == [c["task_co"]]
    assert cur.stages[0].steps_factor == c["production"]["steps_factor"] == c["smoke"]["steps_factor"]
    assert c["swap_drop_min"] == MIN_SWAP_SUCCESS_DROP
    ref = scl.load_config()
    for key in ("probe", "swap", "chunks"):      # probe (a) must reproduce similar_cl's stage 0
        assert c[key] == ref[key]
    assert c["a_only"]["curriculum"] == ref["source_curriculum"] and c["task_a"] == ref["task_a"]
    assert c["production"]["train"] == ref["production"]["train"]
