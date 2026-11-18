"""Instruction-negatives pilot (``configs/analysis/instruction_negatives.yaml``).

The two training runs are ``scripts/run_continual.py`` with ``gpm_neg`` and ``allowlist_ft_neg`` on
``similar_ab_stove``; this module adds what the runner does not record, and judges the rule:

1. :func:`run_hold`: the hold target in the simulator, before any training;
2. :func:`run_panel`: A's scene under held-out, trained-negative, paraphrased and its own
   instructions on gpm's ``stage0.pt``, with motion, and the discrimination gate (``gate.json``);
3. :func:`run_swaps`: each arm's ``stage1.pt``, A's scene under B's instruction and B's under A's;
4. :func:`run_probe`: probe loss, the A <-> B instruction contrast and its share outside A's memory,
   memory ranks and coverage, and A's action drift from stage 0;
5. :func:`run_summary`: the validity checks, the cells, the labels, and ``summary.json``.

Before any rollout or probe the text cache is rebuilt as the runner built it, with the negative
bank in fp32.
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from flowcl.experiments import similar_tasks as st
from flowcl.utils.libero_paths import repo_root
from flowcl.utils.run import atomic_write_text, file_sha256, git_sha

VALID, INVALID, INCOMPLETE = "valid", "invalid", "incomplete"
EXIT_CODES = {VALID: 0, INVALID: 3, INCOMPLETE: 4}
ARMS = ("gpm", "allow")
PROBE_LABELS = ("neg_stage0", "gpm_stage1", "allow_stage1")


def config_path() -> Path:
    return repo_root() / "configs" / "analysis" / "instruction_negatives.yaml"


def load_config(path: str | Path | None = None) -> dict:
    return OmegaConf.to_container(OmegaConf.load(path or config_path()), resolve=True)


def _root(results_root) -> Path:
    return Path(results_root) if results_root else repo_root() / "results"


def out_dir(cfg: dict, root: Path) -> Path:
    return root / cfg["out"]


def run_dir(cfg: dict, root: Path, arm: str) -> Path:
    return root / cfg["runs"][arm]["name"]


def n_episodes(cfg: dict, smoke: bool) -> int:
    return int(cfg["smoke" if smoke else "production"]["n_episodes"])


def _bootstrap() -> dict:
    from flowcl.experiments.similar_cl import _bootstrap as b

    return b()


def _conf(run: Path) -> dict:
    return OmegaConf.to_container(OmegaConf.load(run / "config.yaml"), resolve=True)


def _publish(canonical: Path, filename: str, record: dict) -> None:
    attempt = st._attempt(canonical.parent, canonical.name)
    atomic_write_text(attempt / filename, json.dumps(record, indent=2, default=str) + "\n")
    st._publish(attempt, canonical)


def _read(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


# ---- motion and the hold check (pure parts) ----------------------------------------------------------


def close_start(actions: np.ndarray, g: int, run: int) -> int | None:
    """The first step ``t`` after the gripper command has been > 0 for ``run`` consecutive steps."""
    closed = np.asarray(actions)[:, g] > 0
    count = 0
    for t, c in enumerate(closed):
        count = count + 1 if c else 0
        if count == run:
            return t + 1
    return None


def hold_action(actions: np.ndarray, t: int, g: int) -> np.ndarray:
    """The method's hold action at step ``t`` (``hold_rows``'s row), as float64 for the env."""
    from flowcl.methods.instruction_negatives import hold_rows

    return hold_rows(np.asarray(actions, dtype=np.float32), g)[t].astype(np.float64)


def displacement(eef: np.ndarray, fingers: np.ndarray, settle_step: int) -> dict:
    """Movement after row ``settle_step`` (max over later rows of the distance from that row: the
    end-effector's norm, each finger's |q - q[settle]|), and before it (from the first row: the
    stopping distance and the fingers' travel)."""
    eef, fingers = np.asarray(eef, dtype=float), np.asarray(fingers, dtype=float)
    late_e, late_f = eef[settle_step:], fingers[settle_step:]
    return {"eef_after_m": float(np.linalg.norm(late_e - late_e[0], axis=1).max()) if len(late_e) else 0.0,
            "finger_after_m": float(np.abs(late_f - late_f[0]).max()) if len(late_f) else 0.0,
            "eef_from_start_m": float(np.linalg.norm(eef - eef[0], axis=1).max()),
            "finger_before_m": float(np.abs(fingers[: settle_step + 1] - fingers[0]).max()),
            "n_rows": int(len(eef))}


def motion_summary(motion: dict | None, window: int) -> dict | None:
    """Over the recorded window: end-effector path length and max displacement (m), finger range (m),
    gripper-command sign changes, and whether the episode lasted the whole window."""
    if not motion:
        return None
    eef = np.asarray(motion["eef_pos"], dtype=float)
    fingers = np.asarray(motion["gripper_qpos"], dtype=float)
    cmd = np.sign(np.asarray(motion["gripper_cmd"], dtype=float))
    path = float(np.linalg.norm(np.diff(eef, axis=0), axis=1).sum()) if len(eef) > 1 else 0.0
    return {"path_m": path, "max_displacement_m": float(np.linalg.norm(eef - eef[0], axis=1).max()),
            "finger_range_m": float((fingers.max(axis=0) - fingers.min(axis=0)).max()),
            "gripper_sign_changes": int((np.diff(cmd) != 0).sum()) if len(cmd) > 1 else 0,
            "n_steps": int(len(cmd)), "full_window": int(len(cmd)) == int(window)}


def _median(values) -> float | None:
    values = [v for v in values if v is not None]
    return float(np.median(values)) if values else None


# ---- step: the hold check -------------------------------------------------------------------------------


def run_hold(cfg: dict, *, results_root=None, allow_dirty=False, dataset_dir=None) -> dict:
    from flowcl.data.config import load_embodiment_spec
    from flowcl.data.libero_adapter import load_episode
    from flowcl.data.tasks import TaskRef
    from flowcl.envs.libero_env import LiberoTaskEnv
    from flowcl.experiments.low_update import refuse_dirty
    from flowcl.methods.instruction_negatives import gripper_index

    sha = refuse_dirty(allow_dirty)
    canonical = out_dir(cfg, _root(results_root)) / "hold"
    if (canonical / "hold.json").is_file():
        print("[flowcl] hold check: published, reused", flush=True)
        return json.loads((canonical / "hold.json").read_text())
    hc = cfg["hold_check"]
    spec = load_embodiment_spec("libero_franka")
    g = gripper_index(spec)
    ref = TaskRef.from_key(cfg["task_a"])
    started = time.perf_counter()
    states = []
    env = LiberoTaskEnv(ref.suite, ref.task_idx, spec, image_size=int(spec.observation.image_size[0]))
    try:
        for d in range(int(hc["n_demos"])):
            ep = load_episode(ref.demo_path(dataset_dir), d, spec, task_id=ref.task_key, with_sim_states=True)
            for label, t in (("initial", 0), ("closed", close_start(ep.action, g, int(hc["close_run"])))):
                if t is None:
                    states.append({"demo": d, "state": label, "t": None, "passed": False,
                                   "problem": "no closing run"})
                    continue
                env.env.seed(0)
                env.env.reset()
                obs = env.env.set_init_state(ep.meta["init_state"])
                for a in ep.action[:t]:
                    obs, _r, _d, _i = env.env.step(np.asarray(a, dtype=np.float64))
                hold = hold_action(ep.action, t, g)
                eef, fingers = [obs["robot0_eef_pos"].copy()], [obs["robot0_gripper_qpos"].copy()]
                for _ in range(int(hc["hold_steps"])):
                    obs, _r, _d, _i = env.env.step(hold)
                    eef.append(obs["robot0_eef_pos"].copy())
                    fingers.append(obs["robot0_gripper_qpos"].copy())
                disp = displacement(np.stack(eef), np.stack(fingers), int(hc["settle_step"]))
                ok = disp["eef_after_m"] < float(hc["eef_max_m"]) and disp["finger_after_m"] < float(hc["finger_max_m"])
                states.append({"demo": d, "state": label, "t": int(t), "hold_gripper": float(hold[g]),
                               **disp, "passed": bool(ok)})
                print(f"[flowcl] hold demo {d} {label} t={t}: after settling eef {disp['eef_after_m']:.4f} m, "
                      f"fingers {disp['finger_after_m']:.4f} m (stopping {disp['eef_from_start_m']:.4f} m) -> "
                      f"{'ok' if ok else 'FAIL'}", flush=True)
    finally:
        env.close()
    record = {"git_sha": sha, "task": cfg["task_a"], "bounds": hc, "states": states,
              "passed": bool(states) and all(s["passed"] for s in states),
              "wall_clock_s": time.perf_counter() - started}
    _publish(canonical, "hold.json", record)
    return record


# ---- rollouts ------------------------------------------------------------------------------------------


def _warm(policy, texts, device) -> dict:
    """Encode ``texts`` not yet cached in fp32; returns each text's precision."""
    out = {}
    with torch.no_grad(), torch.autocast(device_type=torch.device(device).type, enabled=False):
        for text in texts:
            out[text] = "cached" if text in policy.text_encoder._cache else "fp32"
            policy.text_encoder([text])
    return out


def _load(ckpt: Path, run: Path, cfg: dict, extra_keys: list[str], extra_texts: list[str], device, dataset_dir):
    """The checkpoint with the runner's text cache, then ``extra_keys``' instructions, the bank and
    ``extra_texts`` in fp32."""
    from flowcl.experiments.language_cl import rebuild_cache
    from flowcl.methods.instruction_negatives import load_bank
    from flowcl.train.checkpoint import load_checkpoint

    conf = _conf(run)
    loaded = load_checkpoint(ckpt, device=device)
    policy = loaded.policy
    policy.eval()
    cache = rebuild_cache(policy, cfg["curriculum"], extra_keys, dataset_dir, device, bool(conf["train"].get("amp")))
    bank = _warm(policy, load_bank(cfg["bank"])["strings"], device)
    extra = _warm(policy, extra_texts, device)
    return loaded, conf, {"tasks": cache, "bank": bank, "extra": extra}


def _rollouts(policy, env, loaded, ns, eval_cfg, text, requested, n, window, own_text) -> dict:
    from flowcl.envs.libero_env import goal_state

    goal = goal_state(requested) if requested else None
    unscorable = None
    if goal is not None and env.missing_goal_objects(goal):
        unscorable, goal = env.missing_goal_objects(goal), None
    rows = []
    for ep in range(int(n)):
        r = env.rollout(policy, ep, loaded.stats, ns, eval_cfg,
                        language=None if text == own_text else text, alt_goal=goal, motion=int(window))
        rows.append({"episode": ep, "seed": int(r.seed), "own_goal": bool(r.success), "n_steps": int(r.n_steps),
                     "requested_reached": r.alt_reached, "requested_at_end": r.alt_at_end,
                     "motion": motion_summary(r.motion, int(window))})
    own = [r["own_goal"] for r in rows]
    req = [r["requested_reached"] for r in rows if r["requested_reached"] is not None]
    return {"text": text, "requested_task": requested, "requested_unscorable": unscorable, "n": len(rows),
            "own_goal_rate": float(np.mean(own)) if own else None,
            "requested_rate": float(np.mean(req)) if req else None,
            "median_max_displacement_m": _median([r["motion"]["max_displacement_m"] for r in rows if r["motion"]]),
            "median_path_m": _median([r["motion"]["path_m"] for r in rows if r["motion"]]),
            "median_finger_range_m": _median([r["motion"]["finger_range_m"] for r in rows if r["motion"]]),
            "full_window_fraction": float(np.mean([r["motion"]["full_window"] for r in rows if r["motion"]]))
            if rows else None,
            "rows": rows}


def panel_conditions(cfg: dict, dataset_dir=None, smoke: bool = False) -> list[dict]:
    from flowcl.experiments.language_cl import instruction_of

    p = cfg["panel"]
    n_held = n_episodes(cfg, True) if smoke else int(p["n_held_out"])
    n_other = n_episodes(cfg, True) if smoke else int(p["n_other"])
    conds = [{"name": f"held_out::{st.slug(k)}", "kind": "held_out_task", "text": instruction_of(k, dataset_dir),
              "requested": k, "n": n_held} for k in cfg["held_out_tasks"]]
    conds += [{"name": f"trained_negative::{s}", "kind": "trained_negative", "text": s, "requested": None,
               "n": n_other} for s in p["trained_negatives"]]
    conds += [{"name": f"held_out_other::{h['text']}", "kind": "held_out_other", "text": h["text"],
               "requested": h["task"], "n": n_other} for h in p["held_out_other"]]
    conds += [{"name": f"paraphrase::{s}", "kind": "paraphrase", "text": s, "requested": None, "n": n_other}
              for s in p["paraphrases"]]
    conds.append({"name": "own", "kind": "own", "text": instruction_of(cfg["task_a"], dataset_dir),
                  "requested": None, "n": n_other})
    return conds


def gate(cfg: dict, r0_a: list[bool], panel: dict) -> dict:
    """The discrimination gate (pure): R0[A] >= learned_min and, under each held-out task
    instruction, A's goal reached in < held_out_reach_max of rollouts."""
    g = cfg["gate"]
    r0 = float(np.mean(r0_a)) if r0_a else None
    held = {name: c["own_goal_rate"] for name, c in panel["conditions"].items() if c["kind"] == "held_out_task"}
    learned = r0 is not None and r0 >= float(g["learned_min"])
    suppressed = bool(held) and all(v is not None and v < float(g["held_out_reach_max"]) for v in held.values())
    return {"passed": bool(learned and suppressed), "R0_A": r0, "learned": learned,
            "held_out_own_goal_rate": held, "suppressed": suppressed, "thresholds": g}


def run_panel(cfg: dict, *, results_root=None, device="cuda", allow_dirty=False, dataset_dir=None,
              smoke: bool = False) -> dict:
    from flowcl.data.tasks import TaskRef
    from flowcl.envs.evaluation import EvaluationReport, eval_config_from_dict
    from flowcl.envs.libero_env import LiberoTaskEnv
    from flowcl.experiments.low_update import refuse_dirty

    sha = refuse_dirty(allow_dirty)
    root = _root(results_root)
    canonical = out_dir(cfg, root) / "panel"
    if (canonical / "panel.json").is_file():
        print("[flowcl] panel: published, reused", flush=True)
        return json.loads((canonical / "panel.json").read_text())
    run = run_dir(cfg, root, "gpm")
    ckpt = run / "checkpoints" / "stage0.pt"
    ckpt_sha = file_sha256(ckpt)
    conds = panel_conditions(cfg, dataset_dir, smoke)
    extra_keys = sorted({c["requested"] for c in conds if c["requested"] and c["requested"] != cfg["task_b"]})
    loaded, conf, cache = _load(ckpt, run, cfg, extra_keys, [c["text"] for c in conds if c["kind"] == "paraphrase"],
                                device, dataset_dir)
    if loaded.stage != 0 or loaded.task_key != cfg["task_a"]:
        raise ValueError(f"{ckpt} is {loaded.task_key} stage {loaded.stage}")
    policy = loaded.policy
    weights = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    eval_cfg = eval_config_from_dict(conf["eval"])
    report = EvaluationReport.load(run / "eval" / "stage0.json").by_task()[cfg["task_a"]]
    ref = TaskRef.from_key(cfg["task_a"])
    own_text = next(c["text"] for c in conds if c["kind"] == "own")
    window = int(cfg["panel"]["motion_steps"])
    started = time.perf_counter()
    out = {}
    with LiberoTaskEnv(ref.suite, ref.task_idx, loaded.spec, image_size=eval_cfg.image_size) as env:
        if own_text.strip().lower() != env.language.strip().lower():
            raise ValueError(f"A's demo instruction {own_text!r} != the env's {env.language!r}")
        for c in conds:
            res = _rollouts(policy, env, loaded, cfg["namespace"], dataclasses.replace(eval_cfg, n_episodes=c["n"]),
                            c["text"], c["requested"], c["n"], window, own_text)
            res["kind"] = c["kind"]
            res["eval_seeds_match"] = [r["seed"] for r in res["rows"]] == list(report.seeds)[: c["n"]]
            out[c["name"]] = res
            print(f"[flowcl] panel {c['name']}: A goal {res['own_goal_rate']}, requested {res['requested_rate']}, "
                  f"median displacement {res['median_max_displacement_m']}", flush=True)
    unchanged = all(torch.equal(v, weights[k]) for k, v in policy.state_dict().items())
    record = {"git_sha": sha, "smoke": smoke, "run": run.name, "checkpoint": str(ckpt), "checkpoint_sha256": ckpt_sha,
              "namespace": cfg["namespace"], "text_cache": cache, "motion_steps": window, "conditions": out,
              "weights_unchanged": unchanged, "wall_clock_s": time.perf_counter() - started}
    record["gate"] = gate(cfg, list(report.successes), record)
    attempt = st._attempt(canonical.parent, canonical.name)
    atomic_write_text(attempt / "panel.json", json.dumps(record, indent=2, default=str) + "\n")
    atomic_write_text(attempt / "gate.json", json.dumps(record["gate"], indent=2) + "\n")
    st._publish(attempt, canonical)
    print(f"[flowcl] gate: {'passed' if record['gate']['passed'] else 'FAILED'} {record['gate']}", flush=True)
    return record


def gate_passed(cfg: dict, results_root=None) -> bool:
    rec = _read(out_dir(cfg, _root(results_root)) / "panel" / "gate.json")
    return bool(rec and rec.get("passed"))


# ---- step: stage-1 swaps -------------------------------------------------------------------------------


def run_swaps(cfg: dict, arm: str, *, results_root=None, device="cuda", allow_dirty=False, dataset_dir=None) -> dict:
    from flowcl.data.tasks import TaskRef
    from flowcl.envs.evaluation import EvaluationReport, eval_config_from_dict
    from flowcl.envs.libero_env import LiberoTaskEnv
    from flowcl.experiments.language_cl import instruction_of
    from flowcl.experiments.low_update import refuse_dirty

    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    sha = refuse_dirty(allow_dirty)
    root = _root(results_root)
    canonical = out_dir(cfg, root) / "swaps" / arm
    if (canonical / "swaps.json").is_file():
        print(f"[flowcl] swaps {arm}: published, reused", flush=True)
        return json.loads((canonical / "swaps.json").read_text())
    run = run_dir(cfg, root, arm)
    ckpt = run / "checkpoints" / "stage1.pt"
    ckpt_sha = file_sha256(ckpt)
    loaded, conf, cache = _load(ckpt, run, cfg, [], [], device, dataset_dir)
    policy = loaded.policy
    weights = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    eval_cfg = eval_config_from_dict(conf["eval"])
    report = EvaluationReport.load(run / "eval" / "stage1.json").by_task()
    a, b = cfg["task_a"], cfg["task_b"]
    window = int(cfg["panel"]["motion_steps"])
    started = time.perf_counter()
    out = {}
    for scene, requested in ((a, b), (b, a)):
        ref = TaskRef.from_key(scene)
        with LiberoTaskEnv(ref.suite, ref.task_idx, loaded.spec, image_size=eval_cfg.image_size) as env:
            own = instruction_of(scene, dataset_dir)
            res = _rollouts(policy, env, loaded, cfg["namespace"], eval_cfg, instruction_of(requested, dataset_dir),
                            requested, eval_cfg.n_episodes, window, own)
        res["correct_successes"] = list(report[scene].successes)
        res["eval_seeds_match"] = [r["seed"] for r in res["rows"]] == list(report[scene].seeds)
        out[f"{scene}->{requested}"] = res
        print(f"[flowcl] swaps {arm} {st.slug(scene)} under {st.slug(requested)}: own goal {res['own_goal_rate']}, "
              f"requested {res['requested_rate']}", flush=True)
    unchanged = all(torch.equal(v, weights[k]) for k, v in policy.state_dict().items())
    record = {"git_sha": sha, "arm": arm, "run": run.name, "checkpoint": str(ckpt), "checkpoint_sha256": ckpt_sha,
              "text_cache": cache, "swaps": out, "weights_unchanged": unchanged,
              "wall_clock_s": time.perf_counter() - started}
    _publish(canonical, "swaps.json", record)
    return record


# ---- step: probes -------------------------------------------------------------------------------------


@torch.no_grad()
def action_drift(policy0, policy1, dataset, cfg: dict, device, n_steps: int, smoke_batches: int | None) -> dict:
    """A's sampled chunks at two checkpoints from identical observations and noise:
    ``sum |c1 - c0| / sum |c0|`` over the first fixed batches."""
    from flowcl.train.trainer import build_dataloader, move_batch
    from flowcl.utils.seeding import derive_seed

    d = cfg["drift"]
    loader = build_dataloader(dataset, batch_size=int(d["batch_size"]), num_workers=0, shuffle=False)
    limit = int(smoke_batches or d["n_batches"])
    diff = scale = 0.0
    n = 0
    with torch.autocast(device_type=torch.device(device).type, enabled=False):
        for i, batch in enumerate(loader):
            if i >= limit:
                break
            batch = move_batch(batch, torch.device(device))
            seed = derive_seed(d["seed_tag"], "drift", i)
            c0 = policy0.sample(batch, n_steps=n_steps, generator=torch.Generator(device="cpu").manual_seed(seed))
            c1 = policy1.sample(batch, n_steps=n_steps, generator=torch.Generator(device="cpu").manual_seed(seed))
            diff += float((c1 - c0).abs().sum())
            scale += float(c0.abs().sum())
            n += int(c0.shape[0])
    return {"relative": diff / scale if scale > 0 else None, "n_samples": n}


def probe_target(cfg: dict, root: Path, label: str) -> dict:
    if label not in PROBE_LABELS:
        raise ValueError(f"unknown probe {label!r}")
    arm, stage = ("gpm", 0) if label == "neg_stage0" else (label.split("_stage")[0], 1)
    gpm = run_dir(cfg, root, "gpm")
    mem = torch.load(gpm / "checkpoints" / "stage0.pt", map_location="cpu", weights_only=False)
    arts = {x["path"]: x["sha256"] for x in (mem.get("extra") or {}).get("method_artifacts") or []}
    run = run_dir(cfg, root, arm)
    return {"run": run, "checkpoint": run / "checkpoints" / f"stage{stage}.pt", "stage": stage,
            "stage0": run / "checkpoints" / "stage0.pt",
            "memory": {"path": str(gpm / "method" / "memory_task0.pt"),
                       "expected_sha256": arts.get("method/memory_task0.pt")}}


def run_probe(cfg: dict, label: str, *, results_root=None, device="cuda", allow_dirty=False, dataset_dir=None,
              smoke_batches: int | None = None, target: dict | None = None) -> dict:
    from flowcl.analysis.probes import probe_loss
    from flowcl.experiments.language_cl import instruction_probe, load_memory, memory_coverage, memory_ranks
    from flowcl.experiments.low_update import refuse_dirty
    from flowcl.experiments.similar_cl import _probe_cfg

    sha = refuse_dirty(allow_dirty)
    root = _root(results_root)
    canonical = out_dir(cfg, root) / "probes" / label
    if (canonical / "probe.json").is_file():
        print(f"[flowcl] probe {label}: published, reused", flush=True)
        return json.loads((canonical / "probe.json").read_text())
    t = target or probe_target(cfg, root, label)  # a resumed run's target names its source's stage 0
    ckpt = t["checkpoint"]
    ckpt_sha = file_sha256(ckpt)
    loaded, conf, cache = _load(ckpt, t["run"], cfg, [], [], device, dataset_dir)
    policy = loaded.policy
    weights = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    smoke = smoke_batches is not None
    sub = st._capture_config({"capture_config": cfg["swap"]["capture_config"]}, smoke)
    cov_sub = st._capture_config({"capture_config": cfg["coverage"]["capture_config"]}, smoke)
    size = max(int(cfg["probe"]["batch_size"]), int(cfg["chunks"]["batch_size"]), int(sub.batch_size),
               int(cfg["drift"]["batch_size"]))
    n_steps = int(conf["eval"].get("euler_steps", 10))
    a, b = cfg["task_a"], cfg["task_b"]
    started = time.perf_counter()
    M, memory = load_memory(t["memory"])
    memory["ranks"] = memory_ranks(M, st._dims(policy)[0])
    ds_a = st._dataset(a, loaded, dataset_dir, None, smoke_batches, size)
    ds_b = st._dataset(b, loaded, dataset_dir, None, smoke_batches, size)
    probe = _probe_cfg(cfg, smoke_batches)
    text = {a: st.instruction(ds_a), b: st.instruction(ds_b)}
    record = {"git_sha": sha, "smoke": smoke, "label": label, "run": t["run"].name, "checkpoint": str(ckpt),
              "checkpoint_sha256": ckpt_sha, "text_cache": cache, "memory": memory,
              "probe_loss": {a: probe_loss(policy, ds_a, probe, device), b: probe_loss(policy, ds_b, probe, device)},
              "instruction": instruction_probe(policy, ds_a, ds_b, text[a], text[b], cfg, sub, device, f"{a}->{b}",
                                               n_steps, smoke_batches, M)}
    if label == "neg_stage0":
        record["memory_coverage"] = memory_coverage(policy, ds_a, M, cov_sub, device, a, cfg)
    else:
        loaded0, _, _ = _load(t["stage0"], t["run"], cfg, [], [], device, dataset_dir)
        record["drift_from_stage0"] = action_drift(loaded0.policy, policy, ds_a, cfg, device, n_steps, smoke_batches)
        del loaded0
    record["weights_unchanged"] = all(torch.equal(v, weights[k]) for k, v in policy.state_dict().items())
    record["wall_clock_s"] = time.perf_counter() - started
    _publish(canonical, "probe.json", record)
    print(f"[flowcl] probe {label}: probe loss {record['probe_loss']}", flush=True)
    return record


def run_probes(cfg: dict, **kw) -> list[str]:
    root = _root(kw.get("results_root"))
    problems = []
    for label in PROBE_LABELS:
        try:
            ckpt = probe_target(cfg, root, label)["checkpoint"]
        except (OSError, KeyError) as exc:
            problems.append(f"{label}: {exc}")
            continue
        if not ckpt.is_file():
            print(f"[flowcl] probe {label}: {ckpt} missing, skipped", flush=True)
            if label != "allow_stage1":
                problems.append(f"{ckpt}: missing")
            continue
        try:
            run_probe(cfg, label, **kw)
        except Exception as exc:  # one probe's failure must not block the others
            problems.append(f"{label}: {type(exc).__name__}: {exc}")
            print(f"[flowcl] probe {label} FAILED: {exc}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return problems


# ---- the judgment (pure) ------------------------------------------------------------------------------


def _rate(x) -> float | None:
    return float(np.mean(x)) if x is not None and len(x) else None


def _paired(a, b, bootstrap: dict) -> dict:
    from flowcl.experiments.reverse_order import _paired as paired

    return paired(list(map(float, a)), list(map(float, b)), bootstrap)


def judge(cfg: dict, cells: dict, gate_rec: dict | None, bootstrap: dict) -> dict:
    """Labels from the cells (pure). ``cells[run][stage][task]`` = successes; runs ``gpm``, ``allow``,
    ``hist_gpm``, ``hist_allow`` (an absent run is missing)."""
    a, b = cfg["task_a"], cfg["task_b"]
    acq, ret = cfg["acquisition"], cfg["retention"]

    def cell(run, stage, task):
        return ((cells.get(run) or {}).get(stage) or {}).get(task)

    out = {"gate": gate_rec}
    if not gate_rec or not gate_rec.get("passed"):
        out["outcome"] = "gate_failed" if gate_rec else "gate_missing"
        out["labels"] = {k: "not_assessed" for k in ("effect_present", "retains", "projection_contributes",
                                                     "allowlist_without_projection_retains")}
        out["acquisition"] = {"R1_B_gpm": _rate(cell("gpm", 1, b))}
        return out
    r1b = {arm: _rate(cell(arm, 1, b)) for arm in ARMS}
    acquisition = {"R1_B": r1b, "gpm_minus_allow": None}
    if None in r1b.values():
        state = "missing"
    elif min(r1b.values()) < float(acq["inconclusive_below"]):
        state = "inconclusive_acquisition"
    else:
        acquisition["gpm_minus_allow"] = _paired(cell("gpm", 1, b), cell("allow", 1, b), bootstrap)
        clean = (min(r1b.values()) >= float(acq["floor"])
                 and r1b["gpm"] - r1b["allow"] >= -float(acq["tolerance"]))
        state = "clean" if clean else "at_acquisition_cost"
    acquisition["state"] = state
    suffix = "" if state == "clean" else "_at_acquisition_cost"

    def label(name: str, holds: bool) -> str:
        if state in ("inconclusive_acquisition", "missing"):
            return "not_assessed"
        return (name + suffix) if holds else "not_shown"

    forget = {arm: (_rate(cell(arm, 0, a)) - _rate(cell(arm, 1, a))
                    if cell(arm, 0, a) is not None and cell(arm, 1, a) is not None else None) for arm in ARMS}
    effect = (_paired(cell("gpm", 1, a), cell("hist_gpm", 1, a), bootstrap)
              if cell("gpm", 1, a) is not None and cell("hist_gpm", 1, a) is not None else None)
    proj = (_paired(cell("gpm", 1, a), cell("allow", 1, a), bootstrap)
            if cell("gpm", 1, a) is not None and cell("allow", 1, a) is not None else None)
    labels = {
        "effect_present": label("effect_present", bool(effect and effect["diff"] >= float(ret["effect_min"])
                                                       and effect["low"] > 0)),
        "retains": label("retains", forget["gpm"] is not None and forget["gpm"] <= float(ret["forgetting_max"])),
        "projection_contributes": label("projection_contributes", bool(proj and proj["low"] > 0)),
        "allowlist_without_projection_retains": label(
            "allowlist_without_projection_retains",
            forget["allow"] is not None and forget["allow"] <= float(ret["forgetting_max"])),
    }
    table = {f"{neg}/{arm}": _rate(cell(run, 1, a)) for neg, arm, run in
             (("off", "allow", "hist_allow"), ("off", "gpm", "hist_gpm"), ("on", "allow", "allow"), ("on", "gpm", "gpm"))}
    interaction = None
    if all(cell(r, 1, a) is not None for r in ("gpm", "allow", "hist_gpm", "hist_allow")):
        on = np.asarray(cell("gpm", 1, a), float) - np.asarray(cell("allow", 1, a), float)
        off = np.asarray(cell("hist_gpm", 1, a), float) - np.asarray(cell("hist_allow", 1, a), float)
        interaction = {**_paired(on, off, bootstrap), "flag": "against_historical_control"}
    out.update({"outcome": "assessed", "acquisition": acquisition, "forgetting_A": forget,
                "effect_vs_historical_gpm": effect and {**effect, "flag": "against_historical_control"},
                "gpm_minus_allow_A": proj, "labels": labels, "table_R1_A": table, "interaction": interaction})
    return out


# ---- validity and the summary ---------------------------------------------------------------------------


def _eval_cells(run: Path, keys: list[str], n: int) -> tuple[dict, dict, list[str]]:
    """``{stage: {task: successes[:n]}}``, the same for seeds, and the problems."""
    from flowcl.envs.evaluation import EvaluationReport

    succ, seeds, problems = {}, {}, []
    for stage in (0, 1):
        path = run / "eval" / f"stage{stage}.json"
        if not path.is_file():
            problems.append(f"{run.name}: no eval/stage{stage}.json")
            continue
        by = EvaluationReport.load(path).by_task()
        for k in keys:
            if k not in by or len(by[k].successes) < n:
                problems.append(f"{run.name}: stage {stage} {k} has fewer than {n} rollouts")
                continue
            succ.setdefault(stage, {})[k] = [bool(x) for x in by[k].successes[:n]]
            seeds.setdefault(stage, {})[k] = list(by[k].seeds[:n])
    return succ, seeds, problems


def _negatives_check(run: Path, bank_sha: str) -> dict:
    """Negatives active at stage 0 (bank cached in fp32, loss logged) and inactive at stage 1, with
    the bank's SHA-256, from the stage checkpoints' method state."""
    problems, curve = {}, None
    for stage, active in ((0, True), (1, False)):
        ckpt = run / "checkpoints" / f"stage{stage}.pt"
        if not ckpt.is_file():
            problems[f"stage{stage}"] = "missing"
            continue
        state = (torch.load(ckpt, map_location="cpu", weights_only=False).get("extra") or {}).get("method_state") or {}
        neg = state.get("instruction_negatives") or {}
        log = (neg.get("logs") or {}).get(str(stage)) or {}
        if bool(log.get("active")) is not active:
            problems[f"stage{stage}_active"] = log.get("active")
        if neg.get("negative_bank_sha256") != bank_sha:
            problems[f"stage{stage}_bank_sha256"] = neg.get("negative_bank_sha256")
        if not active:
            continue
        cache = log.get("text_cache") or {}
        if not cache or any(v != "fp32" for v in cache.values()):
            problems["bank_cache"] = cache
        steps = log.get("steps") or {}
        if not steps:
            problems["negative_loss_log"] = "empty"
        else:
            ordered = [steps[k]["negative_loss_mean"] for k in sorted(steps, key=int)]
            curve = {"first": ordered[0], "last": ordered[-1], "n_logged": len(ordered)}
    return {"passed": not problems, "problems": problems, "negative_loss": curve}


def _run_checks(cfg: dict, run: Path, arm: str, smoke: bool, bank_sha: str) -> dict:
    from flowcl.experiments.similar_cl import frozen_check, identity_check, memory_check

    conf = _conf(run)
    checks = {"namespace": conf.get("seed_namespace_run_id") == cfg["namespace"],
              "method": conf.get("method", {}).get("name") == cfg["runs"][arm]["method"],
              "clean_git_sha": smoke or not str((run / "git_sha").read_text().strip()).endswith("-dirty"),
              "result_json": (run / "result.json").is_file(),
              "negatives": _negatives_check(run, bank_sha)}
    if not smoke:
        train = conf.get("train", {})
        want = cfg["production"]["train"]
        checks["recipe"] = all(train.get(k) == v for k, v in want.items())
    if arm == "gpm":
        checks["memory"] = memory_check(run)
    else:
        from flowcl.train.checkpoint import load_checkpoint

        checks["identity_stage0"] = identity_check(run)
        pairing = _read(run / "t1_pairing.json") or {}
        checks["t1_pairing"] = {"passed": pairing.get("passed") is True,
                                "reference_run": pairing.get("reference_run"),
                                "rel_weight_diff": pairing.get("rel_weight_diff")}
        allowed = set(load_checkpoint(run / "checkpoints" / "stage0.pt", device="cpu").policy.projectable_parameters())
        checks["frozen"] = frozen_check(run, run, allowed, "allowlist_ft")
    ok = all((v.get("passed") if isinstance(v, dict) else bool(v)) for v in checks.values())
    return {"passed": ok, **checks}


def run_summary(cfg: dict, *, results_root=None, historical_root=None, smoke: bool = False,
                out: Path | None = None) -> dict:
    from flowcl.methods.instruction_negatives import load_bank

    root = _root(results_root)
    hist_root = Path(historical_root) if historical_root else repo_root() / "results"
    n = n_episodes(cfg, smoke)
    keys = [cfg["task_a"], cfg["task_b"]]
    bank_sha = load_bank(cfg["bank"])["sha256"]
    incomplete, invalid = [], []
    validity = {}
    hold = _read(out_dir(cfg, root) / "hold" / "hold.json")
    validity["hold"] = {"passed": bool(hold and hold.get("passed")), "present": hold is not None}
    if hold is None:
        incomplete.append("hold check missing")
    elif not hold.get("passed"):
        invalid.append("hold check failed")
    panel = _read(out_dir(cfg, root) / "panel" / "panel.json")
    gate_rec = panel.get("gate") if panel else None
    cells, seeds = {}, {}
    runs = {"gpm": run_dir(cfg, root, "gpm"), "allow": run_dir(cfg, root, "allow"),
            "hist_gpm": hist_root / cfg["historical"]["gpm"], "hist_allow": hist_root / cfg["historical"]["allow"]}
    expected = ["gpm", "hist_gpm", "hist_allow"] + (["allow"] if (gate_rec and gate_rec.get("passed")) or smoke else [])
    for name, path in runs.items():
        if not (path / "result.json").is_file():
            if name in expected:
                incomplete.append(f"{name}: {path.name} has no result.json")
            continue
        succ, sd, problems = _eval_cells(path, keys, n)
        cells[name], seeds[name] = succ, sd
        if problems:
            incomplete.extend(problems)
        if name in ARMS:
            validity[name] = _run_checks(cfg, path, name, smoke, bank_sha)
            if not validity[name]["passed"]:
                invalid.append(f"{name}: run checks failed")
    reference = seeds.get("gpm")
    validity["seeds_identical"] = bool(reference) and all(
        seeds[r] == reference for r in seeds if r in runs)
    if reference and not validity["seeds_identical"]:
        invalid.append("per-episode seeds differ across runs")
    if panel is None:
        if "gpm" in cells:
            incomplete.append("panel missing")
    else:
        gpm_ckpt = runs["gpm"] / "checkpoints" / "stage0.pt"
        validity["panel"] = {"checkpoint": gpm_ckpt.is_file() and panel["checkpoint_sha256"] == file_sha256(gpm_ckpt),
                             "weights_unchanged": panel["weights_unchanged"],
                             "eval_seeds": all(c["eval_seeds_match"] for c in panel["conditions"].values())}
        if not all(validity["panel"].values()):
            invalid.append("panel artifact")
    swaps = {}
    for arm in ARMS:
        rec = _read(out_dir(cfg, root) / "swaps" / arm / "swaps.json")
        if rec is None:
            if arm in cells and arm in expected:
                incomplete.append(f"swaps {arm} missing")
            continue
        ckpt = runs[arm] / "checkpoints" / "stage1.pt"
        ok = (rec["checkpoint_sha256"] == file_sha256(ckpt) and rec["weights_unchanged"]
              and all(s["eval_seeds_match"] for s in rec["swaps"].values()))
        validity[f"swaps_{arm}"] = ok
        if not ok:
            invalid.append(f"swaps {arm} artifact")
        swaps[arm] = {k: {key: v[key] for key in ("own_goal_rate", "requested_rate", "median_max_displacement_m")}
                      for k, v in rec["swaps"].items()}
    probes = {lab: _read(out_dir(cfg, root) / "probes" / lab / "probe.json") for lab in PROBE_LABELS}
    # gate_failed is a valid outcome: the allow run, its swaps and probe are not expected then.
    status = INVALID if invalid else (INCOMPLETE if incomplete else VALID)
    judged = judge(cfg, cells, gate_rec, _bootstrap())
    panel_view = None
    if panel:
        panel_view = {name: {k: c[k] for k in ("kind", "text", "n", "own_goal_rate", "requested_rate",
                                                "median_max_displacement_m", "median_path_m",
                                                "median_finger_range_m", "full_window_fraction")}
                      for name, c in panel["conditions"].items()}
    timing = {}
    for arm in ARMS:
        res = _read(runs[arm] / "result.json")
        if res:
            timing[arm] = [s.get("train_wall_clock_s") for s in res.get("stages", [])]
    for name in ("hist_gpm", "hist_allow"):
        res = _read(runs[name] / "result.json")
        if res:
            timing[name] = [s.get("train_wall_clock_s") for s in res.get("stages", [])]
    record = {"git_sha": git_sha(), "rule_sha256": file_sha256(config_path()), "bank_sha256": bank_sha,
              "smoke": smoke, "status": status, "exit_code": EXIT_CODES[status], "n_episodes": n,
              "namespace": cfg["namespace"], "incomplete": incomplete, "invalid": invalid, "validity": validity,
              "cells": {r: {str(s): {k: _rate(v) for k, v in t.items()} for s, t in c.items()} for r, c in cells.items()},
              "judgment": judged, "panel": panel_view, "stage1_swaps": swaps,
              "probes": {k: (None if v is None else {"probe_loss": v.get("probe_loss"),
                                                     "drift_from_stage0": v.get("drift_from_stage0"),
                                                     "path": f"probes/{k}/probe.json"}) for k, v in probes.items()},
              "stage_wall_clock_s": timing, "config": cfg}
    path = Path(out) if out else out_dir(cfg, root) / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(record, indent=2, default=str) + "\n")
    print(f"[flowcl] summary: {status}; outcome {judged.get('outcome')}; labels {judged.get('labels')}; "
          f"incomplete {incomplete}; invalid {invalid}", flush=True)
    return record
