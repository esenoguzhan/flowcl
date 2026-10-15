"""The recovery measurement study's runner (configs/analysis/recovery_study.yaml; the pure analysis is
:mod:`flowcl.analysis.recovery`).

* :func:`run_task` - one task's worker: nominal rollouts from training-side demo states, the determinism
  gate, then every query's continuations (nominal and perturbed, common random numbers), in resumable
  work files bound to the rule, the checkpoint and the git SHA.
* :func:`benchmark` - per-phase timing (reset + replay, the closed-loop suffix) at an early and a late
  query, for the smoke.
* :func:`validate` - the pre-run simulation report (the whole decision procedure on synthetic worlds).
* :func:`analyze` - Q1-Q4, the negative control and the outcome; publication of results/<out>/ by one
  rename, gated on every task's gate and the CPU suite.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from flowcl.analysis import recovery as rc
from flowcl.utils.libero_paths import repo_root
from flowcl.utils.run import file_sha256, git_sha
from flowcl.utils.seeding import derive_seed


def rule_path(path=None) -> Path:
    return Path(path) if path else repo_root() / "configs" / "analysis" / "recovery_study.yaml"


def load_rule(path=None) -> dict:
    return OmegaConf.to_container(OmegaConf.load(rule_path(path)), resolve=True)


def _root(results_root=None) -> Path:
    return Path(results_root) if results_root else repo_root() / "results"


def _slug(task_key: str) -> str:
    return task_key.replace("/", "__")


def _atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload) + "\n")
    os.replace(tmp, path)


def _hash(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes()).hexdigest()[:16]


def _ctrl(prefix: np.ndarray) -> np.ndarray:
    """The controller-relevant form of executed actions: position/rotation values, the gripper's sign."""
    p = np.asarray(prefix, dtype=np.float64).copy()
    p[:, rc.GRIP] = np.sign(p[:, rc.GRIP])
    return p


def analysis_rule(rule: dict, fallback: bool) -> dict:
    phases = rule["fallback"]["phases"] if fallback else rule["phases"]
    return {**rule, "n_phase": len(phases), "phases_used": phases}


def binding(rule: dict, task: dict, rule_file: Path, smoke: dict | None) -> dict:
    return {"rule_sha256": file_sha256(rule_file), "checkpoint_sha256": task["sha256"], "git_sha": git_sha(),
            "smoke": smoke}


# ---- the worker ------------------------------------------------------------------------------------


def _setup(rule: dict, task: dict, root: Path, device: str, dataset_dir=None):
    from flowcl.data.libero_adapter import iter_episodes
    from flowcl.data.tasks import TaskRef
    from flowcl.envs.libero_env import EvalConfig, LiberoTaskEnv
    from flowcl.train.checkpoint import load_checkpoint

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    ckpt = root / task["checkpoint"]
    if file_sha256(ckpt) != task["sha256"]:
        raise ValueError(f"{ckpt} is not the checkpoint the rule names")
    loaded = load_checkpoint(ckpt, device=device)
    ref = TaskRef.from_key(task["key"])
    demos = list(iter_episodes(ref.demo_path(dataset_dir), loaded.spec, n_demos=int(rule["n_demos_sigma"]),
                               task_id=ref.task_key, with_sim_states=True))
    sigma = np.concatenate([d.action for d in demos]).astype(np.float64).std(axis=0)
    if np.any(sigma <= 0):
        raise ValueError(f"{task['key']}: a zero action SD")
    ev = rule["eval"]
    cfg = EvalConfig(n_episodes=1, max_steps=int(ev["max_steps"]), execute_k=int(ev["execute_k"]),
                     euler_steps=int(ev["euler_steps"]), image_size=int(ev["image_size"]))
    env = LiberoTaskEnv(ref.suite, ref.task_idx, loaded.spec, image_size=cfg.image_size)
    return loaded, env, demos, sigma, cfg


def _gen(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(int(seed))


def _nominal(env, loaded, demos, i, ns, task_key, cfg):
    seed = derive_seed(ns, f"recovery::nominal::{task_key}", i)
    run = env.run_episode(loaded.policy, loaded.stats, demos[i].meta["init_state"], seed, _gen(seed), cfg, record=True)
    return seed, run


def _continue(env, loaded, init_state, env_seed, cont_seed, replay, delta, cfg):
    run = env.run_episode(loaded.policy, loaded.stats, init_state, env_seed, _gen(cont_seed), cfg, replay=replay,
                          perturb_first=delta)
    k = cfg.execute_k
    after = run.actions[run.n_replay:]
    return run, {"success": bool(run.success), "n_steps": int(run.n_steps),
                 "prefix_ctrl": _hash(_ctrl(after[:k])), "suffix": _hash(after[k:])}


def gate(env, loaded, demos, nominals: dict, ns: str, task_key: str, cfg, rule: dict) -> dict:
    """The determinism gate on the first ``gate_rollouts`` successful nominal rollouts."""
    checks = []
    ok_ids = [i for i in sorted(nominals) if nominals[i][1].success][: int(rule["gate_rollouts"])]
    for i in ok_ids:
        seed, run = nominals[i]
        rp = run.replans[len(run.replans) // 2]
        step = rp["step"]
        g = torch.Generator(device="cpu")
        g.set_state(rp["generator_state"])
        again = env.run_episode(loaded.policy, loaded.stats, demos[i].meta["init_state"], seed, g, cfg,
                                replay=run.actions[:step])
        suffix_ok = bool(np.array_equal(again.actions[step:], run.actions[step:]) and again.success == run.success
                         and again.n_steps == run.n_steps)
        bank = rc.perturbation_bank(np.ones(7) * 0.5, cfg.execute_k, rule["perturbations"]["magnitudes"],
                                    int(rule["perturbations"]["n_isotropic_per_magnitude"]), seed)
        d1, d2 = bank[0]["delta"], bank[-1]["delta"]
        cseed = derive_seed(ns, f"recovery::gate::{task_key}", i)
        order_a = [_continue(env, loaded, demos[i].meta["init_state"], seed, cseed, run.actions[:step], d, cfg)[1]
                   for d in (None, d1, d2)]
        order_b = [_continue(env, loaded, demos[i].meta["init_state"], seed, cseed, run.actions[:step], d, cfg)[1]
                   for d in (d2, None, d1)]
        order_ok = order_a == [order_b[1], order_b[2], order_b[0]]
        checks.append({"rollout": i, "query_step": int(step), "suffix_reproduced": suffix_ok, "order_invariant": order_ok})
    hard = getattr(getattr(env.env, "env", None), "hard_reset", None)
    return {"passed": bool(checks) and all(c["suffix_reproduced"] and c["order_invariant"] for c in checks),
            "checks": checks, "hard_reset": hard, "cudnn_deterministic": torch.backends.cudnn.deterministic}


def run_task(rule: dict, task_index: int, *, results_root=None, out_root=None, rule_file=None, device="cuda",
             dataset_dir=None, smoke: dict | None = None, fallback: bool = False) -> dict:
    """One task's worker (resumable). ``smoke`` = {"n_nominal", "phases", "n_seeds"} overrides into
    ``out_root`` (outside results/)."""
    root = _root(results_root)
    base = Path(out_root) if out_root else root
    task = rule["tasks"][int(task_index)]
    ns, key = rule["namespace"], task["key"]
    bind = binding(rule, task, rule_path(rule_file), smoke)
    n_nominal = int(smoke["n_nominal"]) if smoke else int(rule["n_nominal"])
    phases = smoke["phases"] if smoke else (rule["fallback"]["phases"] if fallback else rule["phases"])
    n_seeds = int(smoke["n_seeds"]) if smoke else int(rule["n_seeds"])
    work = base / rule["work"] / _slug(key)
    loaded, env, demos, sigma, cfg = _setup(rule, task, root, device, dataset_dir)
    started = time.perf_counter()
    try:
        nominals = {i: _nominal(env, loaded, demos, i, ns, key, cfg) for i in range(n_nominal)}
        _atomic_json(work / "nominal.json", {"binding": bind, "fallback": fallback, "sigma": sigma.tolist(),
                                             "rollouts": {str(i): {"seed": s, "success": bool(r.success),
                                                                   "n_steps": int(r.n_steps)}
                                                          for i, (s, r) in nominals.items()}})
        g = gate(env, loaded, demos, nominals, ns, key, cfg, rule)
        _atomic_json(work / "gate.json", {"binding": bind, **g})
        if not g["passed"]:
            return {"task": key, "gate": g, "done": False}
        mags, n_iso = rule["perturbations"]["magnitudes"], int(rule["perturbations"]["n_isotropic_per_magnitude"])
        for i in sorted(nominals):
            seed, run = nominals[i]
            if not run.success:
                continue
            wf = work / f"rollout_{i:02d}.json"
            if wf.is_file():
                prev = json.loads(wf.read_text())
                if prev.get("binding") == bind and prev.get("phases") == list(phases):
                    continue
            queries = rc.query_steps([r["step"] for r in run.replans], run.n_steps, phases)
            rows = []
            for q, step in queries:
                bank = rc.perturbation_bank(sigma, cfg.execute_k, mags, n_iso,
                                            derive_seed(ns, f"recovery::bank::{key}", i * 100 + q))
                for m in range(n_seeds):
                    cseed = derive_seed(ns, f"recovery::cont::{key}", (i * 100 + q) * 10 + m)
                    nrun, nom = _continue(env, loaded, demos[i].meta["init_state"], seed, cseed, run.actions[:step], None,
                                          cfg)
                    base_prefix = nrun.prefix["raw"]
                    rows.append({"rollout": i, "query": q, "seed": m, "kind": 0, "group": -1, "success": nom["success"],
                                 "n_steps": nom["n_steps"], "size": 0.0, "flips": 0, "noop": False,
                                 "clipped_fraction": 0.0, "magnitude": 0.0, "sign": 0, "step": int(step),
                                 "prefix_ctrl": nom["prefix_ctrl"], "suffix": nom["suffix"]})
                    for j, b in enumerate(bank):
                        eff = rc.effective(base_prefix, b["delta"], sigma)
                        _, out = _continue(env, loaded, demos[i].meta["init_state"], seed, cseed, run.actions[:step],
                                           b["delta"], cfg)
                        rows.append({"rollout": i, "query": q, "seed": m, "kind": 1,
                                     "group": rc.ALL_GROUPS.index(b["group"]), "success": out["success"],
                                     "n_steps": out["n_steps"], "size": eff["size"], "flips": eff["flips"],
                                     "noop": eff["noop"], "clipped_fraction": eff["clipped_fraction"],
                                     "magnitude": b["magnitude"], "sign": b["sign"], "bank_index": j, "step": int(step),
                                     "prefix_ctrl": out["prefix_ctrl"], "suffix": out["suffix"]})
            _atomic_json(wf, {"binding": bind, "phases": list(phases), "queries": queries,
                              "dropped_phases": len(phases) - len(queries), "rows": rows})
            print(f"[recovery] {key} rollout {i}: {len(rows)} continuations "
                  f"({time.perf_counter() - started:.0f} s so far)", flush=True)
    finally:
        env.close()
    return {"task": key, "gate": g, "done": True, "seconds": time.perf_counter() - started}


def benchmark(rule: dict, task_index: int, *, results_root=None, device="cuda", dataset_dir=None,
              n_cont: int = 6) -> dict:
    """Seconds for reset + replay and for a full continuation, at an early and a late query of one
    successful nominal rollout (the smoke's estimate)."""
    root = _root(results_root)
    task = rule["tasks"][int(task_index)]
    ns, key = rule["namespace"], task["key"]
    loaded, env, demos, sigma, cfg = _setup(rule, task, root, device, dataset_dir)
    try:
        for i in range(int(rule["n_nominal"])):
            t0 = time.perf_counter()
            seed, run = _nominal(env, loaded, demos, i, ns, key, cfg)
            nominal_s = time.perf_counter() - t0
            if run.success:
                break
        else:
            return {"task": key, "error": "no successful nominal rollout"}
        out = {"task": key, "nominal_seconds": nominal_s, "nominal_steps": int(run.n_steps), "queries": {}}
        for name, ph in (("early", 0.0833333333), ("late", 0.9166666667)):
            (_, step), = rc.query_steps([r["step"] for r in run.replans], run.n_steps, [ph])
            replay_cfg = type(cfg)(**{**cfg.__dict__, "max_steps": step})
            t0 = time.perf_counter()
            for _ in range(int(n_cont)):
                env.run_episode(loaded.policy, loaded.stats, demos[i].meta["init_state"], seed, _gen(1), replay_cfg,
                                replay=run.actions[:step])
            replay_s = (time.perf_counter() - t0) / n_cont
            t0 = time.perf_counter()
            steps = []
            for c in range(int(n_cont)):
                r = env.run_episode(loaded.policy, loaded.stats, demos[i].meta["init_state"], seed, _gen(100 + c), cfg,
                                    replay=run.actions[:step])
                steps.append(r.n_steps)
            full_s = (time.perf_counter() - t0) / n_cont
            out["queries"][name] = {"step": int(step), "reset_replay_s": replay_s, "continuation_s": full_s,
                                    "suffix_s": full_s - replay_s, "mean_total_steps": float(np.mean(steps))}
    finally:
        env.close()
    return out


# ---- the pre-run validation --------------------------------------------------------------------------


def validate(rule: dict, *, out: Path, workers: int = 20, n_worlds: int | None = None, kinds=None,
             regimes=("main", "fallback"), rule_file=None) -> dict:
    """The whole decision procedure on synthetic worlds: false GO under the nulls, power under the
    alternatives, both regimes (and both levels)."""
    v = rule["validation"]
    report = {"n_boot": int(v["n_boot"]), "production_n_boot": int(rule["bootstrap"]["n"]), "results": {},
              "rule_sha256": file_sha256(rule_path(rule_file)), "git_sha": git_sha()}
    started = time.perf_counter()
    all_kinds = ("null_uniform", "null_magnitude", "phase_only", "component_only", "mixed")
    for r_i, regime in enumerate(regimes):
        ar = analysis_rule(rule, regime == "fallback")
        for k_i, kind in enumerate(kinds or all_kinds):
            null = kind.startswith("null")
            spec = {"kind": kind, "n_queries": ar["n_phase"], "n_seeds": int(rule["n_seeds"]),
                    "n_rollouts": int(rule["n_nominal"]), "p_success": float(v["p_success"]),
                    "base_logit": float(v["base_logit"]), "effect": float(v["effect"]),
                    "critical_phases": (ar["n_phase"] // 2 - 1, ar["n_phase"] // 2)}
            n = int(n_worlds or (v["n_null_worlds"] if null else v["n_power_worlds"]))
            res = rc.simulate(spec, ar, rule["usable"][regime], n, len(rule["tasks"]), int(v["n_boot"]),
                              int(v["seed"]) + 1000 * r_i + 10 * all_kinds.index(kind), workers=workers)
            report["results"][f"{regime}/{kind}"] = res
            print(f"[recovery] validation {regime}/{kind}: {json.dumps(res['rates'])}", flush=True)
    report["seconds"] = time.perf_counter() - started
    worst = max(r["rates"]["0.95"]["go_rate_cp95"][1] for k, r in report["results"].items() if "/null" in k)
    worst99 = max(r["rates"]["0.99"]["go_rate_cp95"][1] for k, r in report["results"].items() if "/null" in k)
    level = 0.95 if worst <= float(v["max_false_go"]) else (0.99 if worst99 <= float(v["max_false_go"]) else None)
    report["decision"] = {"worst_null_go_cp_upper_95": worst, "worst_null_go_cp_upper_99": worst99,
                          "level": level, "production_may_start": level is not None}
    _atomic_json(Path(out) / "validation.json", report)
    return report


# ---- the analysis --------------------------------------------------------------------------------


def _load_task(work: Path, bind_rule_sha: str) -> dict:
    nominal = json.loads((work / "nominal.json").read_text())
    gate_rec = json.loads((work / "gate.json").read_text())
    rows, queries, dropped = [], 0, 0
    for f in sorted(work.glob("rollout_*.json")):
        payload = json.loads(f.read_text())
        if payload["binding"]["rule_sha256"] != bind_rule_sha:
            raise RuntimeError(f"{f}: bound to another rule")
        rows.extend(payload["rows"])
        queries += len(payload["queries"])
        dropped += payload["dropped_phases"]
    return {"nominal": nominal, "gate": gate_rec, "rows": rows, "n_queries": queries, "dropped_phases": dropped}


def negative_control(rows: list[dict]) -> dict:
    nominal = {(r["rollout"], r["query"], r["seed"]): r for r in rows if r["kind"] == 0}
    noops = [r for r in rows if r["kind"] == 1 and r["noop"]]
    mism = [r for r in noops if (r["success"], r["n_steps"], r["prefix_ctrl"], r["suffix"]) !=
            tuple(nominal[(r["rollout"], r["query"], r["seed"])][k] for k in ("success", "n_steps", "prefix_ctrl", "suffix"))]
    at_risk = [r for r in noops if nominal[(r["rollout"], r["query"], r["seed"])]["success"]]
    inj = sum(1 for r in at_risk if not r["success"])
    return {"noop_continuations": len(noops), "mismatches": len(mism),
            "mismatch_fraction": len(mism) / len(noops) if noops else 0.0,
            "at_risk": len(at_risk), "injuries": inj,
            "injury_ci": rc.clopper_pearson(inj, len(at_risk)) if at_risk else None}


def descriptives(t: rc.Table, rows: list[dict]) -> dict:
    p = rc.perturbed(t)
    keep = ~p["noop"]
    out = {"temporal_s20": rc.s20_cells(p, 0.2, by="query"), "group_injury": rc.group_shares(p),
           "rescues": int((p["rescue"] & keep).sum()), "injuries": int((p["injury"] & keep).sum()),
           "signed_excess_failure": float((p["fail"][keep].astype(float) - (~p["nominal_success"][keep]).astype(float)).mean())
           if keep.any() else None}
    sizes = p["size"][keep]
    if sizes.size:
        edges = np.quantile(sizes, [0.25, 0.5, 0.75])
        bins = np.digitize(p["size"], edges)
        q3 = {}
        for gi, g in enumerate(rc.ALL_GROUPS):
            for b in range(4):
                m = keep & p["nominal_success"] & (p["group"] == gi) & (bins == b)
                if m.any():
                    q3[f"{g}/bin{b}"] = {"trials": int(m.sum()), "injury_rate": float(p["injury"][m].mean())}
        out["q3_group_by_size_bin"] = q3
    mags = np.asarray([r.get("magnitude", 0.0) for r in rows if r["kind"] == 1])
    out["q4_injury_by_magnitude"] = {str(m): float((p["injury"] & keep & (mags == m) & p["nominal_success"]).sum()
                                                   / max(1, (keep & (mags == m) & p["nominal_success"]).sum()))
                                     for m in sorted(set(mags.tolist()))}
    return out


def analyze(rule: dict, *, results_root=None, rule_file=None, suite_result=None, out_root=None) -> dict:
    from flowcl.experiments.behavior_predictors import suite_green

    root = _root(results_root)
    base = Path(out_root) if out_root else root
    rsha = file_sha256(rule_path(rule_file))
    level = float(rule["decision"]["level"])
    tasks, gate_ok, nc_all = {}, True, []
    for task in rule["tasks"]:
        work = base / rule["work"] / _slug(task["key"])
        if not (work / "gate.json").is_file():
            if out_root is None:
                raise RuntimeError(f"{task['key']}: no work files; every task must run before publication")
            tasks[task["key"]] = {"usable": False, "verdict": {"pass": False, "clear_fail": False}, "missing": True}
            continue
        data = _load_task(work, rsha)
        fallback = bool(data["nominal"]["fallback"])
        ar = analysis_rule(rule, fallback)
        gate_ok = gate_ok and bool(data["gate"]["passed"])
        if not data["rows"]:
            tasks[task["key"]] = {"usable": False, "verdict": {"pass": False, "clear_fail": False}, "gate": data["gate"]}
            continue
        t = rc.Table.from_rows(data["rows"])
        n_ok = sum(1 for r in data["nominal"]["rollouts"].values() if r["success"])
        use = rc.usable(t, n_ok, rule["usable"]["fallback" if fallback else "main"])
        point = rc.task_point(t, ar)
        boot = rc.bootstrap_task(t, ar, int(rule["bootstrap"]["n"]), int(rule["bootstrap"]["seed"]))
        verdict = rc.task_verdict(point, boot, ar, level)
        nc = negative_control(data["rows"])
        nc_all.append(nc)
        tasks[task["key"]] = {"usable": use["usable"], "usability": use, "verdict": verdict,
                              "verdict_99": rc.task_verdict(point, boot, ar, 0.99), "negative_control": nc,
                              "descriptive": descriptives(t, data["rows"]), "gate": data["gate"],
                              "nominal_successes": n_ok, "dropped_phases": data["dropped_phases"],
                              "fallback": fallback}
    noops = sum(n["noop_continuations"] for n in nc_all)
    mism = sum(n["mismatches"] for n in nc_all)
    at_risk = sum(n["at_risk"] for n in nc_all)
    inj = sum(n["injuries"] for n in nc_all)
    nc_ci = rc.clopper_pearson(inj, at_risk) if at_risk else None
    nc_ok = (mism / noops if noops else 0.0) <= float(rule["negative_control"]["max_mismatch_fraction"]) and \
        (nc_ci is None or nc_ci[0] <= float(rule["negative_control"]["max_injury_lower"]))
    outcome = rc.decide(tasks, gate_ok, nc_ok)
    report = {"outcome": outcome, "level": level, "gate_ok": gate_ok,
              "negative_control": {"noops": noops, "mismatches": mism, "at_risk": at_risk, "injuries": inj,
                                   "injury_ci": nc_ci, "ok": nc_ok},
              "tasks": tasks, "rule_sha256": rsha}
    provenance = {"git_sha": git_sha()}
    if out_root is not None:
        _atomic_json(Path(out_root) / "analysis_dry.json", {**report, "provenance": provenance})
        return {**report, "dry_run": True}
    if provenance["git_sha"].endswith("-dirty"):
        raise RuntimeError("publication needs a clean tree")
    ok, why = suite_green(suite_result)
    if not ok:
        raise RuntimeError(f"publication is gated on the CPU suite: {why}")
    out_dir = root / rule["out"] / "analysis"
    if out_dir.exists():
        return json.loads((out_dir / "analysis.json").read_text())
    attempt = out_dir.parent / f".analysis_attempt_{os.getpid()}"
    attempt.mkdir(parents=True)
    (attempt / "analysis.json").write_text(json.dumps({**report, "provenance": provenance}, indent=2) + "\n")
    (attempt / "done.json").write_text(json.dumps({"analysis_sha256": file_sha256(attempt / "analysis.json")}) + "\n")
    os.rename(attempt, out_dir)
    return report
