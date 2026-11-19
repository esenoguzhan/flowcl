"""Instruction negatives on the other two B tasks (``configs/analysis/instruction_negatives_curricula.yaml``).

Four stage-1 runs resumed from the stove pilot's stage-A checkpoints (``scripts/run_continual.py
--resume-run ... --start-stage 1``, queued by ``scripts/queue_instruction_negatives_curricula.sh``).
This module adds what the runner does not record, and judges the rule:

1. :func:`run_preflight`: the pins (stove summary, gate, sources, historical files), the two
   sources' bitwise-equal weights, and the machine (clean tree, idle GPU, disk, absent targets);
2. :func:`run_swaps` / :func:`run_probe`: the stove module's swaps and probes on a per-curriculum
   config (:func:`sub_config`), the probes with the source's stage-0 checkpoint and memory;
3. :func:`resume_equality`: the smoke-only check that a resumed run reproduces its source;
4. :func:`run_summary`: the validity checks (mode-specific), the labels and ``summary.json``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from flowcl.experiments import instruction_negatives as inn
from flowcl.utils.libero_paths import repo_root
from flowcl.utils.run import atomic_write_text, file_sha256, git_sha

VALID, INVALID, INCOMPLETE = inn.VALID, inn.INVALID, inn.INCOMPLETE
EXIT_CODES = inn.EXIT_CODES
ARMS = inn.ARMS
CURRICULA = ("cabinet", "milk")
MIN_FREE_GB = 20


def config_path() -> Path:
    return repo_root() / "configs" / "analysis" / "instruction_negatives_curricula.yaml"


def load_config(path: str | Path | None = None) -> dict:
    return OmegaConf.to_container(OmegaConf.load(path or config_path()), resolve=True)


def base_path(cfg: dict) -> Path:
    return repo_root() / "configs" / "analysis" / f"{cfg['base_rule']}.yaml"


def base_config(cfg: dict) -> dict:
    base = inn.load_config(base_path(cfg))
    if base["namespace"] != cfg["namespace"] or base["task_a"] != cfg["task_a"]:
        raise ValueError("the rule's namespace and task_a must equal the stove rule's")
    return base


def sub_config(cfg: dict, short: str, base: dict | None = None) -> dict:
    """One curriculum in the stove module's config shape, so its swaps, probes and judge apply."""
    base = base or base_config(cfg)
    c = cfg["curricula"][short]
    sub = {k: base[k] for k in ("seed", "namespace", "task_a", "bank", "production", "smoke", "panel",
                                "acquisition", "retention", "probe", "swap", "chunks", "coverage", "drift")}
    sub.update({"out": f"{cfg['out']}/{short}", "curriculum": c["curriculum"], "task_b": c["task_b"],
                "held_out_tasks": [c["task_b"]], "kind": c["kind"],
                "runs": {arm: {"method": cfg["methods"][arm], "name": c["runs"][arm]} for arm in ARMS},
                "historical": {arm: c["historical"][arm]["run"] for arm in ARMS}})
    return sub


def _root(path) -> Path:
    return Path(path) if path else repo_root() / "results"


def source_dir(cfg: dict, sources_root: Path, arm: str) -> Path:
    return sources_root / cfg["sources"][arm]["run"]


# ---- pins ---------------------------------------------------------------------------------------------


def pins(cfg: dict, sources_root: Path, smoke: bool) -> dict:
    """The production pins from the rule; in smoke, computed from the smoke sources (recorded)."""
    stove = cfg["stove"]
    if not smoke:
        return {"gpm": {"stage0": cfg["sources"]["gpm"]["stage0_sha256"],
                        "memory": cfg["sources"]["gpm"]["memory_sha256"]},
                "allow": {"stage0": cfg["sources"]["allow"]["stage0_sha256"]},
                "stove_summary": stove["summary_sha256"], "gate": stove["gate_sha256"], "from": "rule"}

    def sha(p: Path):
        return file_sha256(p) if p.is_file() else None

    g, a = source_dir(cfg, sources_root, "gpm"), source_dir(cfg, sources_root, "allow")
    return {"gpm": {"stage0": sha(g / "checkpoints" / "stage0.pt"), "memory": sha(g / "method" / "memory_task0.pt")},
            "allow": {"stage0": sha(a / "checkpoints" / "stage0.pt")},
            "stove_summary": sha(sources_root / stove["summary"]), "gate": sha(sources_root / stove["gate"]),
            "from": f"smoke sources {sources_root}"}


def state_equal(a: Path, b: Path) -> dict:
    sa = torch.load(a, map_location="cpu", weights_only=False)["state_dict"]
    sb = torch.load(b, map_location="cpu", weights_only=False)["state_dict"]
    different = sorted(k for k in set(sa) & set(sb) if not torch.equal(sa[k], sb[k]))
    missing, extra = sorted(set(sa) - set(sb)), sorted(set(sb) - set(sa))
    return {"passed": not (different or missing or extra), "n_tensors": len(sa), "n_different": len(different),
            "different": different, "missing": missing, "extra": extra}


def pin_checks(cfg: dict, sources_root: Path, historical_root: Path, smoke: bool) -> dict:
    """Every pinned input: the stove rule, summary and gate, the sources, the historical files."""
    p = pins(cfg, sources_root, smoke)
    stove = cfg["stove"]
    summary_path = sources_root / stove["summary"]
    summary = inn._read(summary_path)
    checks = {"stove_rule": file_sha256(base_path(cfg)) == stove["rule_sha256"],
              "stove_summary": {"passed": bool(summary_path.is_file() and file_sha256(summary_path) == p["stove_summary"]
                                               and summary and summary.get("status") == VALID
                                               and (smoke or (summary.get("judgment") or {}).get("gate", {}).get("passed"))),
                                "status": summary and summary.get("status"),
                                "gate_passed": summary and (summary.get("judgment") or {}).get("gate", {}).get("passed"),
                                "gate_required": not smoke},
              "gate_file": (sources_root / stove["gate"]).is_file() and file_sha256(sources_root / stove["gate"]) == p["gate"]}
    for arm in ARMS:
        src = source_dir(cfg, sources_root, arm)
        ckpt = src / "checkpoints" / "stage0.pt"
        checks[f"source_{arm}"] = ckpt.is_file() and p[arm]["stage0"] is not None and file_sha256(ckpt) == p[arm]["stage0"]
    mem = source_dir(cfg, sources_root, "gpm") / "method" / "memory_task0.pt"
    checks["source_memory"] = mem.is_file() and p["gpm"]["memory"] is not None and file_sha256(mem) == p["gpm"]["memory"]
    if checks["source_gpm"] and checks["source_allow"]:
        checks["sources_equal"] = state_equal(source_dir(cfg, sources_root, "gpm") / "checkpoints" / "stage0.pt",
                                              source_dir(cfg, sources_root, "allow") / "checkpoints" / "stage0.pt")
    else:
        checks["sources_equal"] = {"passed": False, "reason": "a source checkpoint is missing or off its pin"}
    hist = {}
    for short in CURRICULA:
        for arm in ARMS:
            h = cfg["curricula"][short]["historical"][arm]
            for rel, want in h["sha256"].items():
                f = historical_root / h["run"] / rel
                hist[f"{short}/{arm}/{rel}"] = f.is_file() and file_sha256(f) == want
    checks["historical_files"] = {"passed": all(hist.values()), "files": hist}
    ok = all((v.get("passed") if isinstance(v, dict) else bool(v)) for v in checks.values())
    return {"passed": ok, "pins": p, **checks}


# ---- step: preflight ---------------------------------------------------------------------------------


def _gpu_idle() -> bool | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.returncode == 0 and not out.stdout.strip()


def run_preflight(cfg: dict, *, results_root=None, sources_root=None, historical_root=None,
                  smoke: bool = False) -> dict:
    root, src, hist = _root(results_root), _root(sources_root), _root(historical_root)
    base = base_config(cfg)
    rec = {"git_sha": git_sha(), "smoke": smoke, "pins_checked": pin_checks(cfg, src, hist, smoke)}
    status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=repo_root())
    machine = {"clean_tree": smoke or (status.returncode == 0 and not status.stdout.strip()),
               "gpu_idle": bool(_gpu_idle()),
               "free_gb": shutil.disk_usage(root if root.exists() else repo_root()).free / 1e9}
    machine["disk"] = machine["free_gb"] >= MIN_FREE_GB
    targets = {f"{short}/{arm}": not (root / sub_config(cfg, short, base)["runs"][arm]["name"]).exists()
               for short in CURRICULA for arm in ARMS}
    machine["targets_absent"] = all(targets.values())
    rec.update({"machine": machine, "targets": targets})
    rec["passed"] = bool(rec["pins_checked"]["passed"] and machine["clean_tree"] and machine["gpu_idle"]
                         and machine["disk"] and machine["targets_absent"])
    out = root / cfg["out"]
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out / "preflight.json", json.dumps(rec, indent=2, default=str) + "\n")
    return rec


# ---- steps: swaps, probes, resume equality -------------------------------------------------------------


def run_swaps(cfg: dict, short: str, arm: str, **kw) -> dict:
    if not cfg["curricula"][short]["swaps"]:
        raise ValueError(f"{short} has no swaps in the rule")
    return inn.run_swaps(sub_config(cfg, short), arm, **kw)


def probe_target(cfg: dict, sub: dict, root: Path, sources_root: Path, arm: str, memory_sha: str) -> dict:
    run = inn.run_dir(sub, root, arm)
    return {"run": run, "checkpoint": run / "checkpoints" / "stage1.pt", "stage": 1,
            "stage0": source_dir(cfg, sources_root, arm) / "checkpoints" / "stage0.pt",
            "memory": {"path": str(source_dir(cfg, sources_root, "gpm") / "method" / "memory_task0.pt"),
                       "expected_sha256": memory_sha}}


def run_probes(cfg: dict, *, results_root=None, sources_root=None, smoke: bool = False, **kw) -> list[str]:
    root, src = _root(results_root), _root(sources_root)
    memory_sha = pins(cfg, src, smoke)["gpm"]["memory"]
    base = base_config(cfg)
    problems = []
    for short in CURRICULA:
        sub = sub_config(cfg, short, base)
        for arm in ARMS:
            t = probe_target(cfg, sub, root, src, arm, memory_sha)
            if not t["checkpoint"].is_file():
                problems.append(f"{short} {arm}: {t['checkpoint']} missing")
                print(f"[flowcl] probe {short} {arm}: {t['checkpoint']} missing, skipped", flush=True)
                continue
            try:
                inn.run_probe(sub, f"{arm}_stage1", target=t, results_root=root, **kw)
            except Exception as exc:  # one probe's failure must not block the others
                problems.append(f"{short} {arm}: {type(exc).__name__}: {exc}")
                print(f"[flowcl] probe {short} {arm} FAILED: {exc}", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return problems


def base_probe_target(cfg: dict, sources_root: Path, memory_sha: str) -> dict:
    """The negatives base itself (the GPM source's stage0.pt), probed against a curriculum's B."""
    run = source_dir(cfg, sources_root, "gpm")
    return {"run": run, "checkpoint": run / "checkpoints" / "stage0.pt", "stage": 0,
            "stage0": run / "checkpoints" / "stage0.pt",
            "memory": {"path": str(run / "method" / "memory_task0.pt"), "expected_sha256": memory_sha}}


def run_base_probe(cfg: dict, short: str, *, sources_root=None, smoke: bool = False, **kw) -> dict:
    """Descriptive, added after the queue (not in the rule): the stove pilot probed the base under the
    stove instruction only. This probes it against ``short``'s B (instruction contrast, its share outside
    the memory, probe losses, memory coverage) into ``<out>/<short>/probes/neg_stage0``. The base must be
    at its pin."""
    src = _root(sources_root)
    p = pins(cfg, src, smoke)["gpm"]
    t = base_probe_target(cfg, src, p["memory"])
    got = file_sha256(t["checkpoint"]) if t["checkpoint"].is_file() else None
    if got != p["stage0"]:
        raise ValueError(f"{t['checkpoint']}: sha256 {got} != pinned {p['stage0']}")
    return inn.run_probe(sub_config(cfg, short), "neg_stage0", target=t, **kw)


def resume_equality(source_run: Path, resumed_run: Path, out: Path) -> dict:
    """Smoke only: the resumed stage 1 against the source's (checkpoint, memory, rollouts)."""
    from flowcl.experiments.resume_check import compare_resume

    rec = compare_resume(Path(source_run), Path(resumed_run), 1)
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out / "resume_equality.json", json.dumps(rec, indent=2, default=str) + "\n")
    return rec


# ---- the judgment (pure) ---------------------------------------------------------------------------------


def _cell(cells: dict, run: str, stage: int, task: str):
    return ((cells.get(run) or {}).get(stage) or {}).get(task)


def _diff(cells, x, y, stage, task, boot):
    a, b = _cell(cells, x, stage, task), _cell(cells, y, stage, task)
    return inn._paired(a, b, boot) if a is not None and b is not None else None


def vs_historical(cfg_sub: dict, cells: dict, boot: dict) -> dict:
    """Each new arm against its historical control: plate before B, plate after B, and B (paired)."""
    a, b = cfg_sub["task_a"], cfg_sub["task_b"]
    out = {}
    for arm in ARMS:
        h = f"hist_{arm}"
        out[arm] = {"plate_before_B": _diff(cells, arm, h, 0, a, boot), "plate_after_B": _diff(cells, arm, h, 1, a, boot),
                    "B": _diff(cells, arm, h, 1, b, boot), "flag": "against_historical_control"}
    return out


def endpoint_label(d: dict | None, margin: float) -> str:
    pp = f"{round(100 * margin)}pp"
    if d is None:
        return "missing"
    if d["low"] > -margin:
        return f"no_cost_beyond_{pp}"
    if d["high"] < -margin:
        return f"cost_beyond_{pp}"
    if d["diff"] < -margin:
        return f"observed_drop_over_{pp}"
    return f"within_{pp}_on_observed_rollouts"


def judge_diverse(cfg_sub: dict, cells: dict, rule: dict, boot: dict) -> dict:
    """Milk's no-harm reading: two endpoints against historical GPM, and the retention labels."""
    a, b = cfg_sub["task_a"], cfg_sub["task_b"]
    margin = float(rule["margin"])
    forget_max = float(cfg_sub["retention"]["forgetting_max"])
    r1b = {arm: inn._rate(_cell(cells, arm, 1, b)) for arm in ARMS}
    if None in r1b.values():
        state = "missing"
    elif min(r1b.values()) < float(rule["inconclusive_below"]):
        state = "inconclusive_acquisition"
    else:
        state = "assessed"
    endpoints = {"plate_after_B": _diff(cells, "gpm", "hist_gpm", 1, a, boot),
                 "B": _diff(cells, "gpm", "hist_gpm", 1, b, boot)}
    forget = {arm: (inn._rate(_cell(cells, arm, 0, a)) - inn._rate(_cell(cells, arm, 1, a))
                    if _cell(cells, arm, 0, a) is not None and _cell(cells, arm, 1, a) is not None else None)
              for arm in ARMS}
    proj = _diff(cells, "gpm", "allow", 1, a, boot)
    if state != "assessed":
        labels = {k: "not_assessed" for k in ("plate_after_B", "B", "milk_no_harm", "retains",
                                              "allowlist_without_projection_retains", "projection_contributes")}
    else:
        ep = {k: endpoint_label(d, margin) for k, d in endpoints.items()}
        no_cost = f"no_cost_beyond_{round(100 * margin)}pp"
        labels = {**ep,
                  "milk_no_harm": "milk_no_harm" if all(v == no_cost for v in ep.values()) else "not_shown",
                  "retains": "retains" if forget["gpm"] is not None and forget["gpm"] <= forget_max else "not_shown",
                  "allowlist_without_projection_retains": (
                      "allowlist_without_projection_retains"
                      if forget["allow"] is not None and forget["allow"] <= forget_max else "not_shown"),
                  "projection_contributes": "projection_contributes" if proj and proj["low"] > 0 else "not_shown"}
    return {"outcome": state, "acquisition": {"R1_B": r1b, "state": state},
            "endpoints": {k: d and {**d, "flag": "against_historical_control"} for k, d in endpoints.items()},
            "forgetting_A": forget, "gpm_minus_allow_A": proj, "labels": labels,
            **table_and_interaction(cells, a, boot)}


def table_and_interaction(cells: dict, a: str, boot: dict) -> dict:
    table = {f"{neg}/{arm}": inn._rate(_cell(cells, run, 1, a)) for neg, arm, run in
             (("off", "allow", "hist_allow"), ("off", "gpm", "hist_gpm"), ("on", "allow", "allow"), ("on", "gpm", "gpm"))}
    interaction = None
    if all(_cell(cells, r, 1, a) is not None for r in ("gpm", "allow", "hist_gpm", "hist_allow")):
        on = np.asarray(_cell(cells, "gpm", 1, a), float) - np.asarray(_cell(cells, "allow", 1, a), float)
        off = np.asarray(_cell(cells, "hist_gpm", 1, a), float) - np.asarray(_cell(cells, "hist_allow", 1, a), float)
        interaction = {**inn._paired(on, off, boot), "flag": "against_historical_control"}
    return {"table_R1_A": table, "interaction": interaction}


# ---- validity --------------------------------------------------------------------------------------------


def eval_cells(run: Path, keys_by_stage: dict[int, list[str]], n: int) -> tuple[dict, dict, list[str]]:
    """``{stage: {task: successes[:n]}}``, the seeds, and the problems, for each stage's own tasks: a
    resumed run's stage-0 evaluation is its source's (A and the stove), so stage 0 asks for A only."""
    from flowcl.envs.evaluation import EvaluationReport

    succ, seeds, problems = {}, {}, []
    for stage, keys in keys_by_stage.items():
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


def negatives_inactive(run: Path, bank_sha: str) -> dict:
    ckpt = run / "checkpoints" / "stage1.pt"
    if not ckpt.is_file():
        return {"passed": False, "problems": {"stage1": "missing"}}
    state = (torch.load(ckpt, map_location="cpu", weights_only=False).get("extra") or {}).get("method_state") or {}
    neg = state.get("instruction_negatives") or {}
    log = (neg.get("logs") or {}).get("1") or {}
    problems = {}
    if log.get("active") is not False:
        problems["stage1_active"] = log.get("active")
    if neg.get("negative_bank_sha256") != bank_sha:
        problems["bank_sha256"] = neg.get("negative_bank_sha256")
    return {"passed": not problems, "problems": problems, "logged_stages": sorted((neg.get("logs") or {}).keys())}


def b_task_check(run: Path, sub: dict, n: int) -> dict:
    from flowcl.envs.evaluation import EvaluationReport

    a, b = sub["task_a"], sub["task_b"]
    conf = inn._conf(run)
    res = inn._read(run / "result.json") or {}
    tasks = [t.get("task_key") for t in conf.get("curriculum", {}).get("tasks", [])]
    ev = run / "eval" / "stage1.json"
    by = EvaluationReport.load(ev).by_task() if ev.is_file() else {}
    checks = {"config": tasks == [a, b], "result_json": res.get("task_keys") == [a, b],
              "eval_stage1": b in by and len(by[b].successes) >= n and a in by}
    return {"passed": all(checks.values()), **checks}


def run_checks_resumed(cfg_sub: dict, run: Path, arm: str, source: Path, pin: dict, smoke: bool,
                       bank_sha: str, n: int, negatives: bool = True) -> dict:
    """A resumed stage-1 run. ``negatives=False`` (a plain-GPM run) skips the negatives check; every arm
    but ``allow`` is a GPM arm (memory checks), ``allow`` is checked as allowlist_ft (frozen set)."""
    from flowcl.experiments.similar_cl import frozen_check, memory_check, resume_check
    from flowcl.train.checkpoint import load_checkpoint

    conf = inn._conf(run)
    rec = inn._read(run / "resume.json") or {}
    git = run / "git_sha"
    checks = {"namespace": conf.get("seed_namespace_run_id") == cfg_sub["namespace"],
              "method": conf.get("method", {}).get("name") == cfg_sub["runs"][arm]["method"],
              "clean_git_sha": smoke or (git.is_file() and not git.read_text().strip().endswith("-dirty")),
              "result_json": (run / "result.json").is_file(),
              "resume": resume_check(run, source),
              "resume_pin": rec.get("sha256") == pin["stage0"] and Path(str(rec.get("run"))).name == source.name,
              "b_task": b_task_check(run, cfg_sub, n)}
    if negatives:
        checks["negatives"] = negatives_inactive(run, bank_sha)
    if not smoke:
        train = conf.get("train", {})
        checks["recipe"] = all(train.get(k) == v for k, v in cfg_sub["production"]["train"].items())
    if arm != "allow":
        checks["memory"] = memory_check(run)
        checks["restored_memory"] = (rec.get("method_state") or {}).get("sha256") == pin["memory"]
    else:
        allowed = set(load_checkpoint(source / "checkpoints" / "stage0.pt", device="cpu").policy.projectable_parameters())
        checks["frozen"] = frozen_check(run, source, allowed, "allowlist_ft")
    ok = all((v.get("passed") if isinstance(v, dict) else bool(v)) for v in checks.values())
    return {"passed": ok, **checks}


def projection_basis(runs: dict, sources: dict) -> dict:
    """What a GPM - allowlist difference can be attributed to: equal start, the same trainable set,
    the same optimizer record, one namespace. Projection is what remains."""
    from flowcl.experiments.similar_cl import frozen_check
    from flowcl.train.checkpoint import load_checkpoint

    start = state_equal(sources["gpm"] / "checkpoints" / "stage0.pt", sources["allow"] / "checkpoints" / "stage0.pt")
    allowed = sorted(load_checkpoint(sources["gpm"] / "checkpoints" / "stage0.pt", device="cpu")
                     .policy.projectable_parameters())
    frozen = {arm: frozen_check(runs[arm], sources[arm], set(allowed), "allowlist_ft" if arm == "allow" else "gpm")
              for arm in ARMS}
    res = {arm: inn._read(runs[arm] / "result.json") or {} for arm in ARMS}
    st1 = {arm: (res[arm].get("stages") or [{}, {}])[1] if len(res[arm].get("stages") or []) > 1 else {} for arm in ARMS}
    tc = {arm: st1[arm].get("train_config") or {} for arm in ARMS}
    train_diff = {k: (tc["gpm"].get(k), tc["allow"].get(k)) for k in set(tc["gpm"]) | set(tc["allow"])
                  if tc["gpm"].get(k) != tc["allow"].get(k)}
    params = {arm: st1[arm].get("trainable_params") for arm in ARMS}
    ns = {arm: res[arm].get("seed_namespace_run_id") for arm in ARMS}
    report = (frozen["allow"].get("trainable_set") or {}).get("report") or {}
    checks = {"start_equal": start["passed"],
              "outside_allowlist_unchanged": all(not frozen[arm]["changed"] for arm in ARMS),
              "allowlist_freeze_report": bool(report) and report.get("trainable_tensors") == len(allowed)
              and not set(report.get("newly_frozen") or []) & set(allowed),
              "trainable_params_equal": params["gpm"] is not None and params["gpm"] == params["allow"],
              "train_config_equal": bool(tc["gpm"]) and not train_diff,
              "namespace_equal": ns["gpm"] is not None and ns["gpm"] == ns["allow"]}
    return {"passed": all(checks.values()), **checks, "start": {k: start[k] for k in ("n_tensors", "n_different")},
            "allowlist": {"n_tensors": len(allowed), "names": allowed}, "trainable_params": params,
            "train_config": tc["gpm"], "train_config_diff": train_diff, "namespaces": ns,
            "remaining_difference": "projection (GPM's memory; projected gradient and update)"}


# ---- the summary -----------------------------------------------------------------------------------------


def _curriculum_block(cfg, short, base, root, src, hist, smoke, n, p, bank_sha, gate_rec, boot) -> dict:
    sub = sub_config(cfg, short, base)
    a, b = sub["task_a"], sub["task_b"]
    runs = {"gpm": inn.run_dir(sub, root, "gpm"), "allow": inn.run_dir(sub, root, "allow"),
            "hist_gpm": hist / sub["historical"]["gpm"], "hist_allow": hist / sub["historical"]["allow"]}
    sources = {arm: source_dir(cfg, src, arm) for arm in ARMS}
    cells, seeds, incomplete, invalid, validity = {}, {}, [], [], {}
    for name, path in runs.items():
        if not (path / "result.json").is_file():
            incomplete.append(f"{short} {name}: {path.name} has no result.json")
            continue
        succ, sd, problems = eval_cells(path, {0: [a], 1: [a, b]}, n)
        cells[name], seeds[name] = succ, sd
        incomplete.extend(f"{short} {x}" for x in problems)
        if name in ARMS:
            validity[name] = run_checks_resumed(sub, path, name, sources[name], p[name], smoke, bank_sha, n)
            if not validity[name]["passed"]:
                invalid.append(f"{short} {name}: run checks failed")
    # pairing row by row: the same task, episode index (position) and seed in every run, and A's rows
    # equal to the stove pilot's (the source's stage-0 evaluation)
    _, s_seeds, _ = eval_cells(sources["gpm"], {0: [a], 1: [a]}, n)
    reference = seeds.get("hist_gpm") or seeds.get("gpm")
    rows_ok = bool(reference) and all(sd == reference for sd in seeds.values())
    def same(x, y) -> bool:
        return x is not None and x == y

    a_ok = bool(s_seeds) and all(same((sd.get(st) or {}).get(a), (s_seeds.get(st) or {}).get(a))
                                 for sd in seeds.values() for st in (0, 1))
    validity["row_pairing"] = {"passed": rows_ok and a_ok, "across_runs": rows_ok, "A_against_stove": a_ok,
                               "runs": sorted(seeds)}
    if seeds and not validity["row_pairing"]["passed"]:
        invalid.append(f"{short}: paired rows differ")
    if all(name in cells for name in ARMS):
        validity["projection_basis"] = projection_basis({arm: runs[arm] for arm in ARMS}, sources)
        if not validity["projection_basis"]["passed"]:
            invalid.append(f"{short}: projection basis")
    swaps = {}
    if cfg["curricula"][short]["swaps"]:
        for arm in ARMS:
            rec = inn._read(root / sub["out"] / "swaps" / arm / "swaps.json")
            if rec is None:
                if arm in cells:
                    incomplete.append(f"{short} swaps {arm} missing")
                continue
            ckpt = runs[arm] / "checkpoints" / "stage1.pt"
            ok = (ckpt.is_file() and rec["checkpoint_sha256"] == file_sha256(ckpt) and rec["weights_unchanged"]
                  and all(s["eval_seeds_match"] for s in rec["swaps"].values()))
            validity[f"swaps_{arm}"] = ok
            if not ok:
                invalid.append(f"{short} swaps {arm} artifact")
            swaps[arm] = {k: {key: v[key] for key in ("own_goal_rate", "requested_rate", "median_max_displacement_m")}
                          for k, v in rec["swaps"].items()}
    probes = {}
    for arm in ARMS:
        rec = inn._read(root / sub["out"] / "probes" / f"{arm}_stage1" / "probe.json")
        ckpt = runs[arm] / "checkpoints" / "stage1.pt"
        if rec is None:
            if arm in cells:
                incomplete.append(f"{short} probe {arm} missing")
            continue
        ok = ckpt.is_file() and rec["checkpoint_sha256"] == file_sha256(ckpt) and rec["weights_unchanged"]
        validity[f"probe_{arm}"] = ok
        if not ok:
            invalid.append(f"{short} probe {arm} artifact")
        probes[arm] = {"probe_loss": rec.get("probe_loss"), "drift_from_stage0": rec.get("drift_from_stage0"),
                       "instruction": {d: {"velocity": v["velocity"]["sensitivity"], "chunks": v["chunks"]["relative"],
                                           "trunk_size": v["size_halves"]["trunk"]["median"],
                                           "trunk_outside_memory": (v.get("outside_share_halves") or {}).get("trunk", {}).get("median")}
                                       for d, v in (rec.get("instruction") or {}).items()}}
    if sub["kind"] == "similar":
        judged = inn.judge(sub, cells, gate_rec, boot)
    else:
        judged = judge_diverse(sub, cells, cfg["diverse"], boot)
    judged["vs_historical"] = vs_historical(sub, cells, boot)
    timing = {name: [s.get("train_wall_clock_s") for s in (inn._read(path / "result.json") or {}).get("stages", [])]
              for name, path in runs.items()}
    return {"curriculum": sub["curriculum"], "kind": sub["kind"], "task_b": b,
            "runs": {k: str(v) for k, v in runs.items()}, "incomplete": incomplete, "invalid": invalid,
            "validity": validity,
            "cells": {r: {str(s): {k: inn._rate(v) for k, v in t.items()} for s, t in c.items()} for r, c in cells.items()},
            "judgment": judged, "stage1_swaps": swaps, "probes": probes, "stage_wall_clock_s": timing}


def run_summary(cfg: dict, *, results_root=None, sources_root=None, historical_root=None, smoke: bool = False,
                out: Path | None = None) -> dict:
    from flowcl.methods.instruction_negatives import load_bank

    root, src, hist = _root(results_root), _root(sources_root), _root(historical_root)
    base = base_config(cfg)
    n = inn.n_episodes(base, smoke)
    bank_sha = load_bank(base["bank"])["sha256"]
    boot = inn._bootstrap()
    pinned = pin_checks(cfg, src, hist, smoke)
    p = pinned["pins"]
    stove = inn._read(src / cfg["stove"]["summary"]) or {}
    gate_rec = (stove.get("judgment") or {}).get("gate")
    if smoke:  # the stove smoke's gate fails at 200 steps; the smoke exercises the labels anyway
        gate_rec = {**(gate_rec or {}), "passed": True, "smoke_override": "gate not applied in smoke"}
    incomplete, invalid = [], ([] if pinned["passed"] else ["pinned inputs"])
    blocks = {}
    for short in CURRICULA:
        blk = _curriculum_block(cfg, short, base, root, src, hist, smoke, n, p, bank_sha, gate_rec, boot)
        blocks[short] = blk
        incomplete += blk["incomplete"]
        invalid += blk["invalid"]
    a = cfg["task_a"]
    across = {"stove": {r: {"plate_before_B": (c.get("0") or {}).get(a), "plate_after_B": (c.get("1") or {}).get(a),
                            "B": (c.get("1") or {}).get(base["task_b"])} for r, c in (stove.get("cells") or {}).items()},
              **{short: {r: {"plate_before_B": (c.get("0") or {}).get(a), "plate_after_B": (c.get("1") or {}).get(a),
                             "B": (c.get("1") or {}).get(blocks[short]["task_b"])}
                         for r, c in blocks[short]["cells"].items()} for short in CURRICULA}}
    status = INVALID if invalid else (INCOMPLETE if incomplete else VALID)
    record = {"git_sha": git_sha(), "rule_sha256": file_sha256(config_path()), "bank_sha256": bank_sha,
              "smoke": smoke, "status": status, "exit_code": EXIT_CODES[status], "n_episodes": n,
              "namespace": cfg["namespace"], "pins": p, "pin_checks": pinned, "incomplete": incomplete,
              "invalid": invalid, "curricula": blocks, "across": across, "config": cfg}
    path = Path(out) if out else root / cfg["out"] / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(record, indent=2, default=str) + "\n")
    print(f"[flowcl] instruction-negatives curricula summary: {status} -> {path}", flush=True)
    for short in CURRICULA:
        print(f"[flowcl]   {short}: {blocks[short]['judgment'].get('labels')}", flush=True)
    return record
