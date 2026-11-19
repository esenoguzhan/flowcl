"""Instruction negatives on one A -> B pair, seeds as in the rule, GPM + negatives against fresh plain-GPM
controls (``configs/analysis/instruction_negatives_<rule>.yaml``; first rule: ``wine``).

Per seed, a negatives base and its stage B (one full run), the panel and gate on its stage-0 checkpoint,
and a plain-GPM control (queued by ``scripts/queue_instruction_negatives_pair.sh --rule <rule>``). This
module adds what the runner does not record, and judges the rule:

1. :func:`run_preflight`: the pins (bank, method config, curriculum) and the machine;
2. :func:`run_hold` / :func:`run_panel` / :func:`run_probes` / :func:`run_swaps`: the stove module's
   steps on per-seed configs (:func:`sub_config`);
3. :func:`run_summary`: validity, the labels per seed (the seeds study's :func:`judge_seed`), what is
   reported for every completed seed (:func:`describe_seed`), the reading across seeds, ``summary.json``.
"""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
from pathlib import Path

import torch
from omegaconf import OmegaConf

from flowcl.experiments import instruction_negatives as inn
from flowcl.experiments import instruction_negatives_curricula as cur
from flowcl.experiments import instruction_negatives_seeds as ins
from flowcl.utils.libero_paths import repo_root
from flowcl.utils.run import atomic_write_text, file_sha256, git_sha

VALID, INVALID, INCOMPLETE = inn.VALID, inn.INVALID, inn.INCOMPLETE
EXIT_CODES = inn.EXIT_CODES
ARMS = ("neg", "ctrl")
PROBES = ("neg_stage0", "gpm_stage1", "ctrl_stage1")
SUB_KEYS = ("task_a", "task_b", "held_out_tasks", "bank", "production", "smoke", "hold_check", "panel", "gate",
            "acquisition", "retention", "probe", "swap", "chunks", "coverage", "drift")


def config_path(rule: str) -> Path:
    return repo_root() / "configs" / "analysis" / f"instruction_negatives_{rule}.yaml"


def load_config(rule: str, path: str | Path | None = None) -> dict:
    cfg = OmegaConf.to_container(OmegaConf.load(path or config_path(rule)), resolve=True)
    cfg["rule"] = rule
    return cfg


def _root(path) -> Path:
    return Path(path) if path else repo_root() / "results"


def namespace(cfg: dict, seed: int) -> str:
    from flowcl.train.continual import seed_namespace_run_id

    return seed_namespace_run_id(cfg["curriculum"], int(seed))


def run_name(cfg: dict, arm: str, seed: int) -> str:
    return f"{cfg['curriculum']}__{cfg['arms'][arm]['display']}__seed{int(seed)}"


def sub_config(cfg: dict, seed: int | None) -> dict:
    """One seed in the stove module's config shape (its hold, panel, swaps and probes apply); the stove
    module's arm ``gpm`` is the negatives run. ``seed=None``: the seed-independent hold check's config."""
    sub = copy.deepcopy({k: cfg[k] for k in SUB_KEYS})
    sub.update({"curriculum": cfg["curriculum"], "historical": {}})
    if seed is None:
        sub.update({"seed": None, "namespace": None, "out": cfg["out"], "runs": {}})
        return sub
    sub.update({"seed": int(seed), "namespace": namespace(cfg, seed), "out": f"{cfg['out']}/seed{int(seed)}",
                "runs": {"gpm": {"method": cfg["arms"]["neg"]["method"], "name": run_name(cfg, "neg", seed)},
                         "ctrl": {"method": cfg["arms"]["ctrl"]["method"], "name": run_name(cfg, "ctrl", seed)}}})
    return sub


def seeds_of(cfg: dict, smoke: bool, seeds: list[int] | None) -> list[int]:
    if seeds is not None and not smoke:
        raise ValueError("the seeds can be narrowed in smoke only")
    return [int(s) for s in (seeds if seeds is not None else cfg["seeds"])]


# ---- pins and preflight ----------------------------------------------------------------------------------


def pin_checks(cfg: dict) -> dict:
    """The bank at its pin (and valid), the CLI method config naming it, the curriculum's tasks."""
    from flowcl.data.config import load_method_config
    from flowcl.data.curriculum import load_curriculum
    from flowcl.methods.instruction_negatives import load_bank

    checks = {}
    try:
        bank = load_bank(cfg["bank"])
        checks["bank"] = bank["sha256"] == cfg["pins"]["bank_sha256"]
    except (OSError, ValueError) as exc:
        checks["bank"] = False
        checks["bank_error"] = str(exc)
    name, kwargs = load_method_config(cfg["arms"]["neg"]["cli_method"])
    checks["method_config"] = name == cfg["arms"]["neg"]["method"] and kwargs.get("negative_bank") == cfg["bank"]
    ctrl_name, ctrl_kwargs = load_method_config(cfg["arms"]["ctrl"]["cli_method"])
    checks["ctrl_config"] = (ctrl_name == cfg["arms"]["ctrl"]["method"]
                             and not any(str(k).startswith("negative") for k in ctrl_kwargs))
    tasks = list(load_curriculum(cfg["curriculum"]).task_keys)
    checks["curriculum"] = tasks == [cfg["task_a"], cfg["task_b"]]
    ok = all(v for k, v in checks.items() if k != "bank_error")
    return {"passed": bool(ok), **checks}


def _gpu_idle() -> bool | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.returncode == 0 and not out.stdout.strip()


def run_preflight(cfg: dict, *, results_root=None, smoke: bool = False, seeds: list[int] | None = None,
                  resume: bool = False) -> dict:
    """``resume``: a queue restarted with --from-step; finished steps' outputs exist, so the targets are
    not required absent (the summary checks every reused output)."""
    root = _root(results_root)
    rec = {"git_sha": git_sha(), "smoke": smoke, "resume": resume, "seeds": seeds_of(cfg, smoke, seeds),
           "pins": pin_checks(cfg)}
    status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=repo_root())
    machine = {"clean_tree": smoke or (status.returncode == 0 and not status.stdout.strip()),
               "gpu_idle": bool(_gpu_idle()),
               "free_gb": shutil.disk_usage(root if root.exists() else repo_root()).free / 1e9}
    machine["disk"] = machine["free_gb"] >= float(cfg["min_free_gb"])
    targets = {run_name(cfg, arm, s): not (root / run_name(cfg, arm, s)).exists() for s in rec["seeds"] for arm in ARMS}
    machine["targets_absent"] = all(targets.values())
    rec.update({"machine": machine, "targets": targets})
    rec["passed"] = bool(rec["pins"]["passed"] and machine["clean_tree"] and machine["gpu_idle"] and machine["disk"]
                         and (resume or machine["targets_absent"]))
    out = root / cfg["out"]
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out / ("preflight_resume.json" if resume else "preflight.json"),
                      json.dumps(rec, indent=2, default=str) + "\n")
    return rec


# ---- steps: hold, panel, probes, swaps --------------------------------------------------------------------


def run_hold(cfg: dict, **kw) -> dict:
    return inn.run_hold(sub_config(cfg, None), **kw)


def run_panel(cfg: dict, seed: int, **kw) -> dict:
    return inn.run_panel(sub_config(cfg, seed), **kw)


def probe_targets(cfg: dict, root: Path, seed: int) -> dict:
    """``{label: target}``: the base against B (its own memory), the neg run's stage 1 (the base's memory),
    and the control's stage 1 (its own memory); each run's own stage 0 for drift."""
    out = {}
    for arm, labels in (("neg", ("neg_stage0", "gpm_stage1")), ("ctrl", ("ctrl_stage1",))):
        run = root / run_name(cfg, arm, seed)
        if not (run / "checkpoints" / "stage0.pt").is_file():
            continue
        mem = ins._memory_ref(run)
        for label in labels:
            stage = 0 if label == "neg_stage0" else 1
            out[label] = {"run": run, "checkpoint": run / "checkpoints" / f"stage{stage}.pt", "stage": stage,
                          "stage0": run / "checkpoints" / "stage0.pt", "memory": mem}
    return out


def run_probes(cfg: dict, *, results_root=None, smoke: bool = False, seeds: list[int] | None = None,
               **kw) -> tuple[list[str], list[str]]:
    """``(problems, skipped)``: a probe whose checkpoint does not exist (its run failed) is skipped, not a
    failure; the summary marks that seed incomplete."""
    root = _root(results_root)
    problems, skipped = [], []
    for seed in seeds_of(cfg, smoke, seeds):
        sub = sub_config(cfg, seed)
        targets = probe_targets(cfg, root, seed)
        for label in PROBES:
            t = targets.get(label)
            if t is None or not t["checkpoint"].is_file():
                skipped.append(f"seed {seed} {label}")
                print(f"[flowcl] probe seed {seed} {label}: checkpoint missing, skipped", flush=True)
                continue
            try:
                inn.run_probe(sub, label, target=t, results_root=root, **kw)
            except Exception as exc:  # one probe's failure must not block the others
                problems.append(f"seed {seed} {label}: {type(exc).__name__}: {exc}")
                print(f"[flowcl] probe seed {seed} {label} FAILED: {exc}", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return problems, skipped


def run_swaps(cfg: dict, seed: int, **kw) -> dict:
    return inn.run_swaps(sub_config(cfg, seed), "gpm", **kw)


# ---- what is reported for every completed seed (pure) -------------------------------------------------------


def describe_seed(cells: dict, a: str, b: str, boot: dict) -> dict:
    """Both arms' A before B, A after B, B after B and forgetting; the paired neg - ctrl differences with
    signed intervals; ``improvement_ci_low_above_0`` on A after B. Computed whatever the gate says."""
    def rate(arm, stage, task):
        return inn._rate(ins._cell(cells, arm, stage, task))

    table = {}
    for arm in ARMS:
        before, after = rate(arm, 0, a), rate(arm, 1, a)
        table[arm] = {"A_before_B": before, "A_after_B": after, "B_after_B": rate(arm, 1, b),
                      "forgetting": None if before is None or after is None else before - after}
    diffs = {"A_after_B": ins._pair(cells, 1, a, boot), "B_after_B": ins._pair(cells, 1, b, boot),
             "A_before_B": ins._pair(cells, 0, a, boot)}
    after = diffs["A_after_B"]
    return {"table": table, "neg_minus_ctrl": diffs,
            "improvement_ci_low_above_0": None if after is None else bool(after["low"] > 0)}


def panel_own_reproduces_eval(panel: dict, eval_stage0: Path, a: str) -> dict:
    """The panel's own-instruction rows equal A's ordinary stage-0 evaluation, row by row (seed, success,
    step count): the panel conditions the policy exactly as the evaluation does."""
    from flowcl.envs.evaluation import EvaluationReport

    own = (panel.get("conditions") or {}).get("own") or {}
    rows = own.get("rows") or []
    if not rows or not eval_stage0.is_file():
        return {"passed": False, "reason": "own rows or the stage-0 evaluation missing"}
    ev = EvaluationReport.load(eval_stage0).by_task().get(a)
    if ev is None or len(ev.successes) < len(rows):
        return {"passed": False, "reason": "A's stage-0 evaluation is shorter than the own rows"}
    n = len(rows)
    checks = {"seeds": [r["seed"] for r in rows] == list(ev.seeds[:n]),
              "successes": [r["own_goal"] for r in rows] == [bool(x) for x in ev.successes[:n]],
              "n_steps": [r["n_steps"] for r in rows] == list(ev.n_steps[:n])}
    return {"passed": all(checks.values()), "n_rows": n, **checks}


# ---- validity and the summary ----------------------------------------------------------------------------


def _commit(path: Path, rec: dict | None = None) -> str | None:
    if rec is not None:
        return rec.get("git_sha")
    f = path / "git_sha"
    return f.read_text().strip() if f.is_file() else None


def _seed_block(cfg: dict, root: Path, seed: int, smoke: bool, final: bool, n: int, bank_sha: str, boot: dict) -> dict:
    a, b = cfg["task_a"], cfg["task_b"]
    sub = sub_config(cfg, seed)
    runs = {arm: root / run_name(cfg, arm, seed) for arm in ARMS}
    incomplete, invalid, validity, cells, seeds, commits = [], [], {}, {}, {}, {}
    for arm in ARMS:
        run = runs[arm]
        if not (run / "result.json").is_file():
            incomplete.append(f"seed {seed} {arm}: {run.name} has no result.json")
            continue
        succ, sd, problems = cur.eval_cells(run, {0: [a], 1: [a, b]}, n)
        cells[arm], seeds[arm] = succ, sd
        incomplete.extend(f"seed {seed} {arm}: {x}" for x in problems)
        validity[arm] = (inn._run_checks(sub, run, "gpm", smoke, bank_sha) if arm == "neg"
                         else ins.control_checks(sub, run, smoke))
        commits[arm] = _commit(run)
        if not validity[arm]["passed"]:
            invalid.append(f"seed {seed} {arm}: run checks failed")
    if len(seeds) == len(ARMS):
        validity["row_pairing"] = {"passed": seeds["neg"] == seeds["ctrl"]}
        if not validity["row_pairing"]["passed"]:
            invalid.append(f"seed {seed}: paired rows differ")
    panel = inn._read(root / sub["out"] / "panel" / "panel.json")
    gate_rec = panel.get("gate") if panel else None
    texts = None
    if panel is None:
        if "neg" in cells:
            incomplete.append(f"seed {seed}: panel missing")
    else:
        validity["panel"] = bool(ins._artifact(panel, runs["neg"] / "checkpoints" / "stage0.pt", "panel"))
        validity["panel_own_reproduces_eval"] = panel_own_reproduces_eval(panel, runs["neg"] / "eval" / "stage0.json", a)
        commits["panel"] = _commit(root, panel)
        texts = (panel.get("text_cache") or {}).get("tasks")
        if not (validity["panel"] and validity["panel_own_reproduces_eval"]["passed"]):
            invalid.append(f"seed {seed}: panel artifact")
    probes, targets = {}, probe_targets(cfg, root, seed)
    for label in PROBES:
        rec = inn._read(root / sub["out"] / "probes" / label / "probe.json")
        if rec is None:
            if (label == "ctrl_stage1" and "ctrl" in cells) or (label != "ctrl_stage1" and "neg" in cells):
                incomplete.append(f"seed {seed} probe {label} missing")
            continue
        t = targets.get(label)
        ok = bool(t is not None and ins._artifact(rec, t["checkpoint"]))
        validity[f"probe_{label}"] = ok
        commits[f"probe_{label}"] = _commit(root, rec)
        if not ok:
            invalid.append(f"seed {seed} probe {label} artifact")
        probes[label] = ins._probe_view(rec)
    swaps = None
    rec = inn._read(root / sub["out"] / "swaps" / "gpm" / "swaps.json")
    if rec is None:
        if final and "neg" in cells:
            incomplete.append(f"seed {seed} swaps missing")
    else:
        validity["swaps"] = bool(ins._artifact(rec, runs["neg"] / "checkpoints" / "stage1.pt", "swaps"))
        commits["swaps"] = _commit(root, rec)
        if not validity["swaps"]:
            invalid.append(f"seed {seed} swaps artifact")
        swaps = {k: {key: s[key] for key in ("own_goal_rate", "requested_rate", "median_max_displacement_m")}
                 for k, s in rec["swaps"].items()}
    judged_gate = gate_rec
    if smoke and gate_rec is not None:  # 200 steps cannot pass the gate; the smoke exercises the labels
        judged_gate = {**gate_rec, "passed": True, "smoke_override": "gate not applied in smoke"}
    judged = ins.judge_seed(cfg["labels"], cells, judged_gate, a, b, boot)
    view = None
    if panel:
        view = {name: {k: c[k] for k in ("kind", "text", "n", "own_goal_rate", "requested_rate",
                                         "median_max_displacement_m")} for name, c in panel["conditions"].items()}
    timing = {arm: [s.get("train_wall_clock_s") for s in (inn._read(runs[arm] / "result.json") or {}).get("stages", [])]
              for arm in ARMS}
    return {"seed": seed, "namespace": sub["namespace"], "runs": {k: str(v) for k, v in runs.items()},
            "gate": gate_rec, "panel": view, "policy_texts": texts, "validity": validity, "commits": commits,
            "incomplete": incomplete, "invalid": invalid,
            "cells": {r: {str(s): {k: inn._rate(x) for k, x in t.items()} for s, t in c.items()} for r, c in cells.items()},
            "judgment": judged, "reported": describe_seed(cells, a, b, boot) if cells else None,
            "probes": probes, "stage1_swaps": swaps, "stage_wall_clock_s": timing}


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def run_summary(cfg: dict, *, results_root=None, smoke: bool = False, seeds: list[int] | None = None,
                final: bool = False, out: Path | None = None) -> dict:
    from flowcl.methods.instruction_negatives import load_bank

    root = _root(results_root)
    chosen = seeds_of(cfg, smoke, seeds)
    n = inn.n_episodes(cfg, smoke)
    bank_sha = load_bank(cfg["bank"])["sha256"]
    boot = inn._bootstrap()
    pins = pin_checks(cfg)
    incomplete, invalid = [], ([] if pins["passed"] else ["pinned inputs"])
    hold = inn._read(root / cfg["out"] / "hold" / "hold.json")
    if hold is None:
        incomplete.append("hold check missing")
    elif not hold.get("passed"):
        invalid.append("hold check failed")
    per_seed = {}
    for s in chosen:
        blk = _seed_block(cfg, root, s, smoke, final, n, bank_sha, boot)
        per_seed[str(s)] = blk
        incomplete += blk["incomplete"]
        invalid += blk["invalid"]
    labels = {str(s): (per_seed[str(s)]["judgment"].get("labels") or {}).get("effect_present") for s in chosen}
    reported = {str(s): per_seed[str(s)]["reported"] for s in chosen}
    means = {arm: {k: _mean([(r or {}).get("table", {}).get(arm, {}).get(k) for r in reported.values()])
                   for k in ("A_before_B", "A_after_B", "B_after_B", "forgetting")} for arm in ARMS}
    across = {"reading": ins.across_seeds(labels), "effect_present": labels,
              "improvement_ci_low_above_0": {s: (r or {}).get("improvement_ci_low_above_0") for s, r in reported.items()},
              "mean_over_seeds": means}
    commits = sorted({c for blk in per_seed.values() for c in blk["commits"].values() if c})
    status = INVALID if invalid else (INCOMPLETE if incomplete else VALID)
    record = {"git_sha": git_sha(), "rule": cfg["rule"], "rule_sha256": file_sha256(config_path(cfg["rule"])),
              "bank_sha256": bank_sha, "smoke": smoke, "stage": "final" if final else "preliminary",
              "seeds": chosen, "status": status, "exit_code": EXIT_CODES[status], "n_episodes": n, "pins": pins,
              "hold": None if hold is None else {"passed": hold.get("passed"), "commit": hold.get("git_sha")},
              "commits": commits, "incomplete": incomplete, "invalid": invalid, "per_seed": per_seed,
              "across": across, "config": cfg}
    path = Path(out) if out else root / cfg["out"] / ("summary.json" if final else "summary_preliminary.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(record, indent=2, default=str) + "\n")
    print(f"[flowcl] instruction-negatives pair {cfg['rule']} summary ({record['stage']}): {status} -> {path}", flush=True)
    print(f"[flowcl]   {across['reading']} {labels}; improvement_ci_low_above_0 "
          f"{across['improvement_ci_low_above_0']}", flush=True)
    return record
