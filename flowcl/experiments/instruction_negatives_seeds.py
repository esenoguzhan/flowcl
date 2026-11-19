"""Instruction negatives at seeds 1-2 (``configs/analysis/instruction_negatives_seeds.yaml``).

Per seed, a new negatives base on stove (``gpm_neg``), its panel and gate, cabinet top resumed from it,
and plain-GPM controls on both (queued by ``scripts/queue_instruction_negatives_seeds.sh``). This module
adds what the runner does not record, and judges the rule:

1. :func:`run_preflight`: the pins (earlier rules, bank, hold record, seed 0's summaries) and the
   machine (clean tree, idle GPU, disk, absent targets);
2. :func:`run_panel` / :func:`run_probes` / :func:`run_swaps`: the stove module's steps on per-seed
   configs (:func:`stove_sub`, :func:`cabinet_sub`);
3. :func:`run_summary`: the validity checks, the labels per seed (:func:`judge_seed`), the reading
   across seeds (:func:`across_seeds`) and ``summary.json``.
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
from flowcl.utils.libero_paths import repo_root
from flowcl.utils.run import atomic_write_text, file_sha256, git_sha

VALID, INVALID, INCOMPLETE = inn.VALID, inn.INVALID, inn.INCOMPLETE
EXIT_CODES = inn.EXIT_CODES
ARMS = ("neg", "ctrl")
BS = ("stove", "cabinet")


def config_path() -> Path:
    return repo_root() / "configs" / "analysis" / "instruction_negatives_seeds.yaml"


def load_config(path: str | Path | None = None) -> dict:
    return OmegaConf.to_container(OmegaConf.load(path or config_path()), resolve=True)


def base_path(cfg: dict) -> Path:
    return repo_root() / "configs" / "analysis" / f"{cfg['base_rule']}.yaml"


def base_config(cfg: dict) -> dict:
    base = inn.load_config(base_path(cfg))
    if base["task_a"] != cfg["task_a"] or base["curriculum"] != cfg["curricula"]["stove"]["curriculum"]:
        raise ValueError("the rule's task_a and stove curriculum must equal the stove rule's")
    return base


def _root(path) -> Path:
    return Path(path) if path else repo_root() / "results"


def namespace(cfg: dict, seed: int) -> str:
    from flowcl.train.continual import seed_namespace_run_id

    return seed_namespace_run_id(cfg["namespace_curriculum"], int(seed))


def run_name(cfg: dict, b: str, arm: str, seed: int) -> str:
    return f"{cfg['curricula'][b]['curriculum']}__{cfg['arms'][arm]['display']}__seed{int(seed)}"


def sub_config(cfg: dict, b: str, seed: int, base: dict | None = None) -> dict:
    """One seed and one B in the stove module's config shape (its panel, swaps and probes apply). The
    stove module's arm ``gpm`` is the negatives run; ``ctrl`` is the plain-GPM control."""
    base = copy.deepcopy(base or base_config(cfg))
    c = cfg["curricula"][b]
    base.update({"seed": int(seed), "namespace": namespace(cfg, seed), "out": f"{cfg['out']}/seed{int(seed)}/{b}",
                 "curriculum": c["curriculum"], "task_b": c["task_b"],
                 "runs": {"gpm": {"method": cfg["arms"]["neg"]["method"], "name": run_name(cfg, b, "neg", seed)},
                          "ctrl": {"method": cfg["arms"]["ctrl"]["method"], "name": run_name(cfg, b, "ctrl", seed)}},
                 "historical": {}})
    if b != "stove":  # the panel and gate belong to the stove base; the cabinet config never runs them
        base["held_out_tasks"] = [c["task_b"]]
    return base


def stove_sub(cfg: dict, seed: int, base: dict | None = None) -> dict:
    return sub_config(cfg, "stove", seed, base)


def cabinet_sub(cfg: dict, seed: int, base: dict | None = None) -> dict:
    return sub_config(cfg, "cabinet", seed, base)


def runs_of(cfg: dict, root: Path, seed: int) -> dict:
    """``{(b, arm): run dir}`` for one seed."""
    return {(b, arm): root / run_name(cfg, b, arm, seed) for b in BS for arm in ARMS}


def source_of(cfg: dict, root: Path, seed: int, b: str, arm: str) -> Path:
    """The run whose stage-0 checkpoint a run starts from (itself for the stove runs)."""
    return root / run_name(cfg, "stove", arm, seed)


def seeds_of(cfg: dict, smoke: bool, seeds: list[int] | None) -> list[int]:
    if seeds is not None and not smoke:
        raise ValueError("the seeds can be narrowed in smoke only")
    return [int(s) for s in (seeds if seeds is not None else cfg["seeds"])]


# ---- pins and preflight ----------------------------------------------------------------------------------


def pin_checks(cfg: dict, pinned_root: Path) -> dict:
    """The earlier rules, the bank, the hold record and seed 0's summaries (read from ``pinned_root``,
    the repository's results/ in production and smoke alike)."""
    from flowcl.methods.instruction_negatives import load_bank

    p = cfg["pins"]
    base = base_config(cfg)

    def at(rel: str, sha: str) -> bool:
        f = pinned_root / rel
        return f.is_file() and file_sha256(f) == sha

    hold = inn._read(pinned_root / p["hold"]["path"]) or {}
    checks = {"stove_rule": file_sha256(base_path(cfg)) == p["stove_rule_sha256"],
              "curricula_rule": file_sha256(cur.config_path()) == p["curricula_rule_sha256"],
              "bank": load_bank(base["bank"])["sha256"] == p["bank_sha256"],
              "hold": at(p["hold"]["path"], p["hold"]["sha256"]) and hold.get("passed") is True}
    for key in ("seed0_stove_summary", "seed0_curricula_summary"):
        rec = inn._read(pinned_root / p[key]["path"]) or {}
        checks[key] = at(p[key]["path"], p[key]["sha256"]) and rec.get("status") == VALID
    return {"passed": all(checks.values()), **checks}


def _gpu_idle() -> bool | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.returncode == 0 and not out.stdout.strip()


def run_preflight(cfg: dict, *, results_root=None, pinned_root=None, smoke: bool = False,
                  seeds: list[int] | None = None) -> dict:
    root, pinned = _root(results_root), _root(pinned_root)
    rec = {"git_sha": git_sha(), "smoke": smoke, "seeds": seeds_of(cfg, smoke, seeds), "pins": pin_checks(cfg, pinned)}
    status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=repo_root())
    machine = {"clean_tree": smoke or (status.returncode == 0 and not status.stdout.strip()),
               "gpu_idle": bool(_gpu_idle()),
               "free_gb": shutil.disk_usage(root if root.exists() else repo_root()).free / 1e9}
    machine["disk"] = machine["free_gb"] >= float(cfg["min_free_gb"])
    targets = {path.name: not path.exists() for s in rec["seeds"] for path in runs_of(cfg, root, s).values()}
    machine["targets_absent"] = all(targets.values())
    rec.update({"machine": machine, "targets": targets})
    rec["passed"] = bool(rec["pins"]["passed"] and machine["clean_tree"] and machine["gpu_idle"]
                         and machine["disk"] and machine["targets_absent"])
    out = root / cfg["out"]
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out / "preflight.json", json.dumps(rec, indent=2, default=str) + "\n")
    return rec


# ---- steps: panel, probes, swaps ---------------------------------------------------------------------------


def run_panel(cfg: dict, seed: int, **kw) -> dict:
    return inn.run_panel(stove_sub(cfg, seed), **kw)


def _memory_ref(run: Path) -> dict:
    """The stage-0 memory a stage-0 checkpoint names (path and recorded SHA-256)."""
    ck = torch.load(run / "checkpoints" / "stage0.pt", map_location="cpu", weights_only=False)
    arts = {x["path"]: x["sha256"] for x in (ck.get("extra") or {}).get("method_artifacts") or []}
    return {"path": str(run / "method" / "memory_task0.pt"), "expected_sha256": arts.get("method/memory_task0.pt")}


def probe_targets(cfg: dict, root: Path, seed: int) -> dict:
    """``{(b, label): target}``: the base against each B, and each arm's stage-1 checkpoint (its source's
    stage 0 for drift, its source's memory for the outside share)."""
    out = {}
    refs = {arm: None for arm in ARMS}
    for arm in ARMS:
        src = source_of(cfg, root, seed, "stove", arm)
        if (src / "checkpoints" / "stage0.pt").is_file():
            refs[arm] = _memory_ref(src)
    for b in BS:
        neg_src = source_of(cfg, root, seed, b, "neg")
        if refs["neg"] is not None:
            out[(b, "neg_stage0")] = {"run": neg_src, "checkpoint": neg_src / "checkpoints" / "stage0.pt", "stage": 0,
                                      "stage0": neg_src / "checkpoints" / "stage0.pt", "memory": refs["neg"]}
        for arm, label in (("neg", "gpm_stage1"), ("ctrl", "ctrl_stage1")):
            run = root / run_name(cfg, b, arm, seed)
            src = source_of(cfg, root, seed, b, arm)
            if refs[arm] is not None:
                out[(b, label)] = {"run": run, "checkpoint": run / "checkpoints" / "stage1.pt", "stage": 1,
                                   "stage0": src / "checkpoints" / "stage0.pt", "memory": refs[arm]}
    return out


def run_probes(cfg: dict, *, results_root=None, smoke: bool = False, seeds: list[int] | None = None, **kw) -> list[str]:
    root = _root(results_root)
    base = base_config(cfg)
    problems = []
    for seed in seeds_of(cfg, smoke, seeds):
        targets = probe_targets(cfg, root, seed)
        for b in BS:
            sub = sub_config(cfg, b, seed, base)
            for label in ("neg_stage0", "gpm_stage1", "ctrl_stage1"):
                t = targets.get((b, label))
                if t is None or not t["checkpoint"].is_file():
                    problems.append(f"seed {seed} {b} {label}: checkpoint missing")
                    print(f"[flowcl] probe seed {seed} {b} {label}: checkpoint missing, skipped", flush=True)
                    continue
                try:
                    inn.run_probe(sub, label, target=t, results_root=root, **kw)
                except Exception as exc:  # one probe's failure must not block the others
                    problems.append(f"seed {seed} {b} {label}: {type(exc).__name__}: {exc}")
                    print(f"[flowcl] probe seed {seed} {b} {label} FAILED: {exc}", flush=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    return problems


def run_swaps(cfg: dict, seed: int, b: str, **kw) -> dict:
    return inn.run_swaps(sub_config(cfg, b, seed), "gpm", **kw)


# ---- the judgment (pure) ---------------------------------------------------------------------------------


def _cell(cells: dict, arm: str, stage: int, task: str):
    return ((cells.get(arm) or {}).get(stage) or {}).get(task)


def _pair(cells, stage, task, boot):
    x, y = _cell(cells, "neg", stage, task), _cell(cells, "ctrl", stage, task)
    if x is None or y is None:
        return None
    return {**inn._paired(x, y, boot), "n_rows_differ": int(sum(a != b for a, b in zip(x, y)))}


def judge_seed(rule: dict, cells: dict, gate_rec: dict | None, a: str, b: str, boot: dict) -> dict:
    """Labels for one seed and one B (pure). ``cells[arm][stage][task]`` = successes, arms ``neg``, ``ctrl``."""
    names = ("effect_present", "retains")
    out = {"gate": gate_rec, "projection_contributes": "not_run"}
    if not gate_rec or not gate_rec.get("passed"):
        out.update({"outcome": "gate_failed" if gate_rec else "gate_missing",
                    "labels": {k: "not_assessed" for k in names}})
        return out
    r1b = {arm: inn._rate(_cell(cells, arm, 1, b)) for arm in ARMS}
    if None in r1b.values():
        state = "missing"
    elif min(r1b.values()) < float(rule["inconclusive_below"]):
        state = "inconclusive_acquisition"
    elif min(r1b.values()) >= float(rule["floor"]) and r1b["neg"] - r1b["ctrl"] >= -float(rule["tolerance"]):
        state = "clean"
    else:
        state = "at_acquisition_cost"
    suffix = "" if state == "clean" else "_at_acquisition_cost"

    def label(name: str, holds: bool) -> str:
        if state in ("inconclusive_acquisition", "missing"):
            return "not_assessed"
        return (name + suffix) if holds else "not_shown"

    forget = {arm: (inn._rate(_cell(cells, arm, 0, a)) - inn._rate(_cell(cells, arm, 1, a))
                    if _cell(cells, arm, 0, a) is not None and _cell(cells, arm, 1, a) is not None else None)
              for arm in ARMS}
    effect = _pair(cells, 1, a, boot)
    labels = {"effect_present": label("effect_present", bool(effect and effect["diff"] >= float(rule["effect_min"])
                                                             and effect["low"] > 0)),
              "retains": label("retains", forget["neg"] is not None and forget["neg"] <= float(rule["forgetting_max"]))}
    out.update({"outcome": "assessed", "acquisition": {"R1_B": r1b, "neg_minus_ctrl": _pair(cells, 1, b, boot),
                                                       "state": state},
                "plate_after_B": {arm: inn._rate(_cell(cells, arm, 1, a)) for arm in ARMS},
                "effect_neg_minus_ctrl": effect, "plate_before_B": _pair(cells, 0, a, boot),
                "forgetting_A": forget, "labels": labels})
    return out


def across_seeds(per_seed: dict) -> str:
    """The pre-registered reading for one B from ``{seed: effect_present label}`` (pure)."""
    values = list(per_seed.values())
    if not values or any(v is None or v == "not_assessed" for v in values):
        return "not_assessed"
    k = sum(str(v).startswith("effect_present") for v in values)
    return "replicated" if k == len(values) else ("not_replicated" if k == 0 else "mixed")


# ---- validity and the summary ----------------------------------------------------------------------------


def control_checks(sub: dict, run: Path, smoke: bool) -> dict:
    """A full plain-GPM control run: namespace, method (no negative keys), clean git, result, recipe, memory."""
    from flowcl.experiments.similar_cl import memory_check

    conf = inn._conf(run)
    method = conf.get("method") or {}
    git = run / "git_sha"
    checks = {"namespace": conf.get("seed_namespace_run_id") == sub["namespace"],
              "method": method.get("name") == sub["runs"]["ctrl"]["method"],
              "no_negatives": not any(str(k).startswith("negative") for k in method),
              "clean_git_sha": smoke or (git.is_file() and not git.read_text().strip().endswith("-dirty")),
              "result_json": (run / "result.json").is_file()}
    if not smoke:
        train = conf.get("train", {})
        checks["recipe"] = all(train.get(k) == v for k, v in sub["production"]["train"].items())
    checks["memory"] = memory_check(run)
    ok = all((v.get("passed") if isinstance(v, dict) else bool(v)) for v in checks.values())
    return {"passed": ok, **checks}


def _artifact(rec: dict | None, ckpt: Path, seeds_key: str | None = None) -> bool | None:
    if rec is None:
        return None
    ok = ckpt.is_file() and rec.get("checkpoint_sha256") == file_sha256(ckpt) and rec.get("weights_unchanged") is True
    if seeds_key == "swaps":
        ok = ok and all(s["eval_seeds_match"] for s in rec["swaps"].values())
    if seeds_key == "panel":
        ok = ok and all(c["eval_seeds_match"] for c in rec["conditions"].values())
    return bool(ok)


def _probe_view(rec: dict) -> dict:
    return {"probe_loss": rec.get("probe_loss"), "drift_from_stage0": rec.get("drift_from_stage0"),
            "instruction": {d: {"velocity": v["velocity"]["sensitivity"], "chunks": v["chunks"]["relative"],
                                "trunk_size": v["size_halves"]["trunk"]["median"],
                                "trunk_outside_memory": (v.get("outside_share_halves") or {}).get("trunk", {}).get("median")}
                            for d, v in (rec.get("instruction") or {}).items()}}


def _seed_block(cfg: dict, base: dict, root: Path, seed: int, smoke: bool, n: int, bank_sha: str, boot: dict) -> dict:
    a = cfg["task_a"]
    runs = runs_of(cfg, root, seed)
    incomplete, invalid, validity, blocks = [], [], {}, {}
    panel = inn._read(root / stove_sub(cfg, seed, base)["out"] / "panel" / "panel.json")
    gate_rec = panel.get("gate") if panel else None
    neg_stove = runs[("stove", "neg")]
    if panel is None:
        incomplete.append(f"seed {seed}: panel missing")
    else:
        validity["panel"] = _artifact(panel, neg_stove / "checkpoints" / "stage0.pt", "panel")
        if not validity["panel"]:
            invalid.append(f"seed {seed}: panel artifact")
    judged_gate = gate_rec
    if smoke and gate_rec is not None:  # 200 steps cannot pass the gate; the smoke exercises the labels
        judged_gate = {**gate_rec, "passed": True, "smoke_override": "gate not applied in smoke"}
    all_seeds = {}
    for b in BS:
        sub = sub_config(cfg, b, seed, base)
        task_b = sub["task_b"]
        cells, v = {}, {}
        for arm in ARMS:
            run = runs[(b, arm)]
            if not (run / "result.json").is_file():
                incomplete.append(f"seed {seed} {b} {arm}: {run.name} has no result.json")
                continue
            succ, sd, problems = cur.eval_cells(run, {0: [a], 1: [a, task_b]}, n)
            cells[arm], all_seeds[(b, arm)] = succ, sd
            incomplete.extend(f"seed {seed} {b} {arm}: {x}" for x in problems)
            if not cfg["curricula"][b]["resumed"]:
                v[arm] = (inn._run_checks({**sub, "runs": {"gpm": sub["runs"]["gpm"]}}, run, "gpm", smoke, bank_sha)
                          if arm == "neg" else control_checks(sub, run, smoke))
            else:
                src = source_of(cfg, root, seed, b, arm)
                ck, mem = src / "checkpoints" / "stage0.pt", src / "method" / "memory_task0.pt"
                pin = {"stage0": file_sha256(ck) if ck.is_file() else None,
                       "memory": file_sha256(mem) if mem.is_file() else None}
                arm_sub = {**sub, "runs": {arm: sub["runs"]["gpm" if arm == "neg" else "ctrl"]}}
                v[arm] = cur.run_checks_resumed(arm_sub, run, arm, src, pin, smoke, bank_sha, n,
                                                negatives=arm == "neg")
            if not v[arm]["passed"]:
                invalid.append(f"seed {seed} {b} {arm}: run checks failed")
        probes = {}
        targets = probe_targets(cfg, root, seed)
        for label in ("neg_stage0", "gpm_stage1", "ctrl_stage1"):
            rec = inn._read(root / sub["out"] / "probes" / label / "probe.json")
            t = targets.get((b, label))
            if rec is None:
                incomplete.append(f"seed {seed} {b} probe {label} missing")
                continue
            ok = t is not None and _artifact(rec, t["checkpoint"])
            v[f"probe_{label}"] = bool(ok)
            if not ok:
                invalid.append(f"seed {seed} {b} probe {label} artifact")
            probes[label] = _probe_view(rec)
        swaps = None
        rec = inn._read(root / sub["out"] / "swaps" / "gpm" / "swaps.json")
        if rec is not None:  # descriptive filler after the main summary; absent is not incomplete
            v["swaps"] = _artifact(rec, runs[(b, "neg")] / "checkpoints" / "stage1.pt", "swaps")
            if not v["swaps"]:
                invalid.append(f"seed {seed} {b} swaps artifact")
            swaps = {k: {key: s[key] for key in ("own_goal_rate", "requested_rate", "median_max_displacement_m")}
                     for k, s in rec["swaps"].items()}
        judged = judge_seed(cfg["labels"], cells, judged_gate, a, task_b, boot)
        timing = {arm: [s.get("train_wall_clock_s") for s in (inn._read(runs[(b, arm)] / "result.json") or {}).get("stages", [])]
                  for arm in ARMS}
        blocks[b] = {"curriculum": sub["curriculum"], "task_b": task_b,
                     "runs": {arm: str(runs[(b, arm)]) for arm in ARMS}, "validity": v,
                     "cells": {r: {str(s): {k: inn._rate(x) for k, x in t.items()} for s, t in c.items()}
                               for r, c in cells.items()},
                     "judgment": judged, "probes": probes, "stage1_swaps": swaps, "stage_wall_clock_s": timing}
    # row pairing: per B, neg and ctrl identical (task, episode index = position, seed); plate's rows equal
    # across the stove and cabinet runs at both stages
    pairs = {b: (b, "neg") in all_seeds and (b, "ctrl") in all_seeds and all_seeds[(b, "neg")] == all_seeds[(b, "ctrl")]
             for b in BS}
    plate = [{st: (sd.get(st) or {}).get(a) for st in (0, 1)} for sd in all_seeds.values()]
    plate_ok = len(plate) == 2 * len(BS) and all(p == plate[0] and None not in p.values() for p in plate)
    validity["row_pairing"] = {"passed": all(pairs.values()) and plate_ok, "within_B": pairs, "plate_across_B": plate_ok}
    if all_seeds and not validity["row_pairing"]["passed"]:
        (invalid if len(plate) == 2 * len(BS) else incomplete).append(f"seed {seed}: paired rows differ or missing")
    view = None
    if panel:
        view = {name: {k: c[k] for k in ("kind", "text", "n", "own_goal_rate", "requested_rate",
                                         "median_max_displacement_m")} for name, c in panel["conditions"].items()}
    for b in BS:
        validity[b] = blocks[b]["validity"]
    return {"seed": seed, "namespace": namespace(cfg, seed), "gate": gate_rec, "panel": view, "validity": validity,
            "incomplete": incomplete, "invalid": invalid, "B": blocks}


def seed0_rows(cfg: dict, pinned_root: Path) -> dict:
    """Seed 0's label and cells for each B, from the pinned summaries (historical controls)."""
    a = cfg["task_a"]
    p = cfg["pins"]
    stove = inn._read(pinned_root / p["seed0_stove_summary"]["path"]) or {}
    curricula = inn._read(pinned_root / p["seed0_curricula_summary"]["path"]) or {}
    cab = (curricula.get("curricula") or {}).get("cabinet") or {}
    out = {}
    for b, rec in (("stove", stove), ("cabinet", cab)):
        cells = rec.get("cells") or {}
        task_b = cfg["curricula"][b]["task_b"]
        out[b] = {"effect_present": ((rec.get("judgment") or {}).get("labels") or {}).get("effect_present"),
                  "neg": {"plate_after_B": (cells.get("gpm", {}).get("1") or {}).get(a),
                          "B": (cells.get("gpm", {}).get("1") or {}).get(task_b)},
                  "ctrl": {"plate_after_B": (cells.get("hist_gpm", {}).get("1") or {}).get(a),
                           "B": (cells.get("hist_gpm", {}).get("1") or {}).get(task_b)},
                  "flag": "against_historical_control"}
    return out


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def run_summary(cfg: dict, *, results_root=None, pinned_root=None, smoke: bool = False,
                seeds: list[int] | None = None, out: Path | None = None) -> dict:
    from flowcl.methods.instruction_negatives import load_bank

    root, pinned = _root(results_root), _root(pinned_root)
    base = base_config(cfg)
    chosen = seeds_of(cfg, smoke, seeds)
    n = inn.n_episodes(base, smoke)
    bank_sha = load_bank(base["bank"])["sha256"]
    boot = inn._bootstrap()
    pins = pin_checks(cfg, pinned)
    incomplete, invalid = [], ([] if pins["passed"] else ["pinned inputs"])
    hold = inn._read(pinned / cfg["pins"]["hold"]["path"]) or {}
    per_seed = {}
    for s in chosen:
        blk = _seed_block(cfg, base, root, s, smoke, n, bank_sha, boot)
        per_seed[str(s)] = blk
        incomplete += blk["incomplete"]
        invalid += blk["invalid"]
    s0 = seed0_rows(cfg, pinned)
    across = {}
    for b in BS:
        labels = {str(s): (per_seed[str(s)]["B"][b]["judgment"].get("labels") or {}).get("effect_present")
                  for s in chosen}
        table = {"0": {"neg": s0[b]["neg"], "ctrl": s0[b]["ctrl"], "flag": s0[b]["flag"]}}
        for s in chosen:
            c = per_seed[str(s)]["B"][b]["cells"]
            task_b = cfg["curricula"][b]["task_b"]
            table[str(s)] = {arm: {"plate_after_B": (c.get(arm, {}).get("1") or {}).get(cfg["task_a"]),
                                   "B": (c.get(arm, {}).get("1") or {}).get(task_b)} for arm in ARMS}
        means = {arm: {k: _mean([table[s][arm][k] for s in table]) for k in ("plate_after_B", "B")} for arm in ARMS}
        across[b] = {"reading": across_seeds(labels), "effect_present": labels,
                     "seed0_effect_present": s0[b]["effect_present"], "table": table, "mean_over_seeds": means}
    status = INVALID if invalid else (INCOMPLETE if incomplete else VALID)
    record = {"git_sha": git_sha(), "rule_sha256": file_sha256(config_path()), "bank_sha256": bank_sha,
              "smoke": smoke, "seeds": chosen, "status": status, "exit_code": EXIT_CODES[status], "n_episodes": n,
              "pins": pins, "hold": {"passed": hold.get("passed"), "path": cfg["pins"]["hold"]["path"]},
              "incomplete": incomplete, "invalid": invalid, "per_seed": per_seed, "across": across, "config": cfg}
    path = Path(out) if out else root / cfg["out"] / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(record, indent=2, default=str) + "\n")
    print(f"[flowcl] instruction-negatives seeds summary: {status} -> {path}", flush=True)
    for b in BS:
        print(f"[flowcl]   {b}: {across[b]['reading']} {across[b]['effect_present']}", flush=True)
    return record
