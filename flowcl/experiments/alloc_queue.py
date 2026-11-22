"""The allocation-rules queue's preflight (``scripts/queue_alloc.sh``; rule ``configs/analysis/allocation_rules.yaml``).

Before any GPU work: a clean tree with the rule committed, an idle GPU, enough free disk, no target run
directory or evaluation output yet, the three method configs (``gpm.yaml`` plus the rule and e99's
schedule), and the references by SHA-256 (seq_ft's stage 0 and plain GPM's stages 0-1 and Task-1 memory
as for the f = 0.7 queue; e99's and ne90's files as the rule pins them). For the archived comparator it
also checks, per seed, that e99's recorded identity at stages 0-1 is against plain GPM's pinned stage
hashes, that its Task-1 memory equals plain GPM's basis for basis, that its run config equals a run of
the current code (the f = 0.7 run of the same seed) except the method block and neutral new keys, and
that the offline post-T2 prediction reproduces e99 and ne90 and the registered per-layer ranks.
A smoke checks that its references exist instead.

Env B (ThinkStation-2, torch 2.9.1; the rule's ``environment`` block, amended 22 Nov 2026): the
references are the runs in the rule's ``results_root`` (``results/envB``) and every hash comes from
the rule's pins, not the gamma constants. The comparator e99 was trained there by the current code,
so its config is compared with env-B plain GPM's (they may differ only in e99's identity provenance,
which must name that plain run at stages 0-1). An environment check refuses the launch when the
interpreter's torch, or any reference run's recorded torch, is not the rule's (smoke references
included): an env-A run is never paired with an env-B one.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from omegaconf import OmegaConf

from flowcl.data.config import load_method_config
from flowcl.experiments.allocation_rules import (
    ARMS,
    LABELS,
    config_diff,
    eval_path,
    load_rule,
    namespace,
    out_dir,
    predict,
    results_root,
    rule_path,
    run_dir,
)
from flowcl.experiments.baseline_queue import GPM_MEMORY0, SEQ_FT_STAGE0, _hashes, gpu_compute_pids
from flowcl.experiments.ne70_queue import GPM_STAGE, free_gb
from flowcl.utils.libero_paths import repo_root

# One 4-task GPM-family run is about 3.5 GB; three runs plus evaluations, with margin. A smoke ~2.5 GB/run.
MIN_FREE_GB = {False: 20.0, True: 12.0}
PIN_FILES = {"e99_memory3": ("e99", "method/memory_task3.pt"), "e99_stage3": ("e99", "checkpoints/stage3.pt"),
             "e99_eval3": ("e99", "eval/stage3.json"), "e99_result": ("e99", "result.json"),
             "ne90_memory1": ("ne90", "method/memory_task1.pt")}
# The references' files, pinned by the rule in env B (env A: the gamma constants below).
REFERENCE_PINS = ("seq_ft_stage0", "gpm_stage0", "gpm_stage1", "gpm_memory0")
# e99's run config may differ from plain GPM's only in its identity provenance (env B).
IDENTITY_KEYS = ("provenance.identity_reference_run", "provenance.identity_stages")


def pin_files(rule: dict) -> dict[str, tuple[str, str]]:
    """The comparator's pinned files and the prediction source's T2 artifact."""
    source = rule.get("prediction_source") or "ne90"
    out = {k: v for k, v in PIN_FILES.items() if k != "ne90_memory1"}
    out[f"{source}_memory1"] = (source, "method/memory_task1.pt")
    return out


def recorded_torch(run: Path) -> str | None:
    """The torch version a run recorded in its ``requirements.txt`` (``None`` if absent)."""
    path = Path(run) / "requirements.txt"
    if not path.is_file():
        return None
    for line in path.read_text().splitlines():
        if line.startswith("torch=="):
            return line.split("==", 1)[1].strip()
    return None


def environment_check(rule: dict, runs: list[Path]) -> dict:
    """The interpreter's torch and every reference run's recorded torch are the rule's environment's
    (passes with no ``environment`` block: env A, where the archived runs are the references)."""
    env = rule.get("environment") or {}
    if not env:
        return {"passed": True, "skipped": "the rule names no environment"}
    import torch

    wanted = env["torch"]
    problems = []
    if torch.__version__ != wanted:
        problems.append(f"interpreter torch {torch.__version__}, the rule's environment {wanted}")
    runs_seen = {}
    for run in runs:
        got = recorded_torch(run)
        runs_seen[str(run)] = got
        if got != wanted:
            problems.append(f"{run}: recorded torch {got}, the rule's environment {wanted}")
    return {"passed": not problems, "problems": problems, "environment": env.get("name"), "torch": wanted,
            "runs": runs_seen}


def config_check(seeds: list[int] | None = None) -> dict:
    """Each ``gpm_alloc_<arm>.yaml`` is ``gpm.yaml`` plus its rule and one schedule shared by all three."""
    plain_name, plain = load_method_config("gpm")
    problems, schedules = [], {}
    for arm in ARMS:
        name, kwargs = load_method_config(f"gpm_alloc_{arm}")
        kwargs = dict(kwargs)
        rule = kwargs.pop("allocation_rule", None)
        schedules[arm] = kwargs.pop("capacity_schedule", None)
        if name != plain_name:
            problems.append(f"gpm_alloc_{arm}: registry name {name}, gpm's is {plain_name}")
        if rule != arm:
            problems.append(f"gpm_alloc_{arm}: allocation_rule {rule!r}")
        if kwargs != plain:
            problems.append(f"gpm_alloc_{arm}: differs from gpm.yaml beyond the rule and schedule")
    if len({repr(s) for s in schedules.values()}) != 1:
        problems.append("the three configs carry different schedules")
    for s in seeds or []:
        if namespace(s) not in (schedules[ARMS[0]] or {}):
            problems.append(f"no schedule for {namespace(s)}")
    return {"passed": not problems, "problems": problems}


def rule_committed(root: Path, rule_file: Path) -> dict:
    """The rule is tracked and identical to HEAD (so every run's git sha pins it)."""
    rel = Path(rule_file).resolve().relative_to(Path(root).resolve())
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(rel)], cwd=root,
                             capture_output=True, text=True).returncode == 0
    unchanged = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(rel)], cwd=root).returncode == 0
    return {"passed": tracked and unchanged, "tracked": tracked, "unchanged_vs_head": unchanged, "path": str(rel)}


def _reference_pin(pins: dict, key: str, seed: int) -> str | None:
    """A reference file's SHA-256: the rule's pin when it has one (env B), else the gamma constant."""
    if key in pins:
        return pins[key]
    return {"seq_ft_stage0": SEQ_FT_STAGE0.get(seed), "gpm_stage0": GPM_STAGE.get(seed, {}).get(0),
            "gpm_stage1": GPM_STAGE.get(seed, {}).get(1), "gpm_memory0": GPM_MEMORY0.get(seed)}[key]


def archived_e99_check(seed: int, results: Path, rule: dict | None = None) -> dict:
    """e99's recorded identity against plain GPM's pinned stages, its T1 memory, and its run config
    (env A: against the current code's f = 0.7 run; env B: against plain GPM's, identity keys aside)."""
    import json

    from flowcl.experiments.lora_selection import compare_memories

    rule = rule or {}
    pins = (rule.get("pins") or {}).get(str(seed)) or {}
    e99, plain = run_dir(results, "e99", seed), run_dir(results, "plain", seed)
    problems = []
    recorded = json.loads((e99 / "result.json").read_text()).get("identity_checks") or {}
    for stage in (0, 1):
        entry = recorded.get(str(stage)) or {}
        if entry.get("passed") is not True:
            problems.append(f"stage {stage}: identity not recorded as passed")
        if entry.get("reference_sha256") != _reference_pin(pins, f"gpm_stage{stage}", seed):
            problems.append(f"stage {stage}: identity against {entry.get('reference_sha256')}, not plain GPM's pin")
    memory = compare_memories(plain / "method" / "memory_task0.pt", e99 / "method" / "memory_task0.pt")
    if not memory["passed"]:
        problems.append(f"T1 memory differs from plain GPM's on {len(memory['different'])} layers")
    if rule.get("environment"):
        cfg = config_diff(plain, e99)
        unexpected = [p for p in cfg["problems"] if not p.startswith(tuple(k + ":" for k in IDENTITY_KEYS))]
        if unexpected:
            problems.append(f"run config differs from plain GPM's beyond the identity keys: {unexpected}")
        prov = OmegaConf.to_container(OmegaConf.load(e99 / "config.yaml"), resolve=True).get("provenance") or {}
        if Path(prov.get("identity_reference_run") or "").resolve() != plain.resolve():
            problems.append(f"e99's identity reference {prov.get('identity_reference_run')}, not {plain}")
        if sorted(prov.get("identity_stages") or []) != [0, 1]:
            problems.append(f"e99's identity stages {prov.get('identity_stages')}, not [0, 1]")
        git = (e99 / "git_sha").read_text().strip() if (e99 / "git_sha").is_file() else ""
        if not git or git.endswith("-dirty"):
            problems.append(f"e99's git sha {git!r} is not a clean commit")
    else:
        current = Path(results) / f"seq_hetero__gpm_projected_adam_ne70__seed{seed}"
        cfg = config_diff(current, e99)
        if not cfg["passed"]:
            problems.append(f"run config differs from the current code's f = 0.7 run: {cfg['problems']}")
    return {"passed": not problems, "problems": problems}


def preflight(seeds: list[int], *, results: Path | None = None, smoke: bool = False, refs: Path | None = None,
              root: Path | None = None, rule_file: Path | None = None, gpu_pids=gpu_compute_pids,
              disk=free_gb) -> dict:
    """Every check before the queue's first GPU step; ``passed`` false refuses the launch."""
    root = Path(root) if root else repo_root()
    rule = load_rule(rule_file)
    results = Path(results) if results else results_root(rule)
    refs = Path(refs) if refs else results
    checks: dict = {}
    if not smoke:
        status = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True,
                                text=True, check=True).stdout
        checks["clean_tree"] = {"passed": not status.strip(), "changes": status.splitlines()}
        checks["rule_committed"] = rule_committed(root, rule_path(rule_file))
    pids = gpu_pids()
    checks["gpu_idle"] = {"passed": not pids, "compute_pids": pids}
    free = disk(results if results.exists() else results.parent)
    checks["disk"] = {"passed": free >= MIN_FREE_GB[smoke], "free_gb": round(free, 1),
                      "needed_gb": MIN_FREE_GB[smoke]}
    existing = [str(p) for s in seeds for p in [*(run_dir(results, a, s) for a in ARMS),
                                                 *(eval_path(results, s, a, r) for a in (*ARMS, "e99")
                                                   for r in (0, 1, 2)), out_dir(results, s) / "summary.json"]
                if p.exists()]
    checks["targets_absent"] = {"passed": not existing, "existing": existing}
    checks["config"] = config_check(seeds)
    files, reference_runs = {}, []
    for s in seeds:
        pins = (rule.get("pins") or {}).get(str(s)) or {}
        seq_ft, gpm = refs / f"seq_hetero__seq_ft__seed{s}", refs / f"seq_hetero__{LABELS['plain']}__seed{s}"
        reference_runs += [seq_ft, gpm]
        files.update({f"seq_ft_stage0_s{s}": (seq_ft / "checkpoints" / "stage0.pt",
                                              _reference_pin(pins, "seq_ft_stage0", s)),
                      f"gpm_stage0_s{s}": (gpm / "checkpoints" / "stage0.pt", _reference_pin(pins, "gpm_stage0", s)),
                      f"gpm_stage1_s{s}": (gpm / "checkpoints" / "stage1.pt", _reference_pin(pins, "gpm_stage1", s)),
                      f"gpm_memory0_s{s}": (gpm / "method" / "memory_task0.pt", _reference_pin(pins, "gpm_memory0", s))})
    if smoke:  # the smoke's references are smoke runs: they must exist, their hashes are their own
        missing = [str(p) for p, _ in files.values() if not p.is_file()]
        checks["references"] = {"passed": not missing, "missing": missing}
        checks["environment"] = environment_check(rule, reference_runs)
        return {"passed": all(c["passed"] for c in checks.values()), "seeds": seeds, "smoke": smoke,
                "checks": checks}
    for s in seeds:
        pins = (rule.get("pins") or {}).get(str(s)) or {}
        for key, (role, rel) in pin_files(rule).items():
            files[f"{key}_s{s}"] = (run_dir(results, role, s) / rel, pins.get(key))
        reference_runs.append(run_dir(results, "e99", s))
    checks["references"] = _hashes(files)
    checks["environment"] = environment_check(rule, reference_runs)
    for s in seeds:
        checks[f"archived_e99_s{s}"] = archived_e99_check(s, results, rule)
        p = predict(s, results=results, rule=rule)
        registered = (rule.get("predictions") or {}).get(str(s)) or {}
        mismatched = [a for a in ARMS if (registered.get(a) or {}).get("k_hash") != p["arms"][a]["k_hash"]]
        checks[f"prediction_s{s}"] = {"passed": p["passed"] and not mismatched,
                                      "checks": {k: c["passed"] for k, c in p["checks"].items()},
                                      "unregistered_or_changed": mismatched}
    return {"passed": all(c["passed"] for c in checks.values()), "seeds": seeds, "smoke": smoke, "checks": checks}
