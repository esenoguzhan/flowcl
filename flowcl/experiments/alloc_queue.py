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
"""

from __future__ import annotations

import subprocess
from pathlib import Path

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


def archived_e99_check(seed: int, results: Path) -> dict:
    """e99's recorded identity against plain GPM's pinned stages, its T1 memory, and its run config."""
    import json

    from flowcl.experiments.lora_selection import compare_memories

    e99, plain = run_dir(results, "e99", seed), run_dir(results, "plain", seed)
    current = Path(results) / f"seq_hetero__gpm_projected_adam_ne70__seed{seed}"
    problems = []
    recorded = json.loads((e99 / "result.json").read_text()).get("identity_checks") or {}
    for stage in (0, 1):
        entry = recorded.get(str(stage)) or {}
        if entry.get("passed") is not True:
            problems.append(f"stage {stage}: identity not recorded as passed")
        if entry.get("reference_sha256") != GPM_STAGE.get(seed, {}).get(stage):
            problems.append(f"stage {stage}: identity against {entry.get('reference_sha256')}, not plain GPM's pin")
    memory = compare_memories(plain / "method" / "memory_task0.pt", e99 / "method" / "memory_task0.pt")
    if not memory["passed"]:
        problems.append(f"T1 memory differs from plain GPM's on {len(memory['different'])} layers")
    cfg = config_diff(current, e99)
    if not cfg["passed"]:
        problems.append(f"run config differs from the current code's f = 0.7 run: {cfg['problems']}")
    return {"passed": not problems, "problems": problems}


def preflight(seeds: list[int], *, results: Path | None = None, smoke: bool = False, refs: Path | None = None,
              root: Path | None = None, rule_file: Path | None = None, gpu_pids=gpu_compute_pids,
              disk=free_gb) -> dict:
    """Every check before the queue's first GPU step; ``passed`` false refuses the launch."""
    root = Path(root) if root else repo_root()
    results = Path(results) if results else root / "results"
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
    files = {}
    for s in seeds:
        seq_ft, gpm = refs / f"seq_hetero__seq_ft__seed{s}", refs / f"seq_hetero__{LABELS['plain']}__seed{s}"
        files.update({f"seq_ft_stage0_s{s}": (seq_ft / "checkpoints" / "stage0.pt", SEQ_FT_STAGE0.get(s)),
                      f"gpm_stage0_s{s}": (gpm / "checkpoints" / "stage0.pt", GPM_STAGE.get(s, {}).get(0)),
                      f"gpm_stage1_s{s}": (gpm / "checkpoints" / "stage1.pt", GPM_STAGE.get(s, {}).get(1)),
                      f"gpm_memory0_s{s}": (gpm / "method" / "memory_task0.pt", GPM_MEMORY0.get(s))})
    if smoke:  # the smoke's references are smoke runs: they must exist, their hashes are their own
        missing = [str(p) for p, _ in files.values() if not p.is_file()]
        checks["references"] = {"passed": not missing, "missing": missing}
        return {"passed": all(c["passed"] for c in checks.values()), "seeds": seeds, "smoke": smoke,
                "checks": checks}
    rule = load_rule(rule_file)
    for s in seeds:
        pins = (rule.get("pins") or {}).get(str(s)) or {}
        for key, (role, rel) in PIN_FILES.items():
            files[f"{key}_s{s}"] = (run_dir(results, role, s) / rel, pins.get(key))
    checks["references"] = _hashes(files)
    for s in seeds:
        checks[f"archived_e99_s{s}"] = archived_e99_check(s, results)
        p = predict(s, results=results, rule=rule)
        registered = (rule.get("predictions") or {}).get(str(s)) or {}
        mismatched = [a for a in ARMS if (registered.get(a) or {}).get("k_hash") != p["arms"][a]["k_hash"]]
        checks[f"prediction_s{s}"] = {"passed": p["passed"] and not mismatched,
                                      "checks": {k: c["passed"] for k, c in p["checks"].items()},
                                      "unregistered_or_changed": mismatched}
    return {"passed": all(c["passed"] for c in checks.values()), "seeds": seeds, "smoke": smoke, "checks": checks}
