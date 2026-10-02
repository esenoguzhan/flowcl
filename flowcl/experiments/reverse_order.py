"""C2, the reverse-order control (``configs/analysis/reverse_order.yaml``).

Reads the three reverse-order runs of one seed (seq_ft, plain GPM, adaptive GPM) and their sequence
reports, enforces the rule's validity conditions itself (a sequence report is written whatever its
checks say), judges the two registered cells of plain GPM, and writes ``summary.json`` atomically in
every case. :func:`classify` is the verdict (first match); :func:`judge_cell` one cell;
:func:`joint_contrast` the adaptive-minus-plain contrast between drops, a bootstrap over matched
episode indices with all four outcomes resampled jointly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from flowcl.utils.libero_paths import repo_root
from flowcl.utils.run import atomic_write_text, file_sha256, git_sha

INCOMPLETE, INVALID, INCONCLUSIVE = "incomplete", "invalid", "inconclusive"
BOTH, POSITION, TASK, NEITHER = "both", "position", "task", "neither"
EXIT_CODES = {BOTH: 0, POSITION: 0, TASK: 0, NEITHER: 0, INCONCLUSIVE: 0, INVALID: 3, INCOMPLETE: 4}
VERDICT_TEXT = {
    INCOMPLETE: "A seq_ft or plain-GPM input is missing or malformed: no verdict.",
    INVALID: "The plain-GPM verdict's validity failed (see the checks): no verdict.",
    INCONCLUSIVE: "Plain GPM did not learn a task a cell needs; that cell cannot speak to forgetting.",
    BOTH: "Plain GPM forgot both Goal (position 2) and Object (position 3) in the reverse order.",
    POSITION: "Plain GPM forgot Goal at position 2 (after Object), not Object at position 3: the "
              "position-2 pattern recurs in this curriculum (not separable from the Goal-Object pair).",
    TASK: "Plain GPM forgot Object at position 3 (after Spatial), not Goal at position 2: the "
          "Object pattern recurs with a different successor.",
    NEITHER: "Plain GPM forgot neither registered cell in the reverse order.",
}
FORGOTTEN, RETAINED, NOT_LEARNED = "forgotten", "retained", "not_learned"
ROLES = ("seq_ft", "plain", "adaptive")
N_TASKS = 4


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else repo_root() / "configs" / "analysis" / "reverse_order.yaml"
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def method_spec(cfg: dict, role: str) -> tuple[str, dict, str]:
    """``(registry name, kwargs, display label)`` of a role's method config file."""
    from flowcl.data.config import load_method_config
    from flowcl.methods.base import build_method
    from flowcl.train.continual import method_label

    name, kwargs = load_method_config(cfg["methods"][role])
    return name, dict(kwargs or {}), method_label(build_method(name, **(kwargs or {})), name)


def run_names(cfg: dict, seed: int, curriculum: str | None = None) -> dict[str, str]:
    from flowcl.train.continual import continual_run_id

    curriculum = curriculum or cfg["curriculum"]
    return {role: continual_run_id(method_spec(cfg, role)[2], curriculum, seed) for role in ROLES}


# ---- pure pieces --------------------------------------------------------------------------------------


def thresholds(seqft_diagonal: list[float], margin_pp: float) -> list[float]:
    """``theta_j = R_seqft[j][j] - margin``, full precision (the sequence report's rule)."""
    return [r - margin_pp / 100.0 for r in seqft_diagonal]


def _cell(value: float, low: float, high: float, theta: float) -> dict:
    return {"value": value, "low": low, "high": high, "theta": theta,
            "borderline": bool(low < theta <= high)}


def judge_cell(diagonal: dict, retention: dict, theta: float) -> dict:
    """One registered cell: ``diagonal`` and ``retention`` are ``{value, low, high}`` of plain GPM's
    ``R[j][j]`` and its later ``R[k][j]``."""
    d = _cell(diagonal["value"], diagonal["low"], diagonal["high"], theta)
    r = _cell(retention["value"], retention["low"], retention["high"], theta)
    if d["value"] < theta:
        state = NOT_LEARNED
    elif r["value"] < theta:
        state = FORGOTTEN
    else:
        state = RETAINED
    return {"state": state, "diagonal": d, "retention": r}


def classify(primary: str, cells: dict | None) -> dict:
    """First match wins: ``primary`` is ``ok``, ``incomplete`` or ``invalid``."""
    if primary == INCOMPLETE:
        verdict = INCOMPLETE
    elif primary == INVALID:
        verdict = INVALID
    elif any(c["state"] == NOT_LEARNED for c in cells.values()):
        verdict = INCONCLUSIVE
    else:
        pos, task = cells["position"]["state"] == FORGOTTEN, cells["task"]["state"] == FORGOTTEN
        verdict = BOTH if pos and task else POSITION if pos else TASK if task else NEITHER
    return {"verdict": verdict, "text": VERDICT_TEXT[verdict], "exit_code": EXIT_CODES[verdict]}


def joint_contrast(a_before, a_after, p_before, p_after, bootstrap: dict) -> dict:
    """``(mean a_before - mean a_after) - (mean p_before - mean p_after)`` with a percentile CI from a
    bootstrap over matched episode indices: one index draw resamples all four outcome lists."""
    arrays = [np.asarray(x, dtype=np.float64) for x in (a_before, a_after, p_before, p_after)]
    n = len(arrays[0])
    if any(len(x) != n for x in arrays) or n == 0:
        raise ValueError("the four outcome lists must be equally long and non-empty")
    value = (arrays[0].mean() - arrays[1].mean()) - (arrays[2].mean() - arrays[3].mean())
    rng = np.random.default_rng(int(bootstrap["seed"]))
    idx = rng.integers(0, n, size=(int(bootstrap["n_resamples"]), n))
    stats = ((arrays[0][idx].mean(1) - arrays[1][idx].mean(1))
             - (arrays[2][idx].mean(1) - arrays[3][idx].mean(1)))
    alpha = (1.0 - float(bootstrap["confidence"])) / 2.0
    return {"diff": float(value), "low": float(np.quantile(stats, alpha)),
            "high": float(np.quantile(stats, 1.0 - alpha)), "n": n}


def _paired(a, b, bootstrap: dict) -> dict:
    from flowcl.analysis.metrics import paired_difference_ci

    est = paired_difference_ci(a, b, seed=bootstrap["seed"], n_bootstrap=bootstrap["n_resamples"],
                               confidence=bootstrap["confidence"])
    return {"diff": est.value, "low": est.low, "high": est.high, "n": len(a)}


# ---- loading ------------------------------------------------------------------------------------------


def _load_view(run_dir: Path, n_episodes: int):
    """The run's view, or ``(None, problems)`` if anything a verdict needs is missing or malformed."""
    from flowcl.experiments.sequence_report import load_run

    problems = []
    if not (run_dir / "result.json").is_file():
        return None, [f"{run_dir.name}: no result.json"]
    try:
        view = load_run(run_dir)
    except (OSError, ValueError, KeyError) as exc:
        return None, [f"{run_dir.name}: {exc}"]
    if view.n_tasks != N_TASKS:
        problems.append(f"{run_dir.name}: {view.n_tasks} tasks")
    for i in range(view.n_tasks):
        by_task = view.evals[i].by_task()
        for key in view.task_keys:
            entry = by_task.get(key)
            if entry is None or entry.n_rollouts != n_episodes or len(entry.seeds) != n_episodes:
                problems.append(f"{run_dir.name}: stage {i} {key.split('/')[0]} has "
                                f"{None if entry is None else entry.n_rollouts} rollouts, expected {n_episodes}")
    return (None if problems else view), problems


def _est(view, i: int, j: int) -> dict:
    e = view.cell(i, j).estimate
    return {"value": e.value, "low": e.low, "high": e.high}


# ---- checks -------------------------------------------------------------------------------------------


def production_check(run_dir: Path, cfg: dict, role: str, seed: int, smoke: bool) -> dict:
    """Curriculum, seed, namespace, recipe, evaluation episodes, method config, clean tree."""
    conf = OmegaConf.to_container(OmegaConf.load(run_dir / "config.yaml"), resolve=True)
    name, kwargs, _ = method_spec(cfg, role)
    expected_ns = f"{cfg['curriculum']}__seq_ft__seed{seed}"
    want = cfg["smoke"] if smoke else cfg["production"]
    problems = {}
    if conf["curriculum"]["name"] != cfg["curriculum"]:
        problems["curriculum"] = conf["curriculum"]["name"]
    if conf.get("seed") != seed:
        problems["seed"] = conf.get("seed")
    if conf.get("seed_namespace_run_id") != expected_ns:
        problems["namespace"] = conf.get("seed_namespace_run_id")
    for k, v in want["train"].items():
        if conf["train"].get(k) != v:
            problems[f"train.{k}"] = (conf["train"].get(k), v)
    if conf["eval"].get("n_episodes") != want["n_episodes"]:
        problems["eval.n_episodes"] = (conf["eval"].get("n_episodes"), want["n_episodes"])
    method = {k: v for k, v in conf["method"].items() if k != "display_name"}
    if method != {"name": name, **kwargs}:
        problems["method"] = method
    sha = (run_dir / "git_sha").read_text().strip() if (run_dir / "git_sha").is_file() else None
    clean = sha is not None and not sha.endswith("-dirty")
    if not clean and not smoke:
        problems["git_sha"] = sha
    return {"passed": not problems, "problems": problems, "git_sha": sha,
            "dirty_excused": bool(smoke and not clean), "smoke": smoke}


def stats_check(run_dirs: dict[str, Path], expected_task: str) -> dict:
    """Same stats fingerprint across runs, fitted on the curriculum's first task, and every stage
    checkpoint's ``stats_fingerprint`` equal to its run's ``stats.json``."""
    import torch

    fingerprints, problems = {}, {}
    for role, d in run_dirs.items():
        stats = json.loads((d / "stats.json").read_text())
        fingerprints[role] = stats["fingerprint"]
        if stats.get("fitted_on_task_id") != expected_task:
            problems[f"{role}.fitted_on"] = stats.get("fitted_on_task_id")
        for ckpt in sorted((d / "checkpoints").glob("stage*.pt")):
            fp = torch.load(ckpt, map_location="cpu", weights_only=False).get("stats_fingerprint")
            if fp != stats["fingerprint"]:
                problems[f"{role}.{ckpt.name}"] = fp
    if len(set(fingerprints.values())) > 1:
        problems["fingerprints"] = fingerprints
    return {"passed": not problems, "problems": problems, "fingerprints": fingerprints}


def seed_pairing(a, b) -> dict:
    """Identical per-episode seeds for every (stage, task) both runs evaluated."""
    mismatched = []
    for i in sorted(set(a.evals) & set(b.evals)):
        ta, tb = a.evals[i].by_task(), b.evals[i].by_task()
        for key in sorted(set(ta) | set(tb)):
            if key not in ta or key not in tb or list(ta[key].seeds) != list(tb[key].seeds):
                mismatched.append(f"stage {i} {key.split('/')[0]}")
    return {"passed": not mismatched, "mismatched": mismatched}


def report_check(report_path: Path, smoke: bool) -> dict:
    """Every provenance check of a sequence report passed (``clean_git_sha`` excused in smoke)."""
    if not report_path.is_file():
        return {"passed": False, "missing": str(report_path)}
    report = json.loads(report_path.read_text())
    failed = []
    for name, check in report.get("provenance_checks", {}).items():
        if check is None:
            failed.append(f"{name}: absent")
        elif not check.get("passed"):
            if not (smoke and name == "clean_git_sha"):
                failed.append(name)
    return {"passed": not failed, "failed": failed, "checks": sorted(report.get("provenance_checks", {}))}


def threshold_check(report_path: Path, theta: list[float], seqft_result: Path) -> dict:
    report = json.loads(report_path.read_text())
    block = report.get("threshold_block", {})
    same = (len(block.get("thresholds", [])) == len(theta)
            and all(abs(a - b) <= 1e-12 for a, b in zip(block.get("thresholds", []), theta)))
    sha = file_sha256(seqft_result)
    return {"passed": bool(same and block.get("reference_result_sha256") == sha),
            "report_thresholds": block.get("thresholds"), "summary_thresholds": theta,
            "reference_result_sha256": sha, "report_reference_sha256": block.get("reference_result_sha256")}


def identity_check(run_dir: Path) -> dict:
    out = {}
    for stage in (0, 1):
        path = run_dir / f"identity_stage{stage}.json"
        out[str(stage)] = json.loads(path.read_text())["passed"] if path.is_file() else None
    return {"passed": all(v is True for v in out.values()), "stages": out}


def mechanism(run_dir: Path, stages=(1, 2)) -> dict | None:
    """The Goal and Object memory updates per half, from the run's latest memory history."""
    from flowcl.analysis.subspace import load_bases
    from flowcl.experiments.sequence_report import capacity_by_stage

    memories = sorted((run_dir / "method").glob("memory_task*.pt"),
                      key=lambda p: int(p.stem.removeprefix("memory_task")))
    if not memories:
        return None
    _, meta = load_bases(memories[-1])
    history = meta["memory_history"]
    capacity = capacity_by_stage(history)
    out = {"capacity": capacity}
    for stage in stages:
        rows = history.get(str(stage)) or history.get(stage)
        if rows is None:
            continue
        per_half = {}
        for half, prefix in (("trunk", "trunk."), ("decoder", "flow_head.")):
            sel = [r for n, r in rows.items() if n.startswith(prefix)]
            if not sel:
                continue
            per_half[half] = {
                "median_proj_energy_fraction": float(np.median([r["proj_energy_fraction"] for r in sel])),
                "median_target_fraction": (float(np.median([r["target_fraction"] for r in sel]))
                                           if all("target_fraction" in r for r in sel) else None),
                "median_captured_energy_fraction": float(np.median([r["captured_energy_fraction"] for r in sel])),
                "k_added": int(sum(r["k_added"] for r in sel)),
                "median_rho_after": float(np.median([r["rho_after"] for r in sel])),
            }
        out[str(stage)] = per_half
    return out


def drops(view, bootstrap: dict) -> dict:
    """Goal ``R[1][1] - R[2][1]`` and Object ``R[2][2] - R[3][2]``, rollout-paired."""
    return {
        "goal": _paired(view.cell(1, 1).successes, view.cell(2, 1).successes, bootstrap),
        "object": _paired(view.cell(2, 2).successes, view.cell(3, 2).successes, bootstrap),
    }


# ---- the summary --------------------------------------------------------------------------------------


def run_summary(cfg: dict, seed: int, *, results_root: Path | None = None, out: Path | None = None,
                smoke: bool = False) -> dict:
    """Judge one seed and write ``summary.json`` atomically (see the module docstring)."""
    root = Path(results_root) if results_root else repo_root() / "results"
    out = Path(out) if out else root / cfg["out"].format(seed=seed)
    bootstrap = OmegaConf.to_container(
        OmegaConf.load(repo_root() / "configs" / "eval" / "libero_eval.yaml"), resolve=True)["bootstrap"]
    names = run_names(cfg, seed)
    dirs = {role: root / name for role, name in names.items()}
    n_episodes = (cfg["smoke"] if smoke else cfg["production"])["n_episodes"]
    reports = {role: root / cfg["reports"][role].format(seed=seed) / "report.json" for role in ("plain", "adaptive")}

    summary: dict = {"git_sha": git_sha(), "config": cfg, "seed": seed, "smoke": smoke,
                     "runs": {r: str(d) for r, d in dirs.items()}}
    views, problems = {}, []
    for role in ("seq_ft", "plain"):
        views[role], p = _load_view(dirs[role], n_episodes)
        problems += p
    primary, checks, cells, theta = "ok", {}, None, None
    if problems:
        primary = INCOMPLETE
    else:
        seqft, plain = views["seq_ft"], views["plain"]
        theta = thresholds([seqft.cell(j, j).estimate.value for j in range(N_TASKS)], cfg["margin_pp"])
        try:
            checks = {
                "production_seq_ft": production_check(dirs["seq_ft"], cfg, "seq_ft", seed, smoke),
                "production_plain": production_check(dirs["plain"], cfg, "plain", seed, smoke),
                "statistics": stats_check({r: dirs[r] for r in ("seq_ft", "plain")}, seqft.task_keys[0]),
                "t1_pairing": {"passed": bool((plain.result.get("t1_pairing") or {}).get("passed"))},
                "episode_pairing": seed_pairing(seqft, plain),
                "sequence_report": report_check(reports["plain"], smoke),
            }
            if checks["sequence_report"].get("missing"):
                primary, problems = INCOMPLETE, [checks["sequence_report"]["missing"]]
            else:
                checks["thresholds"] = threshold_check(reports["plain"], theta, dirs["seq_ft"] / "result.json")
                if not all(c["passed"] for c in checks.values()):
                    primary = INVALID
        except (OSError, ValueError, KeyError) as exc:
            primary, problems = INCOMPLETE, [f"malformed input: {exc}"]
        if primary != INCOMPLETE:
            cells = {name: judge_cell(_est(plain, c["task"], c["task"]), _est(plain, c["stage"], c["task"]),
                                      theta[c["task"]])
                     for name, c in cfg["cells"].items()}
    verdict = classify(primary, cells)
    labels = []
    if cells:
        for name, c in cells.items():
            if c["retention"]["borderline"]:
                labels.append(f"borderline_{name}_retention")
            if c["diagonal"]["borderline"]:
                labels.append(f"borderline_{name}_diagonal")
    if theta is not None:
        seqft = views["seq_ft"]
        for j in range(N_TASKS):
            if seqft.cell(j, j).estimate.value < float(cfg["weak_reference_below"]):
                labels.append(f"weak_reference_task{j}")

    adaptive = adaptive_comparison(cfg, seed, dirs, views, reports, n_episodes, smoke, bootstrap)
    reported = {}
    if views.get("plain") is not None and views.get("seq_ft") is not None:
        reported = {
            "drops": {"plain": drops(views["plain"], bootstrap), "seq_ft": drops(views["seq_ft"], bootstrap)},
            "mechanism": {"plain": mechanism(dirs["plain"])},
            "matrices": {r: [[_est(views[r], i, j)["value"] for j in range(N_TASKS)] for i in range(N_TASKS)]
                         for r in ("seq_ft", "plain")},
            "task_keys": views["plain"].task_keys,
            "forward_unpaired": forward_numbers(cfg, seed, root),
        }
    summary.update({"verdict": {**verdict, "labels": labels, "thresholds": theta, "cells": cells},
                    "primary_status": primary, "problems": problems, "checks": checks,
                    "adaptive_comparison": adaptive, "reported": reported})
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out / "summary.json", json.dumps(summary, indent=2, default=str) + "\n")
    print(f"[flowcl] reverse order seed {seed}: {verdict['verdict']} {labels} "
          f"(adaptive comparison: {adaptive['status']})", flush=True)
    return summary


def adaptive_comparison(cfg, seed, dirs, views, reports, n_episodes, smoke, bootstrap) -> dict:
    """Its own completeness and validity; never changes the plain verdict."""
    d = dirs["adaptive"]
    view, problems = _load_view(d, n_episodes) if d.is_dir() else (None, [f"{d.name}: no run"])
    identity = identity_check(d) if d.is_dir() else {"passed": False, "stages": {}}
    # Missing inputs -> incomplete, whatever the reason (an identity failure stops the run before
    # evaluation; the failed identity file is recorded). Present evidence that fails -> invalid.
    if view is None or not reports["adaptive"].is_file():
        return {"status": INCOMPLETE, "identity": identity,
                "problems": problems + ([] if reports["adaptive"].is_file() else [str(reports["adaptive"])])}
    checks = {
        "identity": identity,
        "production": production_check(d, cfg, "adaptive", seed, smoke),
        "sequence_report": report_check(reports["adaptive"], smoke),
    }
    try:
        checks["statistics"] = stats_check({r: dirs[r] for r in ROLES}, view.task_keys[0])
    except (OSError, KeyError) as exc:
        checks["statistics"] = {"passed": False, "problems": str(exc)}
    if views.get("seq_ft") is not None:
        checks["episode_pairing"] = seed_pairing(views["seq_ft"], view)
    status = "valid" if all(c["passed"] for c in checks.values()) else INVALID
    out = {"status": status, "checks": checks}
    plain = views.get("plain")
    if status == "valid" and plain is not None:
        out["drops"] = drops(view, bootstrap)
        out["contrasts"] = {
            "goal_drop": joint_contrast(view.cell(1, 1).successes, view.cell(2, 1).successes,
                                        plain.cell(1, 1).successes, plain.cell(2, 1).successes, bootstrap),
            "object_drop": joint_contrast(view.cell(2, 2).successes, view.cell(3, 2).successes,
                                          plain.cell(2, 2).successes, plain.cell(3, 2).successes, bootstrap),
            "final_retention": {str(j): _paired(view.cell(3, j).successes, plain.cell(3, j).successes, bootstrap)
                                for j in range(N_TASKS - 1)},
        }
        out["mechanism"] = mechanism(d)
        out["matrix"] = [[_est(view, i, j)["value"] for j in range(N_TASKS)] for i in range(N_TASKS)]
    return out


def forward_numbers(cfg: dict, seed: int, root: Path) -> dict:
    """The forward runs' matrices beside the reverse ones: UNPAIRED (different namespaces)."""
    from flowcl.experiments.sequence_report import load_run

    out = {"note": "unpaired: the forward runs use a different seed namespace and task order"}
    for role, name in run_names(cfg, seed, cfg["forward_curriculum"]).items():
        path = root / name
        if (path / "result.json").is_file():
            v = load_run(path)
            out[role] = {"task_keys": v.task_keys,
                         "matrix": [[v.cell(i, j).estimate.value for j in range(v.n_tasks)]
                                    for i in range(v.n_tasks)]}
    return out
