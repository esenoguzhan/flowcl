"""C1 follow-up: D = e99 − adaptive at Object after Goal, resolved with repeated initial states.

C1 (:mod:`flowcl.experiments.high_protection`) was ``unresolved`` on every seed: at 50 rollouts the
paired CIs were too wide for the 10 pp margin. LIBERO has only 50 fixed initial states per task, so this
second-stage test re-evaluates the same stage-2 checkpoints on Object in several repetitions of those
states (``scripts/evaluate.py``): repetition 0 under the C1 namespace (C1's own episodes), repetition
``r >= 1`` under ``<namespace>__rep<r>`` (same states, fresh sampling noise). The rule is
``configs/analysis/high_protection_followup.yaml``:

* **gates** — repetition 0 reproduces C1's stored cell exactly, the checkpoints and eval reports are
  the expected ones, the arms are paired within a repetition and the noise differs across them, and
  the C1 seed report is valid;
* **verdict** — :func:`flowcl.experiments.high_protection.classify` with C1's checks and D whose CI
  resamples initial states (:func:`state_paired_ci`);
* **replication** — :func:`classify_replication`, which has the all-``unresolved`` row C1 lacked.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from flowcl.utils.libero_paths import repo_root
from flowcl.utils.run import atomic_write_text, file_sha256, git_sha

REPLICATION = {
    "inconclusive_invalid": "Inconclusive: at least one seed is invalid.",
    "protection_suffices_replicated": "Every seed: a flat 0.99 target reproduces the transition "
                                      "benefit within the margin.",
    "flat_control_inferior_replicated": "Every seed: the flat control is worse than adaptive by "
                                        "more than the margin.",
    "unresolved_replicated": "Every seed: unresolved at 150 rollouts per arm.",
    "mixed": "Mixed: the seeds' verdicts differ.",
}


def rule_path(path: str | Path | None = None) -> Path:
    return Path(path) if path else repo_root() / "configs" / "analysis" / "high_protection_followup.yaml"


def load_config(path: str | Path | None = None) -> dict:
    return OmegaConf.to_container(OmegaConf.load(rule_path(path)), resolve=True)


def c1_config(cfg: dict) -> dict:
    from flowcl.experiments.high_protection import load_config as load_c1

    return load_c1(repo_root() / "configs" / "analysis" / f"{cfg['c1_rule']}.yaml")


def namespace(c1: dict, seed: int) -> str:
    return f"{c1['curriculum']}__seq_ft__seed{seed}"


def rep_namespace(c1: dict, cfg: dict, seed: int, rep: int) -> str:
    """Repetition 0 is the C1 namespace itself; ``r >= 1`` appends the rule's suffix."""
    base = namespace(c1, seed)
    return base if rep == 0 else base + cfg["rep_suffix"].format(rep=rep)


def eval_path(out_root: Path, cfg: dict, seed: int, arm: str, rep: int) -> Path:
    return out_root / cfg["out_dir"] / f"seed{seed}" / f"{arm}.rep{rep}.eval.json"


def state_paired_ci(a, b, bootstrap: dict) -> dict:
    """``mean(a) − mean(b)`` over ``(repetitions, states)`` outcome arrays; the CI resamples states.

    Each state's mean difference across repetitions is one observation, so repetitions of a state
    stay together; with one repetition this is :func:`~flowcl.analysis.metrics.paired_difference_ci`.
    """
    import numpy as np

    from flowcl.analysis.metrics import bootstrap_ci

    left, right = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError(f"expected equal (repetitions, states) arrays, got {left.shape} and {right.shape}")
    est = bootstrap_ci((left - right).mean(axis=0), seed=bootstrap["seed"],
                       n_bootstrap=bootstrap["n_resamples"], confidence=bootstrap["confidence"])
    return {"diff": est.value, "low": est.low, "high": est.high, "n_states": left.shape[1],
            "repetitions": left.shape[0]}


# ---- gates ---------------------------------------------------------------------------------------


def _cell(report, task_key: str):
    cell = report.by_task().get(task_key)
    if cell is None:
        raise KeyError(f"no {task_key} cell")
    return cell


def reproduction_check(new, stored, n: int) -> dict:
    """The first ``n`` episodes of ``new`` equal ``stored``'s exactly (successes, steps, seeds)."""
    problems = []
    if len(new.successes) < n or len(stored.successes) < n:
        problems.append(f"too few episodes: new {len(new.successes)}, stored {len(stored.successes)}, need {n}")
    else:
        for key in ("successes", "n_steps", "seeds"):
            a, b = list(getattr(new, key))[:n], list(getattr(stored, key))[:n]
            differing = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
            if differing:
                problems.append(f"{key} differ at episodes {differing[:10]}")
    return {"passed": not problems, "n_compared": n, "problems": problems}


def provenance_check(checkpoint: Path, expected_run: str, report, expected_ns: str, stage: int,
                     n_episodes: int, task_key: str) -> dict:
    """The checkpoint is the expected run's stage, and the eval report is under the namespace."""
    problems = []
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("run_id") != expected_run:
        problems.append(f"checkpoint run_id {payload.get('run_id')!r} != {expected_run!r}")
    if payload.get("stage") != stage:
        problems.append(f"checkpoint stage {payload.get('stage')} != {stage}")
    if report.run_id != expected_ns:
        problems.append(f"eval run_id {report.run_id!r} != {expected_ns!r}")
    if report.stage != stage:
        problems.append(f"eval stage {report.stage} != {stage}")
    cell = _cell(report, task_key)
    if len(cell.successes) != n_episodes:
        problems.append(f"{len(cell.successes)} rollouts, expected {n_episodes}")
    return {"passed": not problems, "problems": problems, "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_sha256(checkpoint)}


def c1_report_check(c1_report: dict, rule_sha: str, expected_runs: dict) -> dict:
    problems = []
    if c1_report.get("smoke") is not False:
        problems.append("C1 report is a smoke report")
    if (c1_report.get("inputs") or {}).get("rule", {}).get("sha256") != rule_sha:
        problems.append("C1 report was written under another rule")
    if c1_report.get("runs") != expected_runs:
        problems.append("C1 report names other runs")
    checks = c1_report.get("checks") or {}
    if not checks or not all(checks.values()):
        problems.append(f"C1 checks not all true: {checks}")
    return {"passed": not problems, "problems": problems}


# ---- verdicts (pure) -----------------------------------------------------------------------------


def classify_replication(verdicts: dict[str, str]) -> dict:
    values = set(verdicts.values())
    if any(v.startswith("invalid") for v in values):
        outcome = "inconclusive_invalid"
    elif values == {"protection_suffices"}:
        outcome = "protection_suffices_replicated"
    elif values == {"flat_control_inferior"}:
        outcome = "flat_control_inferior_replicated"
    elif values == {"unresolved"}:
        outcome = "unresolved_replicated"
    else:
        outcome = "mixed"
    return {"replication": outcome, "text": REPLICATION[outcome], "verdicts": dict(verdicts)}


# ---- the report ----------------------------------------------------------------------------------


def build_report(cfg: dict, seed: int, *, runs_root: Path | None = None, out_root: Path | None = None,
                 smoke: bool = False, rule: Path | None = None) -> dict:
    import numpy as np

    from flowcl.envs.evaluation import EvaluationReport
    from flowcl.experiments.high_protection import classify, run_names

    runs_root = Path(runs_root) if runs_root else repo_root() / "results"
    out_root = Path(out_root) if out_root else runs_root
    c1 = c1_config(cfg)
    names = run_names(c1, seed)
    stage, _ = cfg["cell"]
    key = cfg["task_key"]
    n_episodes = cfg["smoke"]["n_episodes"] if smoke else cfg["n_episodes"]
    reps = list(range(cfg["smoke"]["repetitions"] if smoke else cfg["repetitions"]))
    bootstrap = OmegaConf.to_container(
        OmegaConf.load(repo_root() / "configs" / "eval" / "libero_eval.yaml"), resolve=True)["bootstrap"]
    c1_path = runs_root / c1["out"].format(seed=seed)

    report: dict = {"git_sha": git_sha(), "seed": seed, "smoke": smoke, "config": cfg, "runs": names,
                    "rule": {"path": str(rule_path(rule)), "sha256": file_sha256(rule_path(rule))},
                    "c1_report": {"path": str(c1_path),
                                  "sha256": file_sha256(c1_path) if c1_path.is_file() else None},
                    "bootstrap": bootstrap, "gates": {}}
    gates = report["gates"]
    cells: dict[str, dict[int, object]] = {arm: {} for arm in cfg["arms"]}
    for arm in cfg["arms"]:
        checkpoint = runs_root / names[arm] / "checkpoints" / f"stage{stage}.pt"
        for r in reps:
            path = eval_path(out_root, cfg, seed, arm, r)
            try:
                new = EvaluationReport.load(path)
                cells[arm][r] = _cell(new, key)
                gates[f"provenance_{arm}_rep{r}"] = {
                    **provenance_check(checkpoint, names[arm], new, rep_namespace(c1, cfg, seed, r), stage,
                                       n_episodes, key),
                    "eval_sha256": file_sha256(path)}
                if r == 0:
                    stored = EvaluationReport.load(runs_root / names[arm] / "eval" / f"stage{stage}.json")
                    gates[f"reproduction_{arm}"] = reproduction_check(cells[arm][0], _cell(stored, key), n_episodes)
            except (OSError, ValueError, KeyError) as exc:
                gates[f"inputs_{arm}_rep{r}"] = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
    complete = all(len(cells[arm]) == len(reps) for arm in cfg["arms"])
    if complete:
        a_arm, b_arm = cfg["arms"]
        gates["episode_pairing"] = {"passed": all(list(cells[a_arm][r].seeds) == list(cells[b_arm][r].seeds)
                                                  for r in reps)}
        gates["fresh_noise"] = {"passed": all(
            len({tuple(cells[arm][r].seeds) for r in reps}) == len(reps) for arm in cfg["arms"])}
    try:
        c1_report = json.loads(c1_path.read_text())
        gates["c1_report"] = c1_report_check(c1_report, cfg["c1_rule_sha256"], names)
    except (OSError, ValueError) as exc:
        c1_report, gates["c1_report"] = None, {"passed": False, "error": f"{type(exc).__name__}: {exc}"}

    if not (complete and all(g.get("passed") for g in gates.values())):
        report["verdict"] = {"verdict": "invalid_followup", "flags": [],
                             "text": "Invalid follow-up: a gate failed or lacks evidence."}
        return report
    outcomes = {arm: np.array([cells[arm][r].successes for r in reps], dtype=np.float64) for arm in cfg["arms"]}
    e99, adaptive = outcomes["e99"], outcomes["adaptive"]
    D = state_paired_ci(e99, adaptive, bootstrap)
    report["verdict"] = classify(c1_report["checks"], D, c1["noninferiority_margin"])

    def arm_estimate(x):
        from flowcl.analysis.metrics import bootstrap_ci

        est = bootstrap_ci(x.mean(axis=0), seed=bootstrap["seed"], n_bootstrap=bootstrap["n_resamples"],
                           confidence=bootstrap["confidence"])
        return {"value": est.value, "low": est.low, "high": est.high}

    report["primary"] = {"cell": cfg["cell"], "n_states": n_episodes, "repetitions": len(reps), "D": D,
                         "G_from_c1": c1_report["primary"]["G"], "D_c1": c1_report["primary"]["D"],
                         "e99": arm_estimate(e99), "adaptive": arm_estimate(adaptive)}
    report["reported"] = {
        "D_rep0": state_paired_ci(e99[:1], adaptive[:1], bootstrap),
        "D_new_reps": state_paired_ci(e99[1:], adaptive[1:], bootstrap) if len(reps) > 1 else None,
        "D_per_rep": [state_paired_ci(e99[r:r + 1], adaptive[r:r + 1], bootstrap)["diff"] for r in reps],
        "states_same_outcome_every_rep": {arm: float((x.min(axis=0) == x.max(axis=0)).mean())
                                          for arm, x in outcomes.items()},
    }
    return report


def print_report(report: dict) -> None:
    v = report["verdict"]
    print(f"\n[flowcl] C1 follow-up seed {report['seed']}: {v['verdict']} {v['flags']} — {v['text']}",
          flush=True)
    failed = {k: g.get("problems") or g.get("error") for k, g in report["gates"].items() if not g.get("passed")}
    if failed:
        print(f"  failed gates: {failed}")
    p = report.get("primary")
    if p:
        d, d1 = p["D"], p["D_c1"]
        print(f"  {p['n_states']} states x {p['repetitions']} reps: e99 {p['e99']['value']:.3f} adaptive "
              f"{p['adaptive']['value']:.3f}; D {d['diff']:+.3f} [{d['low']:+.3f}, {d['high']:+.3f}] "
              f"(C1, one rep: {d1['diff']:+.2f} [{d1['low']:+.2f}, {d1['high']:+.2f}])")


def run_report(cfg: dict | None = None, seed: int = 0, *, runs_root: Path | None = None,
               out_root: Path | None = None, smoke: bool = False, rule: Path | None = None) -> dict:
    cfg = cfg or load_config(rule)
    if seed not in cfg["seeds"]:
        raise ValueError(f"seed {seed} is not registered ({cfg['seeds']})")
    report = build_report(cfg, seed, runs_root=runs_root, out_root=out_root, smoke=smoke, rule=rule)
    root = Path(out_root) if out_root else (Path(runs_root) if runs_root else repo_root() / "results")
    out = root / cfg["out_dir"] / f"seed{seed}" / "report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out, json.dumps(report, indent=2, default=str) + "\n")
    print_report(report)
    print(f"[flowcl] wrote {out}", flush=True)
    return report


def build_replication(cfg: dict, *, out_root: Path | None = None, rule: Path | None = None) -> dict:
    from flowcl.experiments.high_protection import FLAGS

    root = Path(out_root) if out_root else repo_root() / "results"
    rule_sha = file_sha256(rule_path(rule))
    per_seed, problems = {}, []
    for seed in cfg["seeds"]:
        path = root / cfg["out_dir"] / f"seed{seed}" / "report.json"
        if not path.is_file():
            problems.append(f"seed {seed}: no report at {path}")
            continue
        r = json.loads(path.read_text())
        if r.get("seed") != seed:
            problems.append(f"seed {seed}: report is for seed {r.get('seed')}")
        if r.get("smoke") is not False:
            problems.append(f"seed {seed}: a smoke report")
        if (r.get("rule") or {}).get("sha256") != rule_sha:
            problems.append(f"seed {seed}: written under another rule")
        per_seed[str(seed)] = r
    if problems:
        raise ValueError("replication refused: " + "; ".join(problems))
    outcome = classify_replication({s: r["verdict"]["verdict"] for s, r in per_seed.items()})
    return {
        "git_sha": git_sha(), "rule_sha256": rule_sha, "seeds": cfg["seeds"], **outcome,
        "flags": {f: [int(s) for s, r in per_seed.items() if f in r["verdict"]["flags"]] for f in FLAGS},
        "per_seed": {s: {"verdict": r["verdict"], "primary": r.get("primary"), "reported": r.get("reported")}
                     for s, r in per_seed.items()},
    }


def run_replication(cfg: dict | None = None, *, out_root: Path | None = None, rule: Path | None = None) -> dict:
    cfg = cfg or load_config(rule)
    summary = build_replication(cfg, out_root=out_root, rule=rule)
    root = Path(out_root) if out_root else repo_root() / "results"
    out = root / cfg["replication_out"]
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out, json.dumps(summary, indent=2, default=str) + "\n")
    print(f"\n[flowcl] C1 follow-up replication: {summary['replication']} — {summary['text']}", flush=True)
    for seed, row in summary["per_seed"].items():
        d = (row["primary"] or {}).get("D") or {}
        print(f"  seed {seed}: {row['verdict']['verdict']} {row['verdict']['flags']}; D {d.get('diff')} "
              f"[{d.get('low')}, {d.get('high')}]", flush=True)
    print(f"  flags by seed: {summary['flags']}")
    print(f"[flowcl] wrote {out}", flush=True)
    return summary
