"""Pre-registered outcome of the adaptive-GPM run: the causal test of under-protection.

The forgetting diagnostics (``docs/runs/2026-09-24_gpm_forgetting_diagnostics.md``) found a
roughly uniform residual interference that later tasks tolerated worse than T1. The
adaptive variant (``gpm_projected_adam_ne90``) extends each later task's memory to
``max(eps, p + 0.9 (1 - p))`` of its input energy. Whether that improves the later tasks'
retention is the causal question; this module evaluates the rule pre-registered in
``configs/analysis/adaptive_gpm.yaml``:

* **identity** — stages 0-1 and the T1 memory equal the plain GPM run (p = 0 at T1);
* **per-layer energy** — every applicable layer reached both its recorded target and the
  90% new-energy share (an implementation check);
* **interference** — Object's realised per-layer interference under Goal's update fell to
  at most ``max_ratio`` of the baseline's (manipulation strength);
* **retention x plasticity** — the paired transition gain on Object, judged jointly with
  T3 plasticity, then durable retention and T4 plasticity.

:func:`classify_adaptive` maps the check results to the verdict; everything it needs is
computed from the run artifacts by :func:`build_adaptive_report`.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import torch
from omegaconf import OmegaConf

from flowcl.utils.libero_paths import repo_root
from flowcl.utils.run import atomic_write_text, git_sha

VERDICTS = {
    "invalid_comparison": "Invalid comparison: stages 0-1 or the T1 memory differ from the "
                          "baseline run.",
    "invalid_implementation": "Invalid implementation: the per-layer energy target was not "
                              "met.",
    "manipulation_weak": "Manipulation weak: the energy target was met but the target task's "
                         "realised interference did not fall below the pre-registered ratio; "
                         "the retention result is exploratory.",
    "not_supported": "Under-protection hypothesis not supported: T3 was learned but Object's "
                     "retention across it did not improve.",
    "trade_off": "Stability-plasticity trade-off: Object's retention improved but T3 was not "
                 "learned to its threshold; not support for a working solution.",
    "inconclusive": "Plasticity failure with no retention benefit: the causal test is "
                    "inconclusive.",
    "transition_support": "Causal transition support: Object's retention across T3 improved "
                          "and T3 was learned.",
    "durable_support": "Strong / durable support: transition support, Object retained at the "
                       "final stage, and T4 learned.",
}


def load_adaptive_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else repo_root() / "configs" / "analysis" / "adaptive_gpm.yaml"
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


# ---- the verdict (pure) ----------------------------------------------------------


def classify_adaptive(checks: dict) -> dict:
    """The pre-registered verdict from boolean check results.

    ``checks``: ``identity``, ``energy``, ``interference``, ``t3_plasticity``,
    ``transition_gain``, ``durable_retention``, ``t4_plasticity``.
    """
    if not checks["identity"]:
        return {"verdict": "invalid_comparison", "text": VERDICTS["invalid_comparison"],
                "retention_verdict": None, "exploratory": False, "flags": []}
    if not checks["energy"]:
        return {"verdict": "invalid_implementation", "text": VERDICTS["invalid_implementation"],
                "retention_verdict": None, "exploratory": False, "flags": []}

    flags: list[str] = []
    t3, gain = checks["t3_plasticity"], checks["transition_gain"]
    if t3 and not gain:
        retention = "not_supported"
    elif gain and not t3:
        retention = "trade_off"
    elif not t3 and not gain:
        retention = "inconclusive"
    elif checks["durable_retention"] and checks["t4_plasticity"]:
        retention = "durable_support"
    else:
        retention = "transition_support"
        if not checks["durable_retention"]:
            flags.append("durable Object retention below threshold")
        if not checks["t4_plasticity"]:
            flags.append("T4 plasticity below threshold")

    exploratory = not checks["interference"]
    verdict = "manipulation_weak" if exploratory else retention
    return {
        "verdict": verdict,
        "text": VERDICTS[verdict],
        "retention_verdict": retention,
        "retention_text": VERDICTS[retention],
        "exploratory": exploratory,
        "flags": flags,
    }


# ---- the checks --------------------------------------------------------------------


def energy_check(memory_history: dict, cfg: dict) -> dict:
    """Per layer and per extension: (a) the recorded target is met and equals the
    independently recomputed one; (b) at least ``f`` of the new energy is captured.

    Layers with ``1 - p <= applicability_atol`` have no new energy and are not applicable.
    """
    eps, f, tol = cfg["eps"], cfg["new_energy_fraction"], cfg["tol"]
    history = {str(k): v for k, v in memory_history.items()}
    failures, shares = [], []
    n_applicable = n_not_applicable = 0
    for task in cfg["tasks"]:
        if str(task) not in history:
            failures.append({"task": task, "layer": None, "condition": "extension missing"})
            continue
        for layer, info in history[str(task)].items():
            p, captured = info["proj_energy_fraction"], info["captured_energy_fraction"]
            if 1.0 - p <= cfg["applicability_atol"]:
                n_not_applicable += 1
                continue
            n_applicable += 1
            target = min(1.0, max(eps, p + f * (1.0 - p)))  # independent of the method code
            recorded = info.get("target_fraction")
            row = {"task": task, "layer": layer, "p": p, "captured": captured,
                   "target": target, "recorded_target": recorded}
            if recorded is None or abs(recorded - target) > cfg["target_match_atol"]:
                failures.append({**row, "condition": "recorded target differs"})
            if captured < target - tol:
                failures.append({**row, "condition": "(a) target not met"})
            if captured < p + f * (1.0 - p) - tol:
                failures.append({**row, "condition": "(b) new-energy share below f"})
            shares.append((captured - p) / (1.0 - p))
    return {
        "passed": not failures and n_applicable > 0,
        "n_applicable": n_applicable,
        "n_not_applicable": n_not_applicable,
        "min_new_energy_share": min(shares) if shares else None,
        "median_new_energy_share": statistics.median(shares) if shares else None,
        "failures": failures,
    }


def interference_check(variant_diag: dict, baseline_diag: dict, cfg: dict,
                       variant_run: str, baseline_run: str) -> dict:
    """Variant / baseline median direct ``r`` of the target task, per half."""
    for report, run in ((variant_diag, variant_run), (baseline_diag, baseline_run)):
        if report["method_run_id"] != run:
            raise ValueError(
                f"diagnostics report is for {report['method_run_id']!r}, expected {run!r}"
            )
    name = cfg["comparison"]
    v = variant_diag["comparisons"][name]["reported"]["median_r"]["target"]["direct"]
    b = baseline_diag["comparisons"][name]["reported"]["median_r"]["target"]["direct"]
    ratios = {half: v[half] / b[half] for half in ("trunk", "decoder")}
    return {
        # Inclusive, with the same float-rounding allowance as the rollout thresholds.
        "passed": all(r <= cfg["max_ratio"] + 1e-9 for r in ratios.values()),
        "ratio": ratios,
        "variant_median_r": v,
        "baseline_median_r": b,
        "max_ratio": cfg["max_ratio"],
    }


def identity_check(variant_dir: Path, baseline_dir: Path, stages: list[int]) -> dict:
    """Stage state dicts (re-verified here, not only read from the runner's files) and the
    T1 memory vectors equal the baseline's; whether ``memory_task1`` differs is info."""
    from flowcl.analysis.subspace import load_bases
    from flowcl.experiments.sequence_report import memory_matrix
    from flowcl.train.continual import stage_identity_check

    stage_checks = {
        str(k): stage_identity_check(variant_dir / "checkpoints" / f"stage{k}.pt", baseline_dir, k)
        for k in stages
    }

    def memory(run_dir: Path, task: int) -> dict:
        bases, _ = load_bases(run_dir / "method" / f"memory_task{task}.pt")
        return {n: memory_matrix(b) for n, b in bases.items()}

    v0, b0 = memory(variant_dir, 0), memory(baseline_dir, 0)
    t1_equal = sorted(v0) == sorted(b0) and all(torch.equal(v0[n], b0[n]) for n in v0)
    v1, b1 = memory(variant_dir, 1), memory(baseline_dir, 1)
    t2_differs = any(v1[n].shape != b1[n].shape or not torch.equal(v1[n], b1[n]) for n in v1)
    return {
        "passed": all(c["passed"] for c in stage_checks.values()) and t1_equal,
        "stages": stage_checks,
        "memory_task0_equal": t1_equal,
        "memory_task1_differs": t2_differs,
    }


def _at_least(value: float, threshold: float) -> bool:
    # Inclusive; rates are k/50 and differences of such, so allow float rounding only.
    return value >= threshold - 1e-9


def rollout_checks(transition_paired: dict, t3: float, durable: float, t4: float,
                   cfg: dict) -> dict:
    """The rollout-based checks. Inclusive thresholds; the CI lower bound strictly > 0."""
    return {
        "transition_gain": (
            _at_least(transition_paired["diff"], cfg["retention"]["min_improvement"])
            and transition_paired["low"] > 0.0
        ),
        "t3_plasticity": _at_least(t3, cfg["plasticity"]["t3_threshold"]),
        "durable_retention": _at_least(durable, cfg["retention"]["durable_threshold"]),
        "t4_plasticity": _at_least(t4, cfg["plasticity"]["t4_threshold"]),
    }


# ---- the report ----------------------------------------------------------------------


def build_adaptive_report(cfg: dict, results_root: Path | None = None) -> dict:
    from flowcl.analysis.subspace import load_bases
    from flowcl.experiments.sequence_report import (
        capacity_by_stage,
        classify_sequence,
        criteria_thresholds,
        load_report_config,
        load_run,
        paired_cells,
        registry_groups,
    )

    root = Path(results_root) if results_root else repo_root() / "results"
    dirs = {k: root / cfg[f"{k}_run"] for k in ("variant", "baseline", "reference")}
    runs = {k: load_run(d) for k, d in dirs.items()}
    n = runs["variant"].n_tasks
    bootstrap = OmegaConf.to_container(
        OmegaConf.load(repo_root() / "configs" / "eval" / "libero_eval.yaml"), resolve=True
    )["bootstrap"]

    # The existing thresholds, asserted equal to this rule's copies.
    ref_diag = [runs["reference"].cell(j, j).estimate.value for j in range(n)]
    thresholds = criteria_thresholds(ref_diag, load_report_config()["criteria"])
    expected = {
        "durable_threshold": thresholds[cfg["retention"]["durable_cell"][1]],
        "t3_threshold": thresholds[cfg["plasticity"]["t3_cell"][1]],
        "t4_threshold": thresholds[cfg["plasticity"]["t4_cell"][1]],
    }
    stated = {
        "durable_threshold": cfg["retention"]["durable_threshold"],
        "t3_threshold": cfg["plasticity"]["t3_threshold"],
        "t4_threshold": cfg["plasticity"]["t4_threshold"],
    }
    if any(not math.isclose(expected[k], stated[k]) for k in expected):
        raise ValueError(f"pre-registered thresholds {stated} differ from the criteria {expected}")

    identity = identity_check(dirs["variant"], dirs["baseline"], cfg["identity_stages"])
    _, meta = load_bases(dirs["variant"] / "method" / f"memory_task{n - 1}.pt")
    energy = energy_check(meta["memory_history"], cfg["energy"])
    diag = {k: json.loads((root / p).read_text()) for k, p in cfg["diagnostics"].items()}
    interference = interference_check(
        diag["variant"], diag["baseline"], cfg["interference"],
        dirs["variant"].name, dirs["baseline"].name,
    )

    paired = paired_cells(runs["variant"], runs["baseline"], bootstrap)

    def cell(spec: list[int]) -> dict:
        i, j = spec
        est = runs["variant"].cell(i, j).estimate
        base = runs["baseline"].cell(i, j).estimate
        return {"cell": spec, "variant": est.value, "variant_ci": [est.low, est.high],
                "baseline": base.value, "paired": paired[f"{i},{j}"]}

    transition = cell(cfg["retention"]["transition_cell"])
    durable = cell(cfg["retention"]["durable_cell"])
    t3 = cell(cfg["plasticity"]["t3_cell"])
    t4 = cell(cfg["plasticity"]["t4_cell"])
    goal = cell(cfg["secondary"]["goal_cell"])

    checks = {
        "identity": identity["passed"],
        "energy": energy["passed"],
        "interference": interference["passed"],
        **rollout_checks(transition["paired"], t3["variant"], durable["variant"],
                         t4["variant"], cfg),
    }
    groups = registry_groups(dirs["variant"])
    _, baseline_meta = load_bases(dirs["baseline"] / "method" / f"memory_task{n - 1}.pt")
    estimates = {(i, j): runs["variant"].cell(i, j).estimate for i in range(n) for j in range(n)}
    return {
        "git_sha": git_sha(),
        "config": cfg,
        "verdict": classify_adaptive(checks),
        "checks": checks,
        "identity": identity,
        "energy": energy,
        "interference": interference,
        "cells": {"transition": transition, "durable": durable, "t3": t3, "t4": t4,
                  "secondary_goal": {**goal, "threshold": cfg["secondary"]["goal_threshold"]}},
        "sequence_outcome": classify_sequence(
            estimates, thresholds, load_report_config()["criteria"]["fallback_tasks"]
        ),
        "paired_cells_vs_baseline": paired,
        "capacity": {
            "variant": capacity_by_stage(meta["memory_history"], groups),
            "baseline": capacity_by_stage(baseline_meta["memory_history"], groups),
        },
    }


def print_adaptive_report(report: dict) -> None:
    v, c = report["verdict"], report["checks"]
    print(f"\n[flowcl] adaptive GPM verdict: {v['verdict']} — {v['text']}", flush=True)
    if v.get("exploratory"):
        print(f"  (exploratory retention verdict: {v['retention_verdict']})")
    if v.get("flags"):
        print(f"  flags: {v['flags']}")
    print(f"  checks: {c}")
    e, i = report["energy"], report["interference"]
    print(f"  energy: applicable {e['n_applicable']}, min share {e['min_new_energy_share']}, "
          f"failures {len(e['failures'])}")
    print(f"  interference ratio (variant/baseline, median direct r_Object): {i['ratio']}")
    for name, row in report["cells"].items():
        p = row["paired"]
        print(f"  {name} {row['cell']}: variant {row['variant']:.2f} baseline {row['baseline']:.2f} "
              f"paired {p['diff']:+.2f} [{p['low']:+.2f}, {p['high']:+.2f}]")
    cap = report["capacity"]["variant"]
    for stage, scopes in cap.items():
        print(f"  capacity after T{int(stage) + 1}: trunk rho {scopes['trunk']['median_rho']:.3f} "
              f"(max {scopes['trunk']['max_rho']:.3f}), decoder rho {scopes['decoder']['median_rho']:.3f}, "
              f"exhausted {scopes['exhausted_layers']}")


def run_adaptive_report(cfg: dict | None = None, out: Path | None = None,
                        results_root: Path | None = None) -> dict:
    cfg = cfg or load_adaptive_config()
    report = build_adaptive_report(cfg, results_root)
    root = Path(results_root) if results_root else repo_root() / "results"
    out = Path(out) if out else root / cfg["out"]
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out, json.dumps(report, indent=2, default=str) + "\n")
    print_adaptive_report(report)
    print(f"[flowcl] wrote {out}", flush=True)
    return report
