"""Four-task sequence report: a method run against the ``seq_ft`` reference (build step 8).

Reads two ``run_continual`` directories that share a seed namespace (so every rollout
cell is paired: same initial state, same sampling seed) and reports:

* both retention matrices with CIs, and the paired difference per cell;
* the pre-registered criteria (``configs/analysis/sequence_report.yaml``): per-task
  plasticity ``R[j][j]`` and final retention ``R[T-1][j]`` against ``R_ref[j][j] - 15 pp``,
  and whether the SGP fallback is triggered (plasticity failure on T3 or T4). Point
  estimates decide; a CI straddling a threshold is flagged ``borderline``;
* memory capacity after each task from the method's ``memory_task{τ}.pt`` artifacts —
  **occupancy** ``rho_l = k_l / d_l`` (dimensions protected, non-decreasing) kept apart
  from **proj_energy_fraction** ``tr(M^T K M)/tr K`` (share of the new task's input energy
  already inside memory) — and the number of capacity-exhausted layers;
* per-stage ``c_l`` trajectories (raw gradient, AdamW step) and residual maxima;
* provenance checks: seeds paired, artifact SHA-256s match the checkpoints, frozen
  tensors unchanged from stage 1 on, occupancy non-decreasing, clean git SHA, and the T1
  pairing numbers recorded by the runner.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

import torch
from omegaconf import OmegaConf

from flowcl.analysis.metrics import Estimate, paired_difference_ci
from flowcl.envs.evaluation import EvaluationReport
from flowcl.utils.libero_paths import repo_root
from flowcl.utils.run import file_sha256


# ---- loading -------------------------------------------------------------------


@dataclass
class RunView:
    run_dir: Path
    result: dict
    evals: dict[int, EvaluationReport]

    @property
    def task_keys(self) -> list[str]:
        return self.result["task_keys"]

    @property
    def n_tasks(self) -> int:
        return len(self.task_keys)

    def cell(self, i: int, j: int):
        return self.evals[i].by_task()[self.task_keys[j]]


def load_run(run_dir: str | Path) -> RunView:
    run_dir = Path(run_dir)
    result = json.loads((run_dir / "result.json").read_text())
    n = len(result["task_keys"])
    evals = {i: EvaluationReport.load(run_dir / "eval" / f"stage{i}.json") for i in range(n)}
    return RunView(run_dir, result, evals)


def load_report_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else repo_root() / "configs" / "analysis" / "sequence_report.yaml"
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


# ---- pre-registered criteria ---------------------------------------------------


def criteria_thresholds(reference_diagonal: list[float], criteria: dict) -> list[float]:
    """``R_ref[j][j] - margin`` per task, asserted against the pre-registered values."""
    margin = criteria["margin_pp"] / 100.0
    thresholds = [round(r - margin, 6) for r in reference_diagonal]
    expected = criteria["expected_thresholds"]
    if len(thresholds) != len(expected) or not all(
        math.isclose(a, b) for a, b in zip(thresholds, expected)
    ):
        raise ValueError(
            f"thresholds from the reference diagonal {thresholds} differ from the "
            f"pre-registered {expected}; the reference run changed"
        )
    return thresholds


def _judge(est: Estimate, threshold: float) -> dict:
    return {
        "value": est.value,
        "low": est.low,
        "high": est.high,
        "threshold": threshold,
        "ok": est.value >= threshold,  # inclusive; the point estimate decides
        "borderline": est.low < threshold <= est.high,
    }


def classify_sequence(
    estimates: dict[tuple[int, int], Estimate],
    thresholds: list[float],
    fallback_tasks: list[int],
) -> dict:
    """Per-task plasticity and final retention, plus the SGP-fallback trigger."""
    n = len(thresholds)
    plasticity = {j: _judge(estimates[(j, j)], thresholds[j]) for j in range(n)}
    retention = {j: _judge(estimates[(n - 1, j)], thresholds[j]) for j in range(n - 1)}
    failed_fallback = [j for j in fallback_tasks if not plasticity[j]["ok"]]
    return {
        "plasticity": plasticity,
        "final_retention": retention,
        "plasticity_failures": [j for j in range(n) if not plasticity[j]["ok"]],
        "retention_failures": [j for j in range(n - 1) if not retention[j]["ok"]],
        "sgp_fallback_triggered": bool(failed_fallback),
        "sgp_fallback_tasks": failed_fallback,
        "borderline": {
            "plasticity": [j for j, v in plasticity.items() if v["borderline"]],
            "final_retention": [j for j, v in retention.items() if v["borderline"]],
        },
    }


def paired_cells(method: RunView, reference: RunView, bootstrap: dict) -> dict:
    """Paired difference per cell; raises unless every cell's rollout seeds match."""
    if method.task_keys != reference.task_keys:
        raise ValueError("runs cover different task sequences")
    out = {}
    for i in range(method.n_tasks):
        for j in range(method.n_tasks):
            a, b = method.cell(i, j), reference.cell(i, j)
            if list(a.seeds) != list(b.seeds):
                raise ValueError(
                    f"cell ({i},{j}) rollout seeds differ: the runs are not paired "
                    "(different seed namespaces?)"
                )
            est = paired_difference_ci(
                a.successes, b.successes,
                seed=bootstrap["seed"], n_bootstrap=bootstrap["n_resamples"],
                confidence=bootstrap["confidence"],
            )
            out[f"{i},{j}"] = {"diff": est.value, "low": est.low, "high": est.high}
    return out


# ---- capacity and dynamics -----------------------------------------------------


def _half(name: str) -> str:
    return "trunk" if name.startswith("trunk.") else "decoder"


def capacity_by_stage(memory_history: dict[str, dict], groups: dict[str, str] | None = None) -> dict:
    """Occupancy and energy fractions per stage, per half (and per group if given).

    ``memory_history``: stage -> layer -> ``extend_basis`` info (as stored in the final
    memory artifact's metadata).
    """
    out = {}
    for stage, layers in sorted(memory_history.items(), key=lambda kv: int(kv[0])):
        scopes: dict[str, list[dict]] = {}
        for name, info in layers.items():
            scopes.setdefault(_half(name), []).append(info)
            if groups:
                scopes.setdefault(groups[name], []).append(info)
        out[str(stage)] = {
            scope: {
                "n_layers": len(rows),
                "median_rho": statistics.median(r["rho_after"] for r in rows),
                "min_rho": min(r["rho_after"] for r in rows),
                "max_rho": max(r["rho_after"] for r in rows),
                "median_free_fraction": statistics.median(1 - r["rho_after"] for r in rows),
                "median_proj_energy_fraction": statistics.median(
                    r["proj_energy_fraction"] for r in rows
                ),
                "k_added": sum(r["k_added"] for r in rows),
                "capacity_exhausted": sum(1 for r in rows if r["capacity_exhausted"]),
            }
            for scope, rows in scopes.items()
        }
        out[str(stage)]["exhausted_layers"] = sorted(
            n for n, r in layers.items() if r["capacity_exhausted"]
        )
    return out


def c_trajectories(logs: dict) -> dict:
    """Median-over-layers ``c_l`` per half at each logged step, plus its mean over steps."""
    out = {}
    for key in ("gradient_c", "update_c"):
        per_step = {}
        for step, layers in sorted(logs.get(key, {}).items(), key=lambda kv: int(kv[0])):
            row = {}
            for half in ("trunk", "decoder"):
                vals = [v for n, v in layers.items() if _half(n) == half and v is not None and not math.isnan(v)]
                row[half] = statistics.median(vals) if vals else None
            per_step[step] = row
        means = {}
        for half in ("trunk", "decoder"):
            vals = [r[half] for r in per_step.values() if r[half] is not None]
            means[half] = statistics.fmean(vals) if vals else None
        out[key] = {"mean_of_medians": means, "per_step": per_step}
    return out


# ---- provenance checks ---------------------------------------------------------


def provenance_checks(method: RunView, reference: RunView, verify_checkpoints: bool = True) -> dict:
    from flowcl.analysis.subspace import load_bases

    run_dir = method.run_dir
    checks: dict = {}
    sha = (run_dir / "git_sha").read_text().strip()
    checks["clean_git_sha"] = {"passed": not sha.endswith("-dirty"), "git_sha": sha}
    checks["seed_namespace"] = {
        "passed": method.result.get("seed_namespace_run_id") == reference.run_dir.name,
        "seed_namespace_run_id": method.result.get("seed_namespace_run_id"),
    }
    checks["t1_pairing"] = method.result.get("t1_pairing")

    rho_prev: dict[str, float] = {}
    rho_ok = True
    for i in range(method.n_tasks):
        path = run_dir / "method" / f"memory_task{i}.pt"
        if not path.is_file():
            continue
        bases, _ = load_bases(path)
        for name, basis in bases.items():
            rho = next(iter(basis.rhos.values()))  # the memory stores one eps
            if rho < rho_prev.get(name, 0.0):
                rho_ok = False
            rho_prev[name] = rho
    checks["occupancy_non_decreasing"] = {"passed": rho_ok}

    worst = 0.0
    for i in range(method.n_tasks):
        path = run_dir / "method" / f"gpm_logs_task{i}.json"
        if path.is_file():
            logs = json.loads(path.read_text())
            for r in logs.get("residuals", {}).values():
                worst = max(worst, r["max_residual_over_bound"])
    checks["residuals_within_bound"] = {"passed": worst <= 1.0, "worst_over_bound": worst}

    if verify_checkpoints:
        from flowcl.methods.gpm import allowlist
        from flowcl.train.checkpoint import load_checkpoint

        mismatched, states = [], []
        for i in range(method.n_tasks):
            payload = torch.load(
                run_dir / "checkpoints" / f"stage{i}.pt", map_location="cpu", weights_only=False
            )
            for artifact in payload["extra"].get("method_artifacts", []):
                if file_sha256(run_dir / artifact["path"]) != artifact["sha256"]:
                    mismatched.append(artifact["path"])
            states.append(payload["state_dict"])
        checks["artifact_hashes_match"] = {"passed": not mismatched, "mismatched": mismatched}
        loaded = load_checkpoint(run_dir / "checkpoints" / "stage0.pt")
        allowed = set(allowlist(loaded.policy))
        trainable = {n for n, p in loaded.policy.named_parameters() if p.requires_grad}
        # Freezing starts at the beginning of stage 1, so every non-allowlisted tensor
        # must equal its stage-0 value in every later checkpoint.
        changed = sorted(
            n for n in trainable - allowed
            if any(not torch.equal(states[0][n], states[k][n]) for k in range(1, method.n_tasks))
        )
        checks["frozen_from_stage1"] = {"passed": not changed, "changed": changed}
    return checks


# ---- the report ----------------------------------------------------------------


def build_report(
    method_dir: str | Path,
    reference_dir: str | Path,
    pilot_json: str | Path | None = None,
    config: dict | None = None,
    verify_checkpoints: bool = True,
) -> dict:
    from flowcl.analysis.subspace import load_bases

    config = config or load_report_config()
    bootstrap = OmegaConf.to_container(
        OmegaConf.load(repo_root() / "configs" / "eval" / "libero_eval.yaml"), resolve=True
    )["bootstrap"]
    method, reference = load_run(method_dir), load_run(reference_dir)
    n = method.n_tasks

    ref_diag = [reference.cell(j, j).estimate.value for j in range(n)]
    thresholds = criteria_thresholds(ref_diag, config["criteria"])
    estimates = {(i, j): method.cell(i, j).estimate for i in range(n) for j in range(n)}

    def matrix(run: RunView) -> list[list[str]]:
        return [[run.cell(i, j).estimate.format_pp() for j in range(n)] for i in range(n)]

    report = {
        "method_run_id": method.result.get("method_run_id", method.run_dir.name),
        "method": method.result.get("method"),
        "reference_run_id": reference.run_dir.name,
        "seed_namespace_run_id": method.result.get("seed_namespace_run_id"),
        "task_keys": method.task_keys,
        "criteria": {"thresholds": thresholds, "reference_diagonal": ref_diag, **config["criteria"]},
        "outcome": classify_sequence(estimates, thresholds, config["criteria"]["fallback_tasks"]),
        "matrices": {"method": matrix(method), "reference": matrix(reference)},
        "paired_cells": paired_cells(method, reference, bootstrap),
        "metrics": {"method": method.result["metrics"], "reference": reference.result["metrics"]},
    }

    final_memory = method.run_dir / "method" / f"memory_task{n - 1}.pt"
    if final_memory.is_file():
        _, meta = load_bases(final_memory)
        report["capacity"] = capacity_by_stage(meta["memory_history"])
    report["dynamics"] = {}
    for i in range(n):
        path = method.run_dir / "method" / f"gpm_logs_task{i}.json"
        if path.is_file():
            logs = json.loads(path.read_text())
            dyn = c_trajectories(logs)
            worst = max(
                (r["max_residual_over_bound"] for r in logs.get("residuals", {}).values()),
                default=None,
            )
            report["dynamics"][str(i)] = {
                "projected": logs.get("projected"),
                "gradient_c_mean": dyn["gradient_c"]["mean_of_medians"],
                "update_c_mean": dyn["update_c"]["mean_of_medians"],
                "worst_residual_over_bound": worst,
            }

    if pilot_json and Path(pilot_json).is_file():
        pilot = json.loads(Path(pilot_json).read_text())["arms"]["gpm_projected_adam"]["evaluation"]
        report["pilot_comparison"] = {
            key: {
                "pilot": pilot[key]["value"],
                "sequence_stage1": method.cell(1, method.task_keys.index(key)).estimate.value,
                "paired_diff": _paired(
                    method.cell(1, method.task_keys.index(key)).successes, pilot[key]["successes"], bootstrap
                ),
            }
            for key in pilot
        }

    report["provenance_checks"] = provenance_checks(method, reference, verify_checkpoints)
    return report


def _paired(a, b, bootstrap) -> dict:
    est = paired_difference_ci(
        a, b, seed=bootstrap["seed"], n_bootstrap=bootstrap["n_resamples"],
        confidence=bootstrap["confidence"],
    )
    return {"diff": est.value, "low": est.low, "high": est.high}


def print_report(report: dict) -> None:
    keys = [k.split("/")[0].replace("libero_", "") for k in report["task_keys"]]
    print(f"\n[flowcl] {report['method']} vs {report['reference_run_id']}", flush=True)
    for label in ("method", "reference"):
        print(f"  R ({label}):  " + " | ".join(keys))
        for i, row in enumerate(report["matrices"][label]):
            print(f"    after {keys[i]:8s} " + " | ".join(row))
    o = report["outcome"]
    for j, v in o["plasticity"].items():
        print(f"  plasticity T{j + 1}: {100 * v['value']:.1f}% (>= {100 * v['threshold']:.0f}%) "
              f"{'OK' if v['ok'] else 'FAIL'}{' borderline' if v['borderline'] else ''}")
    for j, v in o["final_retention"].items():
        print(f"  final retention T{j + 1}: {100 * v['value']:.1f}% (>= {100 * v['threshold']:.0f}%) "
              f"{'OK' if v['ok'] else 'FAIL'}{' borderline' if v['borderline'] else ''}")
    print(f"  SGP fallback triggered: {o['sgp_fallback_triggered']} {o['sgp_fallback_tasks']}")
    for stage, cap in report.get("capacity", {}).items():
        t, d = cap["trunk"], cap["decoder"]
        print(f"  memory after T{int(stage) + 1}: trunk rho {t['median_rho']:.3f} (free {t['median_free_fraction']:.3f}, "
              f"exhausted {t['capacity_exhausted']}), decoder rho {d['median_rho']:.3f} (free "
              f"{d['median_free_fraction']:.3f}, exhausted {d['capacity_exhausted']}); new-task energy "
              f"already in memory: trunk {t['median_proj_energy_fraction']:.3f}, decoder "
              f"{d['median_proj_energy_fraction']:.3f}")
    for stage, dyn in report.get("dynamics", {}).items():
        print(f"  stage {stage} dynamics: {dyn}")
    for name, check in report["provenance_checks"].items():
        passed = check.get("passed") if isinstance(check, dict) else None
        print(f"  [{'PASS' if passed else 'INFO' if passed is None else 'FAIL'}] {name}")


def run_sequence_report(
    method_dir: Path, reference_dir: Path, pilot_json: Path | None, out: Path
) -> dict:
    report = build_report(method_dir, reference_dir, pilot_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print_report(report)
    print(f"[flowcl] wrote {out}", flush=True)
    return report
