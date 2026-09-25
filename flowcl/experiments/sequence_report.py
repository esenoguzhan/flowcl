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
  already inside memory) and the share of each task's *new* energy the extension kept —
  per half and per §4.5 group, with the number of capacity-exhausted layers;
* episode-level transitions per task (same episode seeds at every stage): both / lost /
  gained, horizons, timeouts;
* per-stage ``c_l`` trajectories (raw gradient, AdamW step; every logged step) and
  residual maxima and medians;
* the pilot comparison, including the directional overlap of the sequence's T1 memory
  with the pilot's Gate 2 basis at the same eps;
* provenance checks: seeds paired, artifact SHA-256s match the checkpoints, every
  state-dict tensor outside the allowlist unchanged from stage 1 on, occupancy
  non-decreasing, each memory contained in the next (``memory_chained``), clean git SHA,
  and the T1 pairing numbers recorded by the runner.
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


def criteria_thresholds(
    reference_diagonal: list[float], criteria: dict, reference_run_id: str
) -> list[float]:
    """``R_ref[j][j] - margin`` per task, asserted against the values pre-registered for
    this reference run (each method run is judged against its own seed's seq_ft run)."""
    margin = criteria["margin_pp"] / 100.0
    thresholds = [round(r - margin, 6) for r in reference_diagonal]
    registered = criteria["expected_thresholds_by_reference"]
    if reference_run_id not in registered:
        raise ValueError(
            f"no thresholds pre-registered for reference run {reference_run_id!r} "
            f"(have {sorted(registered)}); add them to sequence_report.yaml before judging"
        )
    expected = registered[reference_run_id]
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


def _median_or_none(values) -> float | None:
    kept = [v for v in values if v is not None]
    return statistics.median(kept) if kept else None


def new_energy_protected(info: dict) -> float | None:
    """Share of the task's *new* energy (outside the old memory) that the extension kept.

    ``(captured - proj) / (1 - proj)``: 0.95 for a first task at eps = 0.95, and far less
    for a later task whose energy is mostly inside memory already. ``None`` when the
    memory already held (numerically) all of it.
    """
    proj = info["proj_energy_fraction"]
    if proj >= 1.0 - 1e-12:
        return None
    return (info["captured_energy_fraction"] - proj) / (1.0 - proj)


def capacity_by_stage(memory_history: dict[str, dict], groups: dict[str, str] | None = None) -> dict:
    """Occupancy and energy fractions per stage, per half (and per group if given).

    ``memory_history``: stage -> layer -> ``extend_basis`` info (as stored in the final
    memory artifact's metadata). Dimension-based occupancy (``rho``) and the energy-based
    fractions are kept apart: ``proj_energy_fraction`` is the new task's energy already
    inside memory, ``new_energy_protected`` the share of the remainder the extension
    kept, ``unprotected_energy`` = ``1 - captured``.
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
                "median_new_energy_protected": _median_or_none(
                    new_energy_protected(r) for r in rows
                ),
                "median_unprotected_energy": statistics.median(
                    1.0 - r["captured_energy_fraction"] for r in rows
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


# ---- episode-level transitions -------------------------------------------------


def _failures(cell, max_steps: int | None) -> dict:
    fails = [s for ok, s in zip(cell.successes, cell.n_steps) if not ok]
    return {
        "n": len(fails),
        "timeouts": None if max_steps is None else sum(1 for s in fails if s == max_steps),
    }


def episode_transitions(run: RunView, max_steps: int | None = None) -> dict:
    """Paired, episode-level view of each task across stages.

    Every stage rolls task ``j`` out on the same episode seeds, so a transition compares
    the same 50 episodes. For task ``j``: each consecutive transition ``i -> i+1``
    (``i >= j``), plus ``j -> T-1`` when that spans more than one stage. Episode indices
    are positions in the rollout list (0-based). Raises if seeds differ between stages.
    """
    n = run.n_tasks
    out = {}
    for j in range(n):
        pairs = [(i, i + 1) for i in range(j, n - 1)]
        if n - 1 - j > 1:
            pairs.append((j, n - 1))
        rows = {}
        for a, b in pairs:
            before, after = run.cell(a, j), run.cell(b, j)
            if list(before.seeds) != list(after.seeds):
                raise ValueError(
                    f"task {j}: rollout seeds differ between stages {a} and {b}; the "
                    "transition is not paired"
                )
            sa, sb = before.successes, after.successes
            idx = range(len(sa))
            both = [e for e in idx if sa[e] and sb[e]]
            dh = [after.n_steps[e] - before.n_steps[e] for e in both]
            rows[f"{a}->{b}"] = {
                "both": len(both),
                "lost": sum(1 for e in idx if sa[e] and not sb[e]),
                "gained": sum(1 for e in idx if sb[e] and not sa[e]),
                "lost_episodes": [e for e in idx if sa[e] and not sb[e]],
                "gained_episodes": [e for e in idx if sb[e] and not sa[e]],
                "median_success_horizon": {
                    "before": _median_or_none(before.n_steps[e] for e in idx if sa[e]),
                    "after": _median_or_none(after.n_steps[e] for e in idx if sb[e]),
                },
                "shared_horizon_change": (
                    {"median": statistics.median(dh), "mean": statistics.fmean(dh)}
                    if dh else None
                ),
                "failures": {
                    "before": _failures(before, max_steps),
                    "after": _failures(after, max_steps),
                },
            }
        out[run.task_keys[j]] = rows
    return out


# ---- memory artifacts: chaining and overlap ------------------------------------


def _eps_key(ranks: dict, eps: float):
    """The threshold key in ``ranks`` equal to ``eps`` (float keys); raises if absent."""
    for key in ranks:
        if math.isclose(float(key), eps, rel_tol=0.0, abs_tol=1e-12):
            return key
    raise ValueError(f"no basis at eps={eps}; stored thresholds are {sorted(ranks)}")


def memory_matrix(basis, eps: float | None = None) -> torch.Tensor:
    """The basis at ``eps`` as float64 ``(d, k)``: the prefix ``vectors[:, :ranks[eps]]``.

    A GPM memory artifact stores exactly one threshold, so ``eps`` may be omitted there. A
    Gate 2 basis stores ``U`` up to its largest threshold (0.99), so ``eps`` is required
    and the prefix matters.
    """
    if eps is None:
        if len(basis.ranks) != 1:
            raise ValueError(
                f"{basis.layer}: basis stores thresholds {sorted(basis.ranks)}; say which eps"
            )
        key = next(iter(basis.ranks))
    else:
        key = _eps_key(basis.ranks, eps)
    return basis.vectors[:, : basis.ranks[key]].to(torch.float64)


CONTAINMENT_TOL = 1e-9


def memory_chain_step(
    prev: dict[str, torch.Tensor],
    nxt: dict[str, torch.Tensor],
    history_prev: dict[str, dict],
    history_next: dict[str, dict],
) -> dict:
    """Did task ``τ``'s memory really carry task ``τ-1``'s forward?

    Equal ranks are not enough (unrelated bases can share a rank), so per layer:

    * ``k_before(τ) == k_after(τ-1)`` and each stored basis has its recorded ``k_after``;
    * **containment** ``||M_{τ-1}^T M_τ||_F² / k_{τ-1} >= 1 - CONTAINMENT_TOL``: the old
      memory lies inside the new one (rotation-invariant);
    * ``prefix_identical``: ``M_τ[:, :k_{τ-1}]`` equals ``M_{τ-1}`` exactly — expected,
      since :func:`flowcl.analysis.subspace.extend_basis` appends and never rotates.
    """
    if sorted(prev) != sorted(nxt):
        raise ValueError("memory artifacts cover different layers")
    rank_problems, not_contained, not_prefix = [], [], []
    min_containment = 1.0
    for name, M0 in prev.items():
        M1 = nxt[name]
        k0 = M0.shape[1]
        if (
            history_next[name]["k_before"] != history_prev[name]["k_after"]
            or k0 != history_prev[name]["k_after"]
            or M1.shape[1] != history_next[name]["k_after"]
        ):
            rank_problems.append(name)
        if k0 == 0:
            continue
        containment = float(torch.linalg.matrix_norm(M0.T @ M1) ** 2) / k0
        min_containment = min(min_containment, containment)
        if containment < 1.0 - CONTAINMENT_TOL:
            not_contained.append(name)
        if M1.shape[1] < k0 or not torch.equal(M0, M1[:, :k0]):
            not_prefix.append(name)
    return {
        "passed": not rank_problems and not not_contained,
        "rank_mismatch": rank_problems,
        "not_contained": not_contained,
        "min_containment": min_containment,
        "prefix_identical": not not_prefix,
        "not_prefix_identical": not_prefix,
    }


def basis_overlap(Ma: torch.Tensor, Mb: torch.Tensor) -> dict:
    """``s = ||M_a^T M_b||_F²`` with both directional normalisations.

    ``s / k_a = 1``: ``span(M_a)`` lies inside ``span(M_b)``; ``s / k_b = 1``: the reverse.
    Both 1 only for equal subspaces.
    """
    s = float(torch.linalg.matrix_norm(Ma.to(torch.float64).T @ Mb.to(torch.float64)) ** 2)
    ka, kb = Ma.shape[1], Mb.shape[1]
    return {
        "s": s, "k_a": ka, "k_b": kb,
        "s_over_k_a": s / ka if ka else None,
        "s_over_k_b": s / kb if kb else None,
    }


def t1_memory_overlap(
    gate2_bases: dict, memory_bases: dict, eps: float, groups: dict[str, str] | None = None
) -> dict:
    """Pilot (Gate 2) T1 basis ``a`` against the sequence's ``memory_task0`` ``b``.

    ``M_a = gate2.vectors[:, :gate2.ranks[eps]]`` — the eps prefix the pilot projected
    against, not Gate 2's full 0.99 ``U``. Raises if Gate 2 has no basis at ``eps``.
    """
    per_layer = {
        name: basis_overlap(memory_matrix(gate2_bases[name], eps), memory_matrix(basis))
        for name, basis in memory_bases.items()
    }
    scopes: dict[str, list[str]] = {}
    for name in per_layer:
        scopes.setdefault(_half(name), []).append(name)
        if groups:
            scopes.setdefault(groups[name], []).append(name)
    summary = {}
    for scope, names in scopes.items():
        rows = [per_layer[n] for n in names]
        a = [r["s_over_k_a"] for r in rows if r["s_over_k_a"] is not None]
        b = [r["s_over_k_b"] for r in rows if r["s_over_k_b"] is not None]
        dk = [r["k_b"] - r["k_a"] for r in rows]
        summary[scope] = {
            "n_layers": len(rows),
            "median_s_over_k_a": statistics.median(a) if a else None,
            "min_s_over_k_a": min(a) if a else None,
            "median_s_over_k_b": statistics.median(b) if b else None,
            "min_s_over_k_b": min(b) if b else None,
            "median_k_b_minus_k_a": statistics.median(dk),
            "min_k_b_minus_k_a": min(dk),
            "max_k_b_minus_k_a": max(dk),
        }
    return {"eps": eps, "summary": summary, "per_layer": per_layer}


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
    chain: dict[str, dict] = {}
    prev = None  # (task index, memory matrices, that task's history rows)
    for i in range(method.n_tasks):
        path = run_dir / "method" / f"memory_task{i}.pt"
        if not path.is_file():
            continue
        bases, meta = load_bases(path)
        for name, basis in bases.items():
            rho = next(iter(basis.rhos.values()))  # the memory stores one eps
            if rho < rho_prev.get(name, 0.0):
                rho_ok = False
            rho_prev[name] = rho
        history = {str(k): v for k, v in meta["memory_history"].items()}
        current = (i, {n: memory_matrix(b) for n, b in bases.items()}, history[str(i)])
        if prev is not None:
            chain[f"{prev[0]}->{i}"] = memory_chain_step(prev[1], current[1], prev[2], current[2])
        prev = current
    checks["occupancy_non_decreasing"] = {"passed": rho_ok}
    if chain:
        checks["memory_chained"] = {
            "passed": all(step["passed"] for step in chain.values()),
            "prefix_identical": all(step["prefix_identical"] for step in chain.values()),
            "steps": chain,
        }

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
        # Freezing starts at the beginning of stage 1, so every state-dict tensor outside
        # the allowlist — parameters that trained at T1, never-trained backbones and
        # persistent buffers alike — must equal its stage-0 value in every later checkpoint.
        outside = sorted(set(states[0]) - allowed)
        changed = [
            n for n in outside
            if any(not torch.equal(states[0][n], states[k][n]) for k in range(1, method.n_tasks))
        ]
        unmoved = {
            str(k): sorted(n for n in allowed if torch.equal(states[k - 1][n], states[k][n]))
            for k in range(1, method.n_tasks)
        }
        checks["frozen_from_stage1"] = {
            "passed": not changed,
            "changed": changed,
            "n_tensors_outside_allowlist": len(outside),
            "n_allowlisted": len(allowed),
            "moved_allowlisted": {k: len(allowed) - len(v) for k, v in unmoved.items()},
            "unmoved_allowlisted": unmoved,
        }
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
    thresholds = criteria_thresholds(ref_diag, config["criteria"], reference.run_dir.name)
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

    groups = registry_groups(method.run_dir) if verify_checkpoints else None
    report["episode_transitions"] = episode_transitions(method, _max_steps(method.run_dir))

    final_memory = method.run_dir / "method" / f"memory_task{n - 1}.pt"
    if final_memory.is_file():
        _, meta = load_bases(final_memory)
        report["capacity"] = capacity_by_stage(meta["memory_history"], groups)
    report["dynamics"] = {}
    for i in range(n):
        path = method.run_dir / "method" / f"gpm_logs_task{i}.json"
        if path.is_file():
            logs = json.loads(path.read_text())
            dyn = c_trajectories(logs)
            residuals = [r["max_residual_over_bound"] for r in logs.get("residuals", {}).values()]
            report["dynamics"][str(i)] = {
                "projected": logs.get("projected"),
                "gradient_c_mean": dyn["gradient_c"]["mean_of_medians"],
                "update_c_mean": dyn["update_c"]["mean_of_medians"],
                "worst_residual_over_bound": max(residuals, default=None),
                "median_residual_over_bound": _median_or_none(residuals),
                "gradient_c_per_step": dyn["gradient_c"]["per_step"],
                "update_c_per_step": dyn["update_c"]["per_step"],
            }

    if pilot_json and Path(pilot_json).is_file():
        payload = json.loads(Path(pilot_json).read_text())
        pilot_ns = payload.get("evaluation_seed_run_id")
        method_ns = method.result.get("seed_namespace_run_id")
        if pilot_ns != method_ns:
            # The pilot's episodes are paired only within its own seed namespace; comparing
            # them with another seed's cells would be a silently unpaired difference.
            report["pilot_comparison"] = {
                "skipped": f"pilot rolled out under {pilot_ns!r}, this run under {method_ns!r}"
            }
        else:
            pilot = payload["arms"]["gpm_projected_adam"]["evaluation"]
            report["pilot_comparison"] = {
                key: {
                    "pilot": pilot[key]["value"],
                    "sequence_stage1": method.cell(1, method.task_keys.index(key)).estimate.value,
                    "paired_diff": _paired(
                        method.cell(1, method.task_keys.index(key)).successes,
                        pilot[key]["successes"], bootstrap,
                    ),
                }
                for key in pilot
            }
    gate2_t1 = reference.run_dir / "bases" / "task0.pt"
    memory_t1 = method.run_dir / "method" / "memory_task0.pt"
    if gate2_t1.is_file() and memory_t1.is_file():
        gate2_bases, _ = load_bases(gate2_t1)
        memory_bases, memory_meta = load_bases(memory_t1)
        report.setdefault("pilot_comparison", {})["t1_memory_overlap"] = t1_memory_overlap(
            gate2_bases, memory_bases, memory_meta["eps"], groups
        )

    report["provenance_checks"] = provenance_checks(method, reference, verify_checkpoints)
    return report


def registry_groups(run_dir: Path) -> dict[str, str] | None:
    """Layer -> §4.5 group, from the run's stage-0 checkpoint (``None`` if absent)."""
    path = run_dir / "checkpoints" / "stage0.pt"
    if not path.is_file():
        return None
    from flowcl.train.checkpoint import load_checkpoint

    policy = load_checkpoint(path).policy
    return {entry.name: entry.group for entry in policy.projectable_layers()}


def _max_steps(run_dir: Path) -> int | None:
    """The rollout step limit from the run's ``config.yaml`` (a failure at it is a timeout)."""
    path = run_dir / "config.yaml"
    if not path.is_file():
        return None
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True).get("eval", {}).get("max_steps")


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
    for stage, cap in report.get("capacity", {}).items():
        parts = []
        for scope in ("trunk", "decoder"):
            v = cap[scope]["median_new_energy_protected"]
            parts.append(f"{scope} {'-' if v is None else f'{v:.2f}'}")
        print(f"  new energy protected after T{int(stage) + 1} (median): " + ", ".join(parts)
              + f"; unprotected total: trunk {cap['trunk']['median_unprotected_energy']:.3f}, "
              f"decoder {cap['decoder']['median_unprotected_energy']:.3f}")
    for task, rows in report.get("episode_transitions", {}).items():
        short = task.split("/")[0].replace("libero_", "")
        print(f"  episodes {short}: " + "; ".join(
            f"{t} both {r['both']} lost {r['lost']} gained {r['gained']}" for t, r in rows.items()
        ))
    for stage, dyn in report.get("dynamics", {}).items():
        brief = {k: v for k, v in dyn.items() if not k.endswith("_per_step")}
        print(f"  stage {stage} dynamics: {brief}")
    overlap = report.get("pilot_comparison", {}).get("t1_memory_overlap")
    if overlap:
        for scope in ("trunk", "decoder"):
            s = overlap["summary"][scope]
            print(f"  T1 memory vs Gate 2 (eps {overlap['eps']}), {scope}: s/k_a median "
                  f"{s['median_s_over_k_a']:.3f} (min {s['min_s_over_k_a']:.3f}), s/k_b median "
                  f"{s['median_s_over_k_b']:.3f} (min {s['min_s_over_k_b']:.3f}), k_b - k_a in "
                  f"[{s['min_k_b_minus_k_a']}, {s['max_k_b_minus_k_a']}]")
    for name, check in report["provenance_checks"].items():
        passed = check.get("passed") if isinstance(check, dict) else None
        extra = ""
        if name == "frozen_from_stage1" and passed is not None:
            extra = (f" ({check['n_tensors_outside_allowlist']} tensors outside the allowlist; "
                     f"allowlisted moved per stage {check['moved_allowlisted']})")
        if name == "memory_chained":
            extra = f" (prefix_identical {check['prefix_identical']})"
        print(f"  [{'PASS' if passed else 'INFO' if passed is None else 'FAIL'}] {name}{extra}")


def run_sequence_report(
    method_dir: Path, reference_dir: Path, pilot_json: Path | None, out: Path
) -> dict:
    report = build_report(method_dir, reference_dir, pilot_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print_report(report)
    print(f"[flowcl] wrote {out}", flush=True)
    return report
