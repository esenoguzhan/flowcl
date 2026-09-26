"""Pre-registered outcome of the SGP runs: the SGP baseline and adaptive SGP (seq_hetero).

Four paired runs per seed (same seq_ft seed namespace, so rollout episodes and probe
batches are matched): **P** the SGP baseline (``sgp_projected_adam``, an AdamW adaptation —
never "paper SGP"), **A** adaptive SGP (``sgp_projected_adam_ne90``, the candidate thesis
method), **G** plain GPM and **N** adaptive GPM (both existing). The rule is
``configs/analysis/sgp.yaml``:

* **gates** — identity (P = G at stage 0; A = P at stages 0-1; equal T1 memories; P's T1
  importance is Eq. 2 of G's stored T1 spectrum), implementation (Eq. 2/10 recomputed from
  every stored importance update, the scaled-update residuals, adaptive SGP's energy
  target), the premise (G forgot Object across Goal) and a positive T4 probe-loss gap
  ``L_N − L_G``;
* **O1** (retention preserved) — ``R_A[2][1] − R_G[2][1] >= 0.20`` with the paired CI > 0;
* **O2** (at least half of the T4 probe-loss gap recovered, T3/T4 performance acceptable) —
  ``g = (L_N − L_A)/(L_N − L_G) >= 0.5`` and ``R_A[3][3]``, ``R_A[2][2]`` at their thresholds.
  O2 is *not* rollout recovery;
* **attribution** (secondary) — adaptive memory from the paired ``A − P`` Object contrast,
  scaled projection from ``g`` (loss level) and the paired ``A − N`` T4 contrast (rollout).

:func:`classify_sgp` maps one seed's checks to its verdict, :func:`classify_attribution`
gives the attribution levels, and :func:`classify_sgp_replication` applies the symmetric
replication rule and the two claim levels across seeds 0-2.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from flowcl.utils.libero_paths import repo_root
from flowcl.utils.run import atomic_write_text, git_sha

ROLES = ("P", "A", "G", "N")
ROLE_NAMES = {"P": "SGP baseline (AdamW adaptation)", "A": "adaptive SGP",
              "G": "plain GPM", "N": "adaptive GPM"}
CONTRASTS = {"A-G": ("A", "G"), "A-N": ("A", "N"), "A-P": ("A", "P"),
             "P-G": ("P", "G"), "N-G": ("N", "G")}

VERDICTS = {
    "invalid_comparison": "Invalid comparison: a checked stage, the T1 memory or the T1 "
                          "importance differs from its pair.",
    "invalid_implementation": "Invalid implementation: a stored importance update does not "
                              "recompute, a scaled update left its bound, or adaptive SGP "
                              "missed its energy target.",
    "not_applicable_baseline_retained": "Not applicable: plain GPM did not forget Object "
                                        "across Goal, so there was no retention to preserve.",
    "not_applicable_no_plasticity_gap": "Not applicable: adaptive GPM's T4 probe loss is not "
                                        "above plain GPM's, so there is no gap to recover.",
    "both_objectives_met": "Both practical objectives met: Object retention preserved (O1), "
                           "and at least half of the T4 probe-loss gap recovered with "
                           "acceptable T3/T4 task performance (O2).",
    "retention_only": "Retention preserved (O1), but O2 not met.",
    "plasticity_only": "O2 met but Object retention not preserved: a trade-off.",
    "neither": "Neither objective met.",
}
INVALID = {"invalid_comparison", "invalid_implementation"}
NOT_APPLICABLE = {"not_applicable_baseline_retained", "not_applicable_no_plasticity_gap"}

CLAIMS = {
    0: "No registered claim: adaptive SGP's two practical objectives did not both replicate.",
    1: "Adaptive SGP meets both practical objectives (replicated over seeds 0-2).",
    2: "Adaptive SGP meets both practical objectives, and the improvement is causally "
       "attributable to contributions from both adaptive memory allocation and scaled "
       "projection (scaled projection at the {level} level).",
}


def load_sgp_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else repo_root() / "configs" / "analysis" / "sgp.yaml"
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def resolve_seed(cfg: dict, seed: int) -> dict:
    """The shared rule plus one seed's runs, reports and output (a flat dict)."""
    seeds = {int(k): v for k, v in cfg["seeds"].items()}
    if seed not in seeds:
        raise ValueError(f"seed {seed} is not pre-registered in sgp.yaml ({sorted(seeds)})")
    missing = set(ROLES) - set(seeds[seed]["runs"])
    if missing:
        raise ValueError(f"seed {seed}: runs missing roles {sorted(missing)}")
    shared = {k: v for k, v in cfg.items() if k not in ("seeds", "replication")}
    return {**shared, **seeds[seed], "seed": seed}


def limits(cfg: dict, thresholds: list[float]) -> dict:
    """Each check's threshold: the seed's per-task threshold at the check's task."""
    return {
        "premise": thresholds[cfg["premise"]["baseline_cell"][1]],
        "durable": thresholds[cfg["retention"]["durable_cell"][1]],
        "t4": thresholds[cfg["plasticity"]["t4_cell"][1]],
        "t3": thresholds[cfg["plasticity"]["t3_cell"][1]],
        "min_improvement": cfg["retention"]["min_improvement"],
        "min_gap_recovery": cfg["plasticity"]["min_gap_recovery"],
    }


def consume_all_blocks(blocks: dict[str, dict], reference_run: str, reference_sha256: str,
                       result_task_keys: dict[str, list[str]]) -> dict:
    """Every run's sequence-report threshold block must agree with the first one's."""
    from flowcl.experiments.adaptive_report import consume_threshold_blocks

    roles = list(blocks)
    for other in roles[1:]:
        try:
            consume_threshold_blocks(
                {"variant": blocks[roles[0]], "baseline": blocks[other]},
                reference_run, reference_sha256, result_task_keys,
            )
        except ValueError as err:
            raise ValueError(f"threshold blocks {roles[0]} vs {other}: {err}") from err
    return blocks[roles[0]]


# ---- the verdicts (pure) ------------------------------------------------------------


def _at_least(value: float, threshold: float) -> bool:
    # Inclusive; rates are k/50 and differences of such, so allow float rounding only.
    return value >= threshold - 1e-9


def gap_recovery(L_A: float, L_N: float, L_G: float) -> float | None:
    """``g = (L_N − L_A) / (L_N − L_G)``; ``None`` when there is no gap (``L_N <= L_G``)."""
    gap = L_N - L_G
    return None if gap <= 0.0 else (L_N - L_A) / gap


def objective_checks(transition: dict, durable: float, g: float | None, t4: float, t3: float,
                     baseline_transition: float, lim: dict) -> dict:
    """O1, O2 and their parts (``transition``: paired A − G at the transition cell)."""
    o1 = _at_least(transition["diff"], lim["min_improvement"]) and transition["low"] > 0.0
    o2_parts = {
        "gap_recovery": g is not None and g >= lim["min_gap_recovery"],
        "t4": _at_least(t4, lim["t4"]),
        "t3": _at_least(t3, lim["t3"]),
    }
    return {
        "premise": baseline_transition < lim["premise"] - 1e-9,
        "plasticity_gap": g is not None,
        "o1": o1,
        "durable": _at_least(durable, lim["durable"]),
        "o2": all(o2_parts.values()),
        "o2_parts": o2_parts,
    }


def classify_sgp(checks: dict) -> dict:
    """One seed's verdict: the gates in order, then O1 x O2."""
    def stop(verdict: str) -> dict:
        return {"verdict": verdict, "text": VERDICTS[verdict], "o1": None, "o2": None,
                "flags": []}

    if not checks["identity"]:
        return stop("invalid_comparison")
    if not checks["implementation"]:
        return stop("invalid_implementation")
    if not checks["premise"]:
        return stop("not_applicable_baseline_retained")
    if not checks["plasticity_gap"]:
        return stop("not_applicable_no_plasticity_gap")
    o1, o2 = checks["o1"], checks["o2"]
    verdict = {(True, True): "both_objectives_met", (True, False): "retention_only",
               (False, True): "plasticity_only", (False, False): "neither"}[(o1, o2)]
    flags = []
    if o1 and not checks["durable"]:
        flags.append("durable Object retention below threshold")
    if not o2:
        flags += [f"O2 part failed: {k}" for k, ok in checks["o2_parts"].items() if not ok]
    return {"verdict": verdict, "text": VERDICTS[verdict], "o1": o1, "o2": o2, "flags": flags}


def classify_attribution(memory: dict, g: float | None, scaling_rollout: dict,
                         min_gap_recovery: float) -> dict:
    """Secondary attribution levels (``memory``: paired A − P; ``scaling_rollout``: A − N)."""
    if memory["low"] > 0.0:
        mem = "strong"
    elif memory["diff"] > 0.0:
        mem = "directional"
    else:
        mem = "none"
    if g is None:
        scaling = "not_applicable"
    elif g >= min_gap_recovery:
        scaling = "loss_level"
    elif g > 0.0:
        scaling = "directional"
    else:
        scaling = "none"
    return {"memory": mem, "scaling": scaling, "scaling_rollout": scaling_rollout["low"] > 0.0,
            "memory_contrast": memory, "gap_recovery": g, "scaling_rollout_contrast": scaling_rollout}


def _replicated(values: list[bool]) -> str:
    if all(values):
        return "replicated"
    if not any(values):
        return "consistently_not_met"
    return "not_replicated"


def classify_sgp_replication(per_seed: dict, seeds: list[int]) -> dict:
    """Symmetric replication over every registered seed; then the claim level.

    ``per_seed``: seed -> ``{"verdict": classify_sgp(...), "attribution": ... or None}``.
    """
    if sorted(int(s) for s in per_seed) != sorted(seeds):
        raise ValueError(f"replication needs exactly seeds {sorted(seeds)}, got {sorted(per_seed)}")
    verdicts = {str(s): row["verdict"]["verdict"] for s, row in per_seed.items()}
    values = set(verdicts.values())
    out: dict = {"verdicts": verdicts, "o1": None, "o2": None, "attribution": None,
                 "claim_level": 0, "claim": CLAIMS[0]}
    if values & INVALID:
        return {**out, "validity": "inconclusive_invalid"}
    if values & NOT_APPLICABLE:
        return {**out, "validity": "not_applicable"}
    o1 = _replicated([row["verdict"]["o1"] for row in per_seed.values()])
    o2 = _replicated([row["verdict"]["o2"] for row in per_seed.values()])
    attr = {str(s): row["attribution"] for s, row in per_seed.items()}
    memory = ("replicated_strong" if all(a["memory"] == "strong" for a in attr.values())
              else "not_replicated")
    scaling = ("replicated_loss_level" if all(a["scaling"] == "loss_level" for a in attr.values())
               else "not_replicated")
    rollout = ("replicated_rollout" if all(a["scaling_rollout"] for a in attr.values())
               else "not_replicated")
    level, claim = 0, CLAIMS[0]
    if o1 == o2 == "replicated":
        level, claim = 1, CLAIMS[1]
        if memory == "replicated_strong" and scaling == "replicated_loss_level":
            reached = "loss and rollout" if rollout == "replicated_rollout" else "loss"
            level, claim = 2, CLAIMS[2].format(level=reached)
    return {
        **out,
        "validity": "valid",
        "o1": o1,
        "o2": o2,
        "attribution": {
            "memory": memory,
            "scaling": scaling,
            "scaling_rollout": rollout,
            "per_seed": {s: {"memory": a["memory"], "scaling": a["scaling"],
                             "scaling_rollout": a["scaling_rollout"]} for s, a in attr.items()},
        },
        "claim_level": level,
        "claim": claim,
    }


# ---- bootstrap over matched units ---------------------------------------------------


def episode_interaction(successes: dict[str, list], seeds: dict[str, list], bootstrap: dict) -> dict:
    """``(A − P) − (N − G)`` per matched rollout episode, bootstrapped over episodes."""
    from flowcl.analysis.metrics import bootstrap_ci

    if any(list(seeds[r]) != list(seeds["A"]) for r in ROLES):
        raise ValueError("rollout seeds differ between the four runs: the cells are not paired")
    x = {r: np.asarray(successes[r], dtype=np.float64) for r in ROLES}
    est = bootstrap_ci(x["A"] - x["P"] - x["N"] + x["G"], seed=bootstrap["seed"],
                       n_bootstrap=bootstrap["n_resamples"], confidence=bootstrap["confidence"])
    return {"diff": est.value, "low": est.low, "high": est.high, "n": est.n,
            "unit": "matched rollout episode"}


def _batch_arrays(cells: dict, key: str) -> tuple[dict, np.ndarray]:
    losses = {r: np.asarray(cells[r][key]["losses"], dtype=np.float64) for r in ROLES}
    weights = {r: np.asarray(cells[r][key]["weights"], dtype=np.float64) for r in ROLES}
    if any(not np.array_equal(weights[r], weights["A"]) for r in ROLES):
        raise ValueError(f"probe cell {key}: batch weights differ, the batches are not matched")
    return losses, weights["A"]


def _resample(n: int, boot: dict) -> np.ndarray:
    rng = np.random.default_rng(boot["seed"])
    return rng.integers(0, n, size=(boot["n_resamples"], n))


def _quantiles(values: np.ndarray, point: float, boot: dict) -> tuple[float, float]:
    alpha = 1.0 - boot["confidence"]
    low, high = np.quantile(values, [alpha / 2, 1.0 - alpha / 2])
    return float(min(low, point)), float(max(high, point))


def batch_interaction(cells: dict, key: str, boot: dict) -> dict:
    """``(A − P) − (N − G)`` of the weighted probe loss, bootstrapped over matched batches."""
    losses, w = _batch_arrays(cells, key)
    d = losses["A"] - losses["P"] - losses["N"] + losses["G"]
    point = float((w * d).sum() / w.sum())
    idx = _resample(d.size, boot)
    stats = (w[idx] * d[idx]).sum(axis=1) / w[idx].sum(axis=1)
    low, high = _quantiles(stats, point, boot)
    return {"diff": point, "low": low, "high": high, "n": int(d.size), "unit": "matched probe batch"}


def batch_gap_recovery(cells: dict, key: str, boot: dict, role: str = "A") -> dict:
    """``g`` for ``role`` with a CI from resampling matched probe batches (reported only)."""
    losses, w = _batch_arrays(cells, key)
    idx = _resample(w.size, boot)
    means = {r: (w[idx] * losses[r][idx]).sum(axis=1) / w[idx].sum(axis=1) for r in ROLES}
    full = {r: float((w * losses[r]).sum() / w.sum()) for r in ROLES}
    point = gap_recovery(full[role], full["N"], full["G"])
    gap = means["N"] - means["G"]
    defined = gap > 0
    g = (means["N"][defined] - means[role][defined]) / gap[defined]
    low = high = None
    if point is not None and g.size:
        low, high = _quantiles(g, point, boot)
    return {"g": point, "low": low, "high": high, "n": int(w.size),
            "undefined_resamples": int((~defined).sum()), "unit": "matched probe batch"}


# ---- the checks ---------------------------------------------------------------------


def _eq2(sigma: np.ndarray, alpha: float) -> np.ndarray:
    """Eq. 2, written out here independently of :mod:`flowcl.analysis.importance`."""
    r = sigma / sigma.max()
    return np.minimum((alpha + 1.0) * r / (alpha * r + 1.0), 1.0)


def _memory(run_dir: Path, task: int):
    from flowcl.analysis.subspace import load_bases

    return load_bases(run_dir / "method" / f"memory_task{task}.pt")


def importance_identity(dirs: dict[str, Path], alpha: float, atol: float) -> dict:
    """A's T1 importance equals P's, and P's equals Eq. 2 of G's stored T1 spectrum."""
    g0, _ = _memory(dirs["G"], 0)
    p0, _ = _memory(dirs["P"], 0)
    a0, _ = _memory(dirs["A"], 0)
    missing = sorted(n for n in g0 if n not in p0 or "importance" not in p0[n].meta)
    adaptive_equal = sorted(a0) == sorted(p0) and all(
        "importance" in a0[n].meta and "importance" in p0[n].meta
        and torch.equal(a0[n].meta["importance"], p0[n].meta["importance"]) for n in p0
    )
    worst = 0.0
    for name, basis in g0.items():
        if name in missing:
            continue
        k = basis.vectors.shape[1]
        expected = _eq2(basis.singular_values[:k].to(torch.float64).numpy(), alpha)
        got = p0[name].meta["importance"].to(torch.float64).numpy()
        worst = math.inf if got.shape != expected.shape else max(worst, float(np.abs(got - expected).max()))
    t1_match = not missing and worst <= atol
    return {"passed": adaptive_equal and t1_match, "adaptive_equals_baseline_t1": adaptive_equal,
            "t1_matches_gpm_spectrum": t1_match, "max_abs_diff": worst, "atol": atol,
            "missing": missing}


def importance_check(run_dir: Path, n_tasks: int, alpha: float, atol: float) -> dict:
    """Every stored importance update, every layer and task: Eq. 2 and Eq. 10 recompute."""
    failures: list[dict] = []
    prev: dict[str, np.ndarray] = {}
    n_checked = 0
    for task in range(n_tasks):
        bases, meta = _memory(run_dir, task)
        if meta.get("config", {}).get("alpha") != alpha:
            failures.append({"task": task, "layer": None,
                             "condition": f"alpha {meta.get('config', {}).get('alpha')} != {alpha}"})
        for name, basis in bases.items():
            m = basis.meta
            before = prev.get(name, np.zeros(0))
            if "importance" not in m or "importance_update" not in m \
                    or m.get("importance_update_task") != task:
                failures.append({"task": task, "layer": name,
                                 "condition": "importance record missing or from another task"})
                continue
            n_checked += 1
            lam = m["importance"].to(torch.float64).numpy()
            upd = {k: v.to(torch.float64).numpy() for k, v in m["importance_update"].items()}
            k = basis.vectors.shape[1]
            conds = []
            if lam.shape != (k,):
                conds.append("not one importance per basis vector")
            if upd["lambda_prev"].shape != before.shape or not np.array_equal(upd["lambda_prev"], before):
                conds.append("lambda_prev is not the previous task's importance")
            if upd["sigma_old"].size != before.size or upd["sigma_old"].size + upd["sigma_new"].size != k:
                conds.append("sigma counts do not match the old and new bases")
            sv = basis.singular_values.to(torch.float64).numpy()
            n_new = upd["sigma_new"].size
            if n_new and float(np.abs(upd["sigma_new"] - sv[:n_new]).max()) > atol * max(float(sv.max()), 1.0):
                conds.append("sigma-hat differs from the stored residual spectrum")
            if not conds:
                lam_task = _eq2(np.concatenate([upd["sigma_old"], upd["sigma_new"]]), alpha)
                if abs(float(lam_task.max()) - 1.0) > atol:
                    conds.append("max lambda_task is not 1")
                if float(np.abs(lam_task - upd["lambda_task"]).max()) > atol:
                    conds.append("Eq. 2 does not recompute")
                expected = np.concatenate([np.minimum(before + lam_task[: before.size], 1.0),
                                           lam_task[before.size:]])
                if float(np.abs(expected - lam).max()) > atol:
                    conds.append("Eq. 10 does not recompute")
                if not bool(((lam > 0.0) & (lam <= 1.0)).all()):
                    conds.append("importance outside (0, 1]")
                if before.size and bool((lam[: before.size] < before).any()):
                    conds.append("an old importance decreased")
            failures += [{"task": task, "layer": name, "condition": c} for c in conds]
            prev[name] = lam
    return {"passed": not failures and n_checked > 0, "n_checked": n_checked,
            "n_failures": len(failures), "failures": failures[:50]}


def residual_check(run_dir: Path, n_tasks: int) -> dict:
    """Every projected task's worst scaled-update residual stayed within its bound."""
    worst, missing = 0.0, []
    for task in range(n_tasks):
        path = run_dir / "method" / f"gpm_logs_task{task}.json"
        if not path.is_file():
            missing.append(task)
            continue
        logs = json.loads(path.read_text())
        for row in logs.get("residuals", {}).values():
            worst = max(worst, row["max_residual_over_bound"])
    return {"passed": not missing and worst <= 1.0, "worst_residual_over_bound": worst,
            "missing_logs": missing}


def _half(name: str) -> str:
    return "trunk" if name.startswith("trunk.") else "decoder"


def importance_summary(memory_history: dict) -> dict:
    """Per task and half: medians over layers of λ's median, hard share, protected energy."""
    out = {}
    for task, layers in sorted(memory_history.items(), key=lambda kv: int(kv[0])):
        halves: dict = {}
        for name, info in layers.items():
            imp = info.get("importance")
            if imp is not None:
                halves.setdefault(_half(name), []).append(imp)
        out[str(task)] = {
            half: {key: statistics.median(r[key] for r in rows)
                   for key in ("lambda_median", "hard_fraction", "protected_energy_fraction")}
            for half, rows in halves.items()
        }
    return out


# ---- the report ---------------------------------------------------------------------


def build_sgp_report(cfg: dict, results_root: Path | None = None) -> dict:
    """``cfg``: one seed's resolved rule (:func:`resolve_seed`)."""
    from flowcl.analysis.subspace import load_bases
    from flowcl.experiments.adaptive_report import energy_check, identity_check
    from flowcl.experiments.sequence_report import (
        capacity_by_stage,
        load_run,
        paired_cells,
        registry_groups,
    )
    from flowcl.utils.run import file_sha256

    root = Path(results_root) if results_root else repo_root() / "results"
    dirs = {r: root / cfg["runs"][r] for r in ROLES}
    ref_dir = root / cfg["reference_run"]
    runs = {r: load_run(d) for r, d in dirs.items()}
    reference = load_run(ref_dir)
    n = runs["A"].n_tasks
    bootstrap = OmegaConf.to_container(
        OmegaConf.load(repo_root() / "configs" / "eval" / "libero_eval.yaml"), resolve=True
    )["bootstrap"]

    seq_reports = {r: json.loads((root / cfg["sequence_reports"][r]).read_text()) for r in ROLES}
    diags = {r: json.loads((root / cfg["diagnostics"][r]).read_text()) for r in ROLES}
    for r in ROLES:
        for kind, report in (("sequence report", seq_reports[r]), ("diagnostics", diags[r])):
            if report["method_run_id"] != dirs[r].name:
                raise ValueError(f"{r}: the {kind} is for {report['method_run_id']!r}, "
                                 f"not {dirs[r].name!r}")
    block = consume_all_blocks(
        {r: seq_reports[r]["threshold_block"] for r in ROLES}, ref_dir.name,
        file_sha256(ref_dir / "result.json"),
        {**{r: list(v.task_keys) for r, v in runs.items()}, "reference": list(reference.task_keys)},
    )
    thresholds = block["thresholds"]
    lim = limits(cfg, thresholds)
    alpha = float(cfg["alpha"])

    # Gates.
    ident = cfg["identity"]
    identity = {
        "baseline_vs_gpm": identity_check(dirs["P"], dirs["G"], ident["baseline_vs_gpm_stages"]),
        "adaptive_vs_baseline": identity_check(dirs["A"], dirs["P"],
                                               ident["adaptive_vs_baseline_stages"]),
        "importance": importance_identity(dirs, alpha, ident["importance_atol"]),
    }
    atol = cfg["implementation"]["importance_atol"]
    _, a_meta = load_bases(dirs["A"] / "method" / f"memory_task{n - 1}.pt")
    implementation = {
        "importance": {r: importance_check(dirs[r], n, alpha, atol) for r in ("P", "A")},
        "residuals": {r: residual_check(dirs[r], n) for r in ("P", "A")},
        "energy_adaptive": energy_check(a_meta["memory_history"], cfg["energy"]),
    }
    implementation_ok = (
        all(v["passed"] for v in implementation["importance"].values())
        and all(v["passed"] for v in implementation["residuals"].values())
        and implementation["energy_adaptive"]["passed"]
    )

    # Rollouts (paired) and probe losses (the diagnostics' fixed batches).
    contrasts = {name: paired_cells(runs[a], runs[b], bootstrap) for name, (a, b) in CONTRASTS.items()}

    def cell(spec: list[int], contrast: str) -> dict:
        i, j = spec
        return {"cell": spec, **{r: runs[r].cell(i, j).estimate.value for r in ROLES},
                "paired": contrasts[contrast][f"{i},{j}"]}

    li, lj = cfg["plasticity"]["loss_cell"]
    L = {r: diags[r]["loss_matrix"]["method"][li][lj] for r in ROLES}
    g = gap_recovery(L["A"], L["N"], L["G"])
    transition = cell(cfg["retention"]["transition_cell"], "A-G")
    durable = cell(cfg["retention"]["durable_cell"], "A-G")
    t4 = cell(cfg["plasticity"]["t4_cell"], "A-N")
    t3 = cell(cfg["plasticity"]["t3_cell"], "A-N")
    baseline_transition = runs["G"].cell(*cfg["premise"]["baseline_cell"]).estimate.value
    checks = {
        "identity": all(v["passed"] for v in identity.values()),
        "implementation": implementation_ok,
        **objective_checks(transition["paired"], durable["A"], g, t4["A"], t3["A"],
                           baseline_transition, lim),
    }
    verdict = classify_sgp(checks)
    attribution = None
    if checks["identity"] and checks["implementation"]:
        attribution = classify_attribution(
            cell(cfg["attribution"]["memory_cell"], "A-P")["paired"], g,
            cell(cfg["attribution"]["scaling_rollout_cell"], "A-N")["paired"],
            lim["min_gap_recovery"],
        )

    # Reported, not decisive.
    probe = json.loads((root / cfg["probe_report"]).read_text())
    probe_problems = [
        f"{r}: {probe['runs'].get(r)!r} != {dirs[r].name!r}" for r in ROLES
        if probe["runs"].get(r) != dirs[r].name
    ]
    if probe_problems:
        raise ValueError(f"{cfg['probe_report']} is for other runs: {probe_problems}")
    loss_key = f"{li},{lj}"
    for r in ROLES:
        mean = probe["cells"][r][loss_key]["mean"]
        if abs(mean - L[r]) > cfg["probe_cells"]["rel_tol"] * abs(L[r]):
            raise ValueError(f"{r}: per-batch probe mean {mean} does not reproduce the "
                             f"diagnostics' L[{li}][{lj}] = {L[r]}")
    boot = cfg["batch_bootstrap"]
    ri, rj = cfg["interaction"]["rollout_cell"]
    rollout_cells = {r: runs[r].cell(ri, rj) for r in ROLES}
    ii, ij = cfg["interaction"]["loss_cell"]
    reported = {
        "contrasts": contrasts,
        "interaction": {
            "rollout": {"cell": [ri, rj], **episode_interaction(
                {r: c.successes for r, c in rollout_cells.items()},
                {r: c.seeds for r, c in rollout_cells.items()}, bootstrap)},
            "probe_loss": {"cell": [ii, ij],
                           **batch_interaction(probe["cells"], f"{ii},{ij}", boot)},
        },
        "gap_recovery_batch_ci": {r: batch_gap_recovery(probe["cells"], loss_key, boot, role=r)
                                  for r in ("A", "P")},
        "gap_recovery_baseline": gap_recovery(L["P"], L["N"], L["G"]),
        "probe_loss": {"cell": [li, lj], **L},
        "t4_last50_training_loss": {r: runs[r].result["stages"][n - 1]["mean_last_50_loss"]
                                    for r in ROLES},
        "F_1": {r: runs[r].result["metrics"]["F_1"] for r in ROLES},
        "importance_summary": {
            r: importance_summary(load_bases(dirs[r] / "method" / f"memory_task{n - 1}.pt")[1]
                                  ["memory_history"]) for r in ("P", "A")
        },
        "update_c_mean": {r: {t: d.get("update_c_mean") for t, d in seq_reports[r].get("dynamics", {}).items()}
                          for r in ROLES},
    }
    groups = registry_groups(dirs["A"])
    return {
        "git_sha": git_sha(),
        "seed": cfg.get("seed"),
        "config": cfg,
        "roles": {r: {"run": dirs[r].name, "name": ROLE_NAMES[r]} for r in ROLES},
        "threshold_block": block,
        "limits": lim,
        "verdict": verdict,
        "checks": checks,
        "attribution": attribution,
        "cells": {"transition": transition, "durable": durable, "t4": t4, "t3": t3,
                  "premise": {"cell": cfg["premise"]["baseline_cell"], "G": baseline_transition,
                              "threshold": lim["premise"]}},
        "gap_recovery": {"g": g, "L": L, "gap": L["N"] - L["G"]},
        "identity": identity,
        "implementation": implementation,
        "reported": reported,
        "capacity": {r: capacity_by_stage(
            load_bases(dirs[r] / "method" / f"memory_task{n - 1}.pt")[1]["memory_history"], groups)
            for r in ("P", "A")},
    }


def print_sgp_report(report: dict) -> None:
    v, c = report["verdict"], report["checks"]
    print(f"\n[flowcl] adaptive SGP, seed {report['seed']}: {v['verdict']} — {v['text']}", flush=True)
    if v["flags"]:
        print(f"  flags: {v['flags']}")
    print(f"  gates: identity {c['identity']}, implementation {c['implementation']}, "
          f"premise {c['premise']}, plasticity gap {c['plasticity_gap']}")
    for name in ("transition", "durable", "t3", "t4"):
        row = report["cells"][name]
        p = row["paired"]
        print(f"  {name} {row['cell']}: A {row['A']:.2f} P {row['P']:.2f} G {row['G']:.2f} "
              f"N {row['N']:.2f}; paired {p['diff']:+.2f} [{p['low']:+.2f}, {p['high']:+.2f}]")
    gr = report["gap_recovery"]
    g = "n/a" if gr["g"] is None else f"{gr['g']:.3f}"
    print(f"  T4 probe loss: A {gr['L']['A']:.5f} P {gr['L']['P']:.5f} G {gr['L']['G']:.5f} "
          f"N {gr['L']['N']:.5f}; g = {g}")
    if report["attribution"]:
        a = report["attribution"]
        print(f"  attribution: memory {a['memory']}, scaling {a['scaling']}, "
              f"rollout {a['scaling_rollout']}")


def run_sgp_report(cfg: dict | None = None, seed: int = 0, out: Path | None = None,
                   results_root: Path | None = None) -> dict:
    resolved = resolve_seed(cfg or load_sgp_config(), seed)
    report = build_sgp_report(resolved, results_root)
    root = Path(results_root) if results_root else repo_root() / "results"
    out = Path(out) if out else root / resolved["out"]
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out, json.dumps(report, indent=2, default=str) + "\n")
    print_sgp_report(report)
    print(f"[flowcl] wrote {out}", flush=True)
    return report


def build_sgp_replication(cfg: dict, results_root: Path | None = None) -> dict:
    """Read every registered seed's report (as written) and apply the replication rule."""
    root = Path(results_root) if results_root else repo_root() / "results"
    seeds = [int(s) for s in cfg["replication"]["seeds"]]
    per_seed = {}
    for seed in seeds:
        resolved = resolve_seed(cfg, seed)
        report = json.loads((root / resolved["out"]).read_text())
        if report.get("config", {}).get("runs") != resolved["runs"]:
            raise ValueError(f"seed {seed}: {resolved['out']} is not the report for {resolved['runs']}")
        per_seed[str(seed)] = {
            "verdict": report["verdict"],
            "attribution": report["attribution"],
            "transition_gain": report["cells"]["transition"]["paired"],
            "gap_recovery": report["gap_recovery"]["g"],
            "t4": {r: report["cells"]["t4"][r] for r in ROLES},
            "git_sha": report.get("git_sha"),
        }
    outcome = classify_sgp_replication(per_seed, seeds)
    return {"git_sha": git_sha(), "seeds": seeds, **outcome, "per_seed": per_seed}


def run_sgp_replication(cfg: dict | None = None, results_root: Path | None = None) -> dict:
    cfg = cfg or load_sgp_config()
    summary = build_sgp_replication(cfg, results_root)
    root = Path(results_root) if results_root else repo_root() / "results"
    out = root / cfg["replication"]["out"]
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out, json.dumps(summary, indent=2, default=str) + "\n")
    print(f"\n[flowcl] SGP replication over seeds {summary['seeds']}: validity "
          f"{summary['validity']}; O1 {summary['o1']}; O2 {summary['o2']}", flush=True)
    print(f"  claim level {summary['claim_level']}: {summary['claim']}", flush=True)
    if summary["attribution"]:
        print(f"  attribution: {summary['attribution']}", flush=True)
    print(f"[flowcl] wrote {out}", flush=True)
    return summary
