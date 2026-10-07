"""C1, the high-protection control: does the adaptive allocation matter, or only more protection?

``gpm_projected_adam_e99`` extends T1's memory to ``eps = 0.95`` like plain and adaptive GPM, then
every later task's to a flat ``eps_later = 0.99``. At Object after Goal (``R[2][1]``) the arms
differ only in how the T2 memory is allocated, at approximately matched total protection (the
prediction is in the rule's header). This module evaluates the rule pre-registered in
``configs/analysis/high_protection.yaml``:

* **gates** — identity against plain GPM, the per-layer energy targets, the validity of e99 *and*
  every comparator (production, matrices, statistics, episode pairing, the sequence reports'
  required checks, thresholds), the premise (plain forgot Object), the adaptive reference's
  transition support, and Goal learned under the 0.99 memory;
* **practical verdict** — ``protection_suffices`` / ``flat_control_inferior`` / ``unresolved``
  from G = e99 − plain and D = e99 − adaptive at ``R[2][1]`` (rollout-paired CIs);
* **flags**, independent of the verdict: ``adaptive_advantage``, ``e99_advantage``,
  ``gain_not_reproduced``, ``within_margin_by_point_only``;
* **descriptive** — later cells, ACC/NBT/AUC, capacity per stage (median ρ and the
  parameter-weighted protected fraction), the T2 allocation against the offline prediction.

Every report pins the rule's SHA-256 and the hashes of the files its numbers come from.
:func:`classify` and :func:`classify_replication` are pure.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import torch
from omegaconf import OmegaConf

from flowcl.utils.libero_paths import repo_root
from flowcl.utils.run import atomic_write_text, file_sha256, git_sha

ROLES = ("seq_ft", "plain", "adaptive", "e99")
MEMORY_ROLES = ("plain", "adaptive", "e99")
N_TASKS = 4
RATE_ATOL = 1e-9  # rates are k/50 and differences of such; allow float rounding only

VERDICTS = {
    "invalid_comparison": "Invalid comparison: e99's stages 0-1 or T1 memory differ from plain GPM.",
    "invalid_implementation": "Invalid implementation: e99's memory history does not show the "
                              "registered targets on every layer and task.",
    "invalid_run": "Invalid run: a validity check of e99 or a comparator failed or lacks evidence.",
    "not_applicable": "Not applicable: plain GPM did not forget Object across Goal.",
    "invalid_reference": "Invalid reference: adaptive GPM does not show transition support here.",
    "plasticity_failure": "Plasticity failure: Goal was not learned under the 0.99 memory.",
    "protection_suffices": "A flat 0.99 target reproduces the transition benefit within the "
                           "pre-specified margin; adaptive targets are unnecessary for this cell "
                           "under this protocol.",
    "flat_control_inferior": "The flat 0.99 control is worse than adaptive GPM by more than the "
                             "margin at Object after Goal.",
    "unresolved": "Unresolved: neither non-inferiority within the margin nor inferiority beyond it "
                  "is established.",
}

REPLICATION = {
    "inconclusive_invalid": "Inconclusive: at least one seed is invalid.",
    "not_applicable": "Not applicable: at least one seed's plain GPM did not forget Object.",
    "protection_suffices_replicated": "Every seed: a flat 0.99 target reproduces the transition "
                                      "benefit within the margin.",
    "flat_control_inferior_replicated": "Every seed: the flat control is worse than adaptive by "
                                        "more than the margin.",
    "plasticity_failure_replicated": "Every seed: Goal not learned under the 0.99 memory.",
    "mixed": "Mixed: the seeds' verdicts differ.",
}

FLAGS = ("adaptive_advantage", "e99_advantage", "gain_not_reproduced", "within_margin_by_point_only")


def rule_path(path: str | Path | None = None) -> Path:
    return Path(path) if path else repo_root() / "configs" / "analysis" / "high_protection.yaml"


def load_config(path: str | Path | None = None) -> dict:
    return OmegaConf.to_container(OmegaConf.load(rule_path(path)), resolve=True)


def run_names(cfg: dict, seed: int) -> dict[str, str]:
    from flowcl.experiments.reverse_order import method_spec
    from flowcl.train.continual import continual_run_id

    return {role: continual_run_id(method_spec(cfg, role)[2], cfg["curriculum"], seed) for role in ROLES}


def _at_least(value: float, threshold: float) -> bool:
    return value >= threshold - RATE_ATOL


# ---- the verdict (pure) --------------------------------------------------------------------------


def classify(checks: dict, D: dict | None, margin: float) -> dict:
    """The verdict and flags from the gate results and ``D = e99 - adaptive`` at ``R[2][1]``.

    ``checks``: booleans ``identity``, ``energy``, ``valid``, ``premise`` (plain forgot),
    ``reference`` (adaptive shows transition support), ``t3`` (e99 learned Goal), ``gain`` (G
    reproduced). ``D``: ``{"diff", "low", "high"}`` (``None`` only when a gate stops first).
    """
    def stop(verdict: str) -> dict:
        return {"verdict": verdict, "text": VERDICTS[verdict], "flags": []}

    for key, verdict in (("identity", "invalid_comparison"), ("energy", "invalid_implementation"),
                         ("valid", "invalid_run")):
        if not checks[key]:
            return stop(verdict)
    if not checks["premise"]:
        return stop("not_applicable")
    if not checks["reference"]:
        return stop("invalid_reference")
    flags = []
    if D["high"] < 0.0:
        flags.append("adaptive_advantage")
    if D["low"] > 0.0:
        flags.append("e99_advantage")
    if not checks["gain"]:
        flags.append("gain_not_reproduced")
    if D["diff"] >= -margin - RATE_ATOL and not D["low"] > -margin:
        flags.append("within_margin_by_point_only")
    if not checks["t3"]:
        return {**stop("plasticity_failure"), "flags": flags}
    if checks["gain"] and D["low"] > -margin:
        verdict = "protection_suffices"
    elif D["high"] < -margin:
        verdict = "flat_control_inferior"
    else:
        verdict = "unresolved"
    return {"verdict": verdict, "text": VERDICTS[verdict], "flags": flags}


def classify_replication(verdicts: dict[str, str]) -> dict:
    """The symmetric replication rule over every registered seed's verdict."""
    values = set(verdicts.values())
    if any(v.startswith("invalid_") for v in values):
        outcome = "inconclusive_invalid"
    elif "not_applicable" in values:
        outcome = "not_applicable"
    elif values == {"protection_suffices"}:
        outcome = "protection_suffices_replicated"
    elif values == {"flat_control_inferior"}:
        outcome = "flat_control_inferior_replicated"
    elif values == {"plasticity_failure"}:
        outcome = "plasticity_failure_replicated"
    else:
        outcome = "mixed"
    return {"replication": outcome, "text": REPLICATION[outcome], "verdicts": dict(verdicts)}


# ---- checks ---------------------------------------------------------------------------------------


def energy_check(history: dict, memory_config: dict, expected_layers: list[str], cfg: dict) -> dict:
    """Every task 0-3, exactly the expected layers, every layer's recorded target and capture.

    ``history``: task -> layer -> ``extend_basis`` info, from the final memory artifact.
    Missing evidence fails; no layer is exempt (a flat target has no new-energy share).
    """
    history = {str(k): v for k, v in history.items()}
    failures = []
    if sorted(history, key=int) != [str(t) for t in cfg["tasks"]]:
        failures.append({"condition": "tasks", "found": sorted(history, key=int),
                         "expected": cfg["tasks"]})
    if memory_config.get("eps_later") != cfg["eps_later"]:
        failures.append({"condition": "memory config eps_later", "found": memory_config.get("eps_later")})
    n_checked = 0
    for task in cfg["tasks"]:
        layers = history.get(str(task))
        if layers is None:
            continue
        if sorted(layers) != sorted(expected_layers):
            failures.append({"task": task, "condition": "layers",
                             "missing": sorted(set(expected_layers) - set(layers)),
                             "extra": sorted(set(layers) - set(expected_layers))})
        target = cfg["eps"] if task == 0 else cfg["eps_later"]
        for layer, info in layers.items():
            n_checked += 1
            row = {"task": task, "layer": layer}
            values = [info.get(k) for k in ("proj_energy_fraction", "target_fraction",
                                            "captured_energy_fraction")]
            if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
                failures.append({**row, "condition": "energy values missing or non-finite", "values": values})
                continue
            _, recorded, captured = values
            if abs(recorded - target) > cfg["target_match_atol"]:
                failures.append({**row, "condition": "recorded target", "recorded": recorded,
                                 "target": target})
            if captured < target - cfg["tol"]:
                failures.append({**row, "condition": "target not met", "captured": captured,
                                 "target": target})
    return {"passed": not failures and n_checked > 0, "n_checked": n_checked, "failures": failures}


def production(run_dir: Path, cfg: dict, role: str, seed: int, smoke: bool) -> dict:
    """:func:`reverse_order.production_check`, plus the one registered excuse (see the rule)."""
    from flowcl.experiments.reverse_order import production_check

    if not (run_dir / "config.yaml").is_file():
        return {"passed": False, "problems": {"config.yaml": "missing"}}
    check = production_check(run_dir, cfg, role, seed, smoke)
    expected_ns = f"{cfg['curriculum']}__seq_ft__seed{seed}"
    if (not check["passed"] and role == "seq_ft" and check["problems"] == {"namespace": None}
            and run_dir.name == expected_ns):
        check = {**check, "passed": True, "problems": {},
                 "excused": "seed_namespace_run_id absent (predates the field); the run is the namespace"}
    return check


def required_report_check(path: Path, required: list[str], expected_run: str, smoke: bool) -> dict:
    """Every required provenance check present and passed (``clean_git_sha`` excused in smoke),
    and the report is for the expected run. Missing evidence fails."""
    if not path.is_file():
        return {"passed": False, "missing": str(path)}
    report = json.loads(path.read_text())
    checks = report.get("provenance_checks") or {}
    problems = []
    if report.get("method_run_id") != expected_run:
        problems.append(f"report is for {report.get('method_run_id')!r}, expected {expected_run!r}")
    for name in required:
        check = checks.get(name)
        if not isinstance(check, dict) or "passed" not in check:
            problems.append(f"{name}: absent")
        elif not check["passed"] and not (smoke and name == "clean_git_sha"):
            problems.append(f"{name}: failed")
    return {"passed": not problems, "problems": problems, "present": sorted(checks)}


# ---- capacity -------------------------------------------------------------------------------------


def layer_dims(checkpoint: Path, layers: list[str]) -> dict[str, tuple[int, int]]:
    """``layer -> (d_out, d_in)`` from a stage checkpoint's ``<layer>.weight`` shapes."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    return {name: tuple(state[f"{name}.weight"].shape) for name in layers}


def protected_fractions(history: dict, dims: dict[str, tuple[int, int]]) -> dict:
    """Per stage: GPM's parameter-weighted protected fraction ``Σ d_out k / Σ d_out d_in``, for all
    registry weights and per half (:func:`flowcl.analysis.sensitivity.protected_fraction`)."""
    from flowcl.analysis.sensitivity import _half, protected_fraction

    out = {}
    for stage, layers in sorted(history.items(), key=lambda kv: int(kv[0])):
        if sorted(layers) != sorted(dims):
            raise ValueError(f"stage {stage}: memory layers differ from the checkpoint's registry")
        sets: dict[str, list[str]] = {"all": list(layers)}
        for name in layers:
            if layers[name]["d_in"] != dims[name][1]:
                raise ValueError(f"{name}: memory d_in {layers[name]['d_in']} != weight d_in {dims[name][1]}")
            sets.setdefault(_half(name), []).append(name)
        out[str(stage)] = {
            scope: protected_fraction([dims[n][0] for n in names], [layers[n]["k_after"] for n in names],
                                      [dims[n][0] for n in names], [dims[n][1] for n in names])["gpm"]
            for scope, names in sets.items()
        }
    return out


def predicted_t2_ranks(bases: dict, history: dict, eps_later: float) -> dict[str, int]:
    """e99's T2 ``k_added`` per layer, predicted from the adaptive run's T2 extension.

    The two runs' T2 Grams are equal (identical stage-1 weights and capture seeds). The adaptive
    artifact stores the T2 residual spectrum (``singular_values**2``) and the info, from which
    ``tr K = Σ_{i<=k_added} λ_i / (captured − p)``; e99 adds the smallest ``k`` with
    ``p tr K + Σ_{i<=k} λ_i >= eps_later tr K`` (as :func:`extend_basis`). ``-1`` marks a layer
    the procedure cannot predict (the adaptive extension added nothing but e99 would).
    """
    out = {}
    for name, info in history.items():
        p = info["proj_energy_fraction"]
        if p >= eps_later:
            out[name] = 0
            continue
        if info["k_added"] == 0 or info["captured_energy_fraction"] <= p:
            out[name] = -1
            continue
        lam = bases[name].singular_values.to(torch.float64) ** 2
        lam = lam[lam > 0]
        total = float(lam[: info["k_added"]].sum()) / (info["captured_energy_fraction"] - p)
        cumulative = p * total + torch.cumsum(lam, dim=0)
        target = torch.tensor([eps_later * total], dtype=torch.float64)
        k = int(torch.searchsorted(cumulative, target, side="left")) + 1
        out[name] = max(1, min(k, lam.numel(), info["d_in"] - info["k_before"]))
    return out


def t2_allocation(histories: dict[str, dict], fractions: dict[str, dict], predicted_ranks: dict,
                  eps_later: float, predicted_fraction: dict | None, tolerance_pp: float) -> dict:
    """The T2 extension of adaptive against e99: targets, ranks, realised and predicted protection."""
    from flowcl.analysis.sensitivity import _half

    ad, e99 = histories["adaptive"]["1"], histories["e99"]["1"]
    per_half = {}
    for half in ("trunk", "decoder"):
        names = [n for n in ad if _half(n) == half]
        diffs = [ad[n]["target_fraction"] - eps_later for n in names]
        per_half[half] = {
            "n_layers": len(names),
            "share_adaptive_target_above": sum(d > 0 for d in diffs) / len(names),
            "median_adaptive_minus_flat_target": statistics.median(diffs),
            "k_added": {"adaptive": sum(ad[n]["k_added"] for n in names),
                        "e99": sum(e99[n]["k_added"] for n in names)},
        }
    realised = {role: fractions[role]["1"] for role in ("plain", "adaptive", "e99")}
    delta_pp = {scope: 100.0 * (realised["e99"][scope] - realised["adaptive"][scope])
                for scope in realised["e99"]}
    matched = abs(delta_pp["all"]) <= tolerance_pp
    rank_match = sum(predicted_ranks.get(n) == e99[n]["k_added"] for n in e99)
    return {
        "per_half": per_half,
        "realised_fraction_after_t2": realised,
        "e99_minus_adaptive_pp": delta_pp,
        "protection_matched": matched,
        "matched_total_tolerance_pp": tolerance_pp,
        "reading": ("approximately matched total protection" if matched
                    else f"protection not matched (delta = {delta_pp['all']:+.2f} pp)"),
        "predicted_fraction_pct": predicted_fraction,
        "realised_fraction_pct": {r: 100.0 * realised[r]["all"] for r in ("e99", "adaptive")},
        "ranks_matching_prediction": {"matching": rank_match, "n_layers": len(e99),
                                      "unpredictable": sorted(n for n, k in predicted_ranks.items() if k < 0)},
    }


# ---- the report ----------------------------------------------------------------------------------


def input_hashes(rule: Path, dirs: dict[str, Path], reports: dict[str, Path], bootstrap: dict,
                 cfg: dict) -> dict:
    """SHA-256 of every file the report's numbers come from (absent files are recorded as such)."""
    def sha(path: Path) -> str | None:
        return file_sha256(path) if path.is_file() else None

    runs = {}
    for role, d in dirs.items():
        files = {"result.json": sha(d / "result.json"), "config.yaml": sha(d / "config.yaml")}
        for i in range(N_TASKS):
            files[f"eval/stage{i}.json"] = sha(d / "eval" / f"stage{i}.json")
        if role in MEMORY_ROLES:
            for t in range(N_TASKS):
                files[f"method/memory_task{t}.pt"] = sha(d / "method" / f"memory_task{t}.pt")
        runs[role] = files
    methods = repo_root() / "configs" / "method"
    eval_cfg = repo_root() / "configs" / "eval" / "libero_eval.yaml"
    return {
        "rule": {"path": str(rule), "sha256": file_sha256(rule)},
        "runs": runs,
        "sequence_reports": {role: sha(p) for role, p in reports.items()},
        "method_configs": {role: sha(methods / f"{cfg['methods'][role]}.yaml") for role in ROLES},
        "libero_eval.yaml": sha(eval_cfg),
        "bootstrap": bootstrap,
    }


def _guard(fn, *args, **kwargs) -> dict:
    """A check's result, or a failed check carrying the error (missing or malformed evidence)."""
    try:
        return fn(*args, **kwargs)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        return {"passed": False, "error": f"{type(exc).__name__}: {exc}"}


def build_report(cfg: dict, seed: int, *, results_root: Path | None = None, smoke: bool = False,
                 rule: Path | None = None) -> dict:
    from flowcl.analysis.subspace import load_bases
    from flowcl.experiments.adaptive_report import consume_threshold_blocks, identity_check
    from flowcl.experiments.reverse_order import _load_view, seed_pairing, stats_check
    from flowcl.experiments.sequence_report import capacity_by_stage, paired_cells, registry_groups

    root = Path(results_root) if results_root else repo_root() / "results"
    rule = rule_path(rule)
    names = run_names(cfg, seed)
    dirs = {role: root / name for role, name in names.items()}
    reports = {role: root / cfg["reports"][role][seed] / "report.json" for role in MEMORY_ROLES}
    n_episodes = (cfg["smoke"] if smoke else cfg["production"])["n_episodes"]
    bootstrap = OmegaConf.to_container(
        OmegaConf.load(repo_root() / "configs" / "eval" / "libero_eval.yaml"), resolve=True)["bootstrap"]

    report: dict = {"git_sha": git_sha(), "seed": seed, "smoke": smoke, "config": cfg,
                    "runs": names, "inputs": input_hashes(rule, dirs, reports, bootstrap, cfg)}

    # Gate 1: identity (re-verified here, not only read from the runner's files).
    identity = _guard(identity_check, dirs["e99"], dirs["plain"], cfg["identity_stages"])

    # Gate 2: energy, against plain's layer registry.
    def energy() -> dict:
        _, meta = load_bases(dirs["e99"] / "method" / f"memory_task{N_TASKS - 1}.pt")
        plain_bases, _ = load_bases(dirs["plain"] / "method" / f"memory_task{N_TASKS - 1}.pt")
        return energy_check(meta["memory_history"], meta.get("config", {}), sorted(plain_bases), cfg["energy"])
    energy_result = _guard(energy)

    # Gate 3: validity of e99 and every comparator.
    views, view_problems = {}, []
    for role in ROLES:
        views[role], p = _load_view(dirs[role], n_episodes)
        view_problems += p
    validity = {
        "matrices": {"passed": not view_problems, "problems": view_problems},
        **{f"production_{role}": _guard(production, dirs[role], cfg, role, seed, smoke) for role in ROLES},
        **{f"sequence_report_{role}": required_report_check(reports[role], cfg["required_checks"],
                                                             names[role], smoke)
           for role in MEMORY_ROLES},
    }
    thresholds = None
    if all(v is not None for v in views.values()):
        expected_task = views["seq_ft"].task_keys[0]
        validity["statistics"] = _guard(stats_check, dirs, expected_task)
        validity["episode_pairing"] = {
            other: seed_pairing(views["e99"], views[other]) for other in ("seq_ft", "plain", "adaptive")}
        validity["episode_pairing"]["passed"] = all(v["passed"] for v in validity["episode_pairing"].values())

        def threshold_check() -> dict:
            blocks = {k: json.loads(reports[r].read_text())["threshold_block"]
                      for k, r in (("variant", "e99"), ("baseline", "adaptive"))}
            block = consume_threshold_blocks(blocks, names["seq_ft"], file_sha256(dirs["seq_ft"] / "result.json"),
                                             {role: list(v.task_keys) for role, v in views.items()})
            return {"passed": True, "block": block}
        validity["thresholds"] = _guard(threshold_check)
        if validity["thresholds"]["passed"]:
            thresholds = validity["thresholds"]["block"]["thresholds"]
    valid = all(c.get("passed") for c in validity.values()) and thresholds is not None

    checks = {"identity": bool(identity.get("passed")), "energy": bool(energy_result.get("passed")),
              "valid": valid}
    cells, primary, D = None, None, None
    if valid:
        paired = {"e99_vs_plain": paired_cells(views["e99"], views["plain"], bootstrap),
                  "e99_vs_adaptive": paired_cells(views["e99"], views["adaptive"], bootstrap),
                  "adaptive_vs_plain": paired_cells(views["adaptive"], views["plain"], bootstrap)}

        def cell(spec: list[int]) -> dict:
            i, j = spec
            est = views["e99"].cell(i, j).estimate
            return {"cell": spec, "e99": est.value, "e99_ci": [est.low, est.high],
                    "plain": views["plain"].cell(i, j).estimate.value,
                    "adaptive": views["adaptive"].cell(i, j).estimate.value,
                    "threshold": thresholds[j],
                    "e99_vs_plain": paired["e99_vs_plain"][f"{i},{j}"],
                    "e99_vs_adaptive": paired["e99_vs_adaptive"][f"{i},{j}"]}

        p_i, p_j = cfg["cells"]["primary"]
        t_i, t_j = cfg["cells"]["t3"]
        G = paired["e99_vs_plain"][f"{p_i},{p_j}"]
        D = paired["e99_vs_adaptive"][f"{p_i},{p_j}"]
        G_ad = paired["adaptive_vs_plain"][f"{p_i},{p_j}"]
        theta_obj, theta_goal = thresholds[p_j], thresholds[t_j]
        plain_primary = views["plain"].cell(p_i, p_j).estimate.value
        adaptive_t3 = views["adaptive"].cell(t_i, t_j).estimate.value
        checks.update({
            "premise": plain_primary < theta_obj - RATE_ATOL,
            "reference": (_at_least(G_ad["diff"], cfg["min_improvement"]) and G_ad["low"] > 0.0
                          and _at_least(adaptive_t3, theta_goal)),
            "t3": _at_least(views["e99"].cell(t_i, t_j).estimate.value, theta_goal),
            "gain": _at_least(G["diff"], cfg["min_improvement"]) and G["low"] > 0.0,
        })
        primary = {"cell": cfg["cells"]["primary"], "G": G, "D": D, "adaptive_vs_plain": G_ad,
                   "plain": plain_primary, "theta_object": theta_obj, "theta_goal": theta_goal,
                   "e99_goal_learned": views["e99"].cell(t_i, t_j).estimate.value,
                   "adaptive_goal_learned": adaptive_t3,
                   "D_sign": (D["diff"] > 0) - (D["diff"] < 0)}
        cells = {name: cell(spec) for name, spec in cfg["cells"]["descriptive"].items()}
    else:
        checks.update({"premise": False, "reference": False, "t3": False, "gain": False})

    verdict = classify(checks, D, cfg["noninferiority_margin"])
    report.update({"verdict": verdict, "checks": checks, "identity": identity, "energy": energy_result,
                   "validity": validity, "primary": primary, "cells": cells})

    # Descriptive: metrics, capacity, T2 allocation. Never changes the verdict.
    def descriptive() -> dict:
        metrics = {}
        for role in ROLES:
            m = json.loads((dirs[role] / "result.json").read_text()).get("metrics", {})
            metrics[role] = {"ACC": m.get("F_1"), "NBT": m.get("NBT"), "AUC": m.get("AUC")}
        histories, groups = {}, registry_groups(dirs["e99"])
        for role in MEMORY_ROLES:
            _, meta = load_bases(dirs[role] / "method" / f"memory_task{N_TASKS - 1}.pt")
            histories[role] = {str(k): v for k, v in meta["memory_history"].items()}
        dims = layer_dims(dirs["e99"] / "checkpoints" / f"stage{N_TASKS - 1}.pt", sorted(histories["e99"]["0"]))
        fractions = {role: protected_fractions(histories[role], dims) for role in MEMORY_ROLES}
        ad_bases, ad_meta = load_bases(dirs["adaptive"] / "method" / "memory_task1.pt")
        ranks = predicted_t2_ranks(ad_bases, {k: v for k, v in ad_meta["memory_history"]["1"].items()},
                                   cfg["energy"]["eps_later"])
        predicted = None if smoke else cfg["predicted_t2_protected_fraction"].get(seed)
        return {
            "metrics": metrics,
            "capacity": {role: capacity_by_stage(histories[role], groups) for role in MEMORY_ROLES},
            "protected_fraction": fractions,
            "t2_allocation": t2_allocation(histories, fractions, ranks, cfg["energy"]["eps_later"],
                                           predicted, cfg["matched_total_tolerance_pp"]),
        }
    report["descriptive"] = _guard(descriptive)
    return report


def print_report(report: dict) -> None:
    v = report["verdict"]
    print(f"\n[flowcl] high-protection seed {report['seed']}: {v['verdict']} {v['flags']} — {v['text']}",
          flush=True)
    print(f"  checks: {report['checks']}")
    failed = [k for k, c in report["validity"].items() if not c.get("passed")]
    if failed:
        print(f"  failed validity checks: {failed}")
    if report["primary"]:
        p = report["primary"]
        for key in ("G", "D", "adaptive_vs_plain"):
            print(f"  {key}: {p[key]['diff']:+.2f} [{p[key]['low']:+.2f}, {p[key]['high']:+.2f}]")
    alloc = (report.get("descriptive") or {}).get("t2_allocation")
    if alloc:
        print(f"  after T2: {alloc['reading']}; e99 - adaptive {alloc['e99_minus_adaptive_pp']} pp; "
              f"ranks matching prediction {alloc['ranks_matching_prediction']['matching']}/"
              f"{alloc['ranks_matching_prediction']['n_layers']}")


def run_report(cfg: dict | None = None, seed: int = 0, *, results_root: Path | None = None,
               out: Path | None = None, smoke: bool = False, rule: Path | None = None) -> dict:
    cfg = cfg or load_config(rule)
    if seed not in cfg["seeds"]:
        raise ValueError(f"seed {seed} is not registered ({cfg['seeds']})")
    report = build_report(cfg, seed, results_root=results_root, smoke=smoke, rule=rule)
    root = Path(results_root) if results_root else repo_root() / "results"
    out = Path(out) if out else root / cfg["out"].format(seed=seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out, json.dumps(report, indent=2, default=str) + "\n")
    print_report(report)
    print(f"[flowcl] wrote {out}", flush=True)
    return report


def build_replication(cfg: dict, *, results_root: Path | None = None, rule: Path | None = None) -> dict:
    """Every registered seed's report, as written; refuses smoke, foreign-rule or wrong-run reports."""
    root = Path(results_root) if results_root else repo_root() / "results"
    rule_sha = file_sha256(rule_path(rule))
    per_seed, problems = {}, []
    for seed in cfg["seeds"]:
        path = root / cfg["out"].format(seed=seed)
        if not path.is_file():
            problems.append(f"seed {seed}: no report at {path}")
            continue
        report = json.loads(path.read_text())
        if report.get("seed") != seed:
            problems.append(f"seed {seed}: report is for seed {report.get('seed')}")
        if report.get("smoke") is not False:
            problems.append(f"seed {seed}: a smoke report")
        if (report.get("inputs") or {}).get("rule", {}).get("sha256") != rule_sha:
            problems.append(f"seed {seed}: written under another rule")
        if report.get("runs") != run_names(cfg, seed):
            problems.append(f"seed {seed}: report names other runs")
        per_seed[str(seed)] = report
    if problems:
        raise ValueError("replication refused: " + "; ".join(problems))
    outcome = classify_replication({s: r["verdict"]["verdict"] for s, r in per_seed.items()})
    return {
        "git_sha": git_sha(), "rule_sha256": rule_sha, "seeds": cfg["seeds"], **outcome,
        "flags": {flag: [int(s) for s, r in per_seed.items() if flag in r["verdict"]["flags"]] for flag in FLAGS},
        "per_seed": {s: {"verdict": r["verdict"], "primary": r["primary"],
                         "protection": ((r.get("descriptive") or {}).get("t2_allocation") or {}).get("reading"),
                         "git_sha": r.get("git_sha")}
                     for s, r in per_seed.items()},
    }


def run_replication(cfg: dict | None = None, *, results_root: Path | None = None,
                    rule: Path | None = None) -> dict:
    cfg = cfg or load_config(rule)
    summary = build_replication(cfg, results_root=results_root, rule=rule)
    root = Path(results_root) if results_root else repo_root() / "results"
    out = root / cfg["replication_out"]
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out, json.dumps(summary, indent=2, default=str) + "\n")
    print(f"\n[flowcl] high-protection replication: {summary['replication']} — {summary['text']}", flush=True)
    for seed, row in summary["per_seed"].items():
        p = row["primary"] or {}
        d = p.get("D") or {}
        print(f"  seed {seed}: {row['verdict']['verdict']} {row['verdict']['flags']}; "
              f"D {d.get('diff')} [{d.get('low')}, {d.get('high')}]; {row['protection']}", flush=True)
    print(f"  flags by seed: {summary['flags']}")
    print(f"[flowcl] wrote {out}", flush=True)
    return summary
