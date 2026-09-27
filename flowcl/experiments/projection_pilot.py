"""Exploratory projection pilot on the low-update backbone (seed 0, T1 -> T2 only).

Once optimizer effects are controlled by the low-update recipe, is hard projection still
too restrictive, and where is SGP's stability-plasticity region? Object is trained after
Spatial from seq_ft seed 0's ``stage0.pt`` (paired as in :mod:`flowcl.experiments.low_update`)
under each configuration of ``configs/analysis/projection_pilot.yaml``: freeze-only and hard
GPM (controls), SGP at several strengths and two learning rates (the eligible candidates),
and SGP alpha 5 (negative control). Runs only behind the pilot gate
(:func:`flowcl.experiments.low_update.pilot_gate_open`).

Per run: plasticity and retention (paired against seq_ft, the GPM runs and the low-update
fine-tuning reference), probe losses, the realised update split against the T1 memory
(:func:`flowcl.experiments.gate3.update_interference`), the projection logs, and the
*nominal* protected-gradient scale ``eta (1 - lambda)`` (never read as realised protection).
:func:`select_configuration` applies the registered selection to eligible runs only.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

from omegaconf import OmegaConf

from flowcl.experiments.hparam_study import _success, default_evaluator, paired_against
from flowcl.experiments.low_update import (
    _root,
    load_low_update_config,
    load_published,
    pilot_gate_open,
    refuse_dirty,
    seed_context,
    train_stage,
)
from flowcl.utils.libero_paths import repo_root
from flowcl.utils.run import atomic_write_text, git_sha

TOL = 1e-9


def load_pilot_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else repo_root() / "configs" / "analysis" / "projection_pilot.yaml"
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def run_name(entry: dict, seed: int) -> str:
    alpha = f"{entry['alpha']:g}" if "alpha" in entry else ""
    return f"pilot__{entry['family']}{alpha}__lr{entry['lr']:g}__seed{seed}"


def is_eligible(entry: dict, eligible: dict) -> bool:
    return (entry["family"] == eligible["family"]
            and any(math.isclose(entry.get("alpha", -1), a) for a in eligible["alphas"])
            and any(math.isclose(entry["lr"], lr) for lr in eligible["lrs"]))


def _half(name: str) -> str:
    return "trunk" if name.startswith("trunk.") else "decoder"


def make_method(entry: dict, bases: dict, eps: float):
    """``setup_method``: freeze-only (SeqFT after the §7.4 freeze), hard GPM, or SGP."""
    from flowcl.methods.gpm import GPM, freeze_to_allowlist
    from flowcl.methods.seq_ft import SeqFT
    from flowcl.methods.sgp import SGP

    def setup(policy):
        family = entry["family"]
        if family == "freeze_only":
            freeze_to_allowlist(policy)  # before the optimiser is built, as for GPM
            return SeqFT()
        if family == "gpm":
            method = GPM(eps=eps, update_memory=False)
        elif family == "sgp":
            method = SGP(alpha=float(entry["alpha"]), eps=eps, update_memory=False)
        else:
            raise ValueError(f"unknown pilot family {family!r}")
        method.set_memory(bases)
        return method

    return setup


def make_extra_row(entry: dict, bases: dict, eps: float):
    """Realised update split against the T1 memory, projection logs, nominal scale."""
    import torch

    from flowcl.experiments.gate3 import update_interference

    def extra(policy, method, start_state) -> dict:
        state = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
        split = update_interference(start_state, state, bases, [eps])
        halves: dict = {}
        for name, layer in split["per_layer"].items():
            halves.setdefault(_half(name), []).append(layer["delta_norm"] ** 2)
        realised = {}
        for half, energies in halves.items():
            total = math.sqrt(sum(energies))
            c = split["c_global"][str(eps)].get(half)
            realised[half] = {"delta_norm": total, "parallel_fraction": c,
                              "parallel_norm": None if c is None else c * total,
                              "perpendicular_norm": None if c is None else math.sqrt(max(0.0, 1 - c * c)) * total}
        out = {"update_split": {"eps": eps, "halves": realised,
                                "c_global": split["c_global"][str(eps)]}}
        logs = getattr(method, "task_logs", {}).get(1) if method is not None else None
        if logs:
            residuals = [r["max_residual_over_bound"] for r in logs.get("residuals", {}).values()]
            last = logs["update_c"][max(logs["update_c"])] if logs.get("update_c") else {}
            out["projection_logs"] = {
                "worst_residual_over_bound": max(residuals, default=None),
                "update_c_last_median": {
                    h: statistics.median([v for n, v in last.items() if _half(n) == h and v == v])
                    for h in ("trunk", "decoder")
                    if any(_half(n) == h for n in last)},
            }
        importance = getattr(method, "_importance", None) if entry["family"] == "sgp" else None
        if importance:
            per = {h: [] for h in ("trunk", "decoder")}
            for name, lam in importance.items():
                per[_half(name)].append(float((1.0 - lam.to(torch.float64)).mean()))
            out["nominal_protected_scale"] = {
                "note": "eta (1 - lambda), nominal only: not the realised update",
                **{h: entry["lr"] * statistics.median(v) for h, v in per.items() if v}}
        elif entry["family"] == "gpm":
            out["nominal_protected_scale"] = {"note": "hard projection: lambda = 1",
                                              "trunk": 0.0, "decoder": 0.0}
        return out

    return extra


def run_pilot(cfg: dict, lr: float, results_root=None, device="cuda", allow_dirty=False,
              evaluator=None, bootstrap=None, dataset_dir=None, build_datasets=None,
              low_update_cfg: dict | None = None) -> list[dict]:
    """Every grid entry at ``lr``, behind the pilot gate."""
    from flowcl.analysis.subspace import load_bases

    refuse_dirty(allow_dirty)
    root = _root(results_root)
    lu = low_update_cfg or load_low_update_config()
    if not pilot_gate_open(lu, root):
        raise RuntimeError("the pilot gate is closed (validation failed or the chain under-learns)")
    ctx = seed_context(lu, int(cfg["seed"]), root)
    if evaluator is None:
        evaluator, bootstrap = default_evaluator()
    bases, _ = load_bases(root / cfg["memory_source"])
    rows = []
    for entry in cfg["grid"]:
        if not math.isclose(entry["lr"], lr):
            continue
        name = run_name(entry, ctx.seed)
        rows.append(train_stage(
            ctx, 1, cfg["steps"], entry["lr"], ctx.ref_dir / "checkpoints" / "stage0.pt",
            root / cfg["out"] / name, name, method_spec=entry,
            setup_method=make_method(entry, bases, cfg["eps"]),
            extra_row=make_extra_row(entry, bases, cfg["eps"]),
            evaluator=evaluator, bootstrap=bootstrap, device=device, dataset_dir=dataset_dir,
            build_datasets=build_datasets,
        ))
    return rows


# ---- selection (pure) ---------------------------------------------------------------------------


def select_configuration(rows: list[dict], eligible: dict, object_floor: float) -> dict:
    """Best Spatial retention among eligible runs with Object >= floor; probe loss breaks ties."""
    def key(r):
        return (-r["cells"]["0"]["success"], r["cells"]["0"]["probe_loss"])

    candidates = [r for r in rows if is_eligible(r["method"], eligible)]
    qualifying = [r for r in candidates if r["cells"]["1"]["success"] >= object_floor - TOL]
    chosen = min(qualifying, key=key) if qualifying else None
    projected = [r for r in rows if r["method"]["family"] in ("gpm", "sgp")
                 and r["cells"]["1"]["success"] >= object_floor - TOL]
    best_overall = min(projected, key=key) if projected else None
    return {
        "object_floor": object_floor,
        "eligible": [r["run_id"] for r in candidates],
        "qualifying": [r["run_id"] for r in qualifying],
        "selected": None if chosen is None else {"run_id": chosen["run_id"], "method": chosen["method"]},
        "best_overall_projected": None if best_overall is None else {
            "run_id": best_overall["run_id"], "method": best_overall["method"],
            "note": "descriptive: includes controls, never the selection"},
    }


def run_pilot_summary(cfg: dict, results_root=None, low_update_cfg: dict | None = None,
                      bootstrap: dict | None = None) -> dict:
    from flowcl.envs.evaluation import EvaluationReport

    root = _root(results_root)
    lu = low_update_cfg or load_low_update_config()
    ctx = seed_context(lu, int(cfg["seed"]), root)
    if bootstrap is None:
        _, bootstrap = default_evaluator()
    refs = {}
    for ref in cfg["ft_references"]:
        src = root / ref["run"]
        report = EvaluationReport.load(src / "eval.json")
        if report.run_id != ctx.namespace:
            raise ValueError(f"{ref['run']}: rollouts under {report.run_id!r}, not {ctx.namespace!r}")
        row = json.loads((src / "row.json").read_text())
        refs[ref["lr"]] = {"run": ref["run"], "git_sha": (src / "git_sha").read_text().strip(),
                           "report": report.by_task(),
                           "probe_loss": {"0": row["probe_loss"]["retained"], "1": row["probe_loss"]["new"]}}
    keys = [ctx.curriculum.stages[j].task_key for j in (0, 1)]
    rows = []
    for entry in cfg["grid"]:
        name = run_name(entry, ctx.seed)
        row = load_published(root / cfg["out"] / name, {"run_id": name, "method": entry})
        if row is None:
            raise FileNotFoundError(f"pilot run {name} is not published")
        own = EvaluationReport.load(root / cfg["out"] / name / "eval.json").by_task()
        ft = refs[min(refs, key=lambda lr: abs(lr - entry["lr"]))]
        row["paired_vs_ft_same_lr"] = {
            str(j): paired_against(own[keys[j]], ft["report"][keys[j]], bootstrap) for j in (0, 1)}
        row["eligible"] = is_eligible(entry, cfg["eligible"])
        rows.append(row)
    sel = cfg["selection"]
    ft_ref = refs[min(refs, key=lambda lr: abs(lr - sel["object_reference_lr"]))]
    floor = ft_ref["report"][keys[1]].estimate.value - sel["object_tolerance"]
    out = {
        "git_sha": git_sha(), "config": cfg,
        "ft_references": {f"{lr:g}": {"run": r["run"], "git_sha": r["git_sha"],
                                      **{str(j): {**_success(r["report"][keys[j]]),
                                                  "probe_loss": r["probe_loss"][str(j)]}
                                         for j in (0, 1)}}
                          for lr, r in refs.items()},
        "selection": select_configuration(rows, cfg["eligible"], floor),
        "rows": rows,
    }
    path = root / cfg["out"] / "pilot_summary.json"
    atomic_write_text(path, json.dumps(out, indent=2) + "\n")
    print(f"[flowcl] projection pilot: selected {out['selection']['selected']}", flush=True)
    return out
