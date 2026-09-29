"""SGP sensitivity on the standard recipe: bracket on seed 0, freeze, validate on seeds 1-2.

Fine-tuning optimization is closed (``docs/runs/2026-09-29_optimizer_study.md``); SGP at alpha 25
was too soft (``docs/runs/2026-09-27_sgp_seed0.md``). This study calibrates SGP's basic
protection at T1 -> T2 under the standard recipe (30k steps, lr 1e-4), per
``configs/analysis/sgp_sensitivity.yaml``:

1. **bracket** (seed 0): SGP alpha in {25, 100, 400, 1600} with freeze-only, hard GPM and
   fine-tuning as controls (:func:`run_bracket`; reused arms verified by :func:`reused_arm`);
2. **select** among eligible alphas only: the smallest alpha whose Object stays within
   tolerance of fine-tuning and whose Spatial stays within tolerance of hard GPM
   (:func:`select_alpha`);
3. **freeze**: an immutable selection bundle published by one atomic rename
   (:func:`publish_selection`, :func:`load_selection`);
4. **validate** on seeds held out from alpha selection (:func:`run_validation`,
   :func:`classify_validation`) — no rescue tuning;
5. **summary**, always (:func:`run_summary`): nominal attenuation (:func:`nominal_attenuation`),
   the realised update split, probe units, and the verdict.

Runs train through :func:`flowcl.experiments.low_update.train_stage` (paired, published
atomically) with :func:`flowcl.experiments.projection_pilot.make_method`.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from flowcl.experiments.hparam_study import default_evaluator, paired_against
from flowcl.experiments.low_update import (
    _root,
    abandoned_attempts,
    load_low_update_config,
    load_published,
    refuse_dirty,
    seed_context,
    stage_cells,
    train_stage,
)
from flowcl.utils.libero_paths import repo_root
from flowcl.utils.run import atomic_write_text, file_sha256, git_sha

TOL = 1e-9
VALIDATED, NOT_VALIDATED, NO_ALPHA = "validated", "not_validated", "no_qualifying_alpha"
NO_RESCUE = ("No rescue tuning: if the frozen alpha fails on either held-out seed, fixed-alpha "
             "SGP is not validated; no other alpha is selected using those seeds (a rescue "
             "sweep would be exploratory and need new validation seeds).")


def load_sensitivity_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else repo_root() / "configs" / "analysis" / "sgp_sensitivity.yaml"
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def gpm_run(cfg: dict, seed: int) -> str:
    return cfg["gpm_run"].format(seed=seed)


def new_run_id(arm: dict, seed: int) -> str:
    alpha = f"_a{arm['alpha']:g}" if "alpha" in arm else ""
    return f"sens__{arm['family']}{alpha}__seed{seed}"


def method_spec(arm: dict, cfg: dict, **extra) -> dict:
    spec = {"family": arm["family"], "lr": float(cfg["recipe"]["lr"])}
    if "alpha" in arm:
        spec["alpha"] = float(arm["alpha"])
    return {**spec, **extra}


# ---- pure pieces ------------------------------------------------------------------------------


def nominal_attenuation(bases: dict, alpha: float | None, lr: float, eps: float) -> dict | None:
    """Per half, the median over layers of the energy-weighted ``1 - lambda`` on T1 bases and of
    the nominal protected-gradient scale ``lr * mean(1 - lambda)``. ``alpha=None`` is hard GPM
    (all zero). NOMINAL only: Adam's normalisation sets what actually reaches the weights."""
    from flowcl.analysis.importance import sgp_importance

    halves: dict = {"trunk": {"weighted": [], "scale": []}, "decoder": {"weighted": [], "scale": []}}
    for name, b in bases.items():
        k = b.ranks[eps]
        s = b.singular_values[:k].to(torch.float64)
        lam = torch.ones_like(s) if alpha is None else sgp_importance(s, alpha)
        h = halves["trunk" if name.startswith("trunk.") else "decoder"]
        h["weighted"].append(float(((1 - lam) * s**2).sum() / (s**2).sum()))
        h["scale"].append(lr * float((1 - lam).mean()))
    return {"note": "nominal: not the realised update",
            **{half: {"energy_weighted_unprotected": statistics.median(v["weighted"]),
                      "nominal_protected_scale": statistics.median(v["scale"])}
               for half, v in halves.items() if v["weighted"]}}


def select_alpha(arms: dict, eligible: list, ft_object: float, gpm_spatial: float,
                 rule: dict) -> dict:
    """``arms``: name -> {family, alpha?, object, spatial}. The smallest qualifying eligible alpha."""
    floors = {"object": ft_object - rule["object_tolerance"],
              "spatial": gpm_spatial - rule["spatial_tolerance"]}
    per_arm, qualifying = {}, []
    for name, a in arms.items():
        eligible_arm = a["family"] == "sgp" and any(math.isclose(a.get("alpha", -1), x) for x in eligible)
        qualifies = a["object"] >= floors["object"] - TOL and a["spatial"] >= floors["spatial"] - TOL
        per_arm[name] = {"eligible": eligible_arm, "qualifies": qualifies,
                         "object": a["object"], "spatial": a["spatial"]}
        if eligible_arm and qualifies:
            qualifying.append((a["alpha"], name))
    chosen = min(qualifying) if qualifying else None
    return {"selected_alpha": None if chosen is None else float(chosen[0]),
            "selected_arm": None if chosen is None else chosen[1],
            "floors": floors, "reference": {"ft_object": ft_object, "gpm_spatial": gpm_spatial},
            "qualifying": sorted(n for _, n in qualifying), "per_arm": per_arm, "rule": rule}


def classify_validation(per_seed: dict, seeds: list) -> dict:
    """``per_seed``: seed -> {object, spatial, ft_object, gpm_spatial, tolerances}."""
    missing = [s for s in seeds if s not in per_seed]
    if missing:
        raise ValueError(f"validation needs every held-out seed; missing {missing}")
    out = {}
    for s in seeds:
        r = per_seed[s]
        obj_floor = r["ft_object"] - r["object_tolerance"]
        spa_floor = r["gpm_spatial"] - r["spatial_tolerance"]
        out[str(s)] = {"object": r["object"], "object_floor": obj_floor,
                       "plasticity": r["object"] >= obj_floor - TOL,
                       "spatial": r["spatial"], "spatial_floor": spa_floor,
                       "protection": r["spatial"] >= spa_floor - TOL}
        out[str(s)]["passed"] = out[str(s)]["plasticity"] and out[str(s)]["protection"]
    failed = [s for s, v in out.items() if not v["passed"]]
    return {"verdict": NOT_VALIDATED if failed else VALIDATED, "failed_seeds": failed,
            "per_seed": out, "no_rescue": NO_RESCUE}


# ---- the selection bundle: published atomically, never rewritten ---------------------------------


def _bundle_ok(bundle: Path) -> tuple[dict, str]:
    sel, done = bundle / "selection.json", bundle / "selection.done.json"
    if not sel.is_file() or not done.is_file():
        raise RuntimeError(f"{bundle}: incomplete selection bundle")
    sha = file_sha256(sel)
    recorded = json.loads(done.read_text()).get("selection_sha256")
    if sha != recorded:
        raise RuntimeError(f"{bundle}: selection.json hashes to {sha}, the marker records {recorded}")
    return json.loads(sel.read_text()), sha


def publish_selection(out: Path, selection: dict) -> tuple[dict, str]:
    """Write the bundle into an attempt directory and publish it by one atomic rename. An
    existing canonical bundle is verified and reused, never rewritten."""
    canonical = out / "selection"
    if canonical.exists():
        return _bundle_ok(canonical)
    out.mkdir(parents=True, exist_ok=True)
    attempt = out / f".selection_attempt_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    attempt.mkdir()
    text = json.dumps(selection, indent=2) + "\n"
    (attempt / "selection.json").write_text(text)
    (attempt / "selection.done.json").write_text(json.dumps(
        {"selection_sha256": file_sha256(attempt / "selection.json")}, indent=2) + "\n")
    _bundle_ok(attempt)
    os.rename(attempt, canonical)
    return _bundle_ok(canonical)


def load_selection(out: Path) -> tuple[dict, str]:
    canonical = out / "selection"
    if not canonical.is_dir():
        raise RuntimeError(f"no published selection bundle at {canonical}")
    return _bundle_ok(canonical)


# ---- arms -----------------------------------------------------------------------------------------


def check_start(ctx, root: Path, cfg: dict) -> dict:
    """The start model must equal plain GPM's stage0 (where the T1 memory was captured) bitwise."""
    from flowcl.train.continual import stage_identity_check

    check = stage_identity_check(ctx.ref_dir / "checkpoints" / "stage0.pt",
                                 root / gpm_run(cfg, ctx.seed), 0)
    if not check["passed"]:
        raise RuntimeError(f"seed {ctx.seed}: seq_ft stage0 differs from plain GPM's stage0 "
                           f"({check['n_different']} tensors): the memory would be mispaired")
    return {"passed": True, "reference_sha256": check.get("reference_sha256")}


def t1_memory(cfg: dict, root: Path, seed: int):
    from flowcl.analysis.subspace import load_bases

    path = root / gpm_run(cfg, seed) / "method" / "memory_task0.pt"
    bases, _ = load_bases(path)
    return bases, path


def reused_arm(arm: dict, ctx, root: Path, cfg: dict, bootstrap: dict) -> dict:
    """Verify a reused arm's recipe, pairing and (for SGP) method and memory; build its row."""
    from flowcl.analysis.subspace import load_bases
    from flowcl.envs.evaluation import EvaluationReport
    from flowcl.experiments.sequence_report import memory_matrix

    run = {"ft": ctx.ref_dir.name, "gpm": gpm_run(cfg, ctx.seed)}.get(arm["family"], arm.get("run"))
    run_dir = root / run
    conf = OmegaConf.to_container(OmegaConf.load(run_dir / "config.yaml"), resolve=True)
    train = {"schedule": "cosine", **conf["train"]}  # the field postdates these runs
    problems = {}
    if int(train["steps"]) != int(cfg["recipe"]["steps"]) or not math.isclose(
            float(train["lr"]), float(cfg["recipe"]["lr"])) or train["schedule"] != "cosine":
        problems["recipe"] = {k: train[k] for k in ("steps", "lr", "schedule")}
    ns = conf.get("seed_namespace_run_id", ctx.namespace)
    if ns != ctx.namespace:
        problems["namespace"] = ns
    if arm["family"] == "sgp":
        m = conf.get("method", {})
        if (m.get("name") != "sgp" or not math.isclose(float(m.get("alpha", -1)), float(arm["alpha"]))
                or m.get("eps") != cfg["eps"] or m.get("projection") != "hard_gradient_scaled_update"):
            problems["method"] = m
        own, _ = load_bases(run_dir / "method" / "memory_task0.pt")
        ref, _ = t1_memory(cfg, root, ctx.seed)
        if sorted(own) != sorted(ref) or any(
                not torch.equal(memory_matrix(own[n], cfg["eps"]), memory_matrix(ref[n], cfg["eps"]))
                for n in ref):
            problems["memory"] = "T1 memory differs from plain GPM's"
    elif arm["family"] == "gpm":
        m = conf.get("method", {})
        if m.get("name") != "gpm" or m.get("eps") != cfg["eps"] or m.get("new_energy_fraction"):
            problems["method"] = m
    if problems:
        raise ValueError(f"reused arm {arm['name']} ({run}) does not match: {problems}")

    ckpt = run_dir / "checkpoints" / "stage1.pt"
    report = EvaluationReport.load(run_dir / "eval" / "stage1.json")
    if report.run_id != ctx.namespace:
        raise ValueError(f"{run}: rollouts under {report.run_id!r}, not {ctx.namespace!r}")
    if arm["family"] == "ft":
        probes = {j: ctx.probe_refs["seq_ft"][1][j] for j in (0, 1)}
    elif arm["family"] == "gpm":
        probes = {j: ctx.probe_refs["gpm"][1][j] for j in (0, 1)}
    else:
        diag = json.loads((root / arm["diagnostics"]).read_text())
        if diag["method_run_id"] != run:
            raise ValueError(f"{arm['diagnostics']} is for {diag['method_run_id']!r}, not {run!r}")
        probes = {j: diag["loss_matrix"]["method"][1][j] for j in (0, 1)}
    return {"arm": arm["name"], "family": arm["family"], "alpha": arm.get("alpha"),
            "seed": ctx.seed, "source": {"kind": "reused", "run": run,
                                         "git_sha": (run_dir / "git_sha").read_text().strip(),
                                         "checkpoint_sha256": file_sha256(ckpt)},
            "checkpoint_path": str(ckpt), "eval_path": str(run_dir / "eval" / "stage1.json"),
            "cells": stage_cells(ctx, 1, report, probes, bootstrap)}


def _train_arm(arm: dict, ctx, root: Path, cfg: dict, bases, evaluator, bootstrap, device,
               dataset_dir, build_datasets, **spec_extra) -> dict:
    from flowcl.experiments.projection_pilot import make_method

    run_id = new_run_id(arm, ctx.seed)
    entry = {"family": arm["family"], "lr": float(cfg["recipe"]["lr"]),
             **({"alpha": float(arm["alpha"])} if "alpha" in arm else {})}
    row = train_stage(
        ctx, 1, cfg["recipe"]["steps"], cfg["recipe"]["lr"], ctx.ref_dir / "checkpoints" / "stage0.pt",
        root / cfg["out"] / f"seed{ctx.seed}" / run_id, run_id,
        method_spec=method_spec(arm, cfg, **spec_extra),
        setup_method=make_method(entry, bases, cfg["eps"]),
        evaluator=evaluator, bootstrap=bootstrap, device=device, dataset_dir=dataset_dir,
        build_datasets=build_datasets,
    )
    return row


def run_bracket(cfg: dict, results_root=None, device="cuda", allow_dirty=False, evaluator=None,
                bootstrap=None, dataset_dir=None, build_datasets=None,
                low_update_cfg: dict | None = None) -> list[dict]:
    refuse_dirty(allow_dirty)
    root = _root(results_root)
    seed = int(cfg["bracket_seed"])
    ctx = seed_context(low_update_cfg or load_low_update_config(), seed, root)
    check_start(ctx, root, cfg)
    if evaluator is None:
        evaluator, bootstrap = default_evaluator()
    bases, _ = t1_memory(cfg, root, seed)
    rows = [_train_arm(arm, ctx, root, cfg, bases, evaluator, bootstrap, device, dataset_dir,
                       build_datasets)
            for arm in cfg["arms"] if arm["source"] == "new"]
    atomic_write_text(root / cfg["out"] / f"done_bracket_seed{seed}.json",
                      json.dumps({"git_sha": git_sha(), "runs": [r["run_id"] for r in rows]},
                                 indent=2) + "\n")
    return rows


def bracket_rows(cfg: dict, root: Path, ctx, bootstrap: dict) -> dict:
    rows = {}
    for arm in cfg["arms"]:
        if arm["source"] == "reused":
            rows[arm["name"]] = reused_arm(arm, ctx, root, cfg, bootstrap)
            continue
        run_id = new_run_id(arm, ctx.seed)
        out_dir = root / cfg["out"] / f"seed{ctx.seed}" / run_id
        published = load_published(out_dir, {"run_id": run_id, "method": method_spec(arm, cfg)})
        if published is None:
            raise FileNotFoundError(f"bracket arm {run_id} is not published")
        rows[arm["name"]] = {"arm": arm["name"], "family": arm["family"], "alpha": arm.get("alpha"),
                             "seed": ctx.seed, "source": {"kind": "new", "run": str(out_dir),
                                                          "git_sha": published["git_sha"],
                                                          "checkpoint_sha256": published["checkpoint_sha256"]},
                             "checkpoint_path": str(out_dir / "checkpoints" / "final.pt"),
                             "eval_path": str(out_dir / "eval.json"), "cells": published["cells"]}
    return rows


def run_select(cfg: dict, results_root=None, low_update_cfg: dict | None = None,
               bootstrap: dict | None = None) -> dict:
    root = _root(results_root)
    ctx = seed_context(low_update_cfg or load_low_update_config(), int(cfg["bracket_seed"]), root)
    if bootstrap is None:
        _, bootstrap = default_evaluator()
    rows = bracket_rows(cfg, root, ctx, bootstrap)
    arms = {n: {"family": r["family"], "alpha": r["alpha"], "object": r["cells"]["1"]["success"],
                "spatial": r["cells"]["0"]["success"]} for n, r in rows.items()}
    choice = select_alpha(arms, cfg["eligible_alphas"], arms["fine_tuning"]["object"],
                          arms["gpm"]["spatial"], cfg["selection"])
    selection = {"git_sha": git_sha(), "bracket_seed": ctx.seed, **choice,
                 "checkpoints_sha256": {n: r["source"]["checkpoint_sha256"] for n, r in rows.items()}}
    published, sha = publish_selection(root / cfg["out"], selection)
    print(f"[flowcl] SGP selection: alpha {published['selected_alpha']} (bundle {sha[:12]})",
          flush=True)
    return published


def selection_open(cfg: dict, results_root=None) -> bool:
    try:
        sel, _ = load_selection(_root(results_root) / cfg["out"])
    except RuntimeError:
        return False
    return sel["selected_alpha"] is not None


def run_validation(cfg: dict, seed: int, results_root=None, device="cuda", allow_dirty=False,
                   evaluator=None, bootstrap=None, dataset_dir=None, build_datasets=None,
                   low_update_cfg: dict | None = None) -> dict:
    refuse_dirty(allow_dirty)
    if seed not in cfg["validation_seeds"]:
        raise ValueError(f"seed {seed} is not a validation seed {cfg['validation_seeds']}")
    root = _root(results_root)
    sel, sha = load_selection(root / cfg["out"])
    if sel["selected_alpha"] is None:
        raise RuntimeError("no alpha was selected: nothing to validate")
    ctx = seed_context(low_update_cfg or load_low_update_config(), seed, root)
    check_start(ctx, root, cfg)
    if evaluator is None:
        evaluator, bootstrap = default_evaluator()
    bases, _ = t1_memory(cfg, root, seed)
    arm = {"name": "validation", "family": "sgp", "alpha": sel["selected_alpha"]}
    row = _train_arm(arm, ctx, root, cfg, bases, evaluator, bootstrap, device, dataset_dir,
                     build_datasets, selection_sha256=sha)
    # Re-verify after the run: the bundle must still be the one the run was trained under.
    _, sha_after = load_selection(root / cfg["out"])
    if sha_after != sha:
        raise RuntimeError("the selection bundle changed during validation")
    out = {"git_sha": git_sha(), "seed": seed, "selection_sha256": sha, "row": row}
    atomic_write_text(root / cfg["out"] / f"validation_seed{seed}.json",
                      json.dumps(out, indent=2) + "\n")
    return out


# ---- the summary (always) ----------------------------------------------------------------------------


def _measure(row: dict, ctx, stage0, datasets, bases, cfg: dict, device, tol: dict) -> dict:
    from flowcl.experiments.gate3 import update_interference
    from flowcl.experiments.optimizer_study import probe_units, update_size
    from flowcl.train.checkpoint import load_checkpoint

    start = {k: v.detach().cpu() for k, v in stage0.policy.named_parameters()}
    trainable = {n for n, p in stage0.policy.named_parameters() if p.requires_grad}
    registry = set(stage0.policy.projectable_parameters())
    loaded = load_checkpoint(Path(row["checkpoint_path"]), device=device)
    stored = {j: row["cells"][str(j)]["probe_loss"] for j in (0, 1)}
    units = probe_units(loaded.policy, datasets, ctx.probe, device, stored, tol,
                        f"{row['arm']} seed {row['seed']}")
    end = {k: v.detach().cpu() for k, v in loaded.policy.named_parameters()}
    split = update_interference(start, end, bases, [cfg["eps"]])
    halves = {}
    for name, layer in split["per_layer"].items():
        halves.setdefault("trunk" if name.startswith("trunk.") else "decoder", []).append(
            layer["delta_norm"] ** 2)
    realised = {}
    for half, energies in halves.items():
        total = math.sqrt(sum(energies))
        c = split["c_global"][str(cfg["eps"])].get(half)
        realised[half] = {"delta_norm": total, "parallel_fraction": c,
                          "parallel_norm": None if c is None else c * total,
                          "perpendicular_norm": None if c is None else math.sqrt(max(0.0, 1 - c * c)) * total}
    alpha = row.get("alpha")
    nominal = (nominal_attenuation(bases, None if row["family"] == "gpm" else float(alpha),
                                   float(cfg["recipe"]["lr"]), cfg["eps"])
               if row["family"] in ("sgp", "gpm") else None)
    return {"probe_units": units, "update_size": update_size(start, end, trainable, registry),
            "realised_split": realised, "nominal_attenuation": nominal}


def run_summary(cfg: dict, results_root=None, device="cuda", dataset_dir=None,
                build_datasets=None, low_update_cfg: dict | None = None,
                bootstrap: dict | None = None) -> dict:
    from flowcl.envs.evaluation import EvaluationReport
    from flowcl.train.checkpoint import load_checkpoint
    from flowcl.train.pipeline import build_dataset

    root = _root(results_root)
    lu = low_update_cfg or load_low_update_config()
    if bootstrap is None:
        _, bootstrap = default_evaluator()
    out_dir = root / cfg["out"]
    sel, sha = load_selection(out_dir)
    tol = cfg["probe_tolerance"]

    def seed_setup(seed):
        ctx = seed_context(lu, seed, root)
        stage0 = load_checkpoint(ctx.ref_dir / "checkpoints" / "stage0.pt")
        stages = ctx.curriculum.stages
        make = build_datasets or (lambda st: build_dataset(
            [st.ref], stage0.spec, stage0.stats, n_demos=st.n_demos, dataset_dir=dataset_dir))
        return ctx, stage0, {j: make(stages[j]) for j in (0, 1)}, t1_memory(cfg, root, seed)[0]

    seed0 = int(cfg["bracket_seed"])
    ctx, stage0, datasets, bases = seed_setup(seed0)
    rows = bracket_rows(cfg, root, ctx, bootstrap)
    evals = {n: EvaluationReport.load(r["eval_path"]).by_task() for n, r in rows.items()}
    keys = {j: ctx.curriculum.stages[j].task_key for j in (0, 1)}
    for name, row in rows.items():
        row.update(_measure(row, ctx, stage0, datasets, bases, cfg, device, tol))
        row["paired_vs"] = {ref: {str(j): paired_against(evals[name][keys[j]], evals[ref][keys[j]], bootstrap)
                                  for j in (0, 1)} for ref in ("gpm", "fine_tuning")}
    consistency = None
    pilot = root / cfg["consistency_reference"] / "eval" / "stage1.json"
    if pilot.is_file():
        rep = EvaluationReport.load(pilot).by_task()
        consistency = {"run": cfg["consistency_reference"], "note": "dirty-tree pilot, reported only",
                       "object": rep[keys[1]].estimate.value, "spatial": rep[keys[0]].estimate.value}

    validation = None
    if sel["selected_alpha"] is None:
        verdict = {"verdict": NO_ALPHA, "text": "No eligible alpha matched hard GPM's T1 protection "
                   "within tolerance while keeping plasticity; validation was not run."}
    else:
        per_seed, vrows = {}, {}
        for s in cfg["validation_seeds"]:
            v = json.loads((out_dir / f"validation_seed{s}.json").read_text())
            if v["selection_sha256"] != sha:
                raise RuntimeError(f"validation seed {s} was run under another selection bundle")
            vctx, vstage0, vdata, vbases = seed_setup(s)
            row = {"arm": "validation", "family": "sgp", "alpha": sel["selected_alpha"], "seed": s,
                   "checkpoint_path": str(out_dir / f"seed{s}" / v["row"]["run_id"] / "checkpoints" / "final.pt"),
                   "cells": v["row"]["cells"], "source": {"git_sha": v["row"]["git_sha"],
                                                          "checkpoint_sha256": v["row"]["checkpoint_sha256"]}}
            row.update(_measure(row, vctx, vstage0, vdata, vbases, cfg, device, tol))
            vrows[str(s)] = row
            c = row["cells"]
            per_seed[s] = {"object": c["1"]["success"], "spatial": c["0"]["success"],
                           "ft_object": c["1"]["reference_success"]["seq_ft"],
                           "gpm_spatial": c["0"]["reference_success"]["gpm"],
                           **{f"{k}_tolerance": cfg["validation"][f"{k}_tolerance"]
                              for k in ("object", "spatial")}}
        verdict = classify_validation(per_seed, cfg["validation_seeds"])
        validation = vrows

    out = {"git_sha": git_sha(), "config": cfg, "selection": sel, "selection_sha256": sha,
           "verdict": verdict, "bracket": rows, "validation": validation,
           "consistency_reference": consistency,
           "abandoned_attempts": sorted(
               [p.name for p in out_dir.glob(".selection_attempt_*")]
               + [f"seed{s}/{a}" for s in [seed0, *cfg["validation_seeds"]]
                  for a in abandoned_attempts(out_dir / f"seed{s}")])}
    atomic_write_text(out_dir / "summary.json", json.dumps(out, indent=2, default=str) + "\n")
    print(f"[flowcl] SGP sensitivity: selected alpha {sel['selected_alpha']}; "
          f"verdict {verdict['verdict']}", flush=True)
    return out
