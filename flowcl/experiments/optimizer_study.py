"""Broader optimizer study: the fine-tuning stability-plasticity trade-off on T1 -> T2.

Low-update fine-tuning (5k steps, lr 3e-5) failed held-out validation
(``docs/runs/2026-09-28_low_update.md``). This study maps, rather than selects: does *any*
fine-tuning setting get both plasticity (Object) and retention (Spatial), and what does each
optimizer knob do? The rule is ``configs/analysis/optimizer_study.yaml``:

* a **core grid** (learning rate x budget; 7 settings) on seeds 0-2, the only settings
  eligible for the existence claim (:func:`existence`);
* **one-factor variants** around the anchor (5k, 3e-5) on seeds 0-1 (schedule, weight decay,
  batch size at equal examples seen), descriptive only (:func:`one_factor`);
* frontiers per seed and on the worst seed (:func:`pareto`, :func:`robust_frontier`).

Every run starts from the seed's seq_ft ``stage0.pt`` and is paired as in
:mod:`flowcl.experiments.low_update` (whose :func:`~flowcl.experiments.low_update.train_stage`
trains and publishes it). Existing runs of core settings are verified and reused
(:func:`reused_row`). At the summary, probe losses are recomputed per batch from every
checkpoint (:func:`probe_units`) so probe differences get matched-batch bootstrap intervals,
and the realised update is measured against ``stage0.pt`` (:func:`update_size`).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from flowcl.experiments.hparam_study import default_evaluator, paired_against
from flowcl.experiments.low_update import (
    _root,
    classify_validation,
    load_low_update_config,
    load_published,
    refuse_dirty,
    seed_context,
    stage_cells,
    train_stage,
)
from flowcl.utils.libero_paths import repo_root
from flowcl.utils.run import atomic_write_text, file_sha256, git_sha

RECIPE_KEYS = ("lr", "steps", "batch_size", "weight_decay", "schedule")
LABELS = ("supported_both_seeds", "supported_one_seed", "directionally_consistent", "inconclusive")


def load_study_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else repo_root() / "configs" / "analysis" / "optimizer_study.yaml"
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


# ---- settings -------------------------------------------------------------------------------


def recipe(cfg: dict, **fields) -> dict:
    r = {**cfg["base"], **fields}
    return {"lr": float(r["lr"]), "steps": int(r["steps"]), "batch_size": int(r["batch_size"]),
            "weight_decay": float(r["weight_decay"]), "schedule": str(r["schedule"])}


def setting_name(r: dict) -> str:
    return (f"lr{r['lr']:g}__s{r['steps']}__b{r['batch_size']}__wd{r['weight_decay']:g}"
            f"__{r['schedule']}")


def same_recipe(a: dict, b: dict) -> bool:
    return all(math.isclose(a[k], b[k], rel_tol=1e-12) if isinstance(a[k], float) else a[k] == b[k]
               for k in RECIPE_KEYS)


def settings(cfg: dict) -> dict:
    """``{"core": [recipe...], "variants": [{"name", "recipe"}...], "anchor": recipe}``."""
    core = [recipe(cfg, **c) for c in cfg["core"]]
    anchor = recipe(cfg, **cfg["anchor"])
    if not any(same_recipe(anchor, c) for c in core):
        raise ValueError("the anchor must be a core setting")
    variants = []
    for v in cfg["variants"]:
        fields = {k: val for k, val in v.items() if k != "name"}
        variants.append({"name": v["name"], "recipe": recipe(cfg, **{**cfg["anchor"], **fields})})
    return {"core": core, "variants": variants, "anchor": anchor,
            "standard": recipe(cfg, **cfg["standard"])}


def source_of(cfg: dict, r: dict, seed: int) -> dict:
    """Where the (recipe, seed) row comes from: the standard run, a reused run, or a new one."""
    if same_recipe(r, recipe(cfg, **cfg["standard"])):
        return {"kind": "standard"}
    for entry in cfg["reused"]:
        if entry["seed"] == seed and same_recipe(r, recipe(cfg, lr=entry["lr"], steps=entry["steps"])):
            return {"kind": "reused", "run": entry["run"]}
    return {"kind": "new", "run_id": f"opt__{setting_name(r)}__seed{seed}"}


def runs_to_train(cfg: dict, seed: int, part: str) -> list[dict]:
    s = settings(cfg)
    if part == "grid":
        recipes = s["core"] if seed in cfg["seeds"] else []
    elif part == "variants":
        recipes = [v["recipe"] for v in s["variants"]] if seed in cfg["variant_seeds"] else []
    else:
        raise ValueError(f"unknown part {part!r}")
    return [r for r in recipes if source_of(cfg, r, seed)["kind"] == "new"]


# ---- reused runs --------------------------------------------------------------------------------


def verify_reused(conf: dict, r: dict, ctx) -> dict:
    """The reused run's recipe and pairing, with ``schedule`` normalised to cosine if absent."""
    train = dict(conf["train"])
    normalised = "schedule" not in train
    train.setdefault("schedule", "cosine")  # the only schedule before the field existed
    found = {"lr": float(train["lr"]), "steps": int(train["steps"]),
             "batch_size": int(train["batch_size"]), "weight_decay": float(train["weight_decay"]),
             "schedule": train["schedule"]}
    start = conf.get("start_checkpoint") or Path(conf.get("parent", {}).get("path", "")).name
    problems = {}
    if not same_recipe(found, r):
        problems["recipe"] = (found, r)
    if conf.get("stream_seed") != {"namespace": ctx.namespace, "stage": 1}:
        problems["stream_seed"] = conf.get("stream_seed")
    if conf.get("reference_run") != ctx.ref_dir.name:
        problems["reference_run"] = conf.get("reference_run")
    if start != "stage0.pt":
        problems["start_checkpoint"] = start
    if problems:
        raise ValueError(f"reused run does not match the registered setting: {problems}")
    return {"schedule_normalised": normalised}


def reused_row(cfg: dict, r: dict, seed: int, run: str, ctx, root: Path, bootstrap: dict) -> dict:
    from flowcl.envs.evaluation import EvaluationReport

    src = root / run
    conf = OmegaConf.to_container(OmegaConf.load(src / "config.yaml"), resolve=True)
    note = verify_reused(conf, r, ctx)
    report = EvaluationReport.load(src / "eval.json")
    if report.run_id != ctx.namespace:
        raise ValueError(f"{run}: rollouts under {report.run_id!r}, not {ctx.namespace!r}")
    stored = json.loads((src / "row.json").read_text())
    if "cells" in stored:  # a low_update stage
        probes = {int(j): c["probe_loss"] for j, c in stored["cells"].items()}
    else:                  # an hparam-study variant
        probes = {0: stored["probe_loss"]["retained"], 1: stored["probe_loss"]["new"]}
    ckpt = src / "checkpoints" / "final.pt"
    return {"recipe": r, "seed": seed, "source": {"kind": "reused", "run": run,
                                                  "git_sha": (src / "git_sha").read_text().strip(),
                                                  "checkpoint_sha256": file_sha256(ckpt), **note},
            "checkpoint_path": str(ckpt), "eval_path": str(src / "eval.json"),
            "cells": stage_cells(ctx, 1, report, probes, bootstrap)}


# ---- training ------------------------------------------------------------------------------------


def run_seed(cfg: dict, seed: int, part: str, results_root=None, device="cuda", allow_dirty=False,
             evaluator=None, bootstrap=None, dataset_dir=None, build_datasets=None,
             low_update_cfg: dict | None = None) -> list[dict]:
    """Train (or reuse, if published) every new setting of ``part`` for ``seed``."""
    refuse_dirty(allow_dirty)
    root = _root(results_root)
    ctx = seed_context(low_update_cfg or load_low_update_config(), seed, root)
    if evaluator is None:
        evaluator, bootstrap = default_evaluator()
    rows = []
    for r in runs_to_train(cfg, seed, part):
        run_id = source_of(cfg, r, seed)["run_id"]
        rows.append(train_stage(
            ctx, 1, r["steps"], r["lr"], ctx.ref_dir / "checkpoints" / "stage0.pt",
            root / cfg["out"] / f"seed{seed}" / run_id, run_id,
            train_overrides={k: r[k] for k in ("batch_size", "weight_decay", "schedule")},
            evaluator=evaluator, bootstrap=bootstrap, device=device, dataset_dir=dataset_dir,
            build_datasets=build_datasets,
        ))
    marker = root / cfg["out"] / f"done_{part}_seed{seed}.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(marker, json.dumps({"git_sha": git_sha(), "part": part, "seed": seed,
                                          "runs": [r["run_id"] for r in rows]}, indent=2) + "\n")
    return rows


# ---- measurements -------------------------------------------------------------------------------


def update_size(start: dict, end: dict, trainable: set, registry: set) -> dict:
    """The realised update against ``start`` over the trainable parameters (L2 norms)."""
    moved_frozen = sorted(n for n in start if n not in trainable and n in end
                          and not torch.equal(start[n], end[n]))
    if moved_frozen:
        raise RuntimeError(f"frozen parameters moved: {moved_frozen[:5]}")

    def energy(names):
        d = sum(float(((end[n].double() - start[n].double()) ** 2).sum()) for n in names)
        b = sum(float((start[n].double() ** 2).sum()) for n in names)
        return d, b

    groups = {
        "trainable": sorted(trainable),
        "registry_trunk": sorted(n for n in registry if n.startswith("trunk.")),
        "registry_decoder": sorted(n for n in registry if not n.startswith("trunk.")),
        "non_registry": sorted(trainable - registry),
    }
    out = {}
    for g, names in groups.items():
        d, b = energy(names)
        out[g] = {"delta_norm": math.sqrt(d), "relative": math.sqrt(d / b) if b > 0 else None,
                  "n_tensors": len(names)}
    out["frozen_unchanged"] = True
    return out


def check_probe(recomputed: float, stored: float, tol: dict, where: str) -> None:
    try:
        torch.testing.assert_close(torch.tensor(recomputed, dtype=torch.float64),
                                   torch.tensor(stored, dtype=torch.float64),
                                   rtol=tol["rtol"], atol=tol["atol"])
    except AssertionError as err:
        raise RuntimeError(f"{where}: recomputed probe loss {recomputed!r} does not reproduce the "
                           f"stored {stored!r}: {err}") from None


def probe_units(policy, datasets: dict, probe: dict, device, stored: dict, tol: dict,
                where: str) -> dict:
    """Per-batch probe losses for each task, asserted against the stored aggregates."""
    from flowcl.analysis.probes import probe_batch_losses, weighted_mean

    out = {}
    for j, ds in datasets.items():
        batches = probe_batch_losses(policy, ds, probe, device)
        mean = weighted_mean(batches)
        check_probe(mean, stored[j], tol, f"{where} task {j}")
        out[str(j)] = {"losses": [b[0] for b in batches], "weights": [b[1] for b in batches],
                       "mean": mean, "stored": stored[j]}
    return out


# ---- the pure analyses ------------------------------------------------------------------------------


def matched_batch_diff(a: dict, b: dict, boot: dict) -> dict:
    """Weighted-mean difference ``a - b`` over matched probe batches, with a bootstrap interval
    over those batches (stability over the registered units, not population uncertainty)."""
    la, lb = np.asarray(a["losses"]), np.asarray(b["losses"])
    w = np.asarray(a["weights"])
    if not np.array_equal(w, np.asarray(b["weights"])):
        raise ValueError("probe batch weights differ: the units are not matched")
    d = la - lb
    point = float((w * d).sum() / w.sum())
    rng = np.random.default_rng(boot["seed"])
    idx = rng.integers(0, d.size, size=(boot["n_resamples"], d.size))
    stats = (w[idx] * d[idx]).sum(axis=1) / w[idx].sum(axis=1)
    alpha = 1.0 - boot["confidence"]
    low, high = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return {"diff": point, "low": float(min(low, point)), "high": float(max(high, point)),
            "unit": "matched probe batch"}


def pareto(points: dict, maximize: tuple[bool, bool]) -> list[str]:
    """Names of the non-dominated points (ties are all kept)."""
    def better_eq(p, q):  # p at least as good as q on both axes
        return all((p[i] >= q[i]) if maximize[i] else (p[i] <= q[i]) for i in range(2))

    return sorted(n for n, p in points.items()
                  if not any(better_eq(q, p) and q != p for m, q in points.items() if m != n))


def robust_frontier(by_setting: dict) -> dict:
    """Worst-seed frontier: ``by_setting``: name -> seed -> (object, spatial, obj_loss, spa_loss)."""
    worst_success = {n: (min(v[0] for v in seeds.values()), min(v[1] for v in seeds.values()))
                     for n, seeds in by_setting.items()}
    worst_loss = {n: (max(v[2] for v in seeds.values()), max(v[3] for v in seeds.values()))
                  for n, seeds in by_setting.items()}
    return {"success": {"points": worst_success, "pareto": pareto(worst_success, (True, True))},
            "loss": {"points": worst_loss, "pareto": pareto(worst_loss, (False, False))}}


def existence(rows: dict, core: list[str], seeds: list[int], tolerance: float) -> dict:
    """``rows``: name -> seed -> row. Only core settings with every seed are eligible."""
    out = {}
    for name, by_seed in rows.items():
        if name not in core:
            out[name] = {"eligible": False, "reason": "not a core setting"}
            continue
        missing = [s for s in seeds if s not in by_seed]
        if missing:
            out[name] = {"eligible": False, "reason": f"missing seeds {missing}"}
            continue
        per = {str(s): classify_validation(by_seed[s], tolerance) for s in seeds}
        out[name] = {"eligible": True, "achieves_both": all(p["passed"] for p in per.values()),
                     "per_seed": per}
    return out


def label(intervals: list[dict]) -> str:
    """One label from per-seed paired intervals ``{"diff", "low", "high"}`` (two seeds)."""
    signs = {int(np.sign(i["diff"])) for i in intervals}
    if len(signs) != 1 or 0 in signs:
        return "inconclusive"
    excluding = sum(1 for i in intervals if i["low"] > 0 or i["high"] < 0)
    return {len(intervals): "supported_both_seeds", 0: "directionally_consistent"}.get(
        excluding, "supported_one_seed")


def one_factor(effects: dict) -> dict:
    """``effects``: variant -> metric -> [per-seed interval] -> labels per variant and metric."""
    return {v: {m: {"label": label(iv), "per_seed": iv} for m, iv in metrics.items()}
            for v, metrics in effects.items()}


# ---- the summary ----------------------------------------------------------------------------------


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
    boot, tol = cfg["batch_bootstrap"], cfg["probe_tolerance"]
    s = settings(cfg)
    core_names = [setting_name(r) for r in s["core"]]
    wanted = {**{setting_name(r): (r, cfg["seeds"]) for r in s["core"]},
              **{v["name"]: (v["recipe"], cfg["variant_seeds"]) for v in s["variants"]}}
    rows: dict = {n: {} for n in wanted}
    units: dict = {n: {} for n in wanted}
    evals: dict = {n: {} for n in wanted}
    gpm_points = {}
    for seed in sorted(set(cfg["seeds"]) | set(cfg["variant_seeds"])):
        ctx = seed_context(lu, seed, root)
        stage0 = load_checkpoint(ctx.ref_dir / "checkpoints" / "stage0.pt")
        start_state = {k: v.detach().cpu().clone() for k, v in stage0.policy.named_parameters()}
        trainable = {n for n, p in stage0.policy.named_parameters() if p.requires_grad}
        registry = set(stage0.policy.projectable_parameters())
        stages = ctx.curriculum.stages
        make = build_datasets or (lambda st: build_dataset(
            [st.ref], stage0.spec, stage0.stats, n_demos=st.n_demos, dataset_dir=dataset_dir))
        datasets = {j: make(stages[j]) for j in (0, 1)}
        gpm_eval = ctx.evals["gpm"][1].by_task()
        gpm_points[str(seed)] = {"object": gpm_eval[stages[1].task_key].estimate.value,
                                 "spatial": gpm_eval[stages[0].task_key].estimate.value,
                                 "object_probe": ctx.probe_refs["gpm"][1][1],
                                 "spatial_probe": ctx.probe_refs["gpm"][1][0]}
        for name, (r, seeds) in wanted.items():
            if seed not in seeds:
                continue
            src = source_of(cfg, r, seed)
            if src["kind"] == "standard":
                ckpt = ctx.ref_dir / "checkpoints" / "stage1.pt"
                report = ctx.evals["seq_ft"][1]
                probes = {j: ctx.probe_refs["seq_ft"][1][j] for j in (0, 1)}
                row = {"recipe": r, "seed": seed, "source": {
                    "kind": "standard", "run": ctx.ref_dir.name, "checkpoint_sha256": file_sha256(ckpt)},
                    "cells": stage_cells(ctx, 1, report, probes, bootstrap)}
                eval_path = ctx.ref_dir / "eval" / "stage1.json"
            elif src["kind"] == "reused":
                row = reused_row(cfg, r, seed, src["run"], ctx, root, bootstrap)
                ckpt, eval_path = Path(row["checkpoint_path"]), Path(row["eval_path"])
            else:
                out_dir = root / cfg["out"] / f"seed{seed}" / src["run_id"]
                published = load_published(out_dir, {"run_id": src["run_id"]})
                if published is None:
                    raise FileNotFoundError(f"{src['run_id']} is not published")
                row = {"recipe": r, "seed": seed, "source": {"kind": "new", "run": str(out_dir),
                       "git_sha": published["git_sha"],
                       "checkpoint_sha256": published["checkpoint_sha256"]},
                       "cells": published["cells"]}
                ckpt, eval_path = out_dir / "checkpoints" / "final.pt", out_dir / "eval.json"
            loaded = load_checkpoint(ckpt, device=device)
            if loaded.stats.fingerprint() != stage0.stats.fingerprint():
                raise ValueError(f"{name} seed {seed}: normalisation stats differ from stage0")
            stored = {j: row["cells"][str(j)]["probe_loss"] for j in (0, 1)}
            units[name][seed] = probe_units(loaded.policy, datasets, ctx.probe, device, stored, tol,
                                            f"{name} seed {seed}")
            end_state = {k: v.detach().cpu() for k, v in loaded.policy.named_parameters()}
            row["update_size"] = update_size(start_state, end_state, trainable, registry)
            del loaded, end_state
            rows[name][seed] = row
            evals[name][seed] = EvaluationReport.load(eval_path).by_task()

    # Frontiers over the core grid.
    def point(row, u):
        return (row["cells"]["1"]["success"], row["cells"]["0"]["success"],
                u["1"]["mean"], u["0"]["mean"])

    by_setting = {n: {sd: point(rows[n][sd], units[n][sd]) for sd in rows[n]} for n in core_names}
    per_seed = {}
    for sd in cfg["seeds"]:
        succ = {n: by_setting[n][sd][:2] for n in core_names}
        loss = {n: by_setting[n][sd][2:] for n in core_names}
        per_seed[str(sd)] = {"success": {"points": succ, "pareto": pareto(succ, (True, True))},
                             "loss": {"points": loss, "pareto": pareto(loss, (False, False))}}
    frontiers = {"per_seed": per_seed, "robust": robust_frontier(by_setting)}

    # One-factor effects against the anchor.
    anchor = setting_name(s["anchor"])
    effects = {}
    for v in s["variants"]:
        name = v["name"]
        metrics = {"object_success": [], "spatial_success": [], "object_probe": [], "spatial_probe": []}
        for sd in cfg["variant_seeds"]:
            stages = seed_context(lu, sd, root).curriculum.stages
            keys = {j: stages[j].task_key for j in (0, 1)}
            ev, an = evals[name][sd], evals[anchor][sd]
            metrics["object_success"].append(paired_against(ev[keys[1]], an[keys[1]], bootstrap))
            metrics["spatial_success"].append(paired_against(ev[keys[0]], an[keys[0]], bootstrap))
            metrics["object_probe"].append(matched_batch_diff(units[name][sd]["1"], units[anchor][sd]["1"], boot))
            metrics["spatial_probe"].append(matched_batch_diff(units[name][sd]["0"], units[anchor][sd]["0"], boot))
        effects[name] = metrics

    out = {
        "git_sha": git_sha(), "config": cfg,
        "existence": existence(rows, core_names, cfg["seeds"], cfg["existence"]["object_tolerance"]),
        "frontiers": frontiers,
        "one_factor": one_factor(effects),
        "gpm_points": gpm_points,
        "standard": setting_name(s["standard"]),
        "rows": {n: {str(sd): r for sd, r in by.items()} for n, by in rows.items()},
        "probe_units": {n: {str(sd): u for sd, u in by.items()} for n, by in units.items()},
    }
    path = root / cfg["out"] / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(out, indent=2) + "\n")
    passing = [n for n, e in out["existence"].items() if e.get("achieves_both")]
    print(f"[flowcl] optimizer study: settings achieving both on all seeds: {passing or 'none'}",
          flush=True)
    return out
