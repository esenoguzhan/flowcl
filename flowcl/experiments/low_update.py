"""Low-update continual fine-tuning: held-out validation and a gated four-task chain.

T1 keeps the standard recipe (the shared seq_ft ``stage0.pt``); only the later tasks are
trained with the frozen low-update configuration (``configs/analysis/low_update.yaml``:
5k steps, lr 3e-5), selected on seed 0 by the exploratory budget/learning-rate study and
not tuned further. Every stage trains on seq_ft's own stream seed and rolls out under the
seed's seq_ft namespace, so each cell is paired episode-for-episode with seq_ft, plain GPM
and adaptive GPM of the same seed.

* **validation** (stage 1): :func:`classify_validation` per seed; the verdict
  (:func:`classify_validation_across_seeds`) counts only the held-out seeds 1-2 — seed 0
  selected the configuration and is reported, not counted.
* **chain** (stages 2-3, only if validated): :func:`classify_chain` —
  ``underlearns`` / ``cumulative`` / ``delay_only`` / ``partial``.

Reliability: every stage is written into a sibling attempt directory and published by one
atomic rename once complete (``row.json`` last); a published stage is re-validated before
reuse (configuration and checkpoint SHA-256) and never overwritten; abandoned attempts are
ignored and listed. :func:`train_stage` is shared with :mod:`flowcl.experiments.projection_pilot`;
:func:`pilot_stage` (loss only, no rollouts) with :mod:`flowcl.experiments.sgp_coupling`.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from omegaconf import OmegaConf

from flowcl.experiments.hparam_study import (
    _free,
    _success,
    default_evaluator,
    paired_against,
    reference_namespace,
    stage_seed,
)
from flowcl.utils.libero_paths import repo_root
from flowcl.utils.run import atomic_write_text, file_sha256, git_sha

ROLES = ("seq_ft", "gpm", "adaptive_gpm")
VALIDATED = "validated_on_held_out_seeds"
NOT_VALIDATED = "not_validated"
TOL = 1e-9  # rates are k/50: allow float rounding only


def load_low_update_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else repo_root() / "configs" / "analysis" / "low_update.yaml"
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def _root(results_root) -> Path:
    return Path(results_root) if results_root else repo_root() / "results"


def seed_config(cfg: dict, seed: int) -> dict:
    seeds = {int(k): v for k, v in cfg["seeds"].items()}
    if seed not in seeds:
        raise ValueError(f"seed {seed} is not registered ({sorted(seeds)})")
    return seeds[seed]


def refuse_dirty(allow_dirty: bool) -> str:
    sha = git_sha()
    if sha.endswith("-dirty") and not allow_dirty:
        raise RuntimeError(f"working tree is dirty ({sha}); commit first, or pass allow_dirty "
                           "(recorded)")
    return sha


# ---- the seed's references ------------------------------------------------------------------


@dataclass
class SeedContext:
    seed: int
    ref_dir: Path
    ref_config: dict
    curriculum: object
    namespace: str
    runs: dict            # role -> run directory name
    evals: dict           # role -> stage -> EvaluationReport
    probe: dict           # the diagnostics' fixed probe
    probe_refs: dict      # role -> loss matrix L[stage][task]


def seed_context(cfg: dict, seed: int, root: Path) -> SeedContext:
    """The seed's seq_ft reference, its paired GPM runs and diagnostics, all cross-checked."""
    from flowcl.data.curriculum import load_curriculum
    from flowcl.envs.evaluation import EvaluationReport

    s = seed_config(cfg, seed)
    runs = {"seq_ft": s["reference_run"], "gpm": s["gpm_run"], "adaptive_gpm": s["adaptive_gpm_run"]}
    ref_dir = root / runs["seq_ft"]
    config = OmegaConf.to_container(OmegaConf.load(ref_dir / "config.yaml"), resolve=True)
    result = json.loads((ref_dir / "result.json").read_text())
    curriculum = load_curriculum(config["curriculum"])
    n = len(curriculum.stages)
    evals = {role: {k: EvaluationReport.load(root / run / "eval" / f"stage{k}.json") for k in range(n)}
             for role, run in runs.items()}
    namespace = reference_namespace(
        ref_dir, config, result, curriculum,
        {f"{role}/eval/stage{k}.json": rep for role, by in evals.items() for k, rep in by.items()},
    )
    diags = {k: json.loads((root / p).read_text()) for k, p in s["diagnostics"].items()}
    for key, run in (("gpm", runs["gpm"]), ("adaptive_gpm", runs["adaptive_gpm"])):
        d = diags[key]
        if d["reference_run_id"] != ref_dir.name or d["method_run_id"] != run:
            raise ValueError(f"{s['diagnostics'][key]} pairs {d['method_run_id']!r} with "
                             f"{d['reference_run_id']!r}, expected {run!r} with {ref_dir.name!r}")
    if diags["gpm"]["config"]["probe"] != diags["adaptive_gpm"]["config"]["probe"]:
        raise ValueError("the two diagnostics reports used different probes")
    return SeedContext(
        seed=seed, ref_dir=ref_dir, ref_config=config, curriculum=curriculum,
        namespace=namespace, runs=runs, evals=evals, probe=diags["gpm"]["config"]["probe"],
        probe_refs={"seq_ft": diags["gpm"]["loss_matrix"]["reference"],
                    "gpm": diags["gpm"]["loss_matrix"]["method"],
                    "adaptive_gpm": diags["adaptive_gpm"]["loss_matrix"]["method"]},
    )


def stage_cells(ctx: SeedContext, stage: int, report, probes: dict, bootstrap: dict) -> dict:
    """Every task trained so far: success, paired gains vs the three references, probes."""
    by_task = report.by_task()
    cells = {}
    for j in range(stage + 1):
        key = ctx.curriculum.stages[j].task_key
        te = by_task[key]
        refs = {role: ctx.evals[role][stage].by_task()[key] for role in ROLES}
        cells[str(j)] = {
            "task_key": key, **_success(te),
            "paired": {role: paired_against(te, refs[role], bootstrap) for role in ROLES},
            "reference_success": {role: refs[role].estimate.value for role in ROLES},
            "probe_loss": probes[j],
            "probe_reference": {role: ctx.probe_refs[role][stage][j] for role in ROLES},
        }
    return cells


# ---- attempt directories: publish complete, reuse validated, never overwrite -----------------


def attempt_parent(out_dir: Path) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return out_dir.parent / f".attempt_{stamp}_{os.getpid()}_{out_dir.name}"


def abandoned_attempts(parent: Path) -> list[str]:
    return sorted(p.name for p in parent.glob(".attempt_*")) if parent.is_dir() else []


def publish(run_dir: Path, out_dir: Path) -> Path:
    """Atomically rename a complete attempt to its canonical path (which must not exist)."""
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} already exists; a published run is never overwritten")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    os.rename(run_dir, out_dir)
    try:
        run_dir.parent.rmdir()  # the now-empty attempt parent
    except OSError:
        pass
    return out_dir


def load_published(out_dir: Path, expect: dict) -> dict | None:
    """A published run's row, re-validated; ``None`` if nothing is published there."""
    if not out_dir.exists():
        return None
    row_path = out_dir / "row.json"
    if not row_path.is_file():
        raise RuntimeError(f"{out_dir} exists without row.json: not a published run (move it aside)")
    row = json.loads(row_path.read_text())
    bad = {k: (row.get(k), v) for k, v in expect.items() if row.get(k) != v}
    if bad:
        raise ValueError(f"{out_dir}: published run is incompatible with this configuration: {bad}")
    if row["checkpoint"] is None:  # a loss-only pilot keeps no checkpoint
        return row
    ckpt = out_dir / row["checkpoint"]
    if file_sha256(ckpt) != row["checkpoint_sha256"]:
        raise ValueError(f"{ckpt}: SHA-256 differs from the one recorded at publication")
    return row


# ---- one stage ----------------------------------------------------------------------------------


@dataclass
class StageSetup:
    start: object         # the loaded start checkpoint (its policy and frozen stats)
    parent_sha: str
    datasets: dict        # task index -> dataset
    train_cfg: object
    run: object           # the attempt run directory
    context: object
    generator: torch.Generator


def warm_text_cache(policy, dataset, device: str) -> list[str]:
    """Encode ``dataset``'s instructions once, in fp32 and one string per call, before training.

    That is the state the continual runner trains a stage in: it evaluates every task after
    every stage, so a task's instruction is first encoded (and cached, see
    :class:`flowcl.models.encoders.CachedTextEncoder`) by fp32 rollouts before the task trains.
    A fresh process would first encode it inside an AMP training step instead, and the cached
    embedding would differ in its last bits, enough for the trajectory to drift from step 0.
    """
    texts = sorted({episode.language for episode in dataset.episodes})
    with torch.no_grad(), torch.autocast(device_type=torch.device(device).type, enabled=False):
        for text in texts:
            policy.text_encoder([text])
    return texts


def stage_setup(ctx: SeedContext, stage: int, steps: int, lr: float, start_checkpoint: Path,
                out_dir: Path, run_id: str, expect: dict, *, tasks, train_overrides: dict | None,
                device: str, dataset_dir: Path | None, build_datasets: Callable | None,
                warm_cache: bool = False) -> StageSetup:
    """What every stage run shares: the start checkpoint (with its frozen stats), the datasets
    of ``tasks``, the ``TrainConfig`` (the reference's recipe plus ``train_overrides``), the
    attempt run directory, the method context, and seq_ft's stream generator for ``stage``.
    ``warm_cache`` runs :func:`warm_text_cache` on the stage's dataset (the runner's state)."""
    from flowcl.methods.base import TaskContext
    from flowcl.train.checkpoint import load_checkpoint
    from flowcl.train.pipeline import build_dataset
    from flowcl.train.trainer import TrainConfig
    from flowcl.utils.run import create_run

    parent_sha = file_sha256(start_checkpoint)
    start = load_checkpoint(start_checkpoint, device=device)
    spec, stats = start.spec, start.stats  # the checkpoint's frozen stats
    stages = ctx.curriculum.stages
    make = build_datasets or (lambda st: build_dataset([st.ref], spec, stats, n_demos=st.n_demos,
                                                       dataset_dir=dataset_dir))
    datasets = {j: make(stages[j]) for j in tasks}
    train_cfg = TrainConfig(**{**ctx.ref_config["train"], **(train_overrides or {}),
                               "steps": int(steps), "lr": float(lr), "device": device})
    run = create_run(
        run_id=run_id,
        cfg=OmegaConf.create({
            **expect, "task_key": stages[stage].task_key, "reference_run": ctx.ref_dir.name,
            "parent": {"path": str(start_checkpoint), "sha256": parent_sha},
            "train": dataclasses.asdict(train_cfg),
            "stream_seed": {"namespace": ctx.namespace, "stage": stage},
        }),
        seed=ctx.seed, results_root=attempt_parent(out_dir),
    )
    context = TaskContext(task_key=stages[stage].task_key, dataset=datasets[stage], device=device,
                          seed_namespace_run_id=ctx.namespace, method_run_id=run_id)
    generator = torch.Generator(device="cpu").manual_seed(
        stage_seed(ctx.namespace, stages[stage].task_key, stage))
    if warm_cache:
        warm_text_cache(start.policy, datasets[stage], device)
    return StageSetup(start=start, parent_sha=parent_sha, datasets=datasets, train_cfg=train_cfg,
                      run=run, context=context, generator=generator)


def train_stage(
    ctx: SeedContext,
    stage: int,
    steps: int,
    lr: float,
    start_checkpoint: Path,
    out_dir: Path,
    run_id: str,
    *,
    method_spec: dict | None = None,
    train_overrides: dict | None = None,
    setup_method: Callable | None = None,
    extra_row: Callable | None = None,
    evaluator=None,
    bootstrap: dict | None = None,
    device: str = "cuda",
    dataset_dir: Path | None = None,
    build_datasets: Callable | None = None,
    warm_cache: bool = False,
) -> dict:
    """Train task ``stage`` from ``start_checkpoint``; evaluate tasks ``0..stage``; publish.

    ``train_overrides`` (optional) replaces further ``TrainConfig`` fields (batch size, weight
    decay, schedule) and becomes part of the recipe a published run must match;
    ``setup_method(policy) -> method`` (optional) runs before training (freezing or projection);
    ``extra_row(policy, method, start_state) -> dict`` adds fields before ``row.json``;
    ``warm_cache`` (recorded) trains from the runner's text-cache state (:func:`warm_text_cache`).
    """
    from flowcl.analysis.probes import probe_loss
    from flowcl.train.checkpoint import save_checkpoint
    from flowcl.train.trainer import train_one_task

    expect = {"run_id": run_id, "seed": ctx.seed, "stage": stage, "namespace": ctx.namespace,
              "steps": int(steps), "lr": float(lr), "method": method_spec}
    if train_overrides is not None:
        expect["train_overrides"] = dict(train_overrides)
    if warm_cache:
        expect["warm_text_cache"] = True
    published = load_published(out_dir, expect)
    if published is not None:
        print(f"[flowcl] {run_id}: published and valid, reused", flush=True)
        return published
    if evaluator is None:
        evaluator, bootstrap = default_evaluator()

    st = stage_setup(ctx, stage, steps, lr, start_checkpoint, out_dir, run_id, expect,
                     tasks=range(stage + 1), train_overrides=train_overrides, device=device,
                     dataset_dir=dataset_dir, build_datasets=build_datasets, warm_cache=warm_cache)
    policy, spec, stats, datasets, run = st.start.policy, st.start.spec, st.start.stats, st.datasets, st.run
    stages = ctx.curriculum.stages
    method = setup_method(policy) if setup_method else None
    start_state = ({k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
                   if extra_row else None)
    started = time.perf_counter()
    log = train_one_task(policy, datasets[stage], st.train_cfg, method=method, task_idx=stage,
                         generator=st.generator, context=st.context)
    wall = time.perf_counter() - started
    atomic_write_text(run.artifact("losses.json"), json.dumps(log.losses) + "\n")
    ckpt = save_checkpoint(run.subdir("checkpoints") / "final.pt", policy=policy,
                           policy_config=st.start.payload["policy_config"], spec=spec, stats=stats,
                           run_id=run_id, stage=stage, task_key=stages[stage].task_key,
                           extra={"steps": int(steps), "lr": float(lr), "method": method_spec})
    extra = extra_row(policy, method, start_state) if extra_row else {}
    del start_state
    probes = {j: probe_loss(policy, datasets[j], ctx.probe, device) for j in range(stage + 1)}
    report = evaluator(policy, [stages[j].ref for j in range(stage + 1)], spec, stats,
                       run_id=ctx.namespace, stage=stage)
    report.save(run.artifact("eval.json"))
    parent_sha = st.parent_sha
    del policy, st
    _free(device)
    cells = stage_cells(ctx, stage, report, probes, bootstrap)
    row = {
        **expect, "task_key": stages[stage].task_key, "git_sha": git_sha(),
        "parent": {"path": str(start_checkpoint), "sha256": parent_sha},
        "checkpoint": "checkpoints/final.pt", "checkpoint_sha256": file_sha256(ckpt),
        "cells": cells,
        "plasticity": {"lu": cells[str(stage)]["success"],
                       "seq_ft": cells[str(stage)]["reference_success"]["seq_ft"]},
        "training": {"final_loss": log.final_loss, "mean_last_50_loss": log.mean_last(50),
                     "wall_clock_s": wall},
        **extra,
    }
    atomic_write_text(run.artifact("row.json"), json.dumps(row, indent=2) + "\n")
    publish(run.path, out_dir)
    print(f"[flowcl] {run_id}: " + ", ".join(
        f"{c['task_key'].split('/')[0]} {c['success']:.2f}" for c in cells.values()), flush=True)
    return row


def pilot_stage(
    ctx: SeedContext,
    stage: int,
    steps: int,
    lr: float,
    start_checkpoint: Path,
    out_dir: Path,
    run_id: str,
    *,
    stop_after: int,
    method_spec: dict | None = None,
    train_overrides: dict | None = None,
    setup_method: Callable | None = None,
    extra_row: Callable | None = None,
    device: str = "cuda",
    dataset_dir: Path | None = None,
    build_datasets: Callable | None = None,
    warm_cache: bool = False,
) -> dict:
    """Loss only: the first ``stop_after`` steps of task ``stage``'s ``steps``-step schedule.

    Same setup, recipe and stream as :func:`train_stage` (so a pilot is the prefix of the full
    run), but no rollouts and no checkpoint, and only task ``stage``'s data is built: the row
    holds the training losses and that task's probe loss before and after, nothing about any
    other task. ``extra_row(policy, method, None) -> dict`` adds fields; ``warm_cache`` as for
    :func:`train_stage`. Published atomically.
    """
    from flowcl.analysis.probes import probe_loss
    from flowcl.train.trainer import train_one_task

    overrides = {**(train_overrides or {}), "stop_after": int(stop_after)}
    expect = {"run_id": run_id, "kind": "pilot", "seed": ctx.seed, "stage": stage,
              "namespace": ctx.namespace, "steps": int(steps), "lr": float(lr),
              "method": method_spec, "train_overrides": overrides}
    if warm_cache:
        expect["warm_text_cache"] = True
    published = load_published(out_dir, expect)
    if published is not None:
        print(f"[flowcl] {run_id}: published and valid, reused", flush=True)
        return published

    st = stage_setup(ctx, stage, steps, lr, start_checkpoint, out_dir, run_id, expect,
                     tasks=[stage], train_overrides=overrides, device=device,
                     dataset_dir=dataset_dir, build_datasets=build_datasets, warm_cache=warm_cache)
    policy, dataset = st.start.policy, st.datasets[stage]
    method = setup_method(policy) if setup_method else None
    start_probe = probe_loss(policy, dataset, ctx.probe, device)
    started = time.perf_counter()
    log = train_one_task(policy, dataset, st.train_cfg, method=method, task_idx=stage,
                         generator=st.generator, context=st.context)
    wall = time.perf_counter() - started
    atomic_write_text(st.run.artifact("losses.json"), json.dumps(log.losses) + "\n")
    extra = extra_row(policy, method, None) if extra_row else {}
    end_probe = probe_loss(policy, dataset, ctx.probe, device)
    row = {
        **expect, "task_key": ctx.curriculum.stages[stage].task_key, "git_sha": git_sha(),
        "parent": {"path": str(start_checkpoint), "sha256": st.parent_sha},
        "checkpoint": None,
        "losses_finite": all(math.isfinite(x) for x in log.losses),
        "probe": {"start": start_probe, "end": end_probe},
        "training": {"steps": log.steps, "final_loss": log.final_loss,
                     "mean_last_50_loss": log.mean_last(50), "wall_clock_s": wall},
        **extra,
    }
    run_path = st.run.path
    atomic_write_text(st.run.artifact("row.json"), json.dumps(row, indent=2) + "\n")
    del policy, st, dataset
    _free(device)
    publish(run_path, out_dir)
    print(f"[flowcl] {run_id}: probe {start_probe:.4f} -> {end_probe:.4f} after {log.steps} steps",
          flush=True)
    return row


# ---- seed 0's stage 1: the selection run, from another commit ------------------------------------


def selection_stage1(cfg: dict, ctx: SeedContext, root: Path, bootstrap: dict) -> dict:
    """Seed 0's stage 1 as the hparam study produced it, verified against the frozen config."""
    from flowcl.envs.evaluation import EvaluationReport

    source = seed_config(cfg, ctx.seed)["stage1_source"]
    src = root / source
    conf = OmegaConf.to_container(OmegaConf.load(src / "config.yaml"), resolve=True)
    frozen = cfg["frozen"]
    checks = {
        "steps": (conf["train"]["steps"], int(frozen["steps"])),
        "lr": (conf["train"]["lr"], float(frozen["lr"])),
        "stream_seed": (conf["stream_seed"], {"namespace": ctx.namespace, "stage": 1}),
        "reference_run": (conf["reference_run"], ctx.ref_dir.name),
        "start_checkpoint": (conf["start_checkpoint"], "stage0.pt"),
    }
    bad = {k: v for k, v in checks.items()
           if not (math.isclose(v[0], v[1], rel_tol=1e-12) if k == "lr" else v[0] == v[1])}
    if bad:
        raise ValueError(f"{source} does not match the frozen low-update configuration: {bad}")
    report = EvaluationReport.load(src / "eval.json")
    if report.run_id != ctx.namespace:
        raise ValueError(f"{source}: rollouts under {report.run_id!r}, not {ctx.namespace!r}")
    src_row = json.loads((src / "row.json").read_text())
    probes = {0: src_row["probe_loss"]["retained"], 1: src_row["probe_loss"]["new"]}
    cells = stage_cells(ctx, 1, report, probes, bootstrap)
    ckpt = src / "checkpoints" / "final.pt"
    return {
        "run_id": src.name, "seed": ctx.seed, "stage": 1, "namespace": ctx.namespace,
        "steps": int(frozen["steps"]), "lr": float(frozen["lr"]), "method": None,
        "task_key": ctx.curriculum.stages[1].task_key,
        "selection_run": True,
        "source": {"run": source, "git_sha": (src / "git_sha").read_text().strip(),
                   "checkpoint_sha256": file_sha256(ckpt)},
        "checkpoint_path": str(ckpt),
        "cells": cells,
        "plasticity": {"lu": cells["1"]["success"],
                       "seq_ft": cells["1"]["reference_success"]["seq_ft"]},
    }


def stage_dir(cfg: dict, root: Path, seed: int, stage: int) -> Path:
    return root / cfg["out"] / f"seed{seed}" / f"stage{stage}"


def stage_run_id(seed: int, stage: int) -> str:
    return f"lu__seed{seed}__stage{stage}"


def stage1_checkpoint(cfg: dict, root: Path, seed: int) -> Path:
    source = seed_config(cfg, seed)["stage1_source"]
    if source:
        return root / source / "checkpoints" / "final.pt"
    return stage_dir(cfg, root, seed, 1) / "checkpoints" / "final.pt"


# ---- verdicts (pure) ---------------------------------------------------------------------------


def classify_validation(row: dict, object_tolerance: float) -> dict:
    """Stage 1: Object within tolerance of seq_ft's (inclusive) and a Spatial gain CI > 0."""
    obj, spa = row["cells"]["1"], row["cells"]["0"]
    floor = obj["reference_success"]["seq_ft"] - object_tolerance
    gain = spa["paired"]["seq_ft"]
    plasticity = obj["success"] >= floor - TOL
    retention = gain["low"] > 0.0
    return {"passed": plasticity and retention,
            "plasticity": {"passed": plasticity, "object": obj["success"], "floor": floor,
                           "seq_ft_object": obj["reference_success"]["seq_ft"]},
            "retention": {"passed": retention, "spatial": spa["success"], "paired_gain": gain}}


def classify_validation_across_seeds(per_seed: dict, held_out: list[int], selection_seed: int) -> dict:
    missing = [s for s in held_out if s not in per_seed]
    if missing:
        raise ValueError(f"validation needs every held-out seed; missing {missing}")
    failed = [s for s in held_out if not per_seed[s]["passed"]]
    selection = per_seed.get(selection_seed)
    return {
        "verdict": VALIDATED if not failed else NOT_VALIDATED,
        "held_out_seeds": list(held_out),
        "failed_held_out_seeds": {s: [k for k in ("plasticity", "retention")
                                      if not per_seed[s][k]["passed"]] for s in failed},
        "selection_seed": {"seed": selection_seed,
                           "passed": None if selection is None else selection["passed"],
                           "note": "selected the configuration: reported, not evidence"},
        "per_seed": {str(s): v for s, v in per_seed.items()},
    }


def classify_chain(rows: dict, tolerance: float) -> dict:
    """``rows``: seed -> stage -> row (stages 1-3). The pre-registered classification."""
    underlearns = []
    for s, by_stage in rows.items():
        for k in (2, 3):
            c = by_stage[k]["cells"][str(k)]
            if c["success"] < c["reference_success"]["seq_ft"] - tolerance - TOL:
                underlearns.append({"seed": s, "stage": k, "success": c["success"],
                                    "seq_ft": c["reference_success"]["seq_ft"]})
    lows = {str(s): {str(j): by_stage[3]["cells"][str(j)]["paired"]["seq_ft"]["low"]
                     for j in range(3)} for s, by_stage in rows.items()}
    positive = [low > 0.0 for per in lows.values() for low in per.values()]
    if underlearns:
        cls = "underlearns"
    elif all(positive):
        cls = "cumulative"
    elif not any(positive):
        cls = "delay_only"
    else:
        cls = "partial"
    return {"class": cls, "underlearns": underlearns, "final_gain_ci_low_vs_seq_ft": lows}


# ---- the parts ---------------------------------------------------------------------------------


def run_validation_seed(cfg: dict, seed: int, results_root=None, device="cuda",
                        allow_dirty=False, evaluator=None, bootstrap=None, dataset_dir=None,
                        build_datasets=None) -> dict:
    sha = refuse_dirty(allow_dirty)
    root = _root(results_root)
    ctx = seed_context(cfg, seed, root)
    if evaluator is None:
        evaluator, bootstrap = default_evaluator()
    if seed_config(cfg, seed)["stage1_source"]:
        row = selection_stage1(cfg, ctx, root, bootstrap)
    else:
        row = train_stage(ctx, 1, cfg["frozen"]["steps"], cfg["frozen"]["lr"],
                          ctx.ref_dir / "checkpoints" / "stage0.pt",
                          stage_dir(cfg, root, seed, 1), stage_run_id(seed, 1),
                          evaluator=evaluator, bootstrap=bootstrap, device=device,
                          dataset_dir=dataset_dir, build_datasets=build_datasets)
    out = {"git_sha": sha, "seed": seed, "row": row,
           "classification": classify_validation(row, cfg["validation"]["object_tolerance"])}
    path = root / cfg["out"] / f"validation_seed{seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(out, indent=2) + "\n")
    return out


def run_validation(cfg: dict, results_root=None) -> dict:
    root = _root(results_root)
    v = cfg["validation"]
    per_seed = {}
    for seed in [v["selection_seed"], *v["held_out_seeds"]]:
        path = root / cfg["out"] / f"validation_seed{seed}.json"
        if path.is_file():
            per_seed[seed] = json.loads(path.read_text())["classification"]
    out = {"git_sha": git_sha(), **classify_validation_across_seeds(
        per_seed, v["held_out_seeds"], v["selection_seed"])}
    atomic_write_text(root / cfg["out"] / "validation.json", json.dumps(out, indent=2) + "\n")
    print(f"[flowcl] low-update validation: {out['verdict']}", flush=True)
    return out


def gate_open(cfg: dict, results_root=None) -> bool:
    path = _root(results_root) / cfg["out"] / "validation.json"
    return path.is_file() and json.loads(path.read_text())["verdict"] == VALIDATED


def pilot_gate_open(cfg: dict, results_root=None) -> bool:
    path = _root(results_root) / cfg["out"] / "chain_summary.json"
    return (gate_open(cfg, results_root) and path.is_file()
            and json.loads(path.read_text())["classification"]["class"] != "underlearns")


def run_chain(cfg: dict, seed: int, results_root=None, device="cuda", allow_dirty=False,
              evaluator=None, bootstrap=None, dataset_dir=None, build_datasets=None) -> dict:
    refuse_dirty(allow_dirty)
    root = _root(results_root)
    if not gate_open(cfg, root):
        raise RuntimeError("the validation gate is closed: the chain does not run")
    ctx = seed_context(cfg, seed, root)
    if evaluator is None:
        evaluator, bootstrap = default_evaluator()
    start = stage1_checkpoint(cfg, root, seed)
    rows = {}
    for k in cfg["chain"]["stages"]:
        rows[k] = train_stage(ctx, k, cfg["frozen"]["steps"], cfg["frozen"]["lr"], start,
                              stage_dir(cfg, root, seed, k), stage_run_id(seed, k),
                              evaluator=evaluator, bootstrap=bootstrap, device=device,
                              dataset_dir=dataset_dir, build_datasets=build_datasets)
        start = stage_dir(cfg, root, seed, k) / "checkpoints" / "final.pt"
    return rows


def run_chain_summary(cfg: dict, results_root=None) -> dict:
    root = _root(results_root)
    seeds = [cfg["validation"]["selection_seed"], *cfg["validation"]["held_out_seeds"]]
    rows, provenance = {}, {}
    for s in seeds:
        v = json.loads((root / cfg["out"] / f"validation_seed{s}.json").read_text())
        by_stage = {1: v["row"]}
        for k in cfg["chain"]["stages"]:
            by_stage[k] = json.loads((stage_dir(cfg, root, s, k) / "row.json").read_text())
        rows[s] = by_stage
        provenance[str(s)] = {
            "stage1": v["row"].get("source") or {"git_sha": v["row"]["git_sha"],
                                                 "checkpoint_sha256": v["row"]["checkpoint_sha256"]},
            **{f"stage{k}": {"git_sha": by_stage[k]["git_sha"],
                             "parent_sha256": by_stage[k]["parent"]["sha256"]}
               for k in cfg["chain"]["stages"]},
            "abandoned_attempts": abandoned_attempts(root / cfg["out"] / f"seed{s}"),
        }
    n = len(next(iter(rows.values()))[3]["cells"])
    matrices = {str(s): {str(k): {j: by[k]["cells"][j]["success"] for j in by[k]["cells"]}
                         for k in by} for s, by in rows.items()}
    final = {str(s): {"F_1": sum(by[3]["cells"][str(j)]["success"] for j in range(n)) / n,
                      "final_row": {str(j): {"success": by[3]["cells"][str(j)]["success"],
                                             "paired": by[3]["cells"][str(j)]["paired"]}
                                    for j in range(n)}}
             for s, by in rows.items()}
    out = {"git_sha": git_sha(), "config": cfg,
           "classification": classify_chain(rows, cfg["chain"]["plasticity_tolerance"]),
           "matrices": matrices, "final": final, "provenance": provenance}
    atomic_write_text(root / cfg["out"] / "chain_summary.json", json.dumps(out, indent=2) + "\n")
    print(f"[flowcl] low-update chain: {out['classification']['class']}", flush=True)
    return out
