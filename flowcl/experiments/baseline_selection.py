"""Baseline selection: EWC's λ bracket on the development seed (docs/thesis_plan.md, Workstream B).

The rule is ``configs/analysis/baseline_selection.yaml``. One EWC arm per pre-registered ratio r
trains T2 (Object) from seq_ft seed 0's ``stage0.pt`` through
:func:`flowcl.experiments.low_update.train_stage`, with the runner's text-cache state; each arm's
setup computes the T1 Fisher at the stage-0 weights through :func:`ewc_t1_setup`, the same function
:func:`calibrate` uses. The grid is calibrated before any arm (no EWC-arm outcome used):
``λ_r = 2 r L / D`` with D the T1-Fisher-weighted drift of unregularised fine-tuning and L seq_ft's
stage-1 training loss. :func:`select_lambda` (pure) applies the selection rule, including the single
edge extension; :func:`run_select` freezes the choice in an immutable bundle that must carry this
rule's and the calibration's SHA-256.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import torch
from omegaconf import OmegaConf

from flowcl.experiments.low_update import _root, load_low_update_config, load_published, refuse_dirty
from flowcl.utils.libero_paths import repo_root
from flowcl.utils.run import atomic_write_text, file_sha256, git_sha

TOL = 1e-9                # rates are k/50: allow float rounding only
EXTENSION_NEEDED = 10     # the CLI's exit code when --select owes the edge extension first
FAMILY = "ewc"


def rule_path(path: str | Path | None = None) -> Path:
    return Path(path) if path else repo_root() / "configs" / "analysis" / "baseline_selection.yaml"


def load_config(path: str | Path | None = None) -> dict:
    return OmegaConf.to_container(OmegaConf.load(rule_path(path)), resolve=True)


def ratio_tag(ratio: float) -> str:
    return f"{float(ratio):g}"


def arm_run_id(ratio: float, seed: int) -> str:
    return f"ewcb__r{ratio_tag(ratio)}__seed{seed}"


def extension_ratio(cfg: dict) -> float:
    return float(cfg["ewc"]["edge_extension_factor"]) * max(float(r) for r in cfg["ewc"]["ratios"])


def _out(cfg: dict, out_root: Path) -> Path:
    return Path(out_root) / cfg["out"]


def calibration_path(cfg: dict, out_root: Path) -> Path:
    return _out(cfg, out_root) / "ewc_calibration.json"


# ---- the shared setup -----------------------------------------------------------------------------


def ewc_t1_setup(ctx, lam: float, fisher_samples: int, device: str):
    """``setup(policy, datasets) -> EWC`` with the T1 Fisher computed at the current weights.

    Used by the calibration and by every arm, so both compute the same Fisher: T1's dataset, the
    production seeds ``derive_seed(namespace, "ewc_fisher::<T1>", 0)``, ``train()`` mode."""
    from flowcl.methods.base import TaskContext
    from flowcl.methods.ewc import EWC

    def setup(policy, datasets):
        method = EWC(lam=lam, fisher_samples=fisher_samples)
        t1 = ctx.curriculum.stages[0]
        context = TaskContext(task_key=t1.task_key, dataset=datasets[0], device=device,
                              seed_namespace_run_id=ctx.namespace)
        method.on_task_end(policy, 0, context=context)
        return method

    return setup


# ---- calibration ----------------------------------------------------------------------------------


def calibration_lambdas(ratios: list[float], D: float, L: float) -> dict[str, float]:
    """``{ratio: 2 r L / D}``; refuses non-finite inputs and ``D <= 0`` or ``L <= 0``."""
    for name, value in (("D", D), ("L", L)):
        if not (isinstance(value, (int, float)) and math.isfinite(value)):
            raise ValueError(f"calibration {name} is not finite: {value!r}")
    if D <= 0 or L <= 0:
        raise ValueError(f"calibration needs D > 0 and L > 0, got D = {D!r}, L = {L!r}")
    return {ratio_tag(r): 2.0 * float(r) * float(L) / float(D) for r in ratios}


def calibrate(cfg: dict, *, results_root=None, out_root=None, device: str = "cuda", dataset_dir=None,
              build_datasets=None, low_update_cfg: dict | None = None) -> dict:
    """Compute D, L and the λ grid (no EWC-arm outcome) and write the calibration file."""
    from flowcl.experiments.low_update import seed_context, warm_runner_cache
    from flowcl.methods.ewc import trainable_parameters
    from flowcl.train.checkpoint import load_checkpoint
    from flowcl.train.pipeline import build_dataset

    root = _root(results_root)
    out_root = Path(out_root) if out_root else root
    if cfg["ewc"].get("calibration") is not None:
        raise RuntimeError("the rule is already calibrated; a new calibration needs a new rule")
    seed = int(cfg["bracket_seed"])
    ctx = seed_context(low_update_cfg or load_low_update_config(), seed, root)
    stage0, stage1 = ctx.ref_dir / "checkpoints" / "stage0.pt", ctx.ref_dir / "checkpoints" / "stage1.pt"
    start = load_checkpoint(stage0, device=device)
    stages = ctx.curriculum.stages
    make = build_datasets or (lambda st: build_dataset([st.ref], start.spec, start.stats, n_demos=st.n_demos,
                                                       dataset_dir=dataset_dir))
    datasets = {j: make(stages[j]) for j in (0, 1)}  # what train_stage builds for stage 1
    cache = warm_runner_cache(start.policy, datasets, device, amp=bool(ctx.ref_config["train"].get("amp")))
    method = ewc_t1_setup(ctx, 1.0, int(cfg["ewc"]["fisher_samples"]), device)(start.policy, datasets)
    fisher = method.fisher
    theta0 = dict(trainable_parameters(start.policy))
    state1 = torch.load(stage1, map_location="cpu", weights_only=False)["state_dict"]
    D = 0.0
    for name, f in fisher.items():
        drift = state1[name].to(f.device, torch.float64) - theta0[name].detach().to(torch.float64)
        D += float((f.to(torch.float64) * drift.pow(2)).sum())
    result_path = ctx.ref_dir / "result.json"
    stage_rec = json.loads(result_path.read_text())["stages"][1]
    L = stage_rec[cfg["ewc"]["stage1_loss_key"]]
    lambdas = calibration_lambdas(cfg["ewc"]["ratios"], D, L)
    payload = {
        "git_sha": git_sha(), "seed": seed, "namespace": ctx.namespace,
        "rule": {"path": str(rule_path()), "sha256": file_sha256(rule_path())},
        "stage0": {"path": str(stage0), "sha256": file_sha256(stage0)},
        "stage1": {"path": str(stage1), "sha256": file_sha256(stage1)},
        "result_json": {"path": str(result_path), "sha256": file_sha256(result_path)},
        "L": {"value": L, "key": cfg["ewc"]["stage1_loss_key"]},
        "D": D, "ratios": [float(r) for r in cfg["ewc"]["ratios"]], "lambdas": lambdas,
        "fisher_t1": method.fisher_tasks[0], "text_cache": cache,
        "n_trainable_parameters": int(sum(f.numel() for f in fisher.values())),
    }
    path = calibration_path(cfg, out_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(payload, indent=2, default=str) + "\n")
    print(f"[flowcl] EWC calibration: D {D:.6g}, L {L!r}; λ {lambdas} -> {path} "
          f"(sha256 {file_sha256(path)})", flush=True)
    return payload


def validate_calibration(cfg: dict, out_root: Path) -> tuple[dict, dict, str]:
    """The rule's registered calibration: ``(lambdas, payload, sha256)``; raises on any mismatch."""
    reg = cfg["ewc"].get("calibration")
    if not reg:
        raise RuntimeError("the rule is not calibrated yet (ewc.calibration is null)")
    path = Path(out_root) / reg["file"]
    sha = file_sha256(path)
    if sha != reg["sha256"]:
        raise ValueError(f"{path}: SHA-256 {sha} differs from the rule's {reg['sha256']}")
    payload = json.loads(path.read_text())
    recomputed = calibration_lambdas(payload["ratios"], payload["D"], payload["L"]["value"])
    registered = {ratio_tag(k): float(v) for k, v in reg["lambdas"].items()}
    if registered != recomputed or payload["lambdas"] != recomputed:
        raise ValueError(f"registered λ {registered} differ from the calibration's {recomputed}")
    if [float(r) for r in payload["ratios"]] != [float(r) for r in cfg["ewc"]["ratios"]]:
        raise ValueError("the calibration's ratios differ from the rule's")
    return recomputed, payload, sha


def extension_lambda(payload: dict, cfg: dict) -> float:
    return calibration_lambdas([extension_ratio(cfg)], payload["D"], payload["L"]["value"])[
        ratio_tag(extension_ratio(cfg))]


# ---- arms -----------------------------------------------------------------------------------------


def method_spec(cfg: dict, ratio: float, lam: float, rule_sha: str, calibration_sha: str) -> dict:
    return {"family": FAMILY, "ratio": float(ratio), "lam": float(lam),
            "fisher_samples": int(cfg["ewc"]["fisher_samples"]),
            "rule_sha256": rule_sha, "calibration_sha256": calibration_sha}


def arm_dir(cfg: dict, out_root: Path, ratio: float, seed: int) -> Path:
    return _out(cfg, out_root) / "ewc" / f"seed{seed}" / arm_run_id(ratio, seed)


def run_arm(cfg: dict, ratio: float, *, results_root=None, out_root=None, device: str = "cuda",
            evaluator=None, bootstrap=None, dataset_dir=None, build_datasets=None,
            low_update_cfg: dict | None = None, allow_dirty: bool = False, extension: bool = False,
            smoke_stop_after: int | None = None, smoke_episodes: int | None = None) -> dict:
    """Train and publish one arm (reused if already published under this rule and calibration)."""
    from flowcl.experiments.hparam_study import default_evaluator
    from flowcl.experiments.low_update import seed_context, train_stage

    smoke = smoke_stop_after is not None or smoke_episodes is not None
    refuse_dirty(allow_dirty or smoke)
    root = _root(results_root)
    out_root = Path(out_root) if out_root else root
    lambdas, payload, cal_sha = validate_calibration(cfg, root)
    tag = ratio_tag(ratio)
    if extension:
        if float(ratio) != extension_ratio(cfg):
            raise ValueError(f"the extension ratio is {extension_ratio(cfg)}, not {ratio}")
        lam = extension_lambda(payload, cfg)
    elif tag in lambdas:
        lam = lambdas[tag]
    else:
        raise ValueError(f"ratio {ratio} is not in the rule's grid {sorted(lambdas)}")
    seed = int(cfg["bracket_seed"])
    ctx = seed_context(low_update_cfg or load_low_update_config(), seed, root)
    if smoke:  # 1-episode cells cannot be paired with the 50-episode references
        ctx = dataclasses.replace(ctx, evals={role: {} for role in ctx.evals})
    if evaluator is None:
        evaluator, bootstrap = default_evaluator(smoke_episodes)
    n = int(cfg["ewc"]["fisher_samples"])

    def extra_row(policy, method, start_state):
        return {"ewc": {"ratio": float(ratio), "lam": float(lam), "extension": extension,
                        "t1_fisher": method.fisher_tasks[0],
                        "penalty_log": method.state_dict()["penalty_log"]}}

    return train_stage(
        ctx, 1, cfg["ewc"]["recipe"]["steps"], cfg["ewc"]["recipe"]["lr"],
        ctx.ref_dir / "checkpoints" / "stage0.pt", arm_dir(cfg, out_root, ratio, seed),
        arm_run_id(ratio, seed),
        method_spec=method_spec(cfg, ratio, lam, file_sha256(rule_path()), cal_sha),
        train_overrides={"stop_after": int(smoke_stop_after)} if smoke_stop_after is not None else None,
        setup_method=ewc_t1_setup(ctx, lam, n, device), setup_method_with_data=True,
        extra_row=extra_row, evaluator=evaluator, bootstrap=bootstrap, device=device,
        dataset_dir=dataset_dir, build_datasets=build_datasets, warm_cache="runner",
    )


def smoke_arm(cfg: dict, ratio: float, *, results_root=None, out_root=None, stop_after: int = 200,
              episodes: int = 1, **kw) -> dict:
    """One arm at smoke settings into scratch; its T1 Fisher must equal the calibration's."""
    row = run_arm(cfg, ratio, results_root=results_root, out_root=out_root, smoke_stop_after=stop_after,
                  smoke_episodes=episodes, **kw)
    _, payload, _ = validate_calibration(cfg, _root(results_root))
    got, want = row["ewc"]["t1_fisher"]["fingerprint"], payload["fisher_t1"]["fingerprint"]
    return {"passed": got == want, "arm_fingerprint": got, "calibration_fingerprint": want,
            "run_id": row["run_id"]}


def published_arms(cfg: dict, out_root: Path, rule_sha: str, cal_sha: str, lambdas: dict,
                   payload: dict) -> dict[str, dict | None]:
    """Every grid arm's published row (``None`` if absent) and the extension's, validated."""
    seed = int(cfg["bracket_seed"])
    arms = {}
    for r in cfg["ewc"]["ratios"]:
        spec = method_spec(cfg, r, lambdas[ratio_tag(r)], rule_sha, cal_sha)
        arms[ratio_tag(r)] = load_published(arm_dir(cfg, out_root, r, seed),
                                            {"run_id": arm_run_id(r, seed), "method": spec})
    r_ext = extension_ratio(cfg)
    spec = method_spec(cfg, r_ext, extension_lambda(payload, cfg), rule_sha, cal_sha)
    arms["extension"] = load_published(arm_dir(cfg, out_root, r_ext, seed),
                                       {"run_id": arm_run_id(r_ext, seed), "method": spec})
    return arms


# ---- the rule (pure) ------------------------------------------------------------------------------


def select_lambda(arms: list[dict], floor: float) -> dict:
    """The selection rule over ``[{"ratio", "lam", "spatial", "object", "extension"}]``.

    ``status`` is ``"needs_extension"`` when the largest grid λ wins and no extension arm is given;
    otherwise ``"selected"`` with the chosen arm and its flags."""
    grid = [a for a in arms if not a.get("extension")]
    ext = [a for a in arms if a.get("extension")]
    if not grid:
        raise ValueError("no grid arms")
    eligible = [a for a in arms if a["object"] >= floor - TOL]
    flags: list[str] = []
    if not eligible:
        chosen = min(arms, key=lambda a: a["lam"])
        flags.append("no_qualifying_setting")
    else:
        best = max(a["spatial"] for a in eligible)
        chosen = min((a for a in eligible if a["spatial"] >= best - TOL), key=lambda a: a["lam"])
    largest_grid = max(grid, key=lambda a: a["lam"])
    if not ext and eligible and chosen is largest_grid:
        return {"status": "needs_extension", "selected": None, "flags": ["edge_extension_owed"],
                "eligible": [a["ratio"] for a in eligible]}
    if ext:
        flags.append("edge_extension")
        if chosen is ext[0]:
            flags.append("optimum_at_boundary")
    return {"status": "selected", "selected": chosen, "flags": flags,
            "eligible": [a["ratio"] for a in eligible]}


def _arm_summary(ratio: float, lam: float, row: dict, extension: bool) -> dict:
    cells = row["cells"]
    return {"ratio": float(ratio), "lam": float(lam), "extension": extension, "run_id": row["run_id"],
            "spatial": cells["0"]["success"], "object": cells["1"]["success"],
            "cells": cells, "checkpoint_sha256": row["checkpoint_sha256"]}


def _arm_table(cfg, out_root, rule_sha, cal_sha, lambdas, payload, require_extension=False):
    published = published_arms(cfg, out_root, rule_sha, cal_sha, lambdas, payload)
    missing = [t for t in map(ratio_tag, cfg["ewc"]["ratios"]) if published[t] is None]
    if missing:
        raise FileNotFoundError(f"bracket arms not published: ratios {missing}")
    arms = [_arm_summary(r, lambdas[ratio_tag(r)], published[ratio_tag(r)], False) for r in cfg["ewc"]["ratios"]]
    if published["extension"] is not None:
        arms.append(_arm_summary(extension_ratio(cfg), extension_lambda(payload, cfg),
                                 published["extension"], True))
    elif require_extension:
        raise FileNotFoundError("the edge-extension arm is not published")
    return arms


def _floor(arms: list[dict], tolerance: float) -> tuple[float, float]:
    refs = {a["cells"]["1"]["reference_success"].get("seq_ft") for a in arms}
    if len(refs) != 1 or None in refs:
        raise ValueError(f"the arms disagree on seq_ft's stage-1 Object: {refs}")
    ref = refs.pop()
    return ref, ref - tolerance


def run_select(cfg: dict, *, results_root=None, out_root=None) -> dict:
    """Apply the rule. Returns ``{"status": "needs_extension"}`` (nothing published) or the bundle."""
    from flowcl.experiments.sgp_sensitivity import publish_selection

    root = _root(results_root)
    out_root = Path(out_root) if out_root else root
    lambdas, payload, cal_sha = validate_calibration(cfg, root)
    rule_sha = file_sha256(rule_path())
    arms = _arm_table(cfg, out_root, rule_sha, cal_sha, lambdas, payload)
    seq_ft_object, floor = _floor(arms, float(cfg["ewc"]["object_tolerance"]))
    choice = select_lambda(arms, floor)
    if choice["status"] == "needs_extension":
        print(f"[flowcl] EWC selection: the largest λ wins; the edge extension "
              f"(r = {extension_ratio(cfg):g}) runs before anything is published", flush=True)
        return {"status": "needs_extension", "eligible": choice["eligible"]}
    chosen = choice["selected"]
    selection = {
        "git_sha": git_sha(), "seed": int(cfg["bracket_seed"]), "family": FAMILY,
        "rule_sha256": rule_sha, "calibration_sha256": cal_sha,
        "seq_ft_object": seq_ft_object, "object_floor": floor,
        "arms": {ratio_tag(a["ratio"]): {k: a[k] for k in ("ratio", "lam", "extension", "run_id", "spatial",
                                                           "object", "checkpoint_sha256")} |
                 {"paired_vs_seq_ft": {j: a["cells"][j].get("paired", {}).get("seq_ft") for j in ("0", "1")}}
                 for a in arms},
        "eligible": choice["eligible"],
        "selected": {"ratio": chosen["ratio"], "lam": chosen["lam"], "run_id": chosen["run_id"],
                     "checkpoint": str(arm_dir(cfg, out_root, chosen["ratio"], int(cfg["bracket_seed"]))
                                       / "checkpoints" / "final.pt"),
                     "checkpoint_sha256": chosen["checkpoint_sha256"]},
        "flags": choice["flags"],
    }
    published, sha = publish_selection(_out(cfg, out_root) / "ewc", selection)
    check_bundle(published, rule_sha, cal_sha)
    print(f"[flowcl] EWC selection: r {published['selected']['ratio']:g}, λ {published['selected']['lam']:.6g} "
          f"{published['flags']} (bundle {sha[:12]})", flush=True)
    return {"status": "selected", **published, "bundle_sha256": sha}


def check_bundle(bundle: dict, rule_sha: str, cal_sha: str) -> None:
    """A bundle (new or reused) must carry this rule's and this calibration's SHA-256."""
    if bundle.get("rule_sha256") != rule_sha or bundle.get("calibration_sha256") != cal_sha:
        raise ValueError("the selection bundle was made under another rule or calibration "
                         f"(bundle: {bundle.get('rule_sha256')}, {bundle.get('calibration_sha256')}; "
                         f"now: {rule_sha}, {cal_sha})")


def run_extension(cfg: dict, **kw) -> dict:
    """The single edge-extension arm; refuses unless the grid's rule says it is owed."""
    root = _root(kw.get("results_root"))
    out_root = Path(kw.get("out_root") or root)
    lambdas, payload, cal_sha = validate_calibration(cfg, root)
    arms = _arm_table(cfg, out_root, file_sha256(rule_path()), cal_sha, lambdas, payload)
    grid = [a for a in arms if not a["extension"]]
    _, floor = _floor(grid, float(cfg["ewc"]["object_tolerance"]))
    if select_lambda(grid, floor)["status"] != "needs_extension":
        raise RuntimeError("no edge extension is owed: the largest λ does not win the grid")
    return run_arm(cfg, extension_ratio(cfg), extension=True, **kw)


def verify(cfg: dict, *, results_root=None, out_root=None) -> dict:
    """Every required arm published under this rule and calibration, and the bundle intact."""
    from flowcl.experiments.sgp_sensitivity import load_selection

    root = _root(results_root)
    out_root = Path(out_root) if out_root else root
    problems = []
    try:
        lambdas, payload, cal_sha = validate_calibration(cfg, root)
        rule_sha = file_sha256(rule_path())
        bundle, sha = load_selection(_out(cfg, out_root) / "ewc")
        check_bundle(bundle, rule_sha, cal_sha)
        arms = _arm_table(cfg, out_root, rule_sha, cal_sha, lambdas, payload,
                          require_extension="edge_extension" in bundle["flags"])
        published = {ratio_tag(a["ratio"]): a["checkpoint_sha256"] for a in arms}
        if {k: v["checkpoint_sha256"] for k, v in bundle["arms"].items()} != published:
            problems.append("the bundle's arms differ from the published rows")
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        problems.append(f"{type(exc).__name__}: {exc}")
        sha = None
    return {"passed": not problems, "problems": problems, "bundle_sha256": sha}


def write_smoke_method(cfg: dict, path: Path, *, results_root=None, ratio: float = 1.0,
                       fisher_samples: int = 256) -> Path:
    """A scratch EWC method config with a calibrated λ, for smoke runs of the 4-task runner."""
    lambdas, _, _ = validate_calibration(cfg, _root(results_root))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create({"name": "ewc", "lam": lambdas[ratio_tag(ratio)],
                                     "fisher_samples": int(fisher_samples), "log_interval": 100}), path)
    return path
