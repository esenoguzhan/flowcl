"""LoRA's rank × learning-rate bracket on the development seed (docs/thesis_plan.md, Workstream B).

The rule is ``configs/analysis/baseline_selection_lora.yaml`` (committed before any arm). One arm per
grid cell trains T2 (Object) from seq_ft seed 0's ``stage0.pt`` through
:func:`flowcl.experiments.low_update.train_stage` with the runner's text-cache state, the method being
:class:`flowcl.methods.lora.LoRA` at the arm's rank and learning rate. :func:`select_lora` (pure)
applies the rule, including the edge extensions; :func:`run_select` freezes the choice in an immutable
bundle bound to the rule's SHA-256 and to the inputs every row records; :func:`verify` re-validates
every row's inputs and recomputes the selection from them.

A follow-up rule (``configs/analysis/baseline_selection_lora_v2.yaml``) may **import** an earlier
rule's published arms (section ``imports``: that rule's file and SHA-256 and its bundle's SHA-256)
instead of retraining them. Imported rows stay bound to their own rule; whether one is a grid or an
extension arm is decided by the importing grid; an imported extension arm is only *available* and
enters the selection only if the importing grid's selection owes exactly that cell.

A rule with a ``projection`` section brackets :class:`flowcl.methods.projected_lora.ProjectedLoRA`
instead (``family`` and ``arm_prefix`` name it): its variant, the Task-1 GPM memory it installs (bound by
SHA-256; the start checkpoint must equal that GPM run's stage 0 bitwise), and the one-time Task-1 input
Gram (``kold``, :func:`capture_kold`) for the layer-output-change diagnostic every arm records.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from flowcl.experiments.low_update import _root, load_low_update_config, load_published, refuse_dirty
from flowcl.utils.libero_paths import repo_root
from flowcl.utils.run import file_sha256, git_sha

TOL = 1e-9                # rates are k/50: allow float rounding only
EXTENSION_NEEDED = 10     # the CLI's exit code when --select owes an edge extension first
FAMILY = "lora"
EVAL_CONFIG = Path("configs") / "eval" / "libero_eval.yaml"


def rule_path(path: str | Path | None = None) -> Path:
    return Path(path) if path else repo_root() / "configs" / "analysis" / "baseline_selection_lora.yaml"


def load_config(path: str | Path | None = None) -> dict:
    return OmegaConf.to_container(OmegaConf.load(rule_path(path)), resolve=True)


def lr_tag(lr: float) -> str:
    return f"{float(lr):g}"


def arm_key(rank: int, lr: float) -> str:
    return f"r{int(rank)}__lr{lr_tag(lr)}"


def arm_run_id(rank: int, lr: float, seed: int, prefix: str = "lorab") -> str:
    return f"{prefix}__{arm_key(rank, lr)}__seed{seed}"


def family(cfg: dict) -> str:
    return cfg.get("family", FAMILY)


def arm_prefix(cfg: dict) -> str:
    return cfg.get("arm_prefix", "lorab")


def _out(cfg: dict, out_root: Path) -> Path:
    return Path(out_root) / cfg["out"]


def arm_dir(cfg: dict, out_root: Path, rank: int, lr: float, seed: int) -> Path:
    return _out(cfg, out_root) / "lora" / f"seed{seed}" / arm_run_id(rank, lr, seed, arm_prefix(cfg))


def grid(cfg: dict) -> list[tuple[int, float]]:
    """The 9 grid cells, rank-major (the queue's order)."""
    return [(int(r), float(lr)) for r in cfg["lora"]["ranks"] for lr in cfg["lora"]["lrs"]]


def extension_candidates(cfg: dict) -> list[tuple[int, float]]:
    """Every cell an extension could ever be: beyond either rank end at a grid lr, or beyond either lr
    end at a grid rank."""
    ext = cfg["lora"]["extension"]
    out = [(int(ext["rank"][side]), float(lr)) for side in ("low", "high") for lr in cfg["lora"]["lrs"]
           if ext.get("rank")]
    out += [(int(r), float(ext["lr"][side])) for side in ("low", "high") for r in cfg["lora"]["ranks"]]
    return out


def method_spec(cfg: dict, rank: int, lr: float, rule_sha: str) -> dict:
    from flowcl.models.lora import INIT_SPEC

    spec = {"family": family(cfg), "rank": int(rank), "alpha": float(cfg["lora"]["alpha"]), "lr": float(lr),
            "init": INIT_SPEC, "rule_sha256": rule_sha}
    if cfg.get("projection"):
        proj = cfg["projection"]
        spec["projection"] = {"variant": proj["variant"], "eps": float(proj["eps"]),
                              "memory_sha256": proj["memory"]["sha256"]}
    return spec


def projection_method(projection: dict, rank: int, lr: float, alpha: float):
    from flowcl.methods.projected_lora import ProjectedLoRA

    return ProjectedLoRA(variant=projection["variant"], rank=rank, lr=lr, alpha=alpha, eps=float(projection["eps"]),
                         update_memory=False, capture_config=projection.get("capture_config", "subspace"))


def memory_path(cfg: dict, root: Path) -> Path:
    mem = cfg["projection"]["memory"]
    return Path(root) / mem["run"] / mem["file"]


def kold_path(cfg: dict, root: Path) -> Path:
    return Path(root) / cfg["projection"]["kold"]["file"]


def _json_sha(payload) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def expected_train_config(ctx, lr: float, steps: int, device: str) -> dict:
    """The TrainConfig :func:`low_update.stage_setup` builds for an arm (the reference's recipe)."""
    from flowcl.train.trainer import TrainConfig

    return dataclasses.asdict(TrainConfig(**{**ctx.ref_config["train"], "steps": int(steps), "lr": float(lr),
                                             "device": device}))


def _file(path: Path) -> dict:
    return {"path": str(path), "sha256": file_sha256(path)}


def shared_inputs(cfg: dict, ctx, start_checkpoint: Path, rule_sha: str, memory: Path | None = None) -> dict:
    """What every arm must share, from the current files (``memory``: a smoke's memory in place of the
    rule's; a K_old file not yet captured is recorded as None, which only a smoke can train with)."""
    from flowcl.models.lora import INIT_SPEC

    eval_path = repo_root() / EVAL_CONFIG
    root = ctx.ref_dir.parent
    return {
        "start_checkpoint": _file(Path(start_checkpoint)),
        "reference_result": _file(ctx.ref_dir / "result.json"),
        "reference_evals_stage1": {role: _file(root / run / "eval" / "stage1.json")
                                   for role, run in sorted(ctx.runs.items())},
        "evaluation_config": {**_file(eval_path),
                              "content": OmegaConf.to_container(OmegaConf.load(eval_path), resolve=True)},
        "probe": ctx.probe,
        "namespace": ctx.namespace,
        "init": INIT_SPEC,
        "alpha": float(cfg["lora"]["alpha"]),
        "recipe_steps": int(cfg["lora"]["recipe"]["steps"]),
        "rule_sha256": rule_sha,
        **({"projection": _projection_inputs(cfg, root, memory)} if cfg.get("projection") else {}),
    }


def _projection_inputs(cfg: dict, root: Path, memory: Path | None) -> dict:
    kold = kold_path(cfg, root)
    return {"variant": cfg["projection"]["variant"], "eps": float(cfg["projection"]["eps"]),
            "memory": _file(Path(memory) if memory is not None else memory_path(cfg, root)),
            "kold": _file(kold) if kold.is_file() else None}


def state_fingerprint(state_dict: dict) -> str:
    from flowcl.methods.ewc import tensor_fingerprint

    return tensor_fingerprint(state_dict)


def checkpoint_fingerprint(path: Path) -> str:
    return state_fingerprint(torch.load(path, map_location="cpu", weights_only=False)["state_dict"])


# ---- arms -----------------------------------------------------------------------------------------


def run_arm(cfg: dict, rank: int, lr: float, *, results_root=None, out_root=None, device: str = "cuda",
            evaluator=None, bootstrap=None, dataset_dir=None, build_datasets=None,
            low_update_cfg: dict | None = None, allow_dirty: bool = False, extension: bool = False,
            rule: Path | None = None, smoke_start: Path | None = None, smoke_steps: int | None = None,
            smoke_episodes: int | None = None, smoke_memory: Path | None = None) -> dict:
    """Train and publish one arm (reused if already published under this rule)."""
    from flowcl.experiments.hparam_study import default_evaluator
    from flowcl.experiments.low_update import seed_context, train_stage
    from flowcl.methods.lora import LoRA

    smoke = smoke_start is not None or smoke_steps is not None or smoke_episodes is not None
    refuse_dirty(allow_dirty or smoke)
    rank, lr = int(rank), float(lr)
    cell = (rank, lr_tag(lr))
    if extension:
        if cell not in {(r, lr_tag(x)) for r, x in extension_candidates(cfg)}:
            raise ValueError(f"({rank}, {lr}) is not an extension cell of this rule")
    elif cell not in {(r, lr_tag(x)) for r, x in grid(cfg)}:
        raise ValueError(f"({rank}, {lr}) is not in the rule's grid {grid(cfg)}")
    root = _root(results_root)
    out_root = Path(out_root) if out_root else root
    seed = int(cfg["bracket_seed"])
    if not smoke and cfg.get("imports") and cell in imported_cells(cfg, out_root):
        raise ValueError(f"({rank}, {lr}) is imported from {cfg['imports']['rule']}; it is not retrained")
    ctx = seed_context(low_update_cfg or load_low_update_config(), seed, root)
    if smoke:  # 1-episode cells cannot be paired with the 50-episode references
        ctx = dataclasses.replace(ctx, evals={role: {} for role in ctx.evals})
    if evaluator is None:
        evaluator, bootstrap = default_evaluator(smoke_episodes)
    rule_sha = file_sha256(rule_path(rule))
    start = Path(smoke_start) if smoke_start is not None else ctx.ref_dir / "checkpoints" / "stage0.pt"
    steps = int(smoke_steps) if smoke_steps is not None else int(cfg["lora"]["recipe"]["steps"])
    projection = cfg.get("projection")
    if smoke_memory is not None and not (smoke and projection):
        raise ValueError("smoke_memory is for smokes of a projection rule")
    inputs = shared_inputs(cfg, ctx, start, rule_sha, memory=smoke_memory)
    train_config = expected_train_config(ctx, lr, steps, device)
    alpha = float(cfg["lora"]["alpha"])
    kold = None
    if projection:
        mem = Path(smoke_memory) if smoke_memory is not None else memory_path(cfg, root)
        if smoke_memory is None:
            if file_sha256(mem) != projection["memory"]["sha256"]:
                raise ValueError(f"{mem} is not the memory the rule names ({projection['memory']['sha256']})")
            check_start_is_memory_s(cfg, root, start)
        if inputs["projection"]["kold"] is not None:
            kold = load_kold(kold_path(cfg, root))
        elif not smoke:
            raise FileNotFoundError(f"{kold_path(cfg, root)}: capture K_old (--capture-kold) before any arm")

    def setup(policy):
        if not projection:
            return LoRA(rank=rank, lr=lr, alpha=alpha)
        from flowcl.methods.projected_lora import ProjectedLoRA

        method = projection_method(projection, rank, lr, alpha)
        method.install_memory(mem, file_sha256(mem))
        return method

    def extra_row(policy, method, start_state):
        rec = {"rank": rank, "alpha": alpha, "lr": lr, "extension": extension,
               "history": method.history["1"], "inputs": inputs,
               **({"timings": method.timings["1"]} if projection else {}),
               "train_config": train_config, "train_config_sha256": _json_sha(train_config),
               "state_fingerprint": state_fingerprint(policy.state_dict())}
        if projection and kold is not None:
            rec["output_change"] = registry_output_change(policy, start_state, kold, method._memory)
        return {"lora": rec}

    out_dir = arm_dir(cfg, out_root, rank, lr, seed)
    row = train_stage(
        ctx, 1, steps, lr, start, out_dir, arm_run_id(rank, lr, seed, arm_prefix(cfg)),
        method_spec=method_spec(cfg, rank, lr, rule_sha),
        setup_method=setup,
        extra_row=extra_row, evaluator=evaluator, bootstrap=bootstrap, device=device,
        dataset_dir=dataset_dir, build_datasets=build_datasets, warm_cache="runner",
    )
    actual = _arm_train_config(out_dir)
    if actual != train_config:
        raise RuntimeError(f"{out_dir}: the arm trained under {actual}, not the rule's {train_config}")
    return row


def _arm_train_config(out_dir: Path) -> dict:
    return OmegaConf.to_container(OmegaConf.load(Path(out_dir) / "config.yaml"), resolve=True)["train"]


def published_rows(cfg: dict, out_root: Path, rule_sha: str) -> dict[tuple[int, str], dict]:
    """Every published grid or extension arm, re-validated by :func:`load_published`."""
    seed = int(cfg["bracket_seed"])
    steps = int(cfg["lora"]["recipe"]["steps"])
    rows = {}
    cells = [(r, lr, False) for r, lr in grid(cfg)] + [(r, lr, True) for r, lr in extension_candidates(cfg)]
    for rank, lr, ext in cells:
        row = load_published(arm_dir(cfg, out_root, rank, lr, seed),
                             {"run_id": arm_run_id(rank, lr, seed, arm_prefix(cfg)), "method": method_spec(cfg, rank, lr, rule_sha),
                              "lr": float(lr), "steps": steps})
        if row is not None:
            if bool(row["lora"]["extension"]) != ext:
                raise ValueError(f"{arm_run_id(rank, lr, seed, arm_prefix(cfg))}: published with extension="
                                 f"{row['lora']['extension']}, expected {ext}")
            rows[(rank, lr_tag(lr))] = row
    return rows


def _import_rule_path(cfg: dict) -> Path:
    path = Path(cfg["imports"]["rule"])
    return path if path.is_absolute() else repo_root() / path


def load_imports(cfg: dict, out_root: Path) -> dict:
    """The imported rule, checked: its file's and its bundle's SHA-256 are the ones this rule names, and
    it shares this rule's seed, method constants, recipe and floor."""
    from flowcl.experiments.sgp_sensitivity import load_selection

    imp = cfg["imports"]
    path = _import_rule_path(cfg)
    sha = file_sha256(path)
    if sha != imp["rule_sha256"]:
        raise ValueError(f"the imported rule {path} hashes to {sha}, this rule names {imp['rule_sha256']}")
    icfg = load_config(path)
    for key in ("alpha", "recipe", "object_tolerance"):
        if icfg["lora"][key] != cfg["lora"][key]:
            raise ValueError(f"the imported rule's lora.{key} {icfg['lora'][key]} is not {cfg['lora'][key]}")
    if icfg["bracket_seed"] != cfg["bracket_seed"]:
        raise ValueError("the imported rule's bracket seed differs")
    bundle, bundle_sha = load_selection(_out(icfg, out_root) / "lora")
    if bundle_sha != imp["bundle_sha256"]:
        raise ValueError(f"the imported bundle hashes to {bundle_sha}, this rule names {imp['bundle_sha256']}")
    check_bundle(bundle, sha)
    return {"cfg": icfg, "rule": path, "rule_sha256": sha, "bundle": bundle, "bundle_sha256": bundle_sha}


def _cell_keys(cfg: dict) -> tuple[set, set]:
    grid_keys = {(r, lr_tag(x)) for r, x in grid(cfg)}
    return grid_keys, {(r, lr_tag(x)) for r, x in extension_candidates(cfg)} - grid_keys


def imported_cells(cfg: dict, out_root: Path) -> set:
    """The cells whose arm this rule imports (published under the imported rule, a grid or extension
    cell here)."""
    imp = load_imports(cfg, Path(out_root))
    grid_keys, ext_keys = _cell_keys(cfg)
    return set(published_rows(imp["cfg"], Path(out_root), imp["rule_sha256"])) & (grid_keys | ext_keys)


def collect_rows(cfg: dict, out_root: Path, rule_sha: str) -> tuple[dict, dict]:
    """``(rows, origins)``: this rule's published arms and the imported ones, each with its origin
    (``imported``, its rule's SHA-256 and config). An imported row must equal the imported bundle's arm
    (checkpoint SHA-256 and fingerprint); a cell both imported and published here is an error."""
    rows = published_rows(cfg, out_root, rule_sha)
    origins = {k: {"imported": False, "rule_sha256": rule_sha, "cfg": cfg} for k in rows}
    if not cfg.get("imports"):
        return rows, origins
    imp = load_imports(cfg, out_root)
    grid_keys, ext_keys = _cell_keys(cfg)
    for key, row in published_rows(imp["cfg"], out_root, imp["rule_sha256"]).items():
        if key not in grid_keys | ext_keys:
            continue
        if key in rows:
            raise ValueError(f"cell {key} is both imported and published under this rule")
        recorded = imp["bundle"]["arms"].get(arm_key(*key))
        if recorded is None or (recorded["checkpoint_sha256"], recorded["state_fingerprint"]) != \
                (row["checkpoint_sha256"], row["lora"]["state_fingerprint"]):
            raise ValueError(f"imported arm {key} differs from the imported bundle's")
        rows[key] = row
        origins[key] = {"imported": True, "rule_sha256": imp["rule_sha256"], "cfg": imp["cfg"]}
    return rows, origins


# ---- the rule (pure) ------------------------------------------------------------------------------


def _summary(row: dict, grid_keys: set | None = None) -> dict:
    """The rule's view of a row; ``grid_keys`` (the importing grid) decides whether it is an extension."""
    c = row["cells"]
    key = (int(row["lora"]["rank"]), lr_tag(row["lora"]["lr"]))
    extension = bool(row["lora"]["extension"]) if grid_keys is None else key not in grid_keys
    return {"rank": int(row["lora"]["rank"]), "lr": float(row["lora"]["lr"]),
            "extension": extension, "run_id": row["run_id"],
            "spatial": c["0"]["success"], "object": c["1"]["success"],
            "spatial_probe": c["0"]["probe_loss"], "object_probe": c["1"]["probe_loss"]}


def _apply(arms: list[dict], floor: float) -> tuple[dict, list[str], list[dict], bool]:
    eligible = [a for a in arms if a["object"] >= floor - TOL]
    if eligible:
        chosen = min(eligible, key=lambda a: (-a["spatial"], a["spatial_probe"], a["rank"], a["lr"]))
        flags = ["all_eligible_spatial_zero"] if chosen["spatial"] <= TOL else []
        return chosen, flags, eligible, False
    chosen = min(arms, key=lambda a: (-a["object"], a["object_probe"], a["rank"], a["lr"]))
    return chosen, ["no_qualifying_setting"], [], True


def owed_extensions(chosen: dict, ranks: list, lrs: list, ext: dict) -> list[tuple[int, float]]:
    """The extension cells a grid selection at ``chosen`` owes: one beyond each axis end it lies on (a rule
    with ``extension.rank: null``, a single-rank grid, owes none on the rank axis)."""
    owed = []
    if not ext.get("rank"):
        pass
    elif chosen["rank"] == min(ranks):
        owed.append((int(ext["rank"]["low"]), chosen["lr"]))
    elif chosen["rank"] == max(ranks):
        owed.append((int(ext["rank"]["high"]), chosen["lr"]))
    tags = [lr_tag(x) for x in lrs]
    if lr_tag(chosen["lr"]) == lr_tag(min(lrs)):
        owed.append((chosen["rank"], float(ext["lr"]["low"])))
    elif lr_tag(chosen["lr"]) == lr_tag(max(lrs)):
        owed.append((chosen["rank"], float(ext["lr"]["high"])))
    assert lr_tag(chosen["lr"]) in tags
    return owed


def select_lora(arms: list[dict], floor: float, ranks: list, lrs: list, ext: dict,
                available: set | frozenset = frozenset()) -> dict:
    """The rule over ``[{"rank", "lr", "extension", "spatial", "object", "spatial_probe", "object_probe"}]``.

    The grid alone decides which extensions are owed (none in the no-qualifying fallback). While any is
    missing, ``status`` is ``"needs_extension"`` (nothing is published); otherwise the rule is
    re-applied once over grid and the owed extensions and ``status`` is ``"selected"``. Extension arms
    whose cell is in ``available`` (imported) and not owed are set aside (``available_not_owed``); any
    other unowed extension arm is an error."""
    grid_arms = [a for a in arms if not a["extension"]]
    extra = {(a["rank"], lr_tag(a["lr"])): a for a in arms if a["extension"]}
    want = {(int(r), lr_tag(x)) for r in ranks for x in lrs}
    have = {(a["rank"], lr_tag(a["lr"])) for a in grid_arms}
    if have != want or len(grid_arms) != len(want):
        raise ValueError(f"grid arms {sorted(have)} are not the rule's grid {sorted(want)}")
    first, first_flags, _, fallback = _apply(grid_arms, floor)
    owed = [] if fallback else owed_extensions(first, ranks, lrs, ext)
    owed_keys = [(r, lr_tag(x)) for r, x in owed]
    unowed = set(extra) - set(owed_keys)
    unexpected = sorted(unowed - set(available))
    if unexpected:
        raise ValueError(f"extension arms {unexpected} are published but not owed by the grid's selection")
    set_aside = sorted((k[0], float(k[1])) for k in unowed)
    missing = [o for o, k in zip(owed, owed_keys) if k not in extra]
    first_key = (first["rank"], first["lr"])
    if missing:
        return {"status": "needs_extension", "selected": None, "owed_extensions": owed, "missing": missing,
                "grid_selection": first_key, "flags": first_flags + ["edge_extension_owed"],
                "available_not_owed": set_aside}
    if owed:
        chosen, flags, eligible, _ = _apply(grid_arms + [extra[k] for k in owed_keys], floor)
        flags = flags + ["edge_extension"] + (["optimum_at_boundary"] if chosen["extension"] else [])
    else:
        chosen, flags, eligible, _ = _apply(grid_arms, floor)
    return {"status": "selected", "selected": chosen, "flags": flags, "owed_extensions": owed, "missing": [],
            "grid_selection": first_key, "available_not_owed": set_aside,
            "eligible": sorted((a["rank"], a["lr"]) for a in eligible)}


def _rule_args(cfg: dict) -> dict:
    return {"ranks": [int(r) for r in cfg["lora"]["ranks"]], "lrs": [float(x) for x in cfg["lora"]["lrs"]],
            "ext": cfg["lora"]["extension"]}


# ---- validation of rows ---------------------------------------------------------------------------


def validate_rows(cfg: dict, rows: dict, ctx, out_root: Path, rule_sha: str, *, allow_dirty: bool = False,
                  fingerprints: bool = False, origins: dict | None = None) -> list[str]:
    """Problems with the rows: shared inputs not identical or not the current files' (each row bound to
    its own rule's SHA-256, ``origins``), arm-specific inputs not what the rule implies, dirty
    provenance, or (``fingerprints``) a checkpoint whose tensors no longer match the recorded
    fingerprint."""
    problems = []
    steps = int(cfg["lora"]["recipe"]["steps"])
    start = ctx.ref_dir / "checkpoints" / "stage0.pt"
    current = {}
    for (rank, tag), row in sorted(rows.items()):
        origin = (origins or {}).get((rank, tag)) or {"rule_sha256": rule_sha, "cfg": cfg}
        if origin["rule_sha256"] not in current:
            current[origin["rule_sha256"]] = shared_inputs(cfg, ctx, start, origin["rule_sha256"])
        want_inputs = current[origin["rule_sha256"]]
        name = row["run_id"]
        rec = row["lora"]
        if rec["inputs"] != want_inputs:
            diff = sorted(k for k in set(want_inputs) | set(rec["inputs"])
                          if want_inputs.get(k) != rec["inputs"].get(k))
            problems.append(f"{name}: shared inputs differ from the current files in {diff}")
        if int(rec["rank"]) != rank or lr_tag(rec["lr"]) != tag or float(rec["alpha"]) != float(cfg["lora"]["alpha"]):
            problems.append(f"{name}: rank/lr/alpha {rec['rank']}, {rec['lr']}, {rec['alpha']} do not match its cell")
        want = expected_train_config(ctx, float(rec["lr"]), steps, row_device(rec))
        if rec["train_config"] != want or rec["train_config_sha256"] != _json_sha(want):
            problems.append(f"{name}: recorded TrainConfig is not the rule's for its lr")
        directory = arm_dir(origin["cfg"], out_root, rank, float(rec["lr"]), int(cfg["bracket_seed"]))
        if _arm_train_config(directory) != want:
            problems.append(f"{name}: config.yaml's TrainConfig is not the rule's for its lr")
        if str(row.get("git_sha", "")).endswith("-dirty") and not allow_dirty:
            problems.append(f"{name}: trained on a dirty tree ({row.get('git_sha')})")
        if fingerprints:
            got = checkpoint_fingerprint(directory / row["checkpoint"])
            if got != rec["state_fingerprint"]:
                problems.append(f"{name}: checkpoint tensors no longer match the recorded fingerprint")
    return problems


def row_device(rec: dict) -> str:
    return rec["train_config"]["device"]


def _floor(summaries: list[dict], rows: dict, tolerance: float) -> tuple[float, float]:
    refs = {r["cells"]["1"]["reference_success"].get("seq_ft") for r in rows.values()}
    if len(refs) != 1 or None in refs:
        raise ValueError(f"the arms disagree on seq_ft's stage-1 Object: {refs}")
    ref = refs.pop()
    return ref, ref - tolerance


def _context(cfg: dict, root: Path, low_update_cfg: dict | None):
    from flowcl.experiments.low_update import seed_context

    return seed_context(low_update_cfg or load_low_update_config(), int(cfg["bracket_seed"]), root)


def _evaluate_rule(cfg: dict, rows: dict, origins: dict | None = None) -> tuple[dict, float, float]:
    grid_keys, _ = _cell_keys(cfg)
    missing = sorted(grid_keys - set(rows))
    if missing:
        raise FileNotFoundError(f"bracket arms not published: {missing}")
    summaries = [_summary(r, grid_keys) for r in rows.values()]
    available = {k for k, o in (origins or {}).items() if o["imported"] and k not in grid_keys}
    seq_ft_object, floor = _floor(summaries, rows, float(cfg["lora"]["object_tolerance"]))
    return select_lora(summaries, floor, **_rule_args(cfg), available=available), seq_ft_object, floor


def _validated(cfg: dict, root: Path, out_root: Path, rule_sha: str, low_update_cfg, allow_dirty: bool,
               fingerprints: bool = False) -> tuple[dict, dict, object, list[str]]:
    ctx = _context(cfg, root, low_update_cfg)
    rows, origins = collect_rows(cfg, out_root, rule_sha)
    problems = validate_rows(cfg, rows, ctx, out_root, rule_sha, allow_dirty=allow_dirty,
                             fingerprints=fingerprints, origins=origins)
    return rows, origins, ctx, problems


def _imports_record(cfg: dict, out_root: Path) -> dict | None:
    if not cfg.get("imports"):
        return None
    imp = load_imports(cfg, out_root)
    return {"rule": str(cfg["imports"]["rule"]), "rule_sha256": imp["rule_sha256"],
            "bundle_sha256": imp["bundle_sha256"]}


def run_select(cfg: dict, *, results_root=None, out_root=None, rule: Path | None = None,
               low_update_cfg: dict | None = None, allow_dirty: bool = False) -> dict:
    """Apply the rule. Returns ``{"status": "needs_extension"}`` (nothing published) or the bundle."""
    from flowcl.experiments.sgp_sensitivity import publish_selection

    root = _root(results_root)
    out_root = Path(out_root) if out_root else root
    rule_sha = file_sha256(rule_path(rule))
    rows, origins, ctx, problems = _validated(cfg, root, out_root, rule_sha, low_update_cfg, allow_dirty)
    if problems:
        raise ValueError(f"published arms fail validation: {problems}")
    choice, seq_ft_object, floor = _evaluate_rule(cfg, rows, origins)
    if choice["status"] == "needs_extension":
        print(f"[flowcl] LoRA selection: the grid selects {choice['grid_selection']}; edge extension(s) "
              f"{choice['missing']} run before anything is published", flush=True)
        return {"status": "needs_extension", "missing": choice["missing"], "owed": choice["owed_extensions"]}
    chosen = choice["selected"]
    seed = int(cfg["bracket_seed"])
    key = (chosen["rank"], lr_tag(chosen["lr"]))
    current = shared_inputs(cfg, ctx, ctx.ref_dir / "checkpoints" / "stage0.pt", rule_sha)
    selection = {
        "git_sha": git_sha(), "seed": seed, "family": family(cfg), "rule_sha256": rule_sha,
        "imports": _imports_record(cfg, out_root),
        "inputs": current, "seq_ft_object": seq_ft_object, "object_floor": floor,
        "arms": {arm_key(*k_): _bundle_arm(rows[k_], origins[k_], _cell_keys(cfg)[0]) for k_ in sorted(rows)},
        "grid_selection": list(choice["grid_selection"]),
        "owed_extensions": [list(o) for o in choice["owed_extensions"]],
        "available_not_owed": [list(o) for o in choice["available_not_owed"]],
        "eligible": [list(e) for e in choice["eligible"]],
        "selected": {"rank": chosen["rank"], "lr": chosen["lr"], "alpha": float(cfg["lora"]["alpha"]),
                     "run_id": chosen["run_id"], "imported": origins[key]["imported"],
                     "checkpoint": str(arm_dir(origins[key]["cfg"], out_root, chosen["rank"], chosen["lr"], seed)
                                       / "checkpoints" / "final.pt"),
                     "checkpoint_sha256": rows[key]["checkpoint_sha256"],
                     "state_fingerprint": rows[key]["lora"]["state_fingerprint"]},
        "flags": choice["flags"],
    }
    published, sha = publish_selection(_out(cfg, out_root) / "lora", selection)
    check_bundle(published, rule_sha, family(cfg))
    print(f"[flowcl] LoRA selection: rank {published['selected']['rank']}, lr {published['selected']['lr']:g} "
          f"{published['flags']} (bundle {sha[:12]})", flush=True)
    return {"status": "selected", **published, "bundle_sha256": sha}


def _bundle_arm(row: dict, origin: dict | None = None, grid_keys: set | None = None) -> dict:
    s = _summary(row, grid_keys)
    origin = origin or {"imported": False, "rule_sha256": None}
    return {**s, "checkpoint_sha256": row["checkpoint_sha256"], "state_fingerprint": row["lora"]["state_fingerprint"],
            "trainable_params": row["lora"]["history"]["trainable_params"],
            "git_sha": row.get("git_sha"), "imported": origin["imported"],
            "rule_sha256": origin["rule_sha256"],
            "paired_vs_seq_ft": {j: row["cells"][j].get("paired", {}).get("seq_ft") for j in ("0", "1")}}


def check_bundle(bundle: dict, rule_sha: str, family_name: str = FAMILY) -> None:
    if bundle.get("rule_sha256") != rule_sha or bundle.get("family") != family_name:
        raise ValueError(f"the selection bundle was made under another rule (bundle {bundle.get('rule_sha256')}, "
                         f"now {rule_sha})")


def run_extension(cfg: dict, *, results_root=None, out_root=None, rule: Path | None = None,
                  low_update_cfg: dict | None = None, allow_dirty: bool = False, **kw) -> dict:
    """The next owed edge-extension arm (one per call); refuses unless the grid's rule owes one."""
    root = _root(results_root)
    out_root_p = Path(out_root) if out_root else root
    rule_sha = file_sha256(rule_path(rule))
    rows, origins, _, problems = _validated(cfg, root, out_root_p, rule_sha, low_update_cfg, allow_dirty)
    if problems:
        raise ValueError(f"published arms fail validation: {problems}")
    choice, _, _ = _evaluate_rule(cfg, rows, origins)
    if choice["status"] != "needs_extension":
        raise RuntimeError("no edge extension is owed: the grid's selection lies inside the grid, none "
                           "qualifies, or every owed extension is published")
    rank, lr = choice["missing"][0]
    return run_arm(cfg, rank, lr, extension=True, results_root=results_root, out_root=out_root, rule=rule,
                   low_update_cfg=low_update_cfg, allow_dirty=allow_dirty, **kw)


def verify(cfg: dict, *, results_root=None, out_root=None, rule: Path | None = None,
           low_update_cfg: dict | None = None, allow_dirty: bool = False) -> dict:
    """Every required arm, its inputs and its checkpoint; the bundle; the selection recomputed; and, for
    a rule that imports arms, the imported rule's own verification."""
    from flowcl.experiments.sgp_sensitivity import load_selection

    root = _root(results_root)
    out_root = Path(out_root) if out_root else root
    problems = []
    sha = None
    try:
        rule_sha = file_sha256(rule_path(rule))
        bundle, sha = load_selection(_out(cfg, out_root) / "lora")
        check_bundle(bundle, rule_sha, family(cfg))
        if cfg.get("imports"):
            imp = load_imports(cfg, out_root)
            inner = verify(imp["cfg"], results_root=results_root, out_root=out_root, rule=imp["rule"],
                           low_update_cfg=low_update_cfg, allow_dirty=allow_dirty)
            problems += [f"imported rule: {p}" for p in inner["problems"]]
            if bundle.get("imports") != _imports_record(cfg, out_root):
                problems.append("the bundle's imports differ from this rule's")
        rows, origins, ctx, row_problems = _validated(cfg, root, out_root, rule_sha, low_update_cfg, allow_dirty,
                                                      fingerprints=True)
        problems += row_problems
        choice, seq_ft_object, floor = _evaluate_rule(cfg, rows, origins)
        if choice["status"] != "selected":
            problems.append(f"the rule still owes extensions {choice['missing']}")
        else:
            got = {"selected": [choice["selected"]["rank"], choice["selected"]["lr"]], "flags": choice["flags"],
                   "owed_extensions": [list(o) for o in choice["owed_extensions"]],
                   "available_not_owed": [list(o) for o in choice["available_not_owed"]],
                   "eligible": [list(e) for e in choice["eligible"]],
                   "grid_selection": list(choice["grid_selection"]),
                   "object_floor": floor, "seq_ft_object": seq_ft_object}
            want = {"selected": [bundle["selected"]["rank"], bundle["selected"]["lr"]], "flags": bundle["flags"],
                    "owed_extensions": bundle["owed_extensions"],
                    "available_not_owed": bundle.get("available_not_owed", []), "eligible": bundle["eligible"],
                    "grid_selection": bundle["grid_selection"],
                    "object_floor": bundle["object_floor"], "seq_ft_object": bundle["seq_ft_object"]}
            for k in got:
                if got[k] != want[k]:
                    problems.append(f"recomputed {k} {got[k]} != the bundle's {want[k]}")
        published = {arm_key(*k): (r["checkpoint_sha256"], r["lora"]["state_fingerprint"]) for k, r in rows.items()}
        recorded = {k: (v["checkpoint_sha256"], v["state_fingerprint"]) for k, v in bundle["arms"].items()}
        if published != recorded:
            problems.append("the bundle's arms differ from the published rows")
        current = shared_inputs(cfg, ctx, ctx.ref_dir / "checkpoints" / "stage0.pt", rule_sha)
        if bundle.get("inputs") != current:
            problems.append("the bundle's inputs differ from the current files")
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        problems.append(f"{type(exc).__name__}: {exc}")
    return {"passed": not problems, "problems": problems, "bundle_sha256": sha}


def status(cfg: dict, *, results_root=None, out_root=None, rule: Path | None = None,
           low_update_cfg: dict | None = None, allow_dirty: bool = False) -> dict:
    """Imported, published and missing cells (no training), and the imported rule's verification."""
    root = _root(results_root)
    out_root = Path(out_root) if out_root else root
    rule_sha = file_sha256(rule_path(rule))
    rows, origins, _, problems = _validated(cfg, root, out_root, rule_sha, low_update_cfg, allow_dirty)
    grid_keys, ext_keys = _cell_keys(cfg)
    out = {"rule_sha256": rule_sha,
           "imported": sorted(arm_key(*k) for k, o in origins.items() if o["imported"]),
           "published": sorted(arm_key(*k) for k, o in origins.items() if not o["imported"]),
           "missing_grid": sorted(arm_key(r, float(t)) for r, t in grid_keys - set(rows)),
           "available_extensions": sorted(arm_key(*k) for k in set(rows) & ext_keys),
           "row_problems": problems}
    if cfg.get("imports"):
        imp = load_imports(cfg, out_root)
        out["imports"] = _imports_record(cfg, out_root)
        out["imported_verify"] = verify(imp["cfg"], results_root=results_root, out_root=out_root, rule=imp["rule"],
                                        low_update_cfg=low_update_cfg, allow_dirty=allow_dirty)
    out["ready"] = not problems and (not cfg.get("imports") or out["imported_verify"]["passed"])
    return out


# ---- projection: the start check, K_old and the layer-output-change diagnostic ----------------------


def check_start_is_memory_s(cfg: dict, root: Path, start: Path) -> None:
    """The arms' start checkpoint must equal the stage 0 of the GPM run whose Task-1 memory they use."""
    from flowcl.train.continual import stage_identity_check

    gpm_stage0 = Path(root) / cfg["projection"]["memory"]["run"] / "checkpoints" / "stage0.pt"
    check = stage_identity_check(Path(start), None, 0, reference_checkpoint=gpm_stage0)
    if not check["passed"]:
        raise RuntimeError(f"{start} differs from {gpm_stage0} ({check['n_different']} tensors): the memory "
                           "would be mispaired")


def load_kold(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload["grams"]


def capture_kold(cfg: dict, *, results_root=None, device: str = "cuda", dataset_dir=None,
                 low_update_cfg: dict | None = None, allow_dirty: bool = False, build_datasets=None) -> dict:
    """The one-time full Task-1 input Gram per registry layer at seq_ft seed 0's stage 0 on Task-1 data
    (GPM's capture; a diagnostic, kept apart from adaptation). Immutable once written."""
    from flowcl.experiments.gate2 import capture_task_grams, load_subspace_config
    from flowcl.experiments.low_update import seed_context
    from flowcl.train.checkpoint import load_checkpoint
    from flowcl.train.pipeline import build_dataset
    from flowcl.utils.seeding import derive_seed

    root = _root(results_root)
    path = kold_path(cfg, root)
    if path.exists():
        return {"path": str(path), "sha256": file_sha256(path), "reused": True}
    refuse_dirty(allow_dirty)
    ctx = seed_context(low_update_cfg or load_low_update_config(), int(cfg["bracket_seed"]), root)
    start = ctx.ref_dir / "checkpoints" / "stage0.pt"
    loaded = load_checkpoint(start, device=device)
    stage = ctx.curriculum.stages[0]
    dataset = (build_datasets(stage) if build_datasets is not None else
               build_dataset([stage.ref], loaded.spec, loaded.stats, n_demos=stage.n_demos, dataset_dir=dataset_dir))
    name = cfg["projection"]["kold"]["capture_config"]
    sub = load_subspace_config(name if str(name).endswith(".yaml") else
                               repo_root() / "configs" / "analysis" / f"{name}.yaml")
    t0 = time.perf_counter()
    capture = capture_task_grams(loaded.policy, dataset, sub, device,
                                 probe_seed=derive_seed(ctx.namespace, f"kold_probe::{stage.task_key}", 0),
                                 capture_seed=derive_seed(ctx.namespace, f"kold::{stage.task_key}", 0))
    grams, n = {}, {}
    for entry in loaded.policy.projectable_layers():
        acc = capture.accumulators[entry.name]
        view = capture.primary_view(entry.name)
        grams[entry.name] = acc.gram[view].to(torch.float64).cpu().clone()
        n[entry.name] = int(acc.n[view])
    meta = {"start_checkpoint": _file(start), "task_key": stage.task_key, "namespace": ctx.namespace,
            "capture_config": cfg["projection"]["kold"]["capture_config"], "n_samples": n,
            "seconds": time.perf_counter() - t0, "git_sha": git_sha()}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"grams": grams, "meta": meta}, tmp)
    os.replace(tmp, path)
    return {"path": str(path), "sha256": file_sha256(path), "reused": False, "seconds": meta["seconds"]}


def registry_output_change(policy, start_state: dict, kold: dict, memory: dict | None = None) -> dict:
    """tr(D K_old D^T) / tr(W0 K_old W0^T) per registry layer for D = W_final - W_start (the general form,
    valid for any update); with a memory, also the P⊥-sandwiched form as a check."""
    from flowcl.analysis.projection_diagnostics import output_change

    per_layer, num, den = {}, 0.0, 0.0
    for entry in policy.projectable_layers():
        name = f"{entry.name}.weight"
        W0 = start_state[name].double().cpu()
        D = entry.module.weight.detach().double().cpu() - W0
        P = None
        if memory is not None and entry.name in memory:
            M = memory[entry.name].double()
            P = torch.eye(M.shape[0], dtype=torch.float64) - M @ M.T
        out = output_change(D, W0, kold[entry.name], P)
        per_layer[entry.name] = out
        num += out["energy"]
        den += out["base_energy"]
    rel = [v["relative"] for v in per_layer.values()]
    return {"per_layer": per_layer, "total_relative": num / den if den else None,
            "median_relative": sorted(rel)[len(rel) // 2], "max_relative": max(rel)}


def reference_output_change(start: Path, end: Path, kold_file: Path) -> dict:
    """The same diagnostic for any checkpoint pair (LoRA's or gpm_projected_adam's stage 1)."""
    from flowcl.train.checkpoint import load_checkpoint

    kold = load_kold(kold_file)
    a = torch.load(start, map_location="cpu", weights_only=False)["state_dict"]
    loaded = load_checkpoint(end, device="cpu")
    return {"start": _file(start), "end": _file(end), "kold": _file(kold_file),
            **registry_output_change(loaded.policy, a, kold)}


def compare_memories(a: Path, b: Path) -> dict:
    """Two GPM memory artifacts' bases, tensor for tensor."""
    from flowcl.analysis.subspace import load_bases

    ba, _ = load_bases(a)
    bb, _ = load_bases(b)
    different = sorted(n for n in set(ba) | set(bb) if n not in ba or n not in bb
                       or not torch.equal(ba[n].vectors, bb[n].vectors))
    return {"passed": not different, "layers": len(ba), "different": different}


# ---- smoke and benchmark --------------------------------------------------------------------------


def smoke_arm(cfg: dict, rank: int, lr: float, *, reference_run: Path, identity_run: Path, steps: int = 200,
              episodes: int = 1, out_root: Path, **kw) -> dict:
    """The bracket path against the runner, before the queue: an arm from ``reference_run``'s (a scratch
    seq_ft run's) ``stage0.pt`` at ``steps`` steps must train under the same TrainConfig as
    ``identity_run``'s (a scratch LoRA run's) stage 1 and reproduce its ``stage1.pt`` tensor for tensor."""
    from flowcl.train.continual import stage_identity_check

    reference_run, identity_run = Path(reference_run), Path(identity_run)
    if cfg.get("projection"):   # the runner's own Task-1 memory (equal to a smoke GPM run's, checked apart)
        kw["smoke_memory"] = identity_run / "method" / "memory_task0.pt"
    row = run_arm(cfg, rank, lr, out_root=out_root, smoke_start=reference_run / "checkpoints" / "stage0.pt",
                  smoke_steps=steps, smoke_episodes=episodes, **kw)
    seed = int(cfg["bracket_seed"])
    directory = arm_dir(cfg, out_root, rank, lr, seed)
    runner_stage = json.loads((identity_run / "result.json").read_text())["stages"][1]
    arm_config, runner_config = _arm_train_config(directory), runner_stage.get("train_config")
    config_equal = arm_config == runner_config
    if not config_equal:
        return {"passed": False, "reason": "the arm's TrainConfig differs from the runner's stage-1 TrainConfig",
                "arm": arm_config, "runner": runner_config}
    check = stage_identity_check(directory / "checkpoints" / "final.pt", None, 1,
                                 reference_checkpoint=identity_run / "checkpoints" / "stage1.pt")
    history_equal = row["lora"]["history"] == _runner_history(identity_run)
    return {"passed": bool(check["passed"] and history_equal), "train_config_equal": True,
            "identity": {k: check[k] for k in ("n_tensors", "n_different", "different", "missing", "extra")},
            "history_equal": history_equal, "run_id": row["run_id"]}


def _runner_history(run: Path) -> dict:
    payload = torch.load(run / "checkpoints" / "stage1.pt", map_location="cpu", weights_only=False)
    return payload["extra"]["method_state"]["history"]["1"]


def benchmark(cfg: dict, ranks: list[int], *, steps: int = 300, warmup: int = 50, lr: float = 1e-3,
              results_root=None, out_root: Path, device: str = "cuda", dataset_dir=None,
              low_update_cfg: dict | None = None, overhead_min: float = 20.0) -> dict:
    """Seconds per training step of T2 at each rank (and full fine-tuning, ``rank`` 0), from the
    production ``stage0.pt`` with the production recipe's first ``steps`` steps; and the projected arm
    time (30k steps plus ``overhead_min`` for setup, rollouts and probes). Nothing is published."""
    from flowcl.experiments.low_update import stage_setup
    from flowcl.methods.lora import LoRA
    from flowcl.train.trainer import train_one_task

    root = _root(results_root)
    ctx = _context(cfg, root, low_update_cfg)
    start = ctx.ref_dir / "checkpoints" / "stage0.pt"
    total = int(cfg["lora"]["recipe"]["steps"])
    out = {"steps": steps, "warmup": warmup, "overhead_min": overhead_min, "ranks": {}}
    for rank in ranks:
        run_id = f"bench__r{rank}"
        st = stage_setup(ctx, 1, total, lr, start, Path(out_root) / run_id, run_id, {"run_id": run_id},
                         tasks=[1], train_overrides={"stop_after": steps}, device=device,
                         dataset_dir=dataset_dir, build_datasets=None)
        alpha = float(cfg["lora"]["alpha"])
        if rank <= 0:
            method = None
        elif cfg.get("projection"):
            method = projection_method(cfg["projection"], rank, lr, alpha)
            mem = memory_path(cfg, root)
            method.install_memory(mem, cfg["projection"]["memory"]["sha256"])
        else:
            method = LoRA(rank=rank, lr=lr, alpha=alpha)
        marks = {}

        def on_step(step, outputs, marks=marks):
            if step + 1 in (warmup, steps):
                marks[step + 1] = time.perf_counter()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        train_one_task(st.start.policy, st.datasets[1], st.train_cfg, method=method, task_idx=1,
                       generator=st.generator, context=st.context, on_step=on_step)
        per_step = (marks[steps] - marks[warmup]) / (steps - warmup)
        out["ranks"][str(rank)] = {"s_per_step": per_step, "train_min_30k": per_step * total / 60,
                                   "arm_min_estimate": per_step * total / 60 + overhead_min,
                                   "peak_gpu_bytes": (int(torch.cuda.max_memory_allocated())
                                                      if torch.cuda.is_available() else None)}
        if method is not None and hasattr(method, "timings"):
            out["ranks"][str(rank)]["cost"] = {**{k: v for k, v in method.history.get("1", {}).items()
                                                  if k not in ("per_layer", "merge")},
                                               **method.timings.get("1", {})}
        print(f"[flowcl] benchmark rank {rank}: {per_step * 1000:.1f} ms/step, "
              f"{per_step * total / 60:.1f} min per 30k steps", flush=True)
        del st, method
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out


def write_smoke_method(path: Path, rank: int, lr: float, alpha: float = 16.0, projection: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    node = {"name": "lora", "rank": int(rank), "lr": float(lr), "alpha": float(alpha)}
    if projection:   # the bracket's variant, with GPM's memory rule from Task 1 on (as in a 4-task run)
        node = {"name": "projected_lora", "variant": projection["variant"], **{k: v for k, v in node.items() if k != "name"},
                "eps": float(projection["eps"]), "update_memory": True,
                "capture_config": projection.get("capture_config", "subspace")}
    OmegaConf.save(OmegaConf.create(node), path)
    return path
