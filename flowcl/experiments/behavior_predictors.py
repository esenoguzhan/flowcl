"""Track 2: offline predictors of closed-loop forgetting (docs/thesis_plan.md, Workstream D).

Step 1, the manifest: every development cell (a checkpoint, a trained task, a 50-rollout evaluation)
under ``results/``, with its lineage, deduplicated by tensor content. The rule is the ``manifest``
section of ``configs/analysis/behavior_predictors.yaml``; read it first. :func:`discover` recognises the
sources by layout and records a reason for every evaluation file it does not use;
:func:`build_manifest` fingerprints the checkpoints (cached by path and file SHA-256), resolves the
lineage, deduplicates, and either reports (``dry_run``) or publishes the manifest and its summary
together, atomically.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from omegaconf import OmegaConf

from flowcl.utils.libero_paths import repo_root
from flowcl.utils.run import file_sha256, git_sha

REPEATED_STATES = re.compile(r".*\.rep\d+\.eval\.json$")
DIR_REASONS = {"dobot": "embodiment"}   # any other excluded directory pattern: "invalid_dir"


def config_path(path: str | Path | None = None) -> Path:
    return Path(path) if path else repo_root() / "configs" / "analysis" / "behavior_predictors.yaml"


def load_config(path: str | Path | None = None) -> dict:
    return OmegaConf.to_container(OmegaConf.load(config_path(path)), resolve=True)


def _yaml(path: Path) -> dict:
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


@dataclass
class Source:
    """One evaluation with its checkpoint: the unit the cells of a manifest come from."""

    kind: str
    run: str                        # the run directory, relative to the results root
    evaluation: Path
    checkpoint: Path
    stage: int | None               # a runner/pilot/row stage, or a trajectory's step
    label: str
    curriculum: str | None
    own_tasks: set
    parent: Path | None             # the parent checkpoint (None: a root)
    trajectory: str | None = None   # a trajectory's id (its root)
    files: list = field(default_factory=list)
    bracket: bool = False
    embodiment: str | None = None
    trained: set = field(default_factory=set)


# ---- discovery --------------------------------------------------------------------------------------


def evaluation_files(root: Path, cfg: dict) -> list[Path]:
    """Every evaluation-like file under ``root``: ``eval.json``, ``eval/stage*.json``, ``eval/step*.json``
    and ``*.eval.json``; hidden directories and the skipped top-level ones are never entered."""
    skip = set(cfg["manifest"]["skip_dirs"])
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel = Path(dirpath).relative_to(root)
        dirnames[:] = sorted(d for d in dirnames
                             if not d.startswith(".") and not (rel == Path(".") and d in skip))
        for name in sorted(filenames):
            path = Path(dirpath) / name
            in_eval = Path(dirpath).name == "eval" and name.endswith(".json") and \
                (name.startswith("stage") or name.startswith("step"))
            if name == "eval.json" or name.endswith(".eval.json") or in_eval:
                found.append(path)
    return sorted(found)


def _resolve(path: str | Path, root: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else root.parent / path


def _embodiment(cfg: dict) -> str | None:
    emb = cfg.get("embodiment")
    return emb.get("name") if isinstance(emb, dict) else emb


def _curriculum_tasks(curriculum_cfg) -> tuple:
    from flowcl.data.curriculum import load_curriculum

    curriculum = load_curriculum(curriculum_cfg)
    return curriculum, [{ref.task_key for ref in stage.train_refs} for stage in curriculum.stages]


def classify(path: Path, root: Path, cfg: dict) -> tuple[Source | None, str | None]:
    """A source for evaluation file ``path``, or the reason it is not used."""
    m = cfg["manifest"]
    rel = path.relative_to(root)
    for part in rel.parts[:-1]:
        for pattern in m["exclude_dir_patterns"]:
            if pattern in part.lower():
                return None, DIR_REASONS.get(pattern, "invalid_dir")
    if REPEATED_STATES.match(path.name):
        return None, "repeated_states"
    bracket = rel.parts[0] in set(m["bracket_dirs"])
    if path.parent.name == "eval" and path.name.startswith("stage"):
        return _runner_or_pilot(path, root, bracket)
    if path.parent.name == "eval" and path.name.startswith("step"):
        return _trajectory(path, root)
    if path.name == "eval.json":
        d = path.parent
        if (d / "row.json").is_file():
            return _row(path, root, bracket)
        if d.name.startswith("single__") and (d / "config.yaml").is_file():
            return _single(path, root)
    return None, "unrecognized_layout"


def _runner_or_pilot(path: Path, root: Path, bracket: bool):
    run_dir = path.parent.parent
    k = int(re.fullmatch(r"stage(\d+)\.json", path.name).group(1))
    config_file = run_dir / "config.yaml"
    if not config_file.is_file():
        return None, "unrecognized_layout"
    cfg = _yaml(config_file)
    ckpt = run_dir / "checkpoints" / f"stage{k}.pt"
    resume_file = run_dir / "resume.json"
    resume = json.loads(resume_file.read_text()) if resume_file.is_file() else None
    files = [config_file, path]
    if not ckpt.is_file():
        if resume is not None and k in resume.get("imported_stages", []):
            return None, "resume_import_copy"
        return None, "no_checkpoint"
    files.append(ckpt)
    method = cfg.get("method") or {}
    label = method.get("display_name") or method.get("name") or "unknown"
    if cfg.get("curriculum") is not None:
        curriculum, stage_tasks = _curriculum_tasks(cfg["curriculum"])
        own = set().union(*stage_tasks[:k + 1])
        if k == 0:
            parent = None
        elif resume is not None and k == int(resume["start_stage"]):
            parent = _resolve(resume["checkpoint"], root)
            files.append(resume_file)
        else:
            parent = run_dir / "checkpoints" / f"stage{k - 1}.pt"
        return Source("runner", str(run_dir.relative_to(root)), path, ckpt, k, label, curriculum.name, own,
                      parent, files=files, bracket=bracket, embodiment=_embodiment(cfg)), None
    if cfg.get("start_checkpoint") and cfg.get("task_key"):
        return Source("pilot", str(run_dir.relative_to(root)), path, ckpt, k, f"pilot:{cfg.get('arm', label)}",
                      None, {cfg["task_key"]}, _resolve(cfg["start_checkpoint"], root), files=files,
                      bracket=bracket, embodiment=_embodiment(cfg)), None
    return None, "unrecognized_layout"


def _trajectory(path: Path, root: Path):
    run_dir = path.parent.parent
    config_file = run_dir / "config.yaml"
    if not config_file.is_file():
        return None, "unrecognized_layout"
    cfg = _yaml(config_file)
    if cfg.get("part") != "trajectory":
        return None, "unrecognized_layout"
    step = int(re.fullmatch(r"step(\d+)\.json", path.name).group(1))
    ckpt = run_dir / "checkpoints" / f"step{step}.pt"
    if not ckpt.is_file():
        return None, "no_checkpoint"
    steps = sorted(int(re.fullmatch(r"step(\d+)\.pt", p.name).group(1))
                   for p in (run_dir / "checkpoints").glob("step*.pt"))
    earlier = [s for s in steps if s < step]
    parent = run_dir / "checkpoints" / f"step{earlier[-1]}.pt" if earlier else None
    rel = str(run_dir.relative_to(root))
    return Source("trajectory", rel, path, ckpt, step, "trajectory", None, {cfg["task_key"]}, parent,
                  trajectory=rel, files=[config_file, path, ckpt], embodiment=_embodiment(cfg)), None


def _row(path: Path, root: Path, bracket: bool):
    d = path.parent
    row_file = d / "row.json"
    row = json.loads(row_file.read_text())
    files = [row_file, path]
    if row.get("parent"):
        if not row.get("checkpoint"):
            return None, "no_checkpoint"
        ckpt = d / row["checkpoint"]
        kind, parent, task = "row", _resolve(row["parent"]["path"], root), row["task_key"]
        label = (row.get("method") or {}).get("family") or Path(d.relative_to(root)).parts[0]
    else:
        config_file = d / "config.yaml"
        cfg = _yaml(config_file) if config_file.is_file() else {}
        if not (cfg.get("reference_run") and cfg.get("start_checkpoint") and cfg.get("task_key")):
            return None, "unrecognized_layout"
        ckpt = d / "checkpoints" / "final.pt"
        kind, task = "legacy_row", cfg["task_key"]
        parent = root / cfg["reference_run"] / "checkpoints" / cfg["start_checkpoint"]
        label = Path(d.relative_to(root)).parts[0]
        files.append(config_file)
    if not ckpt.is_file():
        return None, "no_checkpoint"
    files.append(ckpt)
    return Source(kind, str(d.relative_to(root)), path, ckpt, row.get("stage"), label, None, {task}, parent,
                  files=files, bracket=bracket), None


def _single(path: Path, root: Path):
    d = path.parent
    config_file = d / "config.yaml"
    cfg = _yaml(config_file)
    ckpt = d / "checkpoints" / "final.pt"
    if not ckpt.is_file():
        return None, "no_checkpoint"
    return Source("single", str(d.relative_to(root)), path, ckpt, None, "single_task", None, set(cfg["tasks"]),
                  None, files=[config_file, path, ckpt], embodiment=_embodiment(cfg)), None


def reserved_runs(root: Path, cfg: dict) -> list[str]:
    pattern = re.compile(cfg["manifest"]["reserved_test_runs"])
    return sorted(p.name for p in root.iterdir() if p.is_dir() and pattern.match(p.name))


def discover(root: Path, cfg: dict) -> tuple[list[Source], list[dict], list[Path]]:
    """Sources and exclusions for every evaluation file under ``root``."""
    sources, excluded = [], []
    files = evaluation_files(root, cfg)
    for path in files:
        source, reason = classify(path, root, cfg)
        if source is None:
            excluded.append({"evaluation": str(path.relative_to(root)), "reason": reason})
            continue
        if source.embodiment not in (None, cfg["manifest"]["embodiment"]):
            excluded.append({"evaluation": str(path.relative_to(root)), "reason": "embodiment"})
            continue
        sources.append(source)
    return sources, excluded, files


# ---- fingerprints and lineage -----------------------------------------------------------------------


class FingerprintCache:
    """Tensor-content fingerprints of checkpoints (``ewc.tensor_fingerprint`` over the state dict), keyed
    by path and reused only while the file's SHA-256 is unchanged."""

    def __init__(self, path: Path | None):
        self.path = path
        self.entries = json.loads(path.read_text()) if path is not None and path.is_file() else {}
        self.computed = 0

    def get(self, checkpoint: Path) -> tuple[str, str]:
        from flowcl.methods.ewc import tensor_fingerprint

        key = str(Path(checkpoint).resolve())
        sha = file_sha256(checkpoint)
        entry = self.entries.get(key)
        if entry is None or entry["sha256"] != sha:
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
            entry = {"sha256": sha, "fingerprint": tensor_fingerprint(state)}
            del state
            self.entries[key] = entry
            self.computed += 1
            self.save()
        return entry["sha256"], entry["fingerprint"]

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.entries, indent=1, sort_keys=True) + "\n")
        os.replace(tmp, self.path)


def resolve_lineage(sources: list[Source], cache: FingerprintCache, root: Path) -> tuple[dict, list[dict]]:
    """For every source checkpoint: its chain of ancestors (root first) as ``[{path, sha256, fingerprint}]``,
    and the trained tasks (its own plus its ancestors'). A parent that is no source's checkpoint ends the
    chain there and is reported."""
    by_ckpt = {s.checkpoint.resolve(): s for s in sources}
    problems, chains = [], {}

    def node(path: Path) -> dict:
        sha, fp = cache.get(path)
        return {"path": _rel(path, root), "sha256": sha, "fingerprint": fp}

    def chain(path: Path, seen=()) -> list[dict]:
        key = path.resolve()
        if key in chains:
            return chains[key]
        if key in seen:
            raise ValueError(f"lineage cycle at {path}")
        src = by_ckpt.get(key)
        if src is None:
            problems.append({"checkpoint": str(path), "problem": "parent_not_a_source"})
            out = [node(path)]
        elif src.parent is None:
            out = ([{"path": f"trajectory:{src.trajectory}", "sha256": None, "fingerprint": f"trajectory:{src.trajectory}"}]
                   if src.trajectory else []) + [node(path)]
        else:
            if not src.parent.is_file():
                raise FileNotFoundError(f"{src.evaluation}: parent checkpoint {src.parent} is missing")
            out = chain(src.parent, seen + (key,)) + [node(path)]
        chains[key] = out
        return out

    for s in sources:
        chain(s.checkpoint)

    def trained(s: Source, seen=()) -> set:
        out = set(s.own_tasks)
        parent = by_ckpt.get(s.parent.resolve()) if s.parent is not None else None
        if parent is not None and parent.checkpoint.resolve() not in seen:
            out |= trained(parent, seen + (s.checkpoint.resolve(),))
        return out

    for s in sources:
        s.trained = trained(s)
    return chains, problems


def _rel(path: Path, root: Path) -> str:
    resolved, base = Path(path).resolve(), Path(root).resolve()
    return str(resolved.relative_to(base)) if resolved.is_relative_to(base) else str(path)


def first_divergence(chain_fps: tuple, run_chains: set) -> str:
    """thesis_plan §D's literal branch: the first non-root node on exactly one run's chain, else the
    cell's own node."""
    if len(chain_fps) == 1:
        return chain_fps[0]
    for v in chain_fps[1:]:
        if sum(1 for rc in run_chains if v in rc) == 1:
            return v
    return chain_fps[-1]


# ---- the manifest -----------------------------------------------------------------------------------


def _seeds_key(seeds: list) -> str:
    return hashlib.sha256(json.dumps(list(seeds)).encode()).hexdigest()


def build_manifest(cfg: dict | None = None, *, results_root: Path | None = None, rule: Path | None = None,
                   dry_run: bool = False) -> dict:
    """Discover, fingerprint, resolve lineage and deduplicate; publish unless ``dry_run``."""
    cfg = cfg or load_config(rule)
    m = cfg["manifest"]
    root = Path(results_root) if results_root else repo_root() / "results"
    reserved = reserved_runs(root, cfg)
    if reserved:
        raise RuntimeError(f"reserved test runs exist before the freeze: {reserved}; the manifest refuses")
    out_dir = root / m["out"]
    if out_dir.exists() and not dry_run:
        return load_manifest(out_dir)
    sources, excluded, evaluations = discover(root, cfg)
    cache = FingerprintCache(root / m["cache"] / "fingerprints.json")
    chains, lineage_problems = resolve_lineage(sources, cache, root)
    order = {k: i for i, k in enumerate(m["precedence"])}
    runs: dict[str, tuple] = {}
    for s in sources:
        fps = tuple(n["fingerprint"] for n in chains[s.checkpoint.resolve()])
        if len(fps) > len(runs.get(s.run, ())):
            runs[s.run] = fps
    run_chains = set(runs.values())

    candidates = []
    for s in sorted(sources, key=lambda s: (order[s.kind], str(s.evaluation))):
        report = json.loads(s.evaluation.read_text())
        lineage = chains[s.checkpoint.resolve()]
        fps = tuple(n["fingerprint"] for n in lineage)
        files = {_rel(f, root): (lineage[-1]["sha256"] if f == s.checkpoint else file_sha256(f)) for f in s.files}
        for task in report["tasks"]:
            key = task["task_key"]
            reason = None
            if not key.startswith(m["task_prefix"]):
                reason = "embodiment"
            elif int(task["n_rollouts"]) != int(m["n_rollouts"]) or len(task["successes"]) != int(m["n_rollouts"]):
                reason = "n_rollouts"
            elif key not in s.trained:
                reason = "unseen_task"
            if reason:
                excluded.append({"evaluation": str(s.evaluation.relative_to(root)), "task_key": key, "reason": reason})
                continue
            alias = {"kind": s.kind, "run": s.run, "stage": s.stage, "label": s.label, "curriculum": s.curriculum,
                     "evaluation": str(s.evaluation.relative_to(root)),
                     "checkpoint": str(s.checkpoint.relative_to(root)), "bracket": s.bracket,
                     "files": files, "lineage": lineage}
            candidates.append({"fingerprint": fps[-1], "task_key": key, "seeds": list(task["seeds"]),
                               "successes": [bool(x) for x in task["successes"]], "chain": fps, "alias": alias})

    cells, by_key = [], {}
    disagreeing = []
    for c in candidates:   # already in precedence order
        key = (c["fingerprint"], c["task_key"], _seeds_key(c["seeds"]))
        if key in by_key:
            kept = by_key[key]
            kept["aliases"].append(c["alias"])
            kept["bracket"] = kept["bracket"] or c["alias"]["bracket"]
            if c["successes"] != kept["successes"]:
                kept["disagreeing_duplicate"] = True
                disagreeing.append({"cell": kept["id"], "alias": c["alias"]["evaluation"]})
            continue
        fps = c["chain"]
        cell = {"id": f"{c['fingerprint'][:16]}::{c['task_key']}::{key[2][:8]}",
                "fingerprint": c["fingerprint"], "task_key": c["task_key"], "suite": c["task_key"].split("/")[0],
                "n": len(c["successes"]), "successes": c["successes"], "seeds": c["seeds"],
                "success_rate": sum(c["successes"]) / len(c["successes"]),
                "root": fps[0], "root_kind": "trajectory" if fps[0].startswith("trajectory:") else "checkpoint",
                "depth": len(fps) - 1, "branch_depth1": fps[1] if len(fps) > 1 else fps[0],
                "branch_first_divergence": first_divergence(fps, run_chains),
                "bracket": c["alias"]["bracket"], "disagreeing_duplicate": False, "aliases": [c["alias"]]}
        by_key[key] = cell
        cells.append(cell)
    cells.sort(key=lambda c: c["id"])
    used = {s.evaluation for s in sources}
    reconciliation = {"evaluation_files": len(evaluations), "used": len(used),
                      "excluded_files": len({e["evaluation"] for e in excluded if "task_key" not in e})}
    if reconciliation["used"] + reconciliation["excluded_files"] != reconciliation["evaluation_files"]:
        raise RuntimeError(f"evaluation files do not reconcile: {reconciliation}")
    manifest = {
        "version": 1, "git_sha": git_sha(), "rule_sha256": file_sha256(config_path(rule)),
        "results_root": str(root.resolve()), "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "rule": cfg["manifest"], "cells": cells,
        "development_keys": sorted({(c["fingerprint"], c["task_key"]) for c in cells}),
        "excluded": excluded, "lineage_problems": lineage_problems,
        "duplicates": {"collapsed": len(candidates) - len(cells), "disagreeing": disagreeing},
        "reconciliation": reconciliation,
    }
    summary = summarize(manifest)
    if dry_run:
        return {"dry_run": True, "summary": summary, "fingerprints_computed": cache.computed}
    return publish_manifest(out_dir, manifest, summary)


def summarize(manifest: dict) -> dict:
    from collections import Counter

    cells = manifest["cells"]
    rates = sorted(c["success_rate"] for c in cells)
    return {
        "cells": len(cells),
        "by_kind": dict(Counter(c["aliases"][0]["kind"] for c in cells)),
        "by_suite": dict(Counter(c["suite"] for c in cells)),
        "bracket_cells": sum(c["bracket"] for c in cells),
        "roots": len({c["root"] for c in cells}),
        "roots_by_kind": dict(Counter(c["root_kind"] for c in {c["root"]: c for c in cells}.values())),
        "branches_depth1": len({(c["root"], c["branch_depth1"]) for c in cells}),
        "branches_first_divergence": len({(c["root"], c["branch_first_divergence"]) for c in cells}),
        "aliases": sum(len(c["aliases"]) for c in cells),
        "duplicates_collapsed": manifest["duplicates"]["collapsed"],
        "disagreeing_duplicates": len(manifest["duplicates"]["disagreeing"]),
        "excluded_by_reason": dict(Counter(e["reason"] for e in manifest["excluded"])),
        "lineage_problems": len(manifest["lineage_problems"]),
        "success": {"zero": sum(r == 0 for r in rates), "one": sum(r == 1 for r in rates),
                    "between": sum(0 < r < 1 for r in rates),
                    "median": rates[len(rates) // 2] if rates else None},
        "reconciliation": manifest["reconciliation"],
    }


def publish_manifest(out_dir: Path, manifest: dict, summary: dict) -> dict:
    """Write manifest and summary into an attempt directory and publish both by one rename; an existing
    manifest is verified and returned, never rewritten."""
    if out_dir.exists():
        return load_manifest(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    attempt = out_dir.parent / f".manifest_attempt_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    attempt.mkdir()
    (attempt / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    (attempt / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (attempt / "done.json").write_text(json.dumps({
        "manifest_sha256": file_sha256(attempt / "manifest.json"),
        "summary_sha256": file_sha256(attempt / "summary.json")}, indent=2) + "\n")
    os.rename(attempt, out_dir)
    return load_manifest(out_dir)


def load_manifest(out_dir: Path) -> dict:
    """A published manifest, verified against its done-marker."""
    done_file = Path(out_dir) / "done.json"
    if not done_file.is_file():
        raise RuntimeError(f"{out_dir}: no done.json; not a published manifest")
    done = json.loads(done_file.read_text())
    for name in ("manifest", "summary"):
        if file_sha256(Path(out_dir) / f"{name}.json") != done[f"{name}_sha256"]:
            raise RuntimeError(f"{out_dir}/{name}.json does not match its done-marker")
    return {"manifest": json.loads((Path(out_dir) / "manifest.json").read_text()),
            "summary": json.loads((Path(out_dir) / "summary.json").read_text()),
            "manifest_sha256": done["manifest_sha256"]}
