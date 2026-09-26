"""Per-batch probe losses at registered cells, for batch-level paired bootstraps.

The forgetting diagnostics store ``L[i][j]`` only as the weighted mean over the probe's 16
fixed batches. The SGP report's loss-level interaction ``(A − P) − (N − G)`` and its CI for
``g`` resample those batches, so this no-training step recomputes them per batch for the
four runs of one seed, with each run's own diagnostics probe (config and seed tags; the
batches, ``s`` and ``A_0`` are identical for every checkpoint), and **asserts** that every
weighted mean reproduces that run's diagnostics ``loss_matrix`` value to ``rel_tol``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from omegaconf import OmegaConf

from flowcl.analysis.probes import probe_batch_losses, weighted_mean
from flowcl.experiments.sgp_report import ROLES, load_sgp_config, resolve_seed
from flowcl.utils.libero_paths import repo_root
from flowcl.utils.run import atomic_write_text, file_sha256, git_sha


def check_reproduces(mean: float, recorded: float, rel_tol: float, where: str) -> float:
    """Relative difference of a recomputed probe mean; raises beyond ``rel_tol``."""
    rel = abs(mean - recorded) / abs(recorded)
    if rel > rel_tol:
        raise RuntimeError(
            f"{where}: the per-batch probe mean {mean!r} does not reproduce the diagnostics' "
            f"{recorded!r} (relative difference {rel:.3e} > {rel_tol}); the probe, the data "
            "or the checkpoint changed"
        )
    return rel


def run_probe_cells(
    cfg: dict | None = None,
    seed: int = 0,
    results_root: Path | None = None,
    dataset_dir: Path | None = None,
    device: str = "cuda",
    allow_dirty: bool = False,
    out: Path | None = None,
) -> dict:
    from flowcl.data.curriculum import load_curriculum
    from flowcl.train.checkpoint import load_checkpoint
    from flowcl.train.pipeline import build_dataset

    started = time.time()
    resolved = resolve_seed(cfg or load_sgp_config(), seed)
    sha = git_sha()
    if sha.endswith("-dirty") and not allow_dirty:
        raise RuntimeError(f"working tree is dirty ({sha}); commit first, or pass allow_dirty "
                           "(recorded)")
    root = Path(results_root) if results_root else repo_root() / "results"
    dirs = {r: root / resolved["runs"][r] for r in ROLES}
    diags = {r: json.loads((root / resolved["diagnostics"][r]).read_text()) for r in ROLES}
    for r in ROLES:
        if diags[r]["method_run_id"] != dirs[r].name:
            raise ValueError(f"{r}: diagnostics are for {diags[r]['method_run_id']!r}, "
                             f"not {dirs[r].name!r}")
    probes = {r: diags[r]["config"]["probe"] for r in ROLES}
    if any(p != probes["A"] for p in probes.values()):
        raise ValueError(f"the four diagnostics used different probes: {probes}")
    probe = probes["A"]

    run_cfgs = {r: OmegaConf.to_container(OmegaConf.load(d / "config.yaml"), resolve=True)
                for r, d in dirs.items()}
    curricula = {r: load_curriculum(c["curriculum"]) for r, c in run_cfgs.items()}
    task_keys = list(curricula["A"].task_keys)
    if any(list(c.task_keys) != task_keys for c in curricula.values()):
        raise ValueError("the four runs cover different task sequences")
    base = load_checkpoint(dirs["A"] / "checkpoints" / "stage0.pt")
    fingerprint = base.stats.fingerprint()
    cells = [tuple(c) for c in resolved["probe_cells"]["cells"]]
    tasks_needed = sorted({j for _, j in cells})
    stages = curricula["A"].stages
    # Data exactly as training (and the diagnostics) built it: stats frozen at T1.
    datasets = {j: build_dataset([stages[j].ref], base.spec, base.stats,
                                 n_demos=stages[j].n_demos, dataset_dir=dataset_dir)
                for j in tasks_needed}
    del base

    rel_tol = resolved["probe_cells"]["rel_tol"]
    out_cells: dict = {r: {} for r in ROLES}
    inputs = {}
    for r in ROLES:
        for stage in sorted({i for i, _ in cells}):
            path = dirs[r] / "checkpoints" / f"stage{stage}.pt"
            inputs[f"{dirs[r].name}/checkpoints/stage{stage}.pt"] = file_sha256(path)
            loaded = load_checkpoint(path, device=device)
            if loaded.stats.fingerprint() != fingerprint:
                raise ValueError(f"{r} stage {stage} normalises with different stats")
            for i, j in cells:
                if i != stage:
                    continue
                batches = probe_batch_losses(loaded.policy, datasets[j], probe, device)
                mean = weighted_mean(batches)
                recorded = diags[r]["loss_matrix"]["method"][i][j]
                rel = check_reproduces(mean, recorded, rel_tol, f"{r} L[{i}][{j}]")
                out_cells[r][f"{i},{j}"] = {
                    "losses": [b[0] for b in batches], "weights": [b[1] for b in batches],
                    "mean": mean, "diagnostics": recorded, "rel_diff": rel,
                }
            del loaded

    report = {
        "git_sha": sha,
        "allow_dirty": allow_dirty,
        "seed": seed,
        "runs": {r: dirs[r].name for r in ROLES},
        "task_keys": task_keys,
        "probe": probe,
        "cells_requested": [list(c) for c in cells],
        "rel_tol": rel_tol,
        "cells": out_cells,
        "inputs_sha256": inputs,
        "diagnostics_sha256": {r: file_sha256(root / resolved["diagnostics"][r]) for r in ROLES},
        "timings_s": time.time() - started,
    }
    out = Path(out) if out else root / resolved["probe_report"]
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out, json.dumps(report, indent=2) + "\n")
    worst = max(c["rel_diff"] for rows in out_cells.values() for c in rows.values())
    print(f"[flowcl] probe cells for seed {seed}: {len(cells)} cells x {len(ROLES)} runs, "
          f"worst relative difference to the diagnostics {worst:.2e}; wrote {out}", flush=True)
    return report
