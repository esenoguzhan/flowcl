"""Thin CLI: the similar-vs-diverse continual study (stage-0 extras, stage-1 probes, the summary).

Spec §1: no logic in scripts/. See :mod:`flowcl.experiments.similar_cl`; the rule is
``configs/analysis/similar_cl.yaml``. The training runs are scripts/run_continual.py (queued by
scripts/queue_similar_cl.sh). A dirty tree is refused unless ``--allow-dirty``.

Smoke options (need ``--results-root`` outside the repository's ``results/``): ``--smoke-batches N``
(N batches per probe pass) and, for the summary, ``--smoke`` (the smoke recipe and a dirty tree).

    uv run python scripts/similar_cl.py --part stage0
    uv run python scripts/similar_cl.py --part probe --curricula similar_ab_stove similar_ab_milk
    uv run python scripts/similar_cl.py --part summary

Exit codes: stage0 0, or 3 if the weights changed; probe 0, or 1 if a stage1.pt is missing or a probe
failed (every other run is still probed); summary 0 valid, 3 invalid, 4 incomplete.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from flowcl.experiments.similar_cl import (
    EXIT_CODES, config_path, load_config, run_probes, run_stage0, run_summary,
)
from flowcl.utils.libero_paths import repo_root
from flowcl.utils.run import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--part", choices=("stage0", "probe", "summary"), required=True)
    parser.add_argument("--curricula", nargs="+", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--smoke-batches", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    results = (repo_root() / "results").resolve()
    scratch = args.results_root is not None and not (
        args.results_root.resolve() == results or results in args.results_root.resolve().parents)
    if (args.smoke_batches is not None or args.smoke) and not scratch:
        parser.error("smoke options need --results-root outside the repository's results/")
    common = dict(results_root=args.results_root, device=args.device, allow_dirty=args.allow_dirty,
                  dataset_dir=args.dataset_dir, smoke_batches=args.smoke_batches)
    if args.part == "stage0":
        record = run_stage0(cfg, **common)
        sys.exit(0 if record["weights_unchanged"] else 3)
    elif args.part == "probe":
        names = [c["name"] for c in cfg["curricula"]]
        if not args.curricula or any(c not in names for c in args.curricula):
            parser.error(f"--part probe needs --curricula among {names}")
        problems = run_probes(cfg, args.curricula, **common)
        for p in problems:
            print(f"[flowcl] probe problem: {p}", flush=True)
        sys.exit(1 if problems else 0)
    else:
        summary = run_summary(cfg, results_root=args.results_root, smoke=args.smoke,
                              rule_sha256=file_sha256(args.config or config_path()))
        sys.exit(EXIT_CODES[summary["status"]])


if __name__ == "__main__":
    main()
