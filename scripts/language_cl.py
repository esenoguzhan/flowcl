"""Thin CLI: the language-conditioned base study (swap rollouts, probes, the summary).

Spec §1: no logic in scripts/. See :mod:`flowcl.experiments.language_cl`; the rule is
``configs/analysis/language_cl.yaml``. The training runs are scripts/run_continual.py on the
``langbase_cabinet_stove`` curriculum (queued by scripts/queue_language_cl.sh). A dirty tree is
refused unless ``--allow-dirty``.

Smoke options (need ``--results-root`` outside the repository's ``results/``): ``--smoke-batches N``
(N batches per probe pass) and, for the summary, ``--smoke`` (the smoke recipe and a dirty tree).
``--a-only-root`` is where the A-only base of similar_cl lives (default: the results root; a smoke
run points it at the repository's ``results/``, read only).

    uv run python scripts/language_cl.py --part swaps --stage 0 --arm seq_ft
    uv run python scripts/language_cl.py --part swaps --stage 1 --arm gpm
    uv run python scripts/language_cl.py --part probe
    uv run python scripts/language_cl.py --part summary

Exit codes: swaps 0, or 3 if the weights changed; probe 0, or 1 if a checkpoint is missing or a
probe failed (every other probe still runs); summary 0 valid, 3 invalid, 4 incomplete.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from flowcl.experiments.language_cl import (
    EXIT_CODES, PROBE_LABELS, config_path, load_config, run_probes, run_summary, run_swaps,
)
from flowcl.utils.libero_paths import repo_root
from flowcl.utils.run import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--part", choices=("swaps", "probe", "summary"), required=True)
    parser.add_argument("--stage", type=int, choices=(0, 1), default=None)
    parser.add_argument("--arm", choices=("seq_ft", "gpm"), default=None)
    parser.add_argument("--labels", nargs="+", choices=PROBE_LABELS, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--a-only-root", type=Path, default=None)
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
    if args.part == "swaps":
        if args.stage is None or args.arm is None:
            parser.error("--part swaps needs --stage and --arm")
        record = run_swaps(cfg, args.stage, args.arm, results_root=args.results_root, device=args.device,
                           allow_dirty=args.allow_dirty, dataset_dir=args.dataset_dir)
        sys.exit(0 if record["weights_unchanged"] else 3)
    elif args.part == "probe":
        problems = run_probes(cfg, args.labels, results_root=args.results_root, a_only_root=args.a_only_root,
                              device=args.device, allow_dirty=args.allow_dirty, dataset_dir=args.dataset_dir,
                              smoke_batches=args.smoke_batches)
        for p in problems:
            print(f"[flowcl] probe problem: {p}", flush=True)
        sys.exit(1 if problems else 0)
    else:
        summary = run_summary(cfg, results_root=args.results_root, a_only_root=args.a_only_root, smoke=args.smoke,
                              rule_sha256=file_sha256(args.config or config_path()))
        sys.exit(EXIT_CODES[summary["status"]])


if __name__ == "__main__":
    main()
