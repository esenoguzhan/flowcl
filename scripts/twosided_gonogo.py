"""Thin CLI: E1, the two-sided protection go/no-go measurement (captures, summary).

Spec §1: no logic in scripts/. See :mod:`flowcl.experiments.twosided_gonogo`; the source, the
capture protocol, the validity checks and the verdict are fixed in
``configs/analysis/twosided_gonogo.yaml``. A dirty tree is refused unless ``--allow-dirty``.

Exit codes: ``--part capture`` 0 if the published capture is valid, 3 if not;
``--part summary`` 0 for a valid finding (pass or fail), 3 invalid, 4 incomplete.
Anything else is a crash.

Smoke option (needs ``--results-root`` outside the repository's ``results/``; the source run is
always read from the repository's ``results/``): ``--smoke-batches N`` captures only the first N
batches of the task (the memory replay and the samples-per-dimension check are then unverified).

Examples::

    uv run python scripts/twosided_gonogo.py --part capture --task 0
    uv run python scripts/twosided_gonogo.py --part summary
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from flowcl.experiments.twosided_gonogo import (
    EXIT_CODES,
    load_config,
    run_capture,
    run_summary,
)
from flowcl.utils.libero_paths import repo_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--part", choices=("capture", "summary"), required=True)
    parser.add_argument("--task", type=int, default=None, help="task index 0..3 (= T1..T4)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--smoke-batches", type=int, default=None)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.smoke_batches is not None and (
            args.results_root is None
            or args.results_root.resolve() == (repo_root() / "results").resolve()
            or (repo_root() / "results").resolve() in args.results_root.resolve().parents):
        parser.error("--smoke-batches needs --results-root outside the repository's results/")
    if args.part == "capture":
        if args.task is None:
            parser.error("--part capture needs --task")
        record = run_capture(cfg, args.task, results_root=args.results_root, device=args.device,
                             allow_dirty=args.allow_dirty, dataset_dir=args.dataset_dir,
                             smoke_batches=args.smoke_batches)
        sys.exit(0 if record["valid"] else 3)
    summary = run_summary(cfg, results_root=args.results_root)
    sys.exit(EXIT_CODES[summary["verdict"]["verdict"]])


if __name__ == "__main__":
    main()
