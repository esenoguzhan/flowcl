"""Thin CLI: the similar-task diagnostic (A's memory, one task B, the summary).

Spec §1: no logic in scripts/. See :mod:`flowcl.experiments.similar_tasks`; the tasks, measurements
and predictions are fixed in ``configs/analysis/similar_tasks.yaml``. A dirty tree is refused unless
``--allow-dirty``. ``--part tasks`` prints the registered B task keys, one per line (for the queue).

Smoke option (needs ``--results-root`` outside the repository's ``results/``): ``--smoke-batches N``
measures only N batches per pass and relaxes the samples-per-dimension check.

    uv run python scripts/similar_tasks.py --part memory
    uv run python scripts/similar_tasks.py --part measure --task libero_goal/put_the_bowl_on_the_stove
    uv run python scripts/similar_tasks.py --part summary
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from flowcl.experiments.similar_tasks import load_config, run_measure, run_memory, run_summary
from flowcl.utils.libero_paths import repo_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--part", choices=("tasks", "memory", "measure", "summary"), required=True)
    parser.add_argument("--task", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--smoke-batches", type=int, default=None)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    results = (repo_root() / "results").resolve()
    if args.smoke_batches is not None and (args.results_root is None or args.results_root.resolve() == results
                                           or results in args.results_root.resolve().parents):
        parser.error("--smoke-batches needs --results-root outside the repository's results/")
    common = dict(results_root=args.results_root, device=args.device, allow_dirty=args.allow_dirty,
                  dataset_dir=args.dataset_dir, smoke_batches=args.smoke_batches)
    if args.part == "tasks":
        for t in cfg["tasks_b"]:
            print(t["key"])
    elif args.part == "memory":
        record = run_memory(cfg, **common)
        sys.exit(0 if record["self_control"]["passed"] and record["weights_unchanged"] else 3)
    elif args.part == "measure":
        if args.task is None:
            parser.error("--part measure needs --task")
        result = run_measure(cfg, args.task, **common)
        sys.exit(0 if result["weights_unchanged"] else 3)
    else:
        summary = run_summary(cfg, results_root=args.results_root)
        sys.exit(0 if summary["status"] == "complete" else 4)


if __name__ == "__main__":
    main()
