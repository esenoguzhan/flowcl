"""Thin CLI: instruction negatives on the other two B tasks (cabinet top, milk), resumed from the stove
pilot's stage-A checkpoints.

Spec §1: no logic in scripts/. See :mod:`flowcl.experiments.instruction_negatives_curricula`; the rule
is ``configs/analysis/instruction_negatives_curricula.yaml``. The four training runs are
scripts/run_continual.py ``--resume-run <stove source> --start-stage 1`` (queued by
scripts/queue_instruction_negatives_curricula.sh).

``--sources-root`` holds the stove pilot's runs and study folder (default: the repository's
``results/``); ``--historical-root`` the 5 Oct runs (default: ``results/``, read only). Smoke options
need ``--results-root`` outside ``results/``.

    uv run python scripts/instruction_negatives_curricula.py --part preflight   # exit 0 ok, 3 refused
    uv run python scripts/instruction_negatives_curricula.py --part pins        # "<arm> <stage0 sha256>"
    uv run python scripts/instruction_negatives_curricula.py --part swaps --curriculum cabinet --arm gpm
    uv run python scripts/instruction_negatives_curricula.py --part probe
    uv run python scripts/instruction_negatives_curricula.py --part base-probe --curriculum cabinet   # descriptive
    uv run python scripts/instruction_negatives_curricula.py --part summary     # 0 valid, 3 invalid, 4 incomplete
    uv run python scripts/instruction_negatives_curricula.py --part resume-equality --source-run S --resumed-run R
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from flowcl.experiments.instruction_negatives_curricula import (
    ARMS, CURRICULA, EXIT_CODES, load_config, pins, resume_equality, run_base_probe, run_preflight, run_probes,
    run_summary, run_swaps,
)
from flowcl.utils.libero_paths import repo_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--part", required=True,
                        choices=("preflight", "pins", "swaps", "probe", "base-probe", "summary",
                                 "resume-equality"))
    parser.add_argument("--curriculum", choices=CURRICULA, default=None)
    parser.add_argument("--arm", choices=ARMS, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--sources-root", type=Path, default=None)
    parser.add_argument("--historical-root", type=Path, default=None)
    parser.add_argument("--source-run", type=Path, default=None)
    parser.add_argument("--resumed-run", type=Path, default=None)
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
    roots = dict(results_root=args.results_root, sources_root=args.sources_root)
    if args.part == "preflight":
        rec = run_preflight(cfg, historical_root=args.historical_root, smoke=args.smoke, **roots)
        if not rec["passed"]:
            print(f"[flowcl] preflight refused: {rec}", flush=True)
        sys.exit(0 if rec["passed"] else 3)
    if args.part == "pins":
        p = pins(cfg, args.sources_root or repo_root() / "results", args.smoke)
        for arm in ARMS:
            print(f"{arm} {p[arm]['stage0']}")
        sys.exit(0)
    common = dict(results_root=args.results_root, allow_dirty=args.allow_dirty, dataset_dir=args.dataset_dir,
                  device=args.device)
    if args.part == "swaps":
        if args.curriculum is None or args.arm is None:
            parser.error("--part swaps needs --curriculum and --arm")
        record = run_swaps(cfg, args.curriculum, args.arm, **common)
        sys.exit(0 if record["weights_unchanged"] else 3)
    if args.part == "probe":
        problems = run_probes(cfg, sources_root=args.sources_root, smoke=args.smoke,
                              smoke_batches=args.smoke_batches, **common)
        for p in problems:
            print(f"[flowcl] probe problem: {p}", flush=True)
        sys.exit(1 if problems else 0)
    if args.part == "base-probe":
        if args.curriculum is None:
            parser.error("--part base-probe needs --curriculum")
        record = run_base_probe(cfg, args.curriculum, sources_root=args.sources_root, smoke=args.smoke,
                                smoke_batches=args.smoke_batches, **common)
        sys.exit(0 if record["weights_unchanged"] else 3)
    if args.part == "resume-equality":
        if args.source_run is None or args.resumed_run is None or args.results_root is None:
            parser.error("--part resume-equality needs --source-run, --resumed-run and --results-root")
        rec = resume_equality(args.source_run, args.resumed_run, args.results_root / cfg["out"])
        print(f"[flowcl] resume equality: passed={rec['passed']} checkpoint {rec['checkpoint']['n_different']} "
              f"of {rec['checkpoint']['n_tensors']} tensors differ", flush=True)
        sys.exit(0 if rec["passed"] else 3)
    summary = run_summary(cfg, historical_root=args.historical_root, smoke=args.smoke, **roots)
    sys.exit(EXIT_CODES[summary["status"]])


if __name__ == "__main__":
    main()
