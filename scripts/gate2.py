"""Thin CLI: run §10.3 Gate 2 -- is projection geometrically plausible?

Spec §1: no logic in scripts/. See :mod:`flowcl.experiments.gate2`.

Example::

    uv run python scripts/gate2.py --curriculum seq_hetero --seed 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

from flowcl.data.curriculum import load_curriculum
from flowcl.experiments.gate2 import default_checkpoints, load_subspace_config, run_gate2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curriculum", default="seq_hetero")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Verdict checkpoint. Default: the Gate 1 seq_ft stage-0 checkpoint.",
    )
    parser.add_argument(
        "--no-references",
        action="store_true",
        help="Skip the Gate 0 single-task checkpoints (robustness evidence only).",
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing bases files."
    )
    args = parser.parse_args()

    verdict, references = default_checkpoints(
        load_curriculum(args.curriculum), seed=args.seed
    )
    run_gate2(
        verdict_checkpoint=args.checkpoint or verdict,
        reference_checkpoints=[] if args.no_references else references,
        cfg=load_subspace_config(args.config),
        device=args.device,
        out_dir=args.out_dir,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
