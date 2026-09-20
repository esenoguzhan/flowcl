"""Thin CLI: build every results table and plot from ``results/``.

Spec §1: no logic in scripts/. See :mod:`flowcl.analysis.tables`.

Spec §11: every number carries a CI. :func:`flowcl.analysis.tables.format_cell` raises
on a bare float, so a table without uncertainty cannot be produced from here.

Example::

    uv run python scripts/make_tables.py --out-dir results/tables
"""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from flowcl.analysis.tables import build_all
from flowcl.utils.libero_paths import repo_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--eval-config",
        type=Path,
        default=None,
        help="Read the bootstrap settings from here so tables and runs agree.",
    )
    args = parser.parse_args()

    eval_path = args.eval_config or (
        repo_root() / "configs" / "eval" / "libero_eval.yaml"
    )
    bootstrap = OmegaConf.to_container(OmegaConf.load(eval_path), resolve=True).get(
        "bootstrap"
    )

    path = build_all(
        results_root=args.results_root, out_dir=args.out_dir, bootstrap=bootstrap
    )
    print(f"[flowcl] wrote {path}")


if __name__ == "__main__":
    main()
