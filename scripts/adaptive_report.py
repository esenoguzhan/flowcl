"""Thin CLI: the pre-registered outcome of the adaptive-GPM run.

Spec §1: no logic in scripts/. See :mod:`flowcl.experiments.adaptive_report`; the rule is
in ``configs/analysis/adaptive_gpm.yaml``. Needs both forgetting-diagnostics reports.

Example::

    uv run python scripts/adaptive_report.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from flowcl.experiments.adaptive_report import load_adaptive_config, run_adaptive_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    run_adaptive_report(load_adaptive_config(args.config), out=args.out)


if __name__ == "__main__":
    main()
