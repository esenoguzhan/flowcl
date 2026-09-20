"""Thin CLI: download and verify the LIBERO demo datasets.

Spec §1: no logic in scripts/. This file only parses arguments and delegates to
:mod:`flowcl.data.libero_setup`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from flowcl.data import libero_setup


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Destination for demo HDF5s (default: LIBERO's own datasets dir).",
    )
    parser.add_argument(
        "--suites",
        nargs="+",
        choices=sorted(libero_setup.SUITE_TASK_COUNTS),
        default=None,
        help=(
            "Suites to fetch (default: the four used by the curricula; libero_90 is "
            "never fetched by default because no curriculum uses it)."
        ),
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Only configure and verify; never fetch.",
    )
    args = parser.parse_args()

    libero_setup.prepare(
        dataset_dir=args.dataset_dir,
        suites=args.suites,
        skip_download=args.skip_download,
    )


if __name__ == "__main__":
    main()
