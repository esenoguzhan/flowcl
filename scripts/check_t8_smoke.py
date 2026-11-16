"""Thin CLI: check the 8-task resume smoke (see :mod:`flowcl.experiments.t8_smoke`).

Writes ROOT/checks.json and ROOT/reports/*.json; exit 0 when every check passed, 1 otherwise.

Example::

    uv run python scripts/check_t8_smoke.py --root /home/oguzhan/flowcl_smoke/t8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flowcl.experiments.t8_smoke import check_smoke


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    out = check_smoke(args.root)
    (args.root / "checks.json").write_text(json.dumps(out, indent=2, default=str) + "\n")
    summary = {name: section["passed"] for group in ("comparisons", "fields", "reports")
               for name, section in ((f"{group}/{k}", v) for k, v in out[group].items())}
    summary["cut"] = out["cut"]["passed"]
    print(json.dumps({"passed": out["passed"], **summary}, indent=2))
    for group in ("fields", "reports"):
        for name, section in out[group].items():
            for problem in section["problems"]:
                print(f"  {group}/{name}: {problem}")
    raise SystemExit(0 if out["passed"] else 1)


if __name__ == "__main__":
    main()
