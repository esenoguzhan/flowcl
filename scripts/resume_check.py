"""Thin CLI: compare a resumed run with its source (the A1 resume test).

Spec §1: no logic in scripts/. See :mod:`flowcl.experiments.resume_check`; the source, the stage
and the criteria are fixed in ``configs/analysis/resume_check.yaml``. Writes
``results/<out>/report.json``; exits 0 if the resumed stage reproduces the source, 1 if not.

    uv run python scripts/resume_check.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf

from flowcl.experiments.resume_check import compare_resume
from flowcl.utils.libero_paths import repo_root
from flowcl.utils.run import atomic_write_text, file_sha256, git_sha


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=repo_root() / "configs" / "analysis" / "resume_check.yaml")
    parser.add_argument("--results-root", type=Path, default=repo_root() / "results")
    args = parser.parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    src = repo_root() / "results" / cfg["source"]["run"]
    stage = int(cfg["source"]["stage"])
    pinned = {"stage2": f"checkpoints/stage{stage - 1}.pt", "stage3": f"checkpoints/stage{stage}.pt",
              "memory2": f"method/memory_task{stage - 1}.pt", "memory3": f"method/memory_task{stage}.pt",
              "eval3": f"eval/stage{stage}.json"}
    changed = {k: rel for k, rel in pinned.items() if file_sha256(src / rel) != cfg["source"]["sha256"][k]}
    if changed:
        sys.exit(f"the source changed since the config was written: {changed}")
    out = args.results_root / cfg["out"]
    report = {"git_sha": git_sha(), "config": cfg,
              **compare_resume(src, out / cfg["source"]["run"], stage)}
    atomic_write_text(out / "report.json", json.dumps(report, indent=2) + "\n")
    c, m, e = report["checkpoint"], report["method_artifacts"], report["evaluation"]
    print(f"[flowcl] resume check: {'PASS' if report['passed'] else 'FAIL'} - checkpoint "
          f"{c['n_different']} of {c['n_tensors']} tensors differ; "
          + "; ".join(f"{k} {'equal' if v['passed'] else 'DIFFERS'}"
                      + (f" (bytes {'equal' if v.get('bytes_equal') else 'differ'})" if "bytes_equal" in v else "")
                      for k, v in m.items())
          + f"; evaluation {'identical' if e['passed'] else 'DIFFERS'}", flush=True)
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
