"""Choose the deployment recipe from block A of scripts/queue_piper_shadow.sh (rule fixed 24 Nov).

Reads ``start_check.json`` and ``relight_check.json`` of the five block-A runs (``v0`` plain,
``b0`` trimmed at both ends, ``e0`` end only, ``s0`` start only, ``b3_mix`` both + shadows) under
``--root`` and applies the rule fixed in the plan of 24 Nov, before any result:

A run qualifies when
* it departs within 2 s in >= 80 % of the demo-light and of the sun rollouts (start check);
* the executor would refuse at departure in <= 10 %;
* its first moving chunk is a plausible reach in >= 80 % of the held-out episodes (clean);
* clean MAE <= 1.1x v0's and relit approach MAE (front maps) <= 1.1x v0's, and the same two
  bounds hold under front maps + wrist shadows.

Trim: among the qualifying b0, e0 and s0, the best demo-light departure within 2 s; within 0.05,
the higher relit opening recall, then the lower relit approach MAE. Shadows are added if b3_mix
qualifies and its relit opening recall beats b0's by >= 0.05 with a demo-light departure no worse
than b0's - 0.05. Prints ``RECIPE <run> <all-90 task> <extra gate0 flags>`` and exits 0, or prints
``NONE`` and exits 3 when nothing qualifies (block B is then skipped).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TRIM_TASK = {"b0": "new_grape_trim2_336", "e0": "new_grape_trimend_336", "s0": "new_grape_trim_336",
             "v0": "new_grape_336"}
RUN_DIR = {"v0": "new_grape_ho10_336", "b0": "new_grape_trim2_ho10_336", "e0": "new_grape_trimend_ho10_336",
           "s0": "new_grape_trim_ho10_336", "b3_mix": "new_grape_trim2_ho10_336"}


def load(root: Path, run: str) -> dict | None:
    d = root / run / f"single__piper__{RUN_DIR[run]}__seed0"
    try:
        return {"start": json.loads((d / "start_check.json").read_text())["summary"],
                "relight": json.loads((d / "relight_check.json").read_text())["summary"]}
    except (OSError, KeyError, ValueError):
        return None


def qualifies(r: dict, ref: dict) -> tuple[bool, list[str]]:
    why = []
    for g in ("demo_light", "sun"):
        dep = r["start"].get(g, {}).get("departed_2s")
        if dep is None or dep < 0.8:
            why.append(f"{g} departure within 2 s {dep}")
        if r["start"].get(g, {}).get("refused", 1.0) > 0.1:
            why.append(f"{g} refused {r['start'][g]['refused']:.2f}")
    reach = r["relight"]["clean"].get("plausible_reach")
    if reach is None or reach < 0.8:
        why.append(f"plausible reach {reach}")
    for cond in ("front_maps", "front_maps_wrist"):
        if r["relight"]["clean"]["mae"] > 1.1 * ref["relight"]["clean"]["mae"]:
            why.append(f"clean mae {r['relight']['clean']['mae']:.4f} > 1.1x v0")
        a, a0 = r["relight"][cond].get("approach_mae"), ref["relight"][cond].get("approach_mae")
        if a is None or a0 is None or a > 1.1 * a0:
            why.append(f"{cond} approach mae {a} vs v0 {a0}")
    return not why, sorted(set(why))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=Path("results/piper_shadow"))
    p.add_argument("--maps", default="datasets/piper/shadow_maps/table2_train.npz")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    runs = {r: load(args.root, r) for r in RUN_DIR}
    ref = runs["v0"]
    if ref is None:
        print("NONE: v0 has no checks"); sys.exit(3)
    verdict = {}
    for name, r in runs.items():
        if r is None:
            verdict[name] = (False, ["missing checks"])
            continue
        verdict[name] = qualifies(r, ref)
        s = r["start"]
        print(f"{name:7s} qualifies {verdict[name][0]!s:5s} | departure 2 s demo-light {s.get('demo_light', {}).get('departed_2s')}, "
              f"sun {s.get('sun', {}).get('departed_2s')} | relit opening {r['relight']['front_maps'].get('opening_recall')}, "
              f"relit approach {r['relight']['front_maps'].get('approach_mae')} | {'; '.join(verdict[name][1])}")
    trims = [t for t in ("b0", "e0", "s0") if verdict[t][0]]
    if not trims:
        print("NONE: no trimmed run qualifies")
        choice = None
    else:
        dep = lambda t: runs[t]["start"]["demo_light"]["departed_2s"]
        best = max(dep(t) for t in trims)
        tied = [t for t in trims if dep(t) >= best - 0.05]
        tied.sort(key=lambda t: (-(runs[t]["relight"]["front_maps"].get("opening_recall") or 0.0),
                                 runs[t]["relight"]["front_maps"].get("approach_mae") or 1e9))
        trim = tied[0]
        shadows = False
        if verdict["b3_mix"][0] and runs["b0"] is not None:
            rb, r3 = runs["b0"], runs["b3_mix"]
            shadows = ((r3["relight"]["front_maps"].get("opening_recall") or 0.0)
                       >= (rb["relight"]["front_maps"].get("opening_recall") or 0.0) + 0.05
                       and r3["start"]["demo_light"]["departed_2s"] >= rb["start"]["demo_light"]["departed_2s"] - 0.05)
        flags = (f"--augment-shadow 0.5 --augment-shadow-maps {args.maps} --augment-shadow-map-share 0.5"
                 if shadows else "")
        choice = {"trim_run": trim, "shadows": shadows, "task": TRIM_TASK[trim], "flags": flags,
                  "name": f"{trim[0]}{'3_mix' if shadows else '0'}_all90"}
        print(f"RECIPE {choice['name']} {choice['task']} {flags}".rstrip())
    if args.out:
        args.out.write_text(json.dumps({"verdict": {k: {"qualifies": v[0], "why": v[1]} for k, v in verdict.items()},
                                        "choice": choice}, indent=1) + "\n")
    sys.exit(0 if choice else 3)


if __name__ == "__main__":
    main()
