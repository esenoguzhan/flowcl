"""Thin CLI: the allocation-rules pilot (see :mod:`flowcl.experiments.allocation_rules`).

Exit 0 when every check passed, 1 otherwise; a JSON record is printed (``predict`` and ``summary`` also
write files).

Examples::

    uv run python scripts/allocation_rules.py predict --seed 0 --out results/allocation_rules/seed0/prediction.json
    uv run python scripts/allocation_rules.py check --arm room --seed 0 --run-only
    uv run python scripts/allocation_rules.py summary --seed 0
    uv run python scripts/allocation_rules.py smoke-configs --e99-run DIR/seq_hetero__gpm_projected_adam_e99__seed2 \
        --out DIR/configs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flowcl.experiments.allocation_rules import ARMS, check_arm, predict, smoke_configs, summary
from flowcl.utils.run import atomic_write_text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--results-root", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    c = sub.add_parser("check")
    c.add_argument("--arm", choices=ARMS, required=True)
    c.add_argument("--seed", type=int, required=True)
    c.add_argument("--run-only", action="store_true")
    s = sub.add_parser("summary")
    s.add_argument("--seed", type=int, required=True)
    for q in (c, s):
        q.add_argument("--results-root", type=Path, default=None)
        q.add_argument("--smoke", action="store_true")
        q.add_argument("--comparator", type=Path, default=None, help="the e99 run (smoke: a smoke e99)")
        q.add_argument("--plain", type=Path, default=None, help="plain GPM of the same seed (smoke: a smoke run)")
    m = sub.add_parser("smoke-configs")
    m.add_argument("--e99-run", type=Path, required=True)
    m.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.cmd == "predict":
        out = predict(args.seed, results=args.results_root)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(args.out, json.dumps(out, indent=2, default=str) + "\n")
        out = {**out, "arms": {a: {k: v for k, v in r.items() if k != "k_after"} for a, r in out["arms"].items()}}
    elif args.cmd == "check":
        out = check_arm(args.arm, args.seed, results=args.results_root, smoke=args.smoke,
                        comparator=args.comparator, plain=args.plain, run_only=args.run_only)
    elif args.cmd == "summary":
        out = summary(args.seed, results=args.results_root, smoke=args.smoke, comparator=args.comparator,
                      plain=args.plain)
        out["passed"] = bool(out["evaluation_valid"]) and all(r.get("valid") for r in out["arms"].values())
        for arm, r in out["arms"].items():
            d = (r.get("primary") or {}).get("delta_acc")
            tail = f" dACC {d['diff']:+.3f} [{d['low']:+.3f}, {d['high']:+.3f}]" if d else ""
            print(f"[flowcl] {arm}: {r.get('label')}{tail}", flush=True)
        print(f"[flowcl] overall: {out['overall']['outcome']}; promote to seeds 1-2: {out['promote_to_seeds_1_2']}")
    else:
        out = smoke_configs(args.e99_run, args.out)
    print(json.dumps(out, indent=2, default=str))
    raise SystemExit(0 if out.get("passed") else 1)


if __name__ == "__main__":
    main()
