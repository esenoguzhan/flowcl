"""Does a Piper checkpoint leave the start pose? Replayed on real logged inputs.

For every logged rollout in ``--rollouts`` (PiperStudio logs), the observations taken while the
arm was still at its start pose (until a joint is 0.03 rad from the first state, at most
``--max-obs``, about 0.5 s apart) are fed, in order, to the checkpoint through the policy
server's own path (``PiperPolicyService.infer``, frames through the recorder's CRF-30 video). Each
of ``--seeds`` noise seeds walks them until a plan moves a joint more than 0.05 rad within the 8
executed ticks; that plan must also pass PiperStudio's first-jump check at the request's own
entry point (flowcl.deploy.entry_check), else the departure counts as refused (the robot stops).

Reported per light group (24 Nov, by the measured light of piper_session_diag.py scenes: sun up to
14:22, a transition 15:28-15:38, light like the demos' from 15:39), every rollout weighted equally:

* ``departed_2s`` / ``departed_6s``: the share of seeds that departed within 2 s / 6 s of the first
  observation, among rollouts whose logged window reaches that far or that departed earlier;
* ``refused``: the share whose first moving plan the executor would have refused;
* ``immediate``: the share of seeds whose plan at the first observation already moves.

    uv run python scripts/hw/piper_start_check.py --checkpoint results/.../checkpoints/final.pt
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from flowcl.deploy.entry_check import first_jump  # noqa: E402
from flowcl.deploy.piper_server import VIDEO_CRF, PiperPolicyService  # noqa: E402

MOVE_RAD = 0.05
HOME_RAD = 0.03
RATE_HZ = 15.0
EXECUTE_K = 8
GROUPS = (("sun", "000000", "150000"), ("transition", "150000", "153900"), ("demo_light", "153900", "240000"))


def group_of(name: str) -> str:
    return next(g for g, lo, hi in GROUPS if lo <= name[:6] < hi)


def at_home(run: Path, max_obs: int) -> list[dict]:
    inf = [json.loads(line) for line in (run / "inferences.jsonl").read_text().splitlines() if line.strip()]
    inf = [i for i in inf if i.get("actions") and i.get("state")]
    if not inf:
        return []
    s0 = np.asarray(inf[0]["state"][:6])
    out = []
    for i in inf:
        if np.abs(np.asarray(i["state"][:6]) - s0).max() > HOME_RAD or len(out) >= max_obs:
            break
        if all((run / "frames" / f"{i['seq']}_{c}.jpg").exists() for c in ("front", "wrist")):
            out.append(i)
    return out


def check(checkpoint: Path, rollouts: Path, seeds: int, max_obs: int, device: str) -> dict:
    svc = PiperPolicyService(checkpoint, device=device, video_crf=VIDEO_CRF)
    runs = {}
    for run in sorted(rollouts.glob("*_rollout")):
        obs = at_home(run, max_obs)
        if not obs:
            continue
        t0 = obs[0]["t_obs"]
        window = obs[-1]["t_obs"] - t0 + EXECUTE_K / RATE_HZ
        images = [{c: base64.b64encode((run / "frames" / f"{o['seq']}_{c}.jpg").read_bytes()).decode()
                   for c in ("front", "wrist")} for o in obs]
        outcomes, immediate = [], 0
        for s in range(seeds):
            result = ("none", window)
            for k, (o, imgs) in enumerate(zip(obs, images)):
                state = np.asarray(o["state"], dtype=np.float64)
                plan = np.asarray(svc.infer({"instruction": svc.task.language, "state": state.tolist(),
                                             "seed": 1_000_003 * s + k, "images": imgs})["actions"])
                moves = np.abs(plan[EXECUTE_K - 1, :6] - state[:6]).max() > MOVE_RAD
                immediate += bool(moves and k == 0)
                if moves:
                    p = max(o.get("t_reply", o["t_obs"]) - o["t_obs"], 0.0) * RATE_HZ
                    ok = first_jump(plan, p, state, state)["ok"]
                    result = ("departed" if ok else "refused", o["t_obs"] - t0)
                    break
            outcomes.append(result)
        runs[run.name] = {"group": group_of(run.name), "observations": len(obs), "window_s": window,
                          "outcomes": outcomes, "immediate": immediate / seeds}
        print(f"{run.name[:6]} {runs[run.name]['group']:10s} {len(obs):2d} obs ({window:4.1f} s): "
              + " ".join(f"{kind[0]}{t:.1f}" for kind, t in outcomes), flush=True)
    summary = {}
    for g, _, _ in GROUPS:
        rows = [r for r in runs.values() if r["group"] == g]
        if not rows:
            continue
        out = {"rollouts": len(rows), "immediate": float(np.mean([r["immediate"] for r in rows])),
               "refused": float(np.mean([np.mean([k == "refused" for k, _ in r["outcomes"]]) for r in rows]))}
        for limit in (2.0, 6.0):
            shares = []
            for r in rows:
                early = [k == "departed" and t <= limit for k, t in r["outcomes"]]
                if r["window_s"] >= limit or any(early):
                    shares.append(np.mean(early))
            out[f"departed_{limit:g}s"] = float(np.mean(shares)) if shares else None
            out[f"rollouts_{limit:g}s"] = len(shares)
        summary[g] = out
    return {"checkpoint": str(checkpoint), "rollouts_dir": str(rollouts), "seeds": seeds, "max_obs": max_obs,
            "move_rad": MOVE_RAD, "groups": [list(g) for g in GROUPS], "summary": summary, "runs": runs}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--rollouts", type=Path, default=Path("results/piper_rollouts/2026-11-24"))
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--max-obs", type=int, default=20)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, default=None, help="default: start_check.json in the run directory")
    args = p.parse_args()
    report = check(args.checkpoint, args.rollouts, args.seeds, args.max_obs, args.device)
    for g, s in report["summary"].items():
        print(f"[start_check] {g:10s} {s['rollouts']:2d} rollouts: departed within 2 s {s['departed_2s']} "
              f"(of {s['rollouts_2s']}), within 6 s {s['departed_6s']} (of {s['rollouts_6s']}), refused "
              f"{s['refused']:.2f}, immediate {s['immediate']:.2f}", flush=True)
    out = args.out or args.checkpoint.parent.parent / "start_check.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"[start_check] wrote {out}")


if __name__ == "__main__":
    main()
