"""Why a Piper policy fails on the robot: read PiperStudio's rollout logs against the demos.

Five checks, each a subcommand, each printing a summary and writing JSON with --out:

    logs    per run: how far the arm moved, the gripper, the lowest end pose, the limiter,
            how much consecutive chunks disagree, tracking, how often a chunk plans motion
    demos   the same quantities for a recorded source, where they apply
    path    one rollout against the recorded episode its scene was set up from: how far the
            arm's state is from that episode's path, and from any demo, over time
    images  in-process, no server: the same state with the rollout's images or the episode's
            (first plan at the start pose; and, at states the rollout stalled in, both cameras
            or one swapped). Shows whether the images, not the state, change the plan
    sheet   front and wrist frames at the moment each run's gripper closes

    uv run python scripts/hw/piper_rollout_diag.py logs --day ~/piper_rollouts/2026-11-03
    uv run python scripts/hw/piper_rollout_diag.py path --run ~/piper_rollouts/2026-11-03/115601_rollout --episode 20
    uv run python scripts/hw/piper_rollout_diag.py images --run ~/piper_rollouts/2026-11-03/115601_rollout \
        --episode 20 --checkpoint results/piper_color/single__piper__milk_plate__seed0/checkpoints/final.pt

Logs follow docs/piper_deploy_spec.md §7. Joint quantities are over J1-J6 in rad; a chunk
"plans motion" when its first 8 ticks move faster than 0.2 rad/s.
"""

from __future__ import annotations

import argparse
import base64
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from flowcl.data.piper_adapter import read_episode_table, read_frames_table, read_info  # noqa: E402

DEFAULT_SOURCE = Path("~/piper_datasets/pick_milk").expanduser()
MOVING_RAD_S = 0.2
RATE = 15.0


# ---- reading -----------------------------------------------------------------------------

def load_run(run: Path) -> tuple[dict, list[dict], dict]:
    meta = json.loads((run / "meta.json").read_text())
    inf = [json.loads(line) for line in (run / "inferences.jsonl").read_text().splitlines() if line.strip()]
    ticks = dict(np.load(run / "ticks.npz"))
    return meta, inf, ticks


def run_name(meta: dict) -> str:
    """The checkpoint's run folder under results/, e.g. piper_color/single__piper__milk_plate__seed0."""
    if meta.get("mode") == "replay":
        ic = meta.get("initial_condition") or {}
        return f"(replay of {ic.get('dataset')} {ic.get('episode_index')})"
    ck = meta["server_metadata"]["checkpoint"] if meta.get("server_metadata") else ""
    return ck.split("results/")[-1].split("/checkpoints")[0] if ck else "(replay)"


def grasp_index(gripper: np.ndarray, open_m: float = 0.02) -> int | None:
    """The first observation after the gripper first opened past ``open_m`` where it has closed
    below 70 % of its widest opening so far; None when it never opened."""
    opened = np.where(gripper > open_m)[0]
    if not opened.size:
        return None
    widest = np.maximum.accumulate(gripper[opened[0]:])
    closed = np.where(gripper[opened[0]:] < 0.7 * widest)[0]
    return int(opened[0] + closed[0]) if closed.size else len(gripper) - 1


class Source:
    """A recorded LeRobot source: states, actions, end poses and frames per episode."""

    def __init__(self, path: Path) -> None:
        import pyarrow.parquet as pq

        self.path = Path(path)
        self.frames = read_frames_table(self.path)
        self.episodes = read_episode_table(self.path)
        self.info = read_info(self.path)
        files = sorted(glob.glob(str(self.path / "data" / "chunk-*" / "file-*.parquet")))
        self.eef = np.asarray(pq.ParquetDataset(files).read(columns=["observation.eef"]).column(0).to_pylist())

    def rows(self, ep: int) -> slice:
        m = self.episodes[ep]
        return slice(m["dataset_from_index"], m["dataset_to_index"])

    def jpegs(self, ep: int, key: str, ticks: list[int] | None = None) -> list[bytes]:
        """The episode's frames of one camera as JPEG bytes (quality 90), as a camera would send them."""
        import cv2

        m = self.episodes[ep]
        path = self.path / self.info["video_path"].format(
            video_key=key, chunk_index=m[f"videos/{key}/chunk_index"], file_index=m[f"videos/{key}/file_index"])
        first = int(round(m[f"videos/{key}/from_timestamp"] * RATE))
        cap = cv2.VideoCapture(str(path))
        out = []
        for t in ticks if ticks is not None else range(m["length"]):
            cap.set(cv2.CAP_PROP_POS_FRAMES, first + t)
            ok, bgr = cap.read()
            if not ok:
                raise IOError(f"{path}: no frame {first + t}")
            out.append(cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes())
        cap.release()
        return out


def chunk_motion(inf: list[dict]) -> tuple[float, float]:
    """Share of chunks that plan motion, and the median lead of a moving chunk's tick 0 over the
    measured state along its own direction of motion."""
    leads = []
    inf = [r for r in inf if r.get("actions")]
    for r in inf:
        a, st = np.asarray(r["actions"])[:, :6], np.asarray(r["state"][:6])
        v = (a[8] - a[0]) * RATE / 8
        if np.linalg.norm(v) > MOVING_RAD_S:
            leads.append(float((a[0] - st) @ v / np.linalg.norm(v)))
    return len(leads) / len(inf), float(np.median(leads)) if leads else float("nan")


def chunk_disagreement(inf: list[dict]) -> np.ndarray:
    """|new chunk - old chunk| per joint at the instant the new chunk is entered, one row per swap."""
    used = [r for r in inf if r.get("actions") and r.get("used", True)]
    rows = []
    for a, b in zip(used[:-1], used[1:]):
        old, new = np.asarray(a["actions"])[:, :6], np.asarray(b["actions"])[:, :6]
        entry = (b["t_reply"] - b["t_obs"]) * RATE
        at_old = (b["t_obs"] - a["t_obs"]) * RATE + entry
        if at_old > 15:
            continue

        def interp(c, p):
            i = int(np.floor(p))
            f = p - i
            return c[i] * (1 - f) + c[min(i + 1, 15)] * f
        rows.append(np.abs(interp(new, entry) - interp(old, at_old)))
    return np.asarray(rows)


# ---- subcommands -------------------------------------------------------------------------

def cmd_logs(args) -> dict:
    out = {}
    for run in sorted(p for p in Path(args.day).expanduser().iterdir() if (p / "inferences.jsonl").is_file()):
        meta, inf, t = load_run(run)
        if not inf:
            continue
        s = np.asarray([r["state"] for r in inf])
        z = np.asarray([r["end_pose"][2] for r in inf])
        lim = t["limited"].astype(bool)
        swap_t = t["t"][np.r_[False, np.diff(t["chunk"]) != 0]]
        near = np.zeros_like(lim)
        for ts in swap_t:
            near |= (t["t"] >= ts) & (t["t"] < ts + 0.15)
        sent = t["sent_target"][:, :6]
        ok = np.isfinite(sent).all(1)
        track = np.abs(sent[ok] - t["joints"][ok]).max(1)
        dis = chunk_disagreement(inf)
        moving, lead = chunk_motion(inf)
        out[run.name] = row = {
            "run": run_name(meta), "label": meta.get("label", ""), "reason": meta.get("reason"),
            "duration_s": meta.get("duration_s"), "requests": len(inf),
            "max_joint_travel_rad": float(np.abs(s[:, :6] - s[0, :6]).max()),
            "gripper_max_mm": float(s[:, 6].max() * 1000), "end_z_min_mm": float(z.min()),
            "limited_share": float(lim.mean()), "limited_near_swap": float((lim & near).sum() / max(lim.sum(), 1)),
            "swap_disagreement_p50_rad": np.median(dis, 0).round(4).tolist(),
            "swap_disagreement_max_rad": dis.max(0).round(3).tolist(),
            "tracking_p95_rad": float(np.percentile(track, 95)),
            "moving_chunk_share": moving, "moving_chunk_lead_rad": lead,
            "latency_ms": meta.get("latency_ms"),
        }
        print(f"{run.name[:6]} {row['run'][:46]:46s} {row['duration_s'] or 0:5.1f}s req {len(inf):3d} "
              f"travel {row['max_joint_travel_rad']:.2f} grip {row['gripper_max_mm']:4.1f}mm z_min {row['end_z_min_mm']:4.0f} "
              f"| lim {row['limited_share']:.3f} ({row['limited_near_swap']:.2f} at swaps) "
              f"swap p50 {max(row['swap_disagreement_p50_rad']):.3f} max {max(row['swap_disagreement_max_rad']):.2f} "
              f"| track p95 {row['tracking_p95_rad']:.3f} | moving {moving:.2f} lead {lead:+.3f}")
    return out


def cmd_demos(args) -> dict:
    src = Source(args.source)
    f = src.frames
    a, s, ep = f["action"][:, :6], f["state"][:, :6], f["episode_index"]
    same = ep[1:] == ep[:-1]
    vel = np.diff(a, axis=0)[same] * RATE
    lead = (a[1:] - s[1:])[same]
    speed = np.linalg.norm(vel, axis=1)
    mv = speed > MOVING_RAD_S
    gap = np.abs(a - s).max(1)
    idle, gz = [], []
    for e in range(len(src.episodes)):
        r = src.rows(e)
        dev = np.abs(f["action"][r, :6] - f["action"][r.start, :6]).max(1)
        idle.append(int(np.argmax(dev > 0.02)))
        g = grasp_index(f["action"][r, 6])
        if g is not None:
            gz.append(float(src.eef[r.start + g, 2] * 1000))
    out = {
        "command_minus_measured_rad": {"p50": float(np.median(gap)), "p95": float(np.percentile(gap, 95)),
                                       "max": float(gap.max())},
        "moving_tick_share": float(mv.mean()),
        "moving_speed_p50_rad_s": float(np.median(speed[mv])),
        "moving_lead_rad": float(np.median((lead[mv] * vel[mv]).sum(1) / speed[mv])),
        "idle_start_ticks": {"p50": float(np.median(idle)), "min": int(min(idle)), "max": int(max(idle))},
        "grasp_end_z_mm": {"min": min(gz), "p50": float(np.median(gz)), "max": max(gz)},
    }
    print(json.dumps(out, indent=1))
    return out


def cmd_path(args) -> dict:
    src = Source(args.source)
    meta, inf, _ = load_run(Path(args.run).expanduser())
    S, E = src.frames["state"][:, :6], src.frames["episode_index"]
    own = S[src.rows(args.episode)]
    rows = []
    for r in inf[: args.max_obs: args.every]:
        st = np.asarray(r["state"][:6])
        d_own = np.abs(own - st).max(1)
        d_all = np.abs(S - st).max(1)
        j = int(d_all.argmin())
        k = int(d_own.argmin())
        rows.append({"t": r["t_obs"], "to_episode_rad": float(d_own.min()), "episode_tick": k,
                     "to_any_demo_rad": float(d_all.min()), "nearest_demo": int(E[j]),
                     "nearest_tick": int(j - src.episodes[E[j]]["dataset_from_index"]),
                     "j1": float(st[0]), "episode_j1": float(own[k, 0])})
        x = rows[-1]
        print(f"t={x['t']:5.1f}s to ep{args.episode} path {x['to_episode_rad']:.3f} (tick {k:3d}) | to any demo "
              f"{x['to_any_demo_rad']:.3f} (ep{x['nearest_demo']} tick {x['nearest_tick']}) | J1 {x['j1']:+.3f} "
              f"(ep{args.episode} there {x['episode_j1']:+.3f})")
    return {"run": run_name(meta), "episode": args.episode, "rows": rows}


def cmd_images(args) -> dict:
    from flowcl.deploy.piper_server import PiperPolicyService

    run = Path(args.run).expanduser()
    meta, inf, _ = load_run(run)
    src = Source(args.source)
    svc = PiperPolicyService(Path(args.checkpoint), device=args.device)
    keys = {"front": "observation.images.image", "wrist": "observation.images.wrist_image"}
    rows = src.rows(args.episode)
    f = src.frames

    def chunk(state, images: dict[str, bytes], seed: int) -> np.ndarray:
        req = {"instruction": svc.task.language, "state": [float(x) for x in state], "seed": seed,
               "images": {c: base64.b64encode(b).decode() for c, b in images.items()}}
        return np.asarray(svc.infer(req)["actions"])

    def logged(r: dict) -> dict[str, bytes]:
        return {c: (run / "frames" / f"{r['seq']}_{c}.jpg").read_bytes() for c in keys}

    out = {"checkpoint": args.checkpoint, "run": run.name, "episode": args.episode, "first_plan": {}, "swap": {}}
    # first plan: the episode's state at tick T, with its own images and (at T = 0) the rollout's first ones
    for T in (0, 15):
        st, truth = f["state"][rows.start + T], f["action"][rows.start + T: rows.start + T + 16, :6]
        conds = {"episode images": {c: src.jpegs(args.episode, k, [T])[0] for c, k in keys.items()}}
        if T == 0:
            conds["rollout images"] = logged(inf[0])
        for name, imgs in conds.items():
            a = np.stack([chunk(st, imgs, sd)[:, :6] for sd in range(5)])
            res = {"error_rad": float(np.abs(a - truth).mean()), "j1_at_1s": float(a[:, 15, 0].mean()),
                   "j1_at_1s_sd": float(a[:, 15, 0].std()), "episode_j1_at_1s": float(truth[15, 0])}
            out["first_plan"][f"tick {T}, {name}"] = res
            print(f"first plan, tick {T:2d}, {name:15s}: error vs episode {res['error_rad']:.3f} rad, "
                  f"J1 at +1 s {res['j1_at_1s']:+.3f}±{res['j1_at_1s_sd']:.3f} (episode {res['episode_j1_at_1s']:+.3f})")
    # swap: at the rollout's states in [t_lo, t_hi], the nearest state of the episode before its grasp
    ep_state = f["state"][rows, :6]
    opens = int(np.argmax(f["action"][rows, 6] > 0.03))
    ep_front = src.jpegs(args.episode, keys["front"], list(range(opens + 15)))
    ep_wrist = src.jpegs(args.episode, keys["wrist"], list(range(opens + 15)))
    picks = [r for r in inf if args.t_lo <= r["t_obs"] <= args.t_hi][:: args.every]
    res = {k: [] for k in ("rollout", "episode", "episode front + rollout wrist", "rollout front + episode wrist")}
    gaps = []
    for r in picks:
        st = np.asarray(r["state"])
        k = int(np.abs(ep_state[: opens + 15] - st[:6]).max(1).argmin())
        gaps.append(float(np.abs(ep_state[k] - st[:6]).max()))
        mine = logged(r)
        conds = {"rollout": mine, "episode": {"front": ep_front[k], "wrist": ep_wrist[k]},
                 "episode front + rollout wrist": {"front": ep_front[k], "wrist": mine["wrist"]},
                 "rollout front + episode wrist": {"front": mine["front"], "wrist": ep_wrist[k]}}
        for seed in (1, 2, 3):
            for name, imgs in conds.items():
                a = chunk(st, imgs, seed)
                res[name].append((float(np.abs(a[15, :6] - a[0, :6]).max()), float(a[:, 6].max() * 1000)))
    print(f"swap at {len(picks)} observations in {args.t_lo}-{args.t_hi} s; the episode's nearest state is "
          f"{min(gaps):.3f}-{max(gaps):.3f} rad away (episode opens its gripper at tick {opens})")
    out["swap"]["state_gap_rad"] = [min(gaps), max(gaps)]
    for name, v in res.items():
        v = np.asarray(v)
        out["swap"][name] = s = {"plan_move_p50_rad": float(np.median(v[:, 0])), "moving_share": float(np.mean(v[:, 0] > 0.1)),
                                 "gripper_max_p50_mm": float(np.median(v[:, 1])), "opens_share": float(np.mean(v[:, 1] > 20))}
        print(f"  {name:30s} plan moves p50 {s['plan_move_p50_rad']:.3f} rad (>0.1: {s['moving_share']:.2f}); "
              f"widest gripper target p50 {s['gripper_max_p50_mm']:4.0f} mm (opens >20 mm: {s['opens_share']:.2f})")
    return out


def cmd_sheet(args) -> dict:
    import cv2

    rows, out = [], {}
    for run in sorted(p for p in Path(args.day).expanduser().iterdir() if (p / "inferences.jsonl").is_file()):
        meta, inf, _ = load_run(run)
        if not inf or meta.get("mode") == "replay" or (args.masked_only and "unmasked" in run_name(meta)):
            continue
        g = np.asarray([r["state"][6] for r in inf])
        i = grasp_index(g)
        if i is None:
            continue
        r = inf[max(i - 2, 0)]
        out[run.name] = {"run": run_name(meta), "t": r["t_obs"], "end_z_mm": r["end_pose"][2], "gripper_mm": r["state"][6] * 1000}
        imgs = [cv2.imread(str(run / "frames" / f"{r['seq']}_{c}.jpg")) for c in ("front", "wrist")]
        text = f"{run.name[:6]} {run_name(meta).replace('single__piper__', '')[:30]}  z {r['end_pose'][2]:.0f} mm"
        cv2.putText(imgs[0], text, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        rows.append(np.hstack(imgs))
    cv2.imwrite(str(args.image), cv2.resize(np.vstack(rows), None, fx=args.scale, fy=args.scale),
                [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"{len(rows)} runs -> {args.image}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("logs")
    s.add_argument("--day", required=True)
    s = sub.add_parser("demos")
    s = sub.add_parser("path")
    s.add_argument("--run", required=True)
    s.add_argument("--episode", type=int, required=True)
    s.add_argument("--every", type=int, default=3)
    s.add_argument("--max-obs", type=int, default=45)
    s = sub.add_parser("images")
    s.add_argument("--run", required=True)
    s.add_argument("--episode", type=int, required=True)
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--device", default="cuda")
    s.add_argument("--t-lo", type=float, default=5.0)
    s.add_argument("--t-hi", type=float, default=14.5)
    s.add_argument("--every", type=int, default=2)
    s = sub.add_parser("sheet")
    s.add_argument("--day", required=True)
    s.add_argument("--image", type=Path, required=True)
    s.add_argument("--masked-only", action="store_true")
    s.add_argument("--scale", type=float, default=0.5)
    for s in sub.choices.values():
        s.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
        s.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    result = {"logs": cmd_logs, "demos": cmd_demos, "path": cmd_path, "images": cmd_images, "sheet": cmd_sheet}[args.cmd](args)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
