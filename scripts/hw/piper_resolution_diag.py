"""Aim, first plans, smoothness and the return home of Piper rollouts (runs/2026-11-06_piper_resolution.md).

Companion to piper_rollout_diag.py, on the same logs. Each check is a subcommand:

    aim      per run: where the grape was, where the gripper closed, how far apart, and whether the
             grape ended on the plate. Grape tasks only: the grape's pixel in frame 0 maps to the
             table through a homography fitted on the demos (frame 0 pixel to end pose at the close)
    first    per run: the first chunk's entry against the arm, joint by joint, and the demos' start
    smooth   roughness of the arm against demos and replays, and what a chunk change does to the target
    home     where the rule "back at the start pose" fires, and what the plans ask for near home
    blend    in simulation on the logged plans: the target's roughness with a fade at each chunk change
    plans    in-process, no server: the first plan on recorded and live frames, states swapped
    seeds    in-process: the target's step at a chunk change with new or the same noise per request

    uv run python scripts/hw/piper_resolution_diag.py aim --day results/piper_rollouts/2026-11-06 --since 124400
    uv run python scripts/hw/piper_resolution_diag.py home --day results/piper_rollouts/2026-11-06 \
        --figure docs/assets/piper_home_restart.png --figure-run 124619
    uv run python scripts/hw/piper_resolution_diag.py plans --day results/piper_rollouts/2026-11-06 \
        --checkpoint results/single__piper__grape_plate_crop336__seed0/checkpoints/final.pt

--since / --until / --checkpoint-has select runs by folder name (HHMMSS) and checkpoint path, as in
piper_rollout_diag.py. Joint quantities are over J1-J6 in rad. Roughness is the RMS second difference
of a joint on the 15 Hz grid in rad/s^2, averaged over the joints.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from piper_rollout_diag import RATE, Source, first_close, load_run, run_name, select_runs  # noqa: E402

GRAPE_SOURCE = Path("~/piper_datasets/pick_grape").expanduser()
CROP_CONFIG = Path(__file__).resolve().parents[2] / "configs/embodiment/agilex_piper_crop.yaml"
FRONT = "observation.images.image"
WRIST = "observation.images.wrist_image"
KEYS = {"front": FRONT, "wrist": WRIST}
J2, J5 = 1, 4
TICKS_MIN = 50                  # below this a run was refused before it moved
ON_PLATE_PX = 25                # the grape's centre within this of the plate's: on the plate
SEEDS_RUN = "124916_rollout"    # the `seeds` check's run
HOME_NEAR_RAD = 0.25


def name(meta: dict) -> str:
    """A run's short policy name, marked when its chunks were blended."""
    short = run_name(meta).replace("single__piper__", "").replace("__seed0", "").replace("piper_daylight/", "daylight ")
    return short + (" +blend" if meta.get("blend_s") else "")


def grape_tasks(args) -> list[Path]:
    return [r for r in select_runs(args) if "grape" in run_name(load_run(r)[0])]


def table_mask(shape: tuple[int, int]) -> np.ndarray:
    polygon = np.asarray(yaml.safe_load(CROP_CONFIG.read_text())["observation"]["image_crops"]["front"], np.int32)
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    return mask


def largest_blob(img: np.ndarray, keep: np.ndarray) -> tuple[np.ndarray | None, int]:
    """The centre and area of the largest component of ``keep`` on the table, in frame pixels."""
    m = (keep.astype(np.uint8) * 255) & table_mask(img.shape[:2])
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, stats, centres = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None, 0
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return centres[i], int(stats[i, cv2.CC_STAT_AREA])


def grape_blob(img: np.ndarray) -> tuple[np.ndarray | None, int]:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return largest_blob(img, (h >= 110) & (h <= 160) & (s > 70) & (v > 30) & (v < 190))


def plate_blob(img: np.ndarray) -> tuple[np.ndarray | None, int]:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return largest_blob(img, (h >= 85) & (h <= 105) & (s > 110) & (v > 90))


def fit_table(source: Path) -> tuple[np.ndarray, dict]:
    """The homography from a demo's frame 0 grape pixel to the end pose at its close, mm."""
    src = Source(source)
    px, xy = [], []
    for e in range(len(src.episodes)):
        r = src.rows(e)
        img = cv2.imdecode(np.frombuffer(src.jpegs(e, FRONT, [0])[0], np.uint8), cv2.IMREAD_COLOR)
        centre, area = grape_blob(img)
        close = first_close(src.frames["state"][r, 6] * 1000)
        if centre is not None and area >= 100 and close is not None:
            px.append(centre)
            xy.append(src.eef[r.start + close, :2] * 1000)
    px, xy = np.asarray(px, float), np.asarray(xy, float)
    H, inliers = cv2.findHomography(px, xy, cv2.RANSAC, 15.0)
    residual = np.linalg.norm(to_table(H, px) - xy, axis=1)
    return H, {"demos": len(px), "inliers": int(inliers.sum()), "residual_p50_mm": float(np.median(residual)),
               "residual_p90_mm": float(np.percentile(residual, 90))}


def to_table(H: np.ndarray, px: np.ndarray) -> np.ndarray:
    v = (H @ np.c_[px, np.ones(len(px))].T).T
    return v[:, :2] / v[:, 2:]


def at(chunk: np.ndarray, p: float) -> np.ndarray:
    p = min(max(p, 0.0), 14.999)
    i = int(p)
    return chunk[i] + (chunk[i + 1] - chunk[i]) * (p - i)


def roughness(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.diff(x, 2, axis=0) * RATE * RATE) ** 2, axis=0)).mean())


def on_grid(t: np.ndarray, x: np.ndarray) -> np.ndarray:
    grid = np.arange(t[0], t[-1], 1 / RATE)
    return np.stack([np.interp(grid, t, x[:, j]) for j in range(x.shape[1])], 1)


def plans(inf: list[dict]) -> list[dict]:
    return [r for r in inf if r.get("actions")]


# ---- subcommands -------------------------------------------------------------------------

def cmd_aim(args) -> dict:
    H, fit = fit_table(args.source)
    print(f"homography on {fit['demos']} demos ({fit['inliers']} inliers): residual p50 {fit['residual_p50_mm']:.1f} mm, "
          f"p90 {fit['residual_p90_mm']:.1f}")
    rows, tiles, groups = {}, [], {}
    for run in grape_tasks(args):
        meta, inf, t = load_run(run)
        if not inf:
            continue
        first = cv2.imread(str(run / "frames" / f"{inf[0]['seq']}_front.jpg"))
        last = cv2.imread(str(run / "frames" / f"{inf[-1]['seq']}_front.jpg"))
        centre, area = grape_blob(first)
        grape = to_table(H, centre[None])[0] if centre is not None and area >= 100 else None
        close = first_close(t["gripper"] * 1000) if len(t["t"]) >= TICKS_MIN else None
        where = None if close is None else t["end_pose"][close, :2]
        miss = None if grape is None or where is None else float(np.linalg.norm(where - grape))
        end_grape, end_area = grape_blob(last)
        end_plate, plate_area = plate_blob(last)
        gap = None if end_grape is None or end_plate is None else float(np.linalg.norm(end_grape - end_plate))
        rows[run.name] = row = {"policy": name(meta), "miss_mm": miss, "end_gap_px": gap, "end_grape_px": end_area,
                                "end_plate_px": plate_area, "reason": meta.get("reason"),
                                "duration_s": meta.get("duration_s"), "kind": meta.get("kind")}
        placed = gap is not None and gap < ON_PLATE_PX
        print(f"{run.name[:6]} {row['policy']:34s} miss {'-' if miss is None else format(miss, '4.0f')} mm  "
              f"end gap {'-' if gap is None else format(gap, '4.0f')} px  grape {end_area:4d} plate {plate_area:4d}"
              f"  {'on plate' if placed else ''}  {str(row['reason'])[:34]}")
        g = groups.setdefault(row["policy"], {"runs": 0, "miss": [], "placed": 0})
        g["runs"] += 1
        g["placed"] += placed
        if miss is not None:
            g["miss"].append(miss)
        if args.image:
            tile = last[190:480, 150:600].copy()
            cv2.putText(tile, f"{run.name[:6]} {row['policy'][:30]}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 1, cv2.LINE_AA)
            tiles.append(cv2.resize(tile, None, fx=args.scale, fy=args.scale))
    for k, g in groups.items():
        m = g["miss"]
        print(f"  {k:34s} runs {g['runs']:2d}  closed {len(m):2d}  miss p50 {np.median(m) if m else float('nan'):4.0f} mm  "
              f"on plate at the last frame {g['placed']}")
    if args.image and tiles:
        rows_img = [np.hstack(tiles[i:i + 4] + [np.zeros_like(tiles[0])] * (4 - len(tiles[i:i + 4])))
                    for i in range(0, len(tiles), 4)]
        cv2.imwrite(str(args.image), np.vstack(rows_img), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return {"homography": fit, "runs": rows}


def cmd_first(args) -> dict:
    rows, groups = {}, {}
    for run in select_runs(args):
        meta, inf, _ = load_run(run)
        if not plans(inf):
            continue
        r = plans(inf)[0]
        state = np.asarray(r["state"], float)
        p = (r["t_reply"] - r["t_obs"]) * RATE
        entry = at(np.asarray(r["actions"], float), p) - state
        rows[run.name] = {"policy": name(meta), "p": p, "entry_minus_arm_rad": entry[:6].tolist(),
                          "reason": meta.get("reason")}
        groups.setdefault(name(meta), []).append(entry[:6])
    for k, v in groups.items():
        v = np.asarray(v)
        print(f"{k:42s} n {len(v):2d}  J5 entry p50 {np.median(v[:, 4]):+.3f} max {v[:, 4].max():+.3f}  "
              f"J3 {np.median(v[:, 2]):+.3f}  J4 {np.median(v[:, 3]):+.3f}")
    f = Source(args.source).frames
    ep = f["episode_index"]
    start, idle, ahead = [], [], []
    for e in np.unique(ep):
        a, s = f["action"][ep == e, :6], f["state"][ep == e, :6]
        start.append(s[0])
        idle.append(int(np.argmax(np.abs(a - a[0]).max(1) > 0.02)))
        ahead.append(a[1] - s[0])
    print(f"demos: J5 at tick 1 minus the start state p50 {np.median(np.asarray(ahead)[:, 4]):+.3f}, "
          f"max {np.abs(np.asarray(ahead)[:, 4]).max():.3f}; first move after p50 {np.median(idle) / RATE:.2f} s "
          f"({min(idle) / RATE:.2f}-{max(idle) / RATE:.2f})")
    return rows


def cmd_smooth(args) -> dict:
    f = Source(args.source).frames
    ep = f["episode_index"]
    demo = float(np.median([roughness(f["state"][ep == e, :6]) for e in np.unique(ep)]))
    print(f"demos, measured: {demo:.2f}")
    replays = []
    for day in args.replay_day:
        for run in sorted(Path(day).expanduser().glob("*_replay")):
            _, _, t = load_run(run)
            if len(t["t"]) >= 400:
                replays.append(roughness(on_grid(t["t"], t["joints"])))
    if replays:
        print(f"open-loop replays (n {len(replays)}): {np.median(replays):.2f}")
    rough, steps, along, near, far = {}, [], [], [], []
    for run in select_runs(args):
        meta, inf, t = load_run(run)
        if len(t["t"]) < 400:
            continue
        end = min(t["t"][-1], args.first_s)
        ok = t["t"] < end
        x = on_grid(t["t"][ok], t["joints"][ok])
        rough.setdefault(name(meta), []).append(roughness(x))
        used = [r for r in plans(inf) if r["t_reply"] < end]
        for a, b in zip(used[:-1], used[1:]):
            po, pn = (b["t_reply"] - a["t_obs"]) * RATE, (b["t_reply"] - b["t_obs"]) * RATE
            if po > 14:
                continue
            A, B = np.asarray(a["actions"])[:, :6], np.asarray(b["actions"])[:, :6]
            step = at(B, pn) - at(A, po)
            steps.append(np.abs(step).max())
            motion = A[min(int(po) + 2, 15)] - A[max(int(po) - 2, 0)]
            if np.linalg.norm(motion) > 0.06:
                along.append(float(step @ motion / np.linalg.norm(motion)))
        dd = np.diff(x, 2, axis=0) * RATE * RATE
        grid = np.arange(t["t"][ok][0], t["t"][ok][-1], 1 / RATE)[1:-1]
        swaps = np.asarray([r["t_reply"] for r in used])
        by_swap = np.asarray([np.abs(swaps - g).min() < 1.5 / RATE for g in grid[:len(dd)]])
        near.append((dd[by_swap] ** 2).mean())
        far.append((dd[~by_swap] ** 2).mean())
    for k, v in rough.items():
        print(f"{k:42s} n {len(v):2d}  roughness p50 {np.median(v):.2f}")
    along = np.asarray(along)
    print(f"chunk changes: {len(steps)}; worst joint step p50 {np.median(steps):.3f}, p95 {np.percentile(steps, 95):.3f} rad; "
          f"while moving ({len(along)}): backwards in {100 * (along < 0).mean():.0f} %, along the motion p50 {np.median(along):+.3f} rad")
    print(f"squared roughness near chunk changes against elsewhere: {np.median(np.asarray(near) / np.asarray(far)):.2f}x")
    return {"demos": demo, "replays": replays, "roughness": rough}


def cmd_home(args) -> dict:
    out = {}
    for run in select_runs(args):
        meta, inf, t = load_run(run)
        if len(t["t"]) < TICKS_MIN:
            continue
        home = np.radians(np.asarray(meta["start_pose_deg_mm"][:6], float))
        off = np.abs(t["joints"] - home).max(1)
        hit = np.where((np.maximum.accumulate(off) > args.leave) & (off < args.tolerance))[0]
        if not hit.size:
            continue
        i = int(hit[0])
        k = max(i - 20, 0)
        speed = float(np.abs(t["joints"][i] - t["joints"][k]).max() / max(t["t"][i] - t["t"][k], 1e-3))
        close, away = 0, 0
        for r in plans(inf):
            if r["t_obs"] < t["t"][i]:
                continue
            a, s = np.asarray(r["actions"])[:, :6], np.asarray(r["state"])[:6]
            now = np.abs(s - home).max()
            if now < HOME_NEAR_RAD:
                close += 1
                away += int(np.abs(a[15] - home).max() > now + 0.02)
        out[run.name] = {"policy": name(meta), "t_s": float(t["t"][i]), "end_s": float(t["t"][-1]), "speed_rad_s": speed,
                         "gripper_mm": float(t["gripper"][i] * 1000), "plans_near_home": close, "pointing_away": away}
        print(f"{run.name[:6]} {name(meta):34s} fires at {t['t'][i]:5.1f} s of {t['t'][-1]:5.1f} s, {speed:.2f} rad/s, gripper "
              f"{t['gripper'][i] * 1000:3.0f} mm; {away} of {close} later plans near home point away  {meta.get('kind')}")
    if args.figure:
        figure_home(Path(args.day).expanduser() / f"{args.figure_run}_rollout", args.figure)
    return out


def figure_home(run: Path, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    meta, _, t = load_run(run)
    home = np.radians(np.asarray(meta["start_pose_deg_mm"][:6], float))
    ink, muted, grid = "#0b0b0b", "#52514e", "#e7e6e1"
    fig, ax = plt.subplots(figsize=(9, 3.4), dpi=120, facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for joint, colour, label in ((J5, "#2a78d6", "J5"), (J2, "#d9772b", "J2")):
        y = t["joints"][:, joint] - home[joint]
        ax.plot(t["t"], y, color=colour, lw=1.6, label=label, solid_joinstyle="round", solid_capstyle="round")
    ax.axhline(0, color=muted, lw=0.8)
    ax.set_xlabel("seconds since Go", color=muted)
    ax.set_ylabel("rad from the start pose", color=muted)
    ax.set_xlim(t["t"][0], t["t"][-1])
    ax.tick_params(colors=muted, length=0)
    ax.grid(axis="y", color=grid, lw=0.8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    ax.legend(frameon=False, loc="upper right", ncol=2, labelcolor=ink)
    ax.set_title(f"Run {run.name[:6]}: after reaching the start pose, plan after plan sets off and comes back",
                 color=ink, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path)


def cmd_blend(args) -> dict:
    res: dict[float, list[float]] = {s: [] for s in args.lengths}
    for run in select_runs(args):
        meta, inf, _ = load_run(run)
        used = plans(inf)
        if meta.get("blend_s") or len(used) < 12:
            continue
        used = used[:30]
        ch = [(r["t_reply"], r["t_obs"], np.asarray(r["actions"])[:, :6]) for r in used]
        times = np.arange(ch[0][0], ch[-1][0], 1 / 200)
        for length in args.lengths:
            x, k = [], 0
            for now in times:
                while k + 1 < len(ch) and ch[k + 1][0] <= now:
                    k += 1
                y = at(ch[k][2], (now - ch[k][1]) * RATE)
                if length > 0 and k > 0 and now - ch[k][0] < length:
                    a = (now - ch[k][0]) / length
                    a = a * a * (3 - 2 * a)
                    y = at(ch[k - 1][2], (now - ch[k - 1][1]) * RATE) * (1 - a) + y * a
                x.append(y)
            res[length].append(roughness(np.asarray(x)[:: int(200 / RATE)]))
    for length, v in res.items():
        print(f"fade {length:.2f} s: target roughness p50 {np.median(v):.2f} (n {len(v)})")
    return {str(k): v for k, v in res.items()}


def chunk_for(svc, state, images: dict[str, bytes], seed: int) -> np.ndarray:
    req = {"instruction": svc.task.language, "state": [float(x) for x in state], "seed": seed,
           "images": {c: base64.b64encode(b).decode() for c, b in images.items()}}
    return np.asarray(svc.infer(req)["actions"])


def service(checkpoint: Path, device: str):
    import torch

    from flowcl.deploy.piper_server import PiperPolicyService

    torch.set_num_threads(args_threads)
    return PiperPolicyService(checkpoint, device=device)


def cmd_plans(args) -> dict:
    src = Source(args.source)
    svc = service(args.checkpoint, args.device)
    f = src.frames
    recorded = {e: (f["state"][src.rows(e).start], {c: src.jpegs(e, k, [0])[0] for c, k in KEYS.items()}) for e in args.episodes}
    live = {}
    for r in args.live_runs:
        d = Path(args.day).expanduser() / f"{r}_rollout"
        q = json.loads(open(d / "inferences.jsonl").readline())
        live[r] = (np.asarray(q["state"], float), {c: (d / "frames" / f"{q['seq']}_{c}.jpg").read_bytes() for c in KEYS})

    def entry(state, images, seed):
        a = chunk_for(svc, state, images, seed)
        return (a[0] * 0.1 + a[1] * 0.9 - np.asarray(state))[:6]     # the entry at p = 0.9, as on the robot

    out = {k: [] for k in ("recorded frames + recorded state", "live frames + live state", "live frames + recorded state",
                           "recorded frames + live state", "live front + recorded wrist", "recorded front + live wrist")}
    for state, images in recorded.values():
        for s in (0, 1):
            out["recorded frames + recorded state"].append(entry(state, images, s))
    for state, images in live.values():
        for s in (0, 1):
            out["live frames + live state"].append(entry(state, images, s))
    for (lstate, limg), (rstate, rimg) in zip(live.values(), recorded.values()):
        for s in (0, 1):
            out["live frames + recorded state"].append(entry(rstate, limg, s))
            out["recorded frames + live state"].append(entry(lstate, rimg, s))
            out["live front + recorded wrist"].append(entry(rstate, {"front": limg["front"], "wrist": rimg["wrist"]}, s))
            out["recorded front + live wrist"].append(entry(rstate, {"front": rimg["front"], "wrist": limg["wrist"]}, s))
    summary = {}
    for k, v in out.items():
        v = np.asarray(v)
        summary[k] = {"j5_p50": float(np.median(v[:, 4])), "j5_min": float(v[:, 4].min()), "j5_max": float(v[:, 4].max())}
        print(f"{k:34s} J5 entry p50 {np.median(v[:, 4]):+.3f} ({v[:, 4].min():+.3f}..{v[:, 4].max():+.3f})")
    return summary


def cmd_seeds(args) -> dict:
    svc = service(args.checkpoint, args.device)
    d = Path(args.day).expanduser() / SEEDS_RUN
    inf = plans([json.loads(line) for line in open(d / "inferences.jsonl")])

    def chunk(q, seed):
        images = {c: (d / "frames" / f"{q['seq']}_{c}.jpg").read_bytes() for c in KEYS}
        return chunk_for(svc, q["state"], images, seed)[:, :6]

    same, new = [], []
    for i in np.linspace(0, min(len(inf) - 2, 34), 14).astype(int):
        a, b = inf[i], inf[i + 1]
        po, pn = (b["t_reply"] - a["t_obs"]) * RATE, (b["t_reply"] - b["t_obs"]) * RATE
        for s in (0, 1):
            A = chunk(a, s)
            same.append(at(chunk(b, s), pn) - at(A, po))
            new.append(at(chunk(b, s + 10), pn) - at(A, po))
    res = {}
    for k, v in (("new seed each request", new), ("same seed each request", same)):
        w = np.abs(np.asarray(v)).max(1)
        res[k] = {"p50": float(np.median(w)), "p90": float(np.percentile(w, 90))}
        print(f"{k:24s} worst joint step p50 {np.median(w):.3f}, p90 {np.percentile(w, 90):.3f} rad (n {len(w)})")
    return res


args_threads = 6


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    cmds = {"aim": cmd_aim, "first": cmd_first, "smooth": cmd_smooth, "home": cmd_home, "blend": cmd_blend,
            "plans": cmd_plans, "seeds": cmd_seeds}
    for key in cmds:
        s = sub.add_parser(key)
        s.add_argument("--day", required=True)
        s.add_argument("--since", default=None)
        s.add_argument("--until", default=None)
        s.add_argument("--checkpoint-has", default=None)
        s.add_argument("--source", type=Path, default=GRAPE_SOURCE)
        s.add_argument("--out", type=Path, default=None)
        if key == "aim":
            s.add_argument("--image", type=Path, default=None)
            s.add_argument("--scale", type=float, default=0.8)
        if key == "smooth":
            s.add_argument("--replay-day", nargs="*", default=["results/piper_rollouts/2026-11-03", "~/piper_rollouts/2026-11-02"])
            s.add_argument("--first-s", type=float, default=16.0, help="only the first seconds of a run, before any return home")
        if key == "home":
            s.add_argument("--leave", type=float, default=0.5)
            s.add_argument("--tolerance", type=float, default=0.06)
            s.add_argument("--figure", type=Path, default=None)
            s.add_argument("--figure-run", default="124619")
        if key == "blend":
            s.add_argument("--lengths", nargs="*", type=float, default=[0.0, 2 / 15, 4 / 15, 6 / 15])
        if key in ("plans", "seeds"):
            s.add_argument("--checkpoint", type=Path, required=True)
            s.add_argument("--device", default="cpu")
        if key == "plans":
            s.add_argument("--episodes", nargs="*", type=int, default=[0, 10, 20, 30, 39])
            s.add_argument("--live-runs", nargs="*", default=["122255", "122418", "122604", "122611", "122621", "122635"])
    args = parser.parse_args()
    result = cmds[args.cmd](args)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
