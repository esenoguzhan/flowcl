"""A Piper rollout session against its demos: outcomes, scenes and light (24 Nov record).

Complements piper_rollout_diag.py (logs, grip, sheet), which reads the same logs. Three checks:

    outcomes  per rollout: what the gripper did (hover, miss, grasp, opened and never closed),
              where it closed (end z, the grape's size in the wrist frame at the close), how
              long it stayed at the start pose, how high it sat afterwards; and the same
              quantities for the demos
    scenes    the grape and the plate in each rollout's first front frame and in each demo's
              frame 0, by colour: how far from the nearest demo, which rollouts share a scene,
              and the table's light (mean grey, sharp edges, what a linear ramp leaves)
    hovers    for runs that never opened: what the wrist camera shows part-way through

    uv run python scripts/hw/piper_session_diag.py outcomes --day results/piper_rollouts/2026-11-24 \
        --source ~/piper_datasets/new_grape --out results/piper_rollouts/2026-11-24/diag/outcomes.json
    uv run python scripts/hw/piper_session_diag.py scenes --day ... --source ... --image light.jpg
    uv run python scripts/hw/piper_session_diag.py hovers --day ... --image hovers.jpg

Units: end pose in mm, gripper in mm, time in s from the run's start. Colour thresholds are
for the purple grape and the blue plate of the new_grape task (OpenCV HSV).
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from flowcl.data.piper_adapter import _read_parquets, read_episode_table  # noqa: E402

OPEN_MM = 20.0        # the gripper counts as opened above this
CLOSE_FROM_MM = 60.0  # a close starts once the gripper was wider than this (session 1's demos open to 66-77 mm)
CLOSE_DROP_MM = 10.0  # ... at the first tick this far below its widest opening so far
HELD_MM = 30.0        # still this wide 1-2 s after the close: something is held
LEFT_HOME_RAD = 0.05  # the arm has left the start pose once a joint is this far from it
GRAPE_HSV = ((128, 95, 40), (165, 255, 255))
PLATE_HSV = ((88, 130, 60), (108, 255, 255))
TABLE = (slice(220, 480), slice(210, 640))   # front frame rows, cols: table right of and below the arm at home


# ---- reading -------------------------------------------------------------------------------

def rollouts(day: Path) -> list[Path]:
    return sorted(p for p in Path(day).glob("*_rollout") if (p / "ticks.npz").exists())


def load(run: Path) -> tuple[dict, list[dict], dict]:
    meta = json.loads((run / "meta.json").read_text())
    inf = [json.loads(line) for line in (run / "inferences.jsonl").read_text().splitlines() if line.strip()]
    return meta, inf, dict(np.load(run / "ticks.npz"))


def policy_name(meta: dict) -> str:
    ckpt = meta["server_metadata"]["checkpoint"]
    return ckpt.split("single__piper__")[-1].replace("/checkpoints/", ":").removesuffix(".pt")


def frame_seqs(run: Path, camera: str) -> list[int]:
    return sorted(int(re.search(r"(\d+)_" + camera, p).group(1)) for p in glob.glob(str(run / "frames" / f"*_{camera}.jpg")))


def front_mask(meta: dict, hw=(480, 640)) -> np.ndarray:
    """1 where the policy sees the front frame, 0 under the operator mask."""
    m = np.ones(hw, np.uint8)
    for poly in meta["server_metadata"]["image_masks"].get("front", []):
        cv2.fillPoly(m, [np.asarray(poly, np.int32)], 0)
    return m


def demo_table(source: Path) -> dict[str, np.ndarray]:
    t = _read_parquets(sorted((Path(source) / "data").glob("chunk-*/file-*.parquet")))
    col = lambda k: np.asarray(t.column(k).to_pylist())
    return {"state": col("observation.state").astype(np.float64), "eef": col("observation.eef").astype(np.float64),
            "t": col("timestamp").astype(np.float64), "ep": col("episode_index").astype(np.int64)}


def demo_first_frames(source: Path):
    """(episode, BGR frame 0 of the front camera) for every demo."""
    key = "observation.images.image"
    for row in read_episode_table(source):
        video = Path(source) / "videos" / key / f"chunk-{row[f'videos/{key}/chunk_index']:03d}" / \
            f"file-{row[f'videos/{key}/file_index']:03d}.mp4"
        cap = cv2.VideoCapture(str(video))
        cap.set(cv2.CAP_PROP_POS_MSEC, float(row[f"videos/{key}/from_timestamp"]) * 1000 + 30)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"{video}: no frame for episode {row['episode_index']}")
        yield row["episode_index"], frame


# ---- measures ------------------------------------------------------------------------------

def blob(bgr: np.ndarray, hsv_range, mask: np.ndarray | None = None) -> tuple[float, float, int]:
    """Centre x, y and area (px) of the largest blob in the colour range, NaN when there is none."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array(hsv_range[0]), np.array(hsv_range[1]))
    if mask is not None:
        m = m * mask
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, stats, cent = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8))
    if n <= 1:
        return float("nan"), float("nan"), 0
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return float(cent[k][0]), float(cent[k][1]), int(stats[k, cv2.CC_STAT_AREA])


def table_region(demo_frames: list[np.ndarray], mask: np.ndarray) -> np.ndarray:
    """The whole visible table: pixels brighter than 110 grey in the median of the demos' frame 0
    (the arm, the base and the floor are darker), outside the operator mask, eroded by 15 px."""
    median = np.median(np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in demo_frames]), axis=0)
    region = ((median > 110) & (mask > 0)).astype(np.uint8)
    return cv2.erode(region, np.ones((31, 31), np.uint8)) > 0


def table_light(bgr: np.ndarray, region: np.ndarray | None = None) -> dict:
    """A table region without the objects (default TABLE): mean grey; the share of pixels on a sharp
    light edge (Sobel > 40 on the 5 px blurred grey); and the relative std left after fitting
    grey = a + bx + cy to the 15 px blurred grey (the daylight augmentation multiplies by such a ramp)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    objects = cv2.dilate((hsv[..., 1] > 80).astype(np.uint8), np.ones((41, 41), np.uint8))
    if region is None:
        keep = np.zeros(bgr.shape[:2], bool)
        keep[TABLE] = True
    else:
        keep = region.copy()
    keep &= objects == 0
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g5 = cv2.GaussianBlur(grey, (5, 5), 0)
    edges = np.hypot(cv2.Sobel(g5, cv2.CV_32F, 1, 0), cv2.Sobel(g5, cv2.CV_32F, 0, 1))
    g15 = cv2.GaussianBlur(grey, (0, 0), 15)
    ys, xs = np.mgrid[0:grey.shape[0], 0:grey.shape[1]]
    a = np.stack([np.ones(keep.sum()), xs[keep], ys[keep]], 1)
    v = g15[keep]
    coef, *_ = np.linalg.lstsq(a, v, rcond=None)
    return {"grey": float(grey[keep].mean()), "sharp_edge_share": float((edges[keep] > 40).mean()),
            "ramp_residual": float((v - a @ coef).std() / v.mean())}


def first_close(gripper_mm: np.ndarray) -> int | None:
    """Where the first close starts: once the gripper was wider than CLOSE_FROM_MM, the first tick
    CLOSE_DROP_MM below its widest opening so far; None when it never closes."""
    wide = np.where(gripper_mm > CLOSE_FROM_MM)[0]
    if not len(wide):
        return None
    g = gripper_mm[wide[0]:]
    below = np.where(g < np.maximum.accumulate(g) - CLOSE_DROP_MM)[0]
    return int(wide[0] + below[0]) if len(below) else None


def held_width(gripper_mm: np.ndarray, t: np.ndarray, i_close: int) -> float:
    w = gripper_mm[(t > t[i_close] + 1.0) & (t < t[i_close] + 2.0)]
    return float(np.median(w)) if len(w) else float("nan")


def left_home(joints: np.ndarray, t: np.ndarray) -> float:
    moved = np.where(np.abs(joints - joints[0]).max(1) > LEFT_HOME_RAD)[0]
    return float(t[moved[0]]) if len(moved) else float("nan")


def pct(a, q=(0, 5, 25, 50, 75, 95, 100)) -> list[float]:
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return [round(float(x), 3) for x in np.percentile(a, q)] if len(a) else []


# ---- commands ------------------------------------------------------------------------------

def cmd_outcomes(args) -> dict:
    out = {"rollouts": {}, "demos": {}}
    for run in rollouts(args.day):
        meta, inf, k = load(run)
        t, g, ep = k["t"], k["gripper"] * 1000.0, k["end_pose"]
        i_low = int(np.argmin(ep[:, 2]))
        after = ep[t > t[i_low] + 1.0, 2]
        row = {"policy": policy_name(meta), "duration_s": meta["duration_s"], "ended": meta["kind"],
               "left_home_s": left_home(k["joints"], t), "gripper_max_mm": float(g.max()),
               "start_z_mm": float(ep[0, 2]), "lowest_z_mm": float(ep[i_low, 2]), "lowest_t_s": float(t[i_low]),
               "z_after_lowest_p50_mm": float(np.median(after)) if len(after) else None}
        i_c = first_close(g)
        held = held_width(g, t, i_c) if i_c is not None else float("nan")
        if g.max() <= OPEN_MM:
            row["outcome"] = "hover"
        elif i_c is None or not held < CLOSE_FROM_MM:      # never closed, or closed a little and opened again
            row["outcome"] = "opened, never closed"
        else:
            row["outcome"] = "grasp" if held > HELD_MM else "miss"
            # the wrist frame of the last observation at or before the close
            have = set(frame_seqs(run, "wrist"))
            seen = [i for i in inf if i["seq"] in have and i["t_obs"] <= t[i_c]]
            wrist = {}
            if seen:
                s = max(seen, key=lambda i: i["t_obs"])
                x, y, area = blob(cv2.imread(str(run / "frames" / f"{s['seq']}_wrist.jpg")), GRAPE_HSV)
                wrist = {"frame_before_close_s": round(t[i_c] - s["t_obs"], 3), "grape_x": x, "grape_y": y,
                         "grape_area_px": area}
            row.update({"close_t_s": float(t[i_c]), "close_z_mm": float(ep[i_c, 2]), "held_mm": held,
                        "lowest_z_before_close_mm": float(ep[:i_c + 1, 2].min()), "wrist_at_close": wrist})
        out["rollouts"][run.name] = row
        print(f"{run.name[:6]} {row['policy']:28s} {row['outcome']:20s} left home {row['left_home_s']:4.1f} s | "
              f"lowest z {row['lowest_z_mm']:5.1f} at {row['lowest_t_s']:4.1f} s, then p50 "
              f"{row['z_after_lowest_p50_mm'] or float('nan'):5.1f}"
              + (f" | close {row['close_t_s']:4.1f} s at z {row['close_z_mm']:5.1f}, held {row['held_mm']:4.1f} mm, "
                 f"grape {row['wrist_at_close'].get('grape_area_px', 0):5d} px in the wrist frame"
                 if "close_z_mm" in row else ""))

    d = demo_table(args.source)
    close_z, held, open_t, leave, carry = [], [], [], [], []
    for e in np.unique(d["ep"]):
        r = d["ep"] == e
        g, z, t, j = d["state"][r, 6] * 1000.0, d["eef"][r, 2] * 1000.0, d["t"][r], d["state"][r, :6]
        leave.append(left_home(j, t))
        opened = np.where(g > OPEN_MM)[0]
        if len(opened):
            open_t.append(t[opened[0]])
        i_c = first_close(g)
        if i_c is not None:
            close_z.append(z[i_c])
            held.append(held_width(g, t, i_c))
        holding = (g > 35) & (g < 65)
        if holding.sum() > 15:
            carry.append(np.percentile(z[holding], 90))
    close_z, held = np.array(close_z), np.array(held)
    out["demos"] = {"episodes": int(len(np.unique(d["ep"]))), "closes": int(len(close_z)),
                    "close_z_mm": pct(close_z), "held_mm": pct(held), "open_t_s": pct(open_t),
                    "left_home_s": pct(leave), "carry_z_p90_mm": pct(carry),
                    "closes_at_or_above_107_mm": int((close_z >= 107).sum()),
                    "of_those_held": int(((close_z >= 107) & (held > HELD_MM)).sum())}
    print("demos (percentiles 0/5/25/50/75/95/100):")
    for key, v in out["demos"].items():
        print(f"  {key}: {v}")
    return out


def cmd_scenes(args) -> dict:
    runs = rollouts(args.day)
    mask = front_mask(load(runs[0])[0])
    demos, demo_imgs = [], {}
    for e, frame in demo_first_frames(args.source):
        gx, gy, _ = blob(frame, GRAPE_HSV, mask)
        px, py, _ = blob(frame, PLATE_HSV, mask)
        demos.append({"episode": e, "grape": [gx, gy], "plate": [px, py]})
        demo_imgs[e] = frame
    # the light, over two regions: the lower table (TABLE, fixed before looking) and the whole
    # visible table (table_region, added after the frames showed the stripes mostly above TABLE)
    whole = table_region(list(demo_imgs.values()), mask)
    light = lambda frame: {**{f"lower_{k}": v for k, v in table_light(frame).items()},
                           **{f"whole_{k}": v for k, v in table_light(frame, whole).items()}}
    for d in demos:
        d.update(light(demo_imgs[d["episode"]]))
    dg = np.array([d["grape"] for d in demos])
    dp = np.array([d["plate"] for d in demos])
    rows, imgs = {}, {}
    for run in runs:
        meta, _, _ = load(run)
        frame = cv2.imread(str(run / "frames" / f"{frame_seqs(run, 'front')[0]}_front.jpg"))
        gx, gy, ga = blob(frame, GRAPE_HSV, mask)
        px, py, _ = blob(frame, PLATE_HSV, mask)
        to_g = np.hypot(dg[:, 0] - gx, dg[:, 1] - gy)
        to_pair = np.hypot(to_g, np.hypot(dp[:, 0] - px, dp[:, 1] - py))
        rows[run.name] = {"policy": policy_name(meta), "grape": [gx, gy], "grape_area_px": ga, "plate": [px, py],
                          "grape_to_plate_px": float(np.hypot(gx - px, gy - py)),
                          "nearest_demo_grape_px": float(np.nanmin(to_g)), "demo_grapes_within_40_px": int((to_g < 40).sum()),
                          "nearest_demo_pair_px": float(np.nanmin(to_pair)), **light(frame)}
        imgs[run.name] = frame
    # scenes: consecutive runs whose grape and plate both moved less than 15 px
    scene, last = 0, None
    for name, r in rows.items():
        if last is not None and max(np.hypot(*np.subtract(r["grape"], last["grape"])),
                                    np.hypot(*np.subtract(r["plate"], last["plate"]))) >= 15:
            scene += 1
        r["scene"] = scene + 1
        last = r
    for name, r in rows.items():
        print(f"{name[:6]} scene {r['scene']:2d} {r['policy']:28s} grape {r['grape'][0]:4.0f},{r['grape'][1]:4.0f} "
              f"plate {r['plate'][0]:4.0f},{r['plate'][1]:4.0f} | nearest demo grape {r['nearest_demo_grape_px']:4.0f} px, "
              f"{r['demo_grapes_within_40_px']:2d} within 40 | pair {r['nearest_demo_pair_px']:4.0f} | "
              f"whole table: grey {r['whole_grey']:5.1f} edges {r['whole_sharp_edge_share']:.3f} "
              f"ramp residual {r['whole_ramp_residual']:.3f}")
    gp = np.hypot(dg[:, 0] - dp[:, 0], dg[:, 1] - dp[:, 1])
    summary = {key: {"demos": pct([d[key] for d in demos]), "rollouts": pct([r[key] for r in rows.values()])}
               for key in [f"{part}_{k}" for part in ("lower", "whole") for k in ("grey", "sharp_edge_share", "ramp_residual")]}
    summary["whole_table_px"] = int(whole.sum())
    summary["grape_to_plate_px"] = {"demos": pct(gp), "rollouts": pct([r["grape_to_plate_px"] for r in rows.values()])}
    for key, v in summary.items():
        if isinstance(v, dict):
            print(f"{key} (0/5/25/50/75/95/100): demos {v['demos']}  rollouts {v['rollouts']}")
    print(f"whole table: {summary['whole_table_px']} px of the 640x480 front frame")
    if args.image:
        cv2.imwrite(str(args.image.with_name(args.image.stem + "_region.jpg")),
                    cv2.addWeighted(demo_imgs[args.demo_episodes[0]], 0.6,
                                    np.dstack([whole.astype(np.uint8) * 255] * 3), 0.4, 0))
    if args.image:
        pick_d = [e for e in args.demo_episodes if e in demo_imgs]
        names = list(imgs)
        pick_r = [names[int(i * (len(names) - 1) / (len(pick_d) - 1))] for i in range(len(pick_d))]
        def tile(img, text):
            img = cv2.resize(img, (320, 240))
            cv2.rectangle(img, (0, 0), (320, 18), (0, 0, 0), -1)
            cv2.putText(img, text, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            return img
        top = np.hstack([tile(demo_imgs[e].copy(), f"demo ep {e}, frame 0") for e in pick_d])
        bottom = np.hstack([tile(imgs[n].copy(), f"rollout {n[:6]}, first frame") for n in pick_r])
        cv2.imwrite(str(args.image), np.vstack([top, bottom]), [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"-> {args.image}")
    return {"rollouts": rows, "demos": demos, "summary": summary}


def cmd_hovers(args) -> dict:
    out, tiles = {}, []
    for run in rollouts(args.day):
        meta, inf, k = load(run)
        if (k["gripper"] * 1000.0).max() > OPEN_MM:
            continue
        seqs = frame_seqs(run, "front")
        s = seqs[int(args.at * (len(seqs) - 1))]
        t_obs = {i["seq"]: i["t_obs"] for i in inf}.get(s, float("nan"))
        i = min(int(np.searchsorted(k["t"], t_obs)), len(k["t"]) - 1)
        front = cv2.imread(str(run / "frames" / f"{s}_front.jpg"))
        wrist = cv2.imread(str(run / "frames" / f"{s}_wrist.jpg"))
        g_area, p_area = blob(wrist, GRAPE_HSV)[2], blob(wrist, PLATE_HSV)[2]
        shows = [name for name, a in (("plate", p_area), ("grape", g_area)) if a > 500] or ["neither"]
        out[run.name] = {"policy": policy_name(meta), "t_s": t_obs, "end_z_mm": float(k["end_pose"][i, 2]),
                         "wrist_plate_px": p_area, "wrist_grape_px": g_area, "wrist_shows": shows}
        print(f"{run.name[:6]} {out[run.name]['policy']:28s} t {t_obs:4.1f} s z {out[run.name]['end_z_mm']:5.1f} | "
              f"wrist: plate {p_area:6d} px, grape {g_area:6d} px -> {' and '.join(shows)}")
        f = cv2.resize(front, (320, 240))
        cv2.putText(f, f"{run.name[:6]} t {t_obs:.1f} s z {out[run.name]['end_z_mm']:.0f} mm", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        tiles.append(np.vstack([f, cv2.resize(wrist, (320, 240))]))
    if args.image and tiles:
        while len(tiles) % args.cols:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(args.image), np.vstack([np.hstack(tiles[i:i + args.cols]) for i in range(0, len(tiles), args.cols)]),
                    [cv2.IMWRITE_JPEG_QUALITY, 80])
        print(f"-> {args.image}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("outcomes", "scenes", "hovers"):
        s = sub.add_parser(name)
        s.add_argument("--day", type=Path, required=True)
        s.add_argument("--source", type=Path, default=Path("~/piper_datasets/new_grape").expanduser())
        s.add_argument("--out", type=Path, default=None)
        if name != "outcomes":
            s.add_argument("--image", type=Path, default=None)
    sub.choices["scenes"].add_argument("--demo-episodes", type=int, nargs="+", default=[0, 20, 45, 70, 89])
    sub.choices["hovers"].add_argument("--at", type=float, default=0.6, help="how far through the run's frames")
    sub.choices["hovers"].add_argument("--cols", type=int, default=5)
    args = p.parse_args()
    args.source = args.source.expanduser()
    result = {"outcomes": cmd_outcomes, "scenes": cmd_scenes, "hovers": cmd_hovers}[args.cmd](args)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1, default=float))


if __name__ == "__main__":
    main()
