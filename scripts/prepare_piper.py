"""Check, review and cache the AgileX Piper recordings of a registry task.

Three modes, for ``flowcl/data/piper_tasks.py``'s task (default ``sausage_pot_lid``):

* ``--review``: sheets for checking the front-camera mask before anything is built.
  Each tile is one recorded frame at full resolution with the mask drawn on it (red),
  next to the policy's two inputs for that tick, made by the same
  :func:`flowcl.data.images.policy_frame` the cache uses (shown at 2x, nearest
  neighbour). Frames are each episode's highest-gripper tick away from the home pose,
  the ticks where joint 1 swings widest, and random ticks.
* ``--build``: fingerprint check, then the cache ``datasets/piper/cache/<task>.h5``, a
  report (``datasets/piper/report_<task>.json``, also printed) and initial-condition
  photos for the held-out episodes (``datasets/piper/photos/<task>/``).
* ``--cache-sheet``: the policy inputs read back from the built cache.

    uv run python scripts/prepare_piper.py --review
    uv run python scripts/prepare_piper.py --build
    uv run python scripts/prepare_piper.py --cache-sheet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flowcl.data.config import load_embodiment_spec  # noqa: E402
from flowcl.data.images import policy_frame  # noqa: E402
from flowcl.data.piper_adapter import (  # noqa: E402
    build_cache,
    iter_cache_episodes,
    read_episode_table,
    read_info,
    video_spans,
)
from flowcl.data.piper_tasks import cache_path, get_task, piper_root, raw_root  # noqa: E402

EMBODIMENT = "agilex_piper"


def kept_episodes(task) -> list[tuple[str, int]]:
    """``(folder, raw episode index)`` in cache order."""
    out = []
    for src in task.sources:
        n = len(read_episode_table(raw_root() / src.folder))
        out.extend((src.folder, e) for e in range(n) if e not in src.exclude)
    return out


def read_column(folder: str, key: str) -> tuple[np.ndarray, np.ndarray]:
    """One data column of a folder and its episode index per row."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    paths = sorted((raw_root() / folder / "data").glob("chunk-*/file-*.parquet"))
    table = pa.concat_tables([pq.read_table(p, columns=[key, "episode_index"]) for p in paths])
    values = np.asarray(table.column(key).to_pylist(), dtype=np.float64)
    return values, np.asarray(table.column("episode_index").to_pylist())


def full_frames(folder: str, camera: str, wanted: dict[int, list[int]]) -> dict[tuple[int, int], np.ndarray]:
    """Recorded RGB frames ``(episode, tick) -> (480, 640, 3)``, decoded sequentially."""
    import cv2

    source = raw_root() / folder
    info = read_info(source)
    spans = video_spans(read_episode_table(source), camera, source, info)
    out = {}
    for path, items in spans.items():
        targets = {first + t: (e, t) for e, first, n in items for t in wanted.get(e, []) if t < n}
        if not targets:
            continue
        cap = cv2.VideoCapture(str(path))
        k = 0
        while targets and k <= max(targets):
            ok, bgr = cap.read()
            if not ok:
                raise ValueError(f"{path}: ended at frame {k}")
            if k in targets:
                out[targets.pop(k)] = np.ascontiguousarray(bgr[:, :, ::-1])
            k += 1
        cap.release()
    return out


def review_tile(spec, front: np.ndarray, wrist: np.ndarray, label: str) -> np.ndarray:
    """Full frame with the mask outlined, plus the two policy inputs at 2x."""
    import cv2

    overlay = front.copy()
    polygons = [np.asarray(p, dtype=np.int32) for p in spec.observation.masks_for("front")]
    shaded = overlay.copy()
    cv2.fillPoly(shaded, polygons, (255, 0, 0))
    overlay = cv2.addWeighted(shaded, 0.35, overlay, 0.65, 0)
    cv2.polylines(overlay, polygons, True, (255, 0, 0), 2)
    cv2.putText(overlay, label, (8, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    inputs = [cv2.resize(policy_frame(img, cam, spec), (256, 256), interpolation=cv2.INTER_NEAREST)
              for cam, img in (("front", front), ("wrist", wrist))]
    below = np.concatenate([*inputs, np.zeros((256, overlay.shape[1] - 512, 3), np.uint8)], axis=1)
    return np.concatenate([overlay, below], axis=0)


def write_sheet(tiles: list[np.ndarray], path: Path, cols: int) -> None:
    import cv2

    h, w = tiles[0].shape[:2]
    rows = -(-len(tiles) // cols)
    sheet = np.zeros((rows * h, cols * w, 3), np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = tile
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sheet[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    print(f"[prepare_piper] wrote {path} ({len(tiles)} tiles)", flush=True)


def review(task, spec, seed: int) -> None:
    rng = np.random.default_rng(seed)
    order = kept_episodes(task)
    picks: dict[str, dict[int, set]] = {}
    reasons: dict[tuple[str, int, int], str] = {}
    by_folder: dict[str, list[int]] = {}
    for folder, e in order:
        by_folder.setdefault(folder, []).append(e)
    joint1_scores = []
    for folder, episodes in by_folder.items():
        eef, ep_of_row = read_column(folder, "observation.eef")
        state, _ = read_column(folder, "observation.state")
        for e in episodes:
            rows = np.flatnonzero(ep_of_row == e)
            # The home pose is the highest of all, so take the highest tick away from it.
            away = np.abs(state[rows, :6] - state[rows[0], :6]).max(axis=1) > 0.3
            z = np.where(away, eef[rows, 2], -np.inf)
            t_high = int(np.argmax(z))
            picks.setdefault(folder, {}).setdefault(e, set()).add(t_high)
            reasons[(folder, e, t_high)] = "highest"
            j1 = state[rows, 0]
            for t in (int(np.argmin(j1)), int(np.argmax(j1))):
                joint1_scores.append((abs(j1[t] - np.median(state[:, 0])), folder, e, t))
            for t in rng.choice(len(rows), size=1, replace=False):
                picks[folder][e].add(int(t))
                reasons.setdefault((folder, e, int(t)), "random")
    for _, folder, e, t in sorted(joint1_scores, reverse=True)[:12]:
        picks[folder][e].add(t)
        reasons[(folder, e, t)] = "joint1 wide"
    tiles = []
    for folder, episodes in by_folder.items():
        wanted = {e: sorted(picks[folder][e]) for e in episodes}
        fronts = full_frames(folder, "front", wanted)
        wrists = full_frames(folder, "wrist", wanted)
        for e in episodes:
            for t in wanted[e]:
                label = f"{folder} ep{e} t{t} {reasons.get((folder, e, t), '')}"
                tiles.append(review_tile(spec, fronts[(e, t)], wrists[(e, t)], label))
    out = piper_root() / "review" / task.name
    per_sheet = 12
    for i in range(0, len(tiles), per_sheet):
        write_sheet(tiles[i:i + per_sheet], out / f"mask_review_{i // per_sheet:02d}.jpg", cols=3)


def report(task, spec, summary: dict) -> dict:
    eps = list(iter_cache_episodes(cache_path(task.name), spec))
    actions = [e.action for e in eps[:task.n_train]]
    stacked = np.concatenate(actions)
    steps = np.concatenate([np.abs(np.diff(a[:, :6], axis=0)).max(axis=1) for a in actions])
    lag = {}
    for k in range(0, 5):
        diffs = [np.abs(e.action[:len(e.action) - k, :6] - e.state[k:, :6]) for e in eps]
        lag[k] = float(np.concatenate(diffs).mean())
    return {
        **summary,
        "n_train": task.n_train,
        "holdout": task.holdout,
        "episodes": [{"cache_index": e.meta["episode_index"], "source": e.meta["source"],
                      "raw_episode_index": e.meta["raw_episode_index"], "length": e.length}
                     for e in eps],
        "train_frames": int(stacked.shape[0]),
        "train_action_mean": stacked.mean(0).round(4).tolist(),
        "train_action_std": stacked.std(0).round(4).tolist(),
        "train_action_min": stacked.min(0).round(4).tolist(),
        "train_action_max": stacked.max(0).round(4).tolist(),
        "joint_step_p99_rad": float(np.percentile(steps, 99)),
        "joint_step_max_rad": float(steps.max()),
        "mean_abs_action_minus_state_at_lag": lag,
    }


def photos(task) -> None:
    """First recorded frames (unmasked) of the held-out episodes, for setting the scene."""
    import cv2

    order = kept_episodes(task)
    out = piper_root() / "photos" / task.name
    out.mkdir(parents=True, exist_ok=True)
    for i in range(task.n_train, task.n_episodes):
        folder, e = order[i]
        for cam in ("front", "wrist"):
            frame = full_frames(folder, cam, {e: [0]})[(e, 0)]
            cv2.imwrite(str(out / f"ep{i:03d}_{folder}_{e}_{cam}.png"), frame[:, :, ::-1])
    print(f"[prepare_piper] initial-condition photos in {out}", flush=True)


def cache_sheet(task, spec, seed: int) -> None:
    import cv2

    rng = np.random.default_rng(seed)
    tiles = []
    for e in iter_cache_episodes(cache_path(task.name), spec):
        for t in sorted(rng.choice(e.length, size=2, replace=False)):
            pair = [cv2.resize(e.images[c][t], (256, 256), interpolation=cv2.INTER_NEAREST) for c in spec.cameras]
            tile = np.concatenate(pair, axis=1)
            cv2.putText(tile, f"cache ep{e.meta['episode_index']} t{t}", (4, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            tiles.append(tile)
    write_sheet(tiles[:48], piper_root() / "review" / task.name / "cache_inputs.jpg", cols=6)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="sausage_pot_lid")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--review", action="store_true")
    mode.add_argument("--build", action="store_true")
    mode.add_argument("--cache-sheet", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    task = get_task(args.task)
    spec = load_embodiment_spec(EMBODIMENT)
    if args.review:
        review(task, spec, seed=args.seed)
    elif args.build:
        summary = build_cache(task, raw_root(), cache_path(task.name), spec)
        out = report(task, spec, summary)
        path = piper_root() / f"report_{task.name}.json"
        path.write_text(json.dumps(out, indent=2) + "\n")
        print(json.dumps({k: v for k, v in out.items() if k != "episodes"}, indent=2))
        print(f"[prepare_piper] report {path}", flush=True)
        photos(task)
    else:
        cache_sheet(task, spec, seed=args.seed)


if __name__ == "__main__":
    main()
