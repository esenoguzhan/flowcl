"""Side-by-side sheet: what the policy saw live vs. the training frames of an episode.

The server dumps its first inputs (``--dump-requests``) as ``request_*.npz`` with the
128x128 policy frames. This puts each dump next to the cached first frame of a recorded
episode (normally the held-out episode whose photo the scene was set from), for checking
crop, flip, colour and camera order at the resolution the policy actually uses.

    uv run python scripts/hw/compare_frames.py --dump-dir results/dobot_serve/<run>__final \
        --task puzzle_red_retrieve --episode 110
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from flowcl.data.dobot_tasks import cache_path  # noqa: E402

CAMERAS = ("top", "left_wrist", "right_wrist")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    with h5py.File(cache_path(args.task), "r") as f:
        g = f["episodes"][f"{args.episode:06d}"]
        reference = np.hstack([np.asarray(g[f"images/{c}"][0]) for c in CAMERAS])
    rows = [reference]
    dumps = sorted(args.dump_dir.glob("request_*.npz"))
    if not dumps:
        raise SystemExit(f"no request_*.npz in {args.dump_dir}")
    for path in dumps:
        d = np.load(path)
        rows.append(np.hstack([d[f"img_{c}"] for c in CAMERAS]))
    sheet = np.vstack(rows)[:, :, ::-1]  # RGB -> BGR for imwrite
    sheet = cv2.resize(sheet, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    cv2.putText(sheet, f"dataset ep{args.episode} t=0", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    for i, path in enumerate(dumps, start=1):
        cv2.putText(sheet, f"live {path.stem}", (6, 20 + 256 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    out = args.out or args.dump_dir / f"compare_ep{args.episode:03d}.png"
    cv2.imwrite(str(out), sheet)
    print(f"wrote {out}: top row = dataset, below = live ({len(dumps)} dumps); columns top | left | right")


if __name__ == "__main__":
    main()
