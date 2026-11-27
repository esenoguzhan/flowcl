"""Build the second table's shadow-map banks (flowcl.data.shadow_maps), preview and validate them.

Two banks for the front camera of ``agilex_piper_table2_336``, written to ``--out-dir``:

* **train**: from the light recording (``--light-dir``: front/*.jpg and frames.jsonl, 24 Nov
  15:03-15:07, arm at home, no actions), the frames with no grape or plate in view, ``--n-train``
  of them picked evenly in time. It is effectively one sun pattern; variety comes from the
  augmentation's warps (flowcl.data.augment.warped_maps).
* **eval**: the first front frame of every rollout in ``--rollouts`` started before ``--sun-until``
  (24 Nov 14:08-14:22, the sun session). Never trained on; used by piper_relight_check.py.

The reference (the demos' own light) is the median frame 0 of the task's **training** episodes only,
so the held-out demos stay unseen. Printed checks:

* each train map laid over the reference at 336 against its own sunny frame at 336, on the table;
* the sharp-edge share (Sobel > 40 on a 5 px blur, the table region at 336) of real sun frames
  (through the server's CRF-30 video and policy_frame) against demo frames with train maps applied
  through the augmentation's warps.

    uv run python scripts/hw/piper_shadow_maps.py --task new_grape_ho10_336
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
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from flowcl.data.augment import warped_maps  # noqa: E402
from flowcl.data.images import policy_frame  # noqa: E402
from flowcl.data.piper_adapter import read_episode_table, read_info  # noqa: E402
from flowcl.data.piper_tasks import get_task, piper_root, raw_root  # noqa: E402
from flowcl.data.shadow_maps import load_bank, ratio_map, reference_frame, save_bank, support_mask  # noqa: E402
from flowcl.data.config import load_embodiment_spec  # noqa: E402
from flowcl.deploy.piper_server import VIDEO_CRF, video_roundtrip  # noqa: E402

GRAPE_HSV = ((128, 95, 40), (165, 255, 255))
PLATE_HSV = ((88, 130, 60), (108, 255, 255))
FRONT_KEY = "observation.images.image"


def episode_frames(source: Path, episodes: list[int], at: float = 0.0) -> dict[int, np.ndarray]:
    """RGB front frames of raw ``episodes``, at fraction ``at`` of each episode (0 = frame 0)."""
    info, table = read_info(source), {r["episode_index"]: r for r in read_episode_table(source)}
    out = {}
    for e in episodes:
        r = table[e]
        video = source / info["video_path"].format(video_key=FRONT_KEY, chunk_index=r[f"videos/{FRONT_KEY}/chunk_index"],
                                                   file_index=r[f"videos/{FRONT_KEY}/file_index"])
        cap = cv2.VideoCapture(str(video))
        first = int(round(r[f"videos/{FRONT_KEY}/from_timestamp"] * info["fps"]))
        cap.set(cv2.CAP_PROP_POS_FRAMES, first + int(at * (r["length"] - 1)))
        ok, bgr = cap.read()
        cap.release()
        if not ok:
            raise IOError(f"{video}: no frame for episode {e}")
        out[e] = bgr[:, :, ::-1].copy()
    return out


def blob_area(rgb: np.ndarray, hsv_range, masked: np.ndarray) -> int:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    m = cv2.inRange(hsv, np.array(hsv_range[0]), np.array(hsv_range[1]))
    m[masked] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8))
    return int(stats[1:, cv2.CC_STAT_AREA].max()) if n > 1 else 0


def sharp_edge_share(rgb: np.ndarray, region: np.ndarray) -> float:
    grey = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32), (5, 5), 0)
    edges = np.hypot(cv2.Sobel(grey, cv2.CV_32F, 1, 0), cv2.Sobel(grey, cv2.CV_32F, 0, 1))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    keep = region & (cv2.dilate((hsv[..., 1] > 80).astype(np.uint8), np.ones((21, 21), np.uint8)) == 0)
    return float((edges[keep] > 40).mean())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="new_grape_ho10_336")
    p.add_argument("--light-dir", type=Path, default=Path("~/piper_datasets/table2_light/2026-11-24").expanduser())
    p.add_argument("--rollouts", type=Path, default=Path("results/piper_rollouts/2026-11-24"))
    p.add_argument("--sun-until", default="150000", help="rollout folders (HHMMSS) before this are the sun session")
    p.add_argument("--n-train", type=int, default=12)
    p.add_argument("--out-dir", type=Path, default=None, help="default: datasets/piper/shadow_maps")
    args = p.parse_args()

    task = get_task(args.task)
    spec = load_embodiment_spec(task.embodiment)
    size = tuple(spec.observation.image_size)
    hw = tuple(spec.observation.mask_source_hw)
    masked = np.zeros(hw, np.uint8)
    for poly in spec.observation.masks_for("front"):
        cv2.fillPoly(masked, [np.asarray(poly, np.int32)], 1)
    masked = masked > 0
    source = raw_root() / task.sources[0].folder
    held = set(task.sources[0].held_out)
    n_raw = len(read_episode_table(source))
    train_eps = [e for e in range(n_raw) if e not in held and e not in task.sources[0].exclude]
    out_dir = args.out_dir or piper_root() / "shadow_maps"

    reference = reference_frame(list(episode_frames(source, train_eps).values()))
    support = support_mask(reference, masked)
    print(f"reference: median frame 0 of {len(train_eps)} training episodes; support {support.mean():.1%} of the frame")

    # train frames: no grape or plate in view, evenly in time
    rows = [json.loads(line) for line in (args.light_dir / "frames.jsonl").read_text().splitlines() if line.strip()]
    empty = []
    for r in rows:
        rgb = cv2.imread(str(args.light_dir / r["front"]))[:, :, ::-1]
        if blob_area(rgb, GRAPE_HSV, masked) < 150 and blob_area(rgb, PLATE_HSV, masked) < 150:
            empty.append((r, rgb))
    pick = [empty[int(round(i * (len(empty) - 1) / (args.n_train - 1)))] for i in range(args.n_train)]
    print(f"train: {len(empty)} of {len(rows)} light frames with no grape or plate; picked "
          f"{[r['time'][11:] for r, _ in pick]}")
    train_maps = [ratio_map(rgb, reference, support, size) for _, rgb in pick]

    # eval frames: the sun session's first front frames
    runs = sorted(d for d in glob.glob(str(args.rollouts / "*_rollout")) if Path(d).name[:6] < args.sun_until)
    eval_frames = []
    for d in runs:
        seqs = sorted(int(re.match(r"(\d+)_front", Path(f).name).group(1)) for f in glob.glob(f"{d}/frames/*_front.jpg"))
        eval_frames.append((Path(d).name, cv2.imread(f"{d}/frames/{seqs[0]}_front.jpg")[:, :, ::-1].copy()))
    eval_maps = [ratio_map(rgb, reference, support, size) for _, rgb in eval_frames]
    print(f"eval: {len(eval_frames)} sun rollouts' first frames ({eval_frames[0][0][:6]}-{eval_frames[-1][0][:6]})")

    common = {"embodiment": spec.name, "image_size": list(size), "task": task.name,
              "reference": {"source": str(source), "episodes": train_eps, "frame": 0},
              "params": {"clip": [0.25, 2.0], "bright": 110, "erode_px": 15, "blur_sigma": 1.5, "feather_px": 15}}
    sha_t = save_bank(out_dir / "table2_train.npz", train_maps,
                      {**common, "frames": [str(args.light_dir / r["front"]) for r, _ in pick]})
    sha_e = save_bank(out_dir / "table2_eval.npz", eval_maps, {**common, "frames": [n for n, _ in eval_frames]})
    print(f"wrote {out_dir / 'table2_train.npz'} (maps sha256 {sha_t[:16]}), {out_dir / 'table2_eval.npz'} ({sha_e[:16]})")

    # checks at 336
    ref336 = policy_frame(np.clip(reference, 0, 255).astype(np.uint8), "front", spec).astype(np.float32)
    sup336 = cv2.resize(support.astype(np.uint8), (size[1], size[0]), interpolation=cv2.INTER_NEAREST) > 0
    err = [float(np.abs(np.clip(ref336 * m, 0, 255) - policy_frame(rgb, "front", spec).astype(np.float32))[sup336].mean())
           for m, (_, rgb) in zip(train_maps, pick)]
    print(f"train map x reference vs its sunny frame, on the table at 336: mean |error| {min(err):.1f}-{max(err):.1f} grey")
    real = [sharp_edge_share(policy_frame(video_roundtrip(rgb, VIDEO_CRF), "front", spec), sup336) for _, rgb in eval_frames]
    demo336 = [policy_frame(f, "front", spec) for f in episode_frames(source, train_eps[::8]).values()]
    bank = torch.as_tensor(np.stack(train_maps), dtype=torch.float16)
    light = warped_maps(bank, len(demo336), torch.Generator().manual_seed(0)).numpy()
    aug = [sharp_edge_share(np.clip(d * l, 0, 255).astype(np.uint8), sup336) for d, l in zip(demo336, light)]
    plain = [sharp_edge_share(d, sup336) for d in demo336]
    print(f"sharp-edge share at 336: real sun frames {min(real):.3f}-{max(real):.3f}; demos {min(plain):.3f}-{max(plain):.3f}; "
          f"demos with train maps {min(aug):.3f}-{max(aug):.3f}")

    # preview: maps over demo frames mid-reach (the arm extended), at 336
    mid = list(episode_frames(source, train_eps[::20], at=0.35).values())
    tiles = []
    for bank_name, maps in (("train", train_maps), ("eval", eval_maps)):
        row = [policy_frame(mid[0], "front", spec)]
        for i, f in enumerate(mid):
            m = maps[(i * 5) % len(maps)]
            row.append(np.clip(policy_frame(f, "front", spec) * m, 0, 255).astype(np.uint8))
        tiles.append(np.hstack(row))
        cv2.putText(tiles[-1], bank_name, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    preview = out_dir / "table2_maps_preview.jpg"
    cv2.imwrite(str(preview), np.vstack(tiles)[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"preview -> {preview} (first tile: no map)")
    _ = load_bank(out_dir / "table2_train.npz")


if __name__ == "__main__":
    main()
