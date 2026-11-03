"""How much a Piper checkpoint reads the front camera's masked corner (no retraining).

The corner that ``agilex_piper``'s mask blacks out holds the operator and the leader arm,
whose pose is the recorded action. A checkpoint trained without the mask sees them move
in every demo; on the robot the leader arm stands still. This check runs the offline
open-loop check (``piper_offline_check.py``) on training episodes twice with the same
noise: on the recorded frames, and with the corner held at the episode's first frame
(the leader at rest, as in a rollout), and reports both errors and how far the plans
move between the two. A masked checkpoint sees black there either way, so its plans
barely move; the follower's shoulder, partly inside the corner, is held too.

    uv run python scripts/hw/piper_corner_probe.py --task milk_plate_unmasked \
        --checkpoint results/single__piper__milk_plate_unmasked__seed0/checkpoints/final.pt
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from piper_offline_check import JOINTS, check_episode, episode_seed, summarize  # noqa: E402

from flowcl.analysis import open_loop  # noqa: E402
from flowcl.data.config import load_embodiment_spec  # noqa: E402
from flowcl.data.images import policy_frame  # noqa: E402
from flowcl.data.piper_adapter import iter_cache_episodes, verify_cache  # noqa: E402
from flowcl.data.piper_tasks import cache_path, get_task  # noqa: E402
from flowcl.train.checkpoint import load_checkpoint  # noqa: E402

CAMERA = "front"


def corner_region(mask_from: str, image_size: tuple[int, int]) -> np.ndarray:
    """``(H, W)`` bool: the policy-frame pixels the mask touches, partly or fully."""
    spec = load_embodiment_spec(mask_from)
    if tuple(spec.observation.image_size) != tuple(image_size):
        raise ValueError(f"{mask_from} frames are {spec.observation.image_size}, the checkpoint's {image_size}")
    h, w = spec.observation.mask_source_hw
    white = policy_frame(np.full((h, w, 3), 255, np.uint8), CAMERA, spec)
    return (white < 255).any(axis=-1)


def corner_still(episode, region: np.ndarray):
    """The episode with the front camera's corner held at its first frame."""
    front = episode.images[CAMERA].copy()
    front[:, region] = front[0, region]
    return dataclasses.replace(episode, images={**episode.images, CAMERA: front})


def plan_change(policy, spec, stats, recorded, still, device, seed: int, batch_size: int) -> dict:
    """Mean and p95 joint change of the 16-tick plans between the two inputs."""
    a = open_loop.predict_chunks(policy, spec, stats, recorded, device, seed, batch_size)
    b = open_loop.predict_chunks(policy, spec, stats, still, device, seed, batch_size)
    change = np.abs(a.predicted - b.predicted)[..., JOINTS][a.valid]
    return {"plan_change_mean_rad": float(change.mean()), "plan_change_p95_rad": float(np.percentile(change, 95))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--mask-from", default="agilex_piper",
                        help="The embodiment whose front-camera mask defines the corner.")
    parser.add_argument("--n-train-episodes", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out", type=Path, default=None,
                        help="Default: corner_probe.json in the checkpoint's run directory.")
    args = parser.parse_args()

    task = get_task(args.task)
    path = cache_path(task.name)
    verify_cache(path, task)
    loaded = load_checkpoint(args.checkpoint, device=args.device)
    policy, spec, stats = loaded.policy, loaded.spec, loaded.stats
    device = torch.device(args.device)
    region = corner_region(args.mask_from, spec.observation.image_size)
    rows = {"recorded": [], "corner_still": []}
    changes = []
    for ep in iter_cache_episodes(path, spec, episode_indices=list(range(args.n_train_episodes))):
        seed = episode_seed(args.seed, ep)
        still = corner_still(ep, region)
        rows["recorded"].append(check_episode(policy, spec, stats, ep, device, seed, args.batch_size))
        rows["corner_still"].append(check_episode(policy, spec, stats, still, device, seed, args.batch_size))
        changes.append({"episode_index": ep.meta["episode_index"],
                        **plan_change(policy, spec, stats, ep, still, device, seed, args.batch_size)})
    report = {"checkpoint": str(args.checkpoint), "task": task.name, "seed": args.seed,
              "mask_from": args.mask_from, "corner_pixels": int(region.sum()),
              "plan_change": changes}
    for name, group in rows.items():
        report[name] = {"summary": summarize(group), "per_episode": group}
        s = report[name]["summary"]
        print(f"[corner_probe] {name:12s} joint MAE {s['joint_mae_executed_rad']:.4f} rad over the executed "
              f"{spec.action.execute_k} ticks, worst joint {s['worst_joint_mae_executed_rad']:.4f}, "
              f"gripper {s['gripper_mae_executed_mm']:.2f} mm", flush=True)
    mean = float(np.mean([c["plan_change_mean_rad"] for c in changes]))
    print(f"[corner_probe] plans move {mean:.4f} rad on average when the corner is held "
          f"({region.sum()} of {region.size} pixels)", flush=True)
    out = args.out or args.checkpoint.parent.parent / "corner_probe.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[corner_probe] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
