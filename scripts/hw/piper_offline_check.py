"""Open-loop check of a Piper checkpoint on recorded episodes (no robot, no server).

Every ``execute_k`` ticks of an episode, the policy gets the recorded frames and state
and predicts a chunk with fixed sampling noise; the prediction is compared with the
recorded actions. Held-out episodes (the registry's last ``holdout``) against the same
number of training episodes is a first look at generalisation. Open-loop error is not
closed-loop success: the robot decides.

    uv run python scripts/hw/piper_offline_check.py \
        --checkpoint results/single__piper__sausage_pot_lid__seed0/checkpoints/final.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from flowcl.data.dataset import ChunkedActionDataset, collate_chunks  # noqa: E402
from flowcl.data.piper_adapter import iter_cache_episodes, verify_cache  # noqa: E402
from flowcl.data.piper_tasks import cache_path, get_task  # noqa: E402
from flowcl.train.checkpoint import load_checkpoint  # noqa: E402
from flowcl.train.trainer import move_batch  # noqa: E402

JOINTS = slice(0, 6)
GRIPPER = 6


@torch.no_grad()
def check_episode(policy, spec, stats, episode, device, seed: int, batch_size: int) -> dict:
    """Errors of the predicted chunks at ticks 0, k, 2k, ... of one episode."""
    k, horizon = spec.action.execute_k, spec.action.chunk_horizon
    dataset = ChunkedActionDataset([episode], spec, stats, horizon=horizon)
    ticks = list(range(0, episode.length, k))
    generator = torch.Generator().manual_seed(seed)
    predicted, recorded, valid = [], [], []
    for i in range(0, len(ticks), batch_size):
        batch = collate_chunks([dataset[t] for t in ticks[i:i + batch_size]])
        size = batch["actions"].shape[0]
        noise = torch.randn((size, horizon, spec.d_action), generator=generator).to(device)
        chunk = policy.sample(move_batch(batch, device), noise=noise).float().cpu().numpy()
        predicted.append(stats.denormalize_action(chunk))
        recorded.append(stats.denormalize_action(batch["actions"].numpy()))
        valid.append(batch["action_mask"].numpy().astype(bool))
    predicted, recorded, valid = map(np.concatenate, (predicted, recorded, valid))
    err = np.abs(predicted - recorded)
    executed = valid.copy()
    executed[:, k:] = False
    joint_k = err[..., JOINTS][executed]
    return {
        "episode_index": episode.meta["episode_index"],
        "source": episode.meta["source"],
        "raw_episode_index": episode.meta["raw_episode_index"],
        "n_chunks": len(ticks),
        "joint_mae_executed_rad": float(joint_k.mean()),
        "joint_mae_chunk_rad": float(err[..., JOINTS][valid].mean()),
        "worst_joint_mae_executed_rad": float(joint_k.mean(axis=0).max()),
        "gripper_mae_executed_mm": float(1000 * err[..., GRIPPER][executed].mean()),
    }


def summarize(rows: list[dict]) -> dict:
    keys = ("joint_mae_executed_rad", "joint_mae_chunk_rad", "worst_joint_mae_executed_rad",
            "gripper_mae_executed_mm")
    weights = np.array([r["n_chunks"] for r in rows], dtype=float)
    out = {"episodes": [r["episode_index"] for r in rows], "n_chunks": int(weights.sum())}
    for key in keys:
        values = np.array([r[key] for r in rows])
        out[key] = float((values * weights).sum() / weights.sum())
        out[f"{key}_range"] = [float(values.min()), float(values.max())]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task", default="sausage_pot_lid")
    parser.add_argument("--n-train-episodes", type=int, default=None,
                        help="Training episodes to compare (the first n); default = the hold-out size.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out", type=Path, default=None,
                        help="Default: offline_check.json in the checkpoint's run directory.")
    args = parser.parse_args()

    task = get_task(args.task)
    path = cache_path(task.name)
    verify_cache(path, task)
    loaded = load_checkpoint(args.checkpoint, device=args.device)
    policy, spec, stats = loaded.policy, loaded.spec, loaded.stats
    device = torch.device(args.device)
    n_train = args.n_train_episodes or task.holdout
    groups = {
        "held_out": list(range(task.n_train, task.n_episodes)),
        "training": list(range(n_train)),
    }
    report = {"checkpoint": str(args.checkpoint), "task": task.name, "seed": args.seed,
              "execute_k": spec.action.execute_k, "chunk_horizon": spec.action.chunk_horizon}
    for name, indices in groups.items():
        rows = [check_episode(policy, spec, stats, ep, device, args.seed + ep.meta["episode_index"], args.batch_size)
                for ep in iter_cache_episodes(path, spec, episode_indices=indices)]
        report[name] = {"summary": summarize(rows), "per_episode": rows}
        s = report[name]["summary"]
        print(f"[offline_check] {name:9s} episodes {indices[0]}-{indices[-1]}: joint MAE "
              f"{s['joint_mae_executed_rad']:.4f} rad over the executed {spec.action.execute_k} ticks "
              f"({s['joint_mae_chunk_rad']:.4f} over the chunk), gripper {s['gripper_mae_executed_mm']:.2f} mm",
              flush=True)
    out = args.out or args.checkpoint.parent.parent / "offline_check.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[offline_check] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
