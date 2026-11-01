"""Open-loop check of a Piper checkpoint on recorded episodes (no robot, no server).

Every ``execute_k`` ticks of an episode, the policy gets the recorded frames and state
and predicts a chunk with fixed sampling noise; the prediction is compared with the
recorded actions. Held-out episodes (the registry's last ``holdout``) against the same
number of training episodes is a first look at generalisation. Open-loop error is not
closed-loop success: the robot decides.

Two do-nothing baselines on the same ticks give the scale: holding the current measured
joints, and repeating the last recorded command. A policy that does not beat them on
held-out episodes has not learned anything that transfers.

``--task`` picks the episodes; a checkpoint trained on another task of the same
recordings can be checked on them too (with ``--out``), since every Piper task shares the
embodiment and its mask.

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

from flowcl.analysis import open_loop  # noqa: E402
from flowcl.data.piper_adapter import iter_cache_episodes, verify_cache  # noqa: E402
from flowcl.data.piper_tasks import cache_path, get_task  # noqa: E402
from flowcl.train.checkpoint import load_checkpoint  # noqa: E402

JOINTS = slice(0, 6)
GRIPPER = 6


@torch.no_grad()
def check_episode(policy, spec, stats, episode, device, seed: int, batch_size: int) -> dict:
    """Errors of the predicted chunks at ticks 0, k, 2k, ... of one episode."""
    chunks = open_loop.predict_chunks(policy, spec, stats, episode, device, seed, batch_size)
    ticks, predicted, recorded, valid = chunks.ticks, chunks.predicted, chunks.recorded, chunks.valid
    err = np.abs(predicted - recorded)
    executed = chunks.executed(spec.action.execute_k)
    joint_k = err[..., JOINTS][executed]
    hold_state = episode.state[ticks][:, None, JOINTS]
    hold_last = episode.action[np.maximum(ticks - 1, 0)][:, None, JOINTS]
    return {
        "episode_index": episode.meta["episode_index"],
        "source": episode.meta["source"],
        "raw_episode_index": episode.meta["raw_episode_index"],
        "n_chunks": len(ticks),
        "joint_mae_executed_rad": float(joint_k.mean()),
        "joint_mae_chunk_rad": float(err[..., JOINTS][valid].mean()),
        "worst_joint_mae_executed_rad": float(joint_k.mean(axis=0).max()),
        "gripper_mae_executed_mm": float(1000 * err[..., GRIPPER][executed].mean()),
        "hold_state_mae_executed_rad": float(np.abs(recorded[..., JOINTS] - hold_state)[executed].mean()),
        "hold_last_command_mae_executed_rad": float(np.abs(recorded[..., JOINTS] - hold_last)[executed].mean()),
    }


def episode_seed(seed: int, episode) -> int:
    """Sampling noise keyed by the recording, so every task and checkpoint sees the same noise
    on the same raw episode."""
    return open_loop.episode_seed(seed, f"{episode.meta['source']}/{episode.meta['raw_episode_index']}")


SUMMARY_KEYS = ("joint_mae_executed_rad", "joint_mae_chunk_rad", "worst_joint_mae_executed_rad",
                "gripper_mae_executed_mm", "hold_state_mae_executed_rad",
                "hold_last_command_mae_executed_rad")


def summarize(rows: list[dict]) -> dict:
    return open_loop.summarize(rows, SUMMARY_KEYS, id_key="episode_index")


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
        rows = [check_episode(policy, spec, stats, ep, device, episode_seed(args.seed, ep), args.batch_size)
                for ep in iter_cache_episodes(path, spec, episode_indices=indices)]
        report[name] = {"summary": summarize(rows), "per_episode": rows}
        s = report[name]["summary"]
        print(f"[offline_check] {name:9s} episodes {indices[0]}-{indices[-1]}: joint MAE "
              f"{s['joint_mae_executed_rad']:.4f} rad over the executed {spec.action.execute_k} ticks "
              f"({s['joint_mae_chunk_rad']:.4f} over the chunk), gripper {s['gripper_mae_executed_mm']:.2f} mm; "
              f"holding the state {s['hold_state_mae_executed_rad']:.4f}, the last command "
              f"{s['hold_last_command_mae_executed_rad']:.4f}", flush=True)
    out = args.out or args.checkpoint.parent.parent / "offline_check.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[offline_check] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
