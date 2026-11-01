"""Open-loop check of a LIBERO checkpoint on recorded demos it did not train on.

The same check as ``scripts/hw/piper_offline_check.py``, for calibrating it: a LIBERO
policy's closed-loop success is known, so this shows whether a policy that succeeds also
beats doing nothing on held-out demos (record ``docs/runs/2026-11-02_libero_offline_calibration.md``).
LIBERO actions are OSC pose deltas, so the errors are in action units and come in two views
(:func:`flowcl.analysis.open_loop.delta_metrics`); the integrated one is the analogue of the
Piper check's absolute joint targets.

The run must have trained on demos 0..n-1 (``--n-demos n``) with every held-out demo at n or
above; its ``config.yaml`` is read to check that.

    uv run python scripts/libero_offline_check.py \
        --checkpoint results/libero_calib/n20/single__libero_10__..._seed0/checkpoints/final.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from flowcl.analysis import open_loop
from flowcl.data.libero_adapter import load_episode
from flowcl.data.tasks import TaskRef
from flowcl.train.checkpoint import load_checkpoint

SUMMARY_KEYS = (
    "integrated_policy_mae_executed", "integrated_standing_still_mae_executed",
    "integrated_repeat_last_mae_executed", "policy_mae_executed", "policy_mae_chunk",
    "zero_motion_mae_executed", "repeat_last_mae_executed", "gripper_mae_executed",
    "integrated_policy_by_tick", "integrated_standing_still_by_tick",
    "integrated_repeat_last_by_tick",
)


def demo_range(text: str) -> list[int]:
    """``"40-49"`` -> demos 40..49."""
    first, _, last = text.partition("-")
    return list(range(int(first), int(last or first) + 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--held-out", type=demo_range, default=demo_range("40-49"))
    parser.add_argument("--training", type=demo_range, default=demo_range("0-9"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out", type=Path, default=None,
                        help="Default: offline_check.json in the checkpoint's run directory.")
    args = parser.parse_args()

    run_dir = args.checkpoint.parent.parent
    n_trained = OmegaConf.load(run_dir / "config.yaml").data.n_demos
    open_loop.assert_unseen(n_trained, args.held_out)
    loaded = load_checkpoint(args.checkpoint, device=args.device)
    policy, spec, stats = loaded.policy, loaded.spec, loaded.stats
    device = torch.device(args.device)
    ref = TaskRef.from_key(loaded.task_key)
    path = ref.demo_path()
    k = spec.action.execute_k
    report = {"checkpoint": str(args.checkpoint), "task": ref.task_key, "n_demos_trained": n_trained,
              "seed": args.seed, "execute_k": k, "chunk_horizon": spec.action.chunk_horizon}
    for name, indices in (("held_out", args.held_out), ("training", args.training)):
        rows = []
        for demo_idx in indices:
            episode = load_episode(path, demo_idx, spec, task_id=ref.task_key)
            seed = open_loop.episode_seed(args.seed, f"{ref.task_key}/{demo_idx}")
            chunks = open_loop.predict_chunks(policy, spec, stats, episode, device, seed, args.batch_size)
            rows.append({"demo_idx": demo_idx, **open_loop.delta_metrics(chunks, episode.action, k)})
        s = open_loop.summarize(rows, SUMMARY_KEYS, id_key="demo_idx")
        s["integrated_ratio_to_standing_still"] = (
            s["integrated_policy_mae_executed"] / s["integrated_standing_still_mae_executed"])
        report[name] = {"summary": s, "per_episode": rows}
        print(f"[offline_check] {name:9s} demos {indices[0]}-{indices[-1]}: integrated pose error "
              f"{s['integrated_policy_mae_executed']:.4f} over the executed {k} ticks, standing still "
              f"{s['integrated_standing_still_mae_executed']:.4f} (ratio "
              f"{s['integrated_ratio_to_standing_still']:.2f}), repeating the last delta "
              f"{s['integrated_repeat_last_mae_executed']:.4f}; per tick {s['policy_mae_executed']:.4f}, "
              f"zero motion {s['zero_motion_mae_executed']:.4f}, repeating the last delta "
              f"{s['repeat_last_mae_executed']:.4f}; gripper {s['gripper_mae_executed']:.4f}", flush=True)
    out = args.out or run_dir / "offline_check.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[offline_check] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
