"""Thin CLI: evaluate a checkpoint on LIBERO tasks under the §8.1 protocol.

Spec §1: no logic in scripts/. Argument parsing and delegation only; the rollout
protocol lives in :mod:`flowcl.envs.evaluation`.

Example::

    uv run python scripts/evaluate.py \\
        --checkpoint results/gate0/checkpoints/stage0.pt \\
        --tasks libero_object/pick_up_the_milk_and_place_it_in_the_basket \\
        --out results/gate0/eval_stage0.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from flowcl.data.tasks import resolve_tasks
from flowcl.envs.evaluation import eval_config_from_dict, evaluate_tasks
from flowcl.train.checkpoint import load_checkpoint
from flowcl.utils.libero_paths import repo_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--tasks",
        nargs="+",
        required=True,
        metavar="SUITE/TASK_NAME",
        help="Task keys to evaluate, e.g. libero_object/pick_up_the_milk...",
    )
    parser.add_argument(
        "--eval-config",
        type=Path,
        default=None,
        help="Defaults to configs/eval/libero_eval.yaml.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "Overrides the checkpoint's run_id. The initial-state seeds derive from "
            "it (§8.3), so changing it changes which states are evaluated."
        ),
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=None,
        help="Override the protocol's rollout count. For smoke tests only.",
    )
    args = parser.parse_args()

    eval_path = args.eval_config or (
        repo_root() / "configs" / "eval" / "libero_eval.yaml"
    )
    eval_payload = OmegaConf.to_container(OmegaConf.load(eval_path), resolve=True)
    if args.n_episodes is not None:
        eval_payload["n_episodes"] = args.n_episodes
    cfg = eval_config_from_dict(eval_payload)

    loaded = load_checkpoint(args.checkpoint, device=args.device)
    run_id = args.run_id or loaded.run_id
    if not run_id:
        raise ValueError(
            f"{args.checkpoint} records no run_id and --run-id was not given; the "
            "initial-state seeds derive from it (§8.3)."
        )

    refs = resolve_tasks(args.tasks)
    report = evaluate_tasks(
        loaded.policy,
        refs,
        loaded.spec,
        loaded.stats,
        run_id=run_id,
        cfg=cfg,
        bootstrap=eval_payload.get("bootstrap"),
        stage=loaded.stage,
    )

    out = args.out or args.checkpoint.with_suffix(".eval.json")
    print(f"[flowcl] wrote {report.save(out)}")


if __name__ == "__main__":
    main()
