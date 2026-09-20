"""Thin CLI: run §10.3 Gate 0 -- is single-task success adequate?

Spec §1: no logic in scripts/. Argument parsing and delegation only; the experiment
lives in :mod:`flowcl.experiments.gate0`.

Example::

    uv run python scripts/gate0.py --curriculum seq_hetero --train-steps 30000
"""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from flowcl.data.config import load_embodiment_spec
from flowcl.data.curriculum import load_curriculum
from flowcl.data.tasks import resolve_tasks
from flowcl.envs.evaluation import eval_config_from_dict
from flowcl.experiments.gate0 import run_gate0
from flowcl.train.trainer import TrainConfig
from flowcl.utils.libero_paths import repo_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--curriculum",
        help="Curriculum config name, e.g. seq_hetero. Gate 0 covers its tasks.",
    )
    source.add_argument(
        "--tasks",
        nargs="+",
        metavar="SUITE/TASK_NAME",
        help="Explicit task keys, for evaluating a single task in isolation.",
    )

    parser.add_argument("--embodiment", default="libero_franka")
    parser.add_argument("--policy", default="flowpolicy_base")
    parser.add_argument("--eval-config", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", type=Path, default=None)

    parser.add_argument("--train-steps", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--n-demos",
        type=int,
        default=None,
        help="Demos per task; default is all 50 (§3.2).",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=None,
        help="Override the §8.1 rollout count. Smoke tests only.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Skip pretrained encoder weights. Wiring smoke tests only -- a Gate 0 "
        "verdict from randomly initialised encoders is meaningless.",
    )
    args = parser.parse_args()

    spec = load_embodiment_spec(args.embodiment)

    if args.curriculum:
        curriculum = load_curriculum(args.curriculum)
        refs = curriculum.refs
        n_demos = args.n_demos or curriculum.stages[0].n_demos
    else:
        refs = resolve_tasks(args.tasks)
        n_demos = args.n_demos

    eval_path = args.eval_config or (
        repo_root() / "configs" / "eval" / "libero_eval.yaml"
    )
    eval_payload = OmegaConf.to_container(OmegaConf.load(eval_path), resolve=True)
    if args.n_episodes is not None:
        eval_payload["n_episodes"] = args.n_episodes

    run_gate0(
        refs,
        spec=spec,
        policy_config=args.policy,
        train_cfg=TrainConfig(
            steps=args.train_steps,
            batch_size=args.batch_size,
            lr=args.lr,
            num_workers=args.num_workers,
            device=args.device,
            amp=args.amp,
        ),
        eval_cfg=eval_config_from_dict(eval_payload),
        seed=args.seed,
        n_demos=n_demos,
        bootstrap=eval_payload.get("bootstrap"),
        pretrained=not args.no_pretrained,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
