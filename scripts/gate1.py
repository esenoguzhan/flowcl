"""Thin CLI: run §10.3 Gate 1 -- does forgetting exist?

Spec §1: no logic in scripts/. See :mod:`flowcl.experiments.gate1`.

Example::

    uv run python scripts/gate1.py --curricula seq_hetero seq_correlated
"""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from flowcl.data.config import load_embodiment_spec
from flowcl.data.curriculum import load_curriculum
from flowcl.envs.evaluation import eval_config_from_dict
from flowcl.experiments.gate1 import run_gate1
from flowcl.train.trainer import TrainConfig
from flowcl.utils.libero_paths import repo_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--curricula",
        nargs="+",
        default=["seq_hetero", "seq_correlated"],
        help="Candidate sequences (§5). Gate 1 passes if any one forgets enough.",
    )
    parser.add_argument("--embodiment", default="libero_franka")
    parser.add_argument("--policy", default="flowpolicy_base")
    parser.add_argument("--eval-config", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", type=Path, default=None)

    parser.add_argument("--steps-per-task", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--n-episodes", type=int, default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument(
        "--no-single-task-baseline",
        action="store_true",
        help="Skip the §10.4 gap column; the verdict still uses NBT.",
    )
    args = parser.parse_args()

    eval_path = args.eval_config or (
        repo_root() / "configs" / "eval" / "libero_eval.yaml"
    )
    eval_payload = OmegaConf.to_container(OmegaConf.load(eval_path), resolve=True)
    if args.n_episodes is not None:
        eval_payload["n_episodes"] = args.n_episodes

    run_gate1(
        [load_curriculum(name) for name in args.curricula],
        spec=load_embodiment_spec(args.embodiment),
        policy_config=args.policy,
        train_cfg=TrainConfig(
            steps=args.steps_per_task,
            batch_size=args.batch_size,
            lr=args.lr,
            num_workers=args.num_workers,
            device=args.device,
            amp=args.amp,
        ),
        eval_cfg=eval_config_from_dict(eval_payload),
        seed=args.seed,
        bootstrap=eval_payload.get("bootstrap"),
        pretrained=not args.no_pretrained,
        use_single_task_baseline=not args.no_single_task_baseline,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
