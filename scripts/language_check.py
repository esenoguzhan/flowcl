"""Thin CLI: the §4.1 language-discriminability check that gates ``seq_correlated``.

Spec §1: no logic in scripts/. See :mod:`flowcl.experiments.language`.

Example::

    uv run python scripts/language_check.py --train-steps 30000 --n-episodes 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from flowcl.data.config import load_embodiment_spec
from flowcl.envs.evaluation import eval_config_from_dict
from flowcl.experiments.language import DEFAULT_TASK_PAIR, run_language_check
from flowcl.train.trainer import TrainConfig
from flowcl.utils.libero_paths import repo_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(DEFAULT_TASK_PAIR),
        help="Tasks to joint-train. Default: two libero_object tasks.",
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
    parser.add_argument("--n-demos", type=int, default=None)
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=20,
        help=(
            "Rollouts per arm per task. Each initial state is rolled out twice (once "
            "per instruction), so the cost is 2x this per task."
        ),
    )
    parser.add_argument(
        "--divergence-only",
        action="store_true",
        help="Skip rollouts; run only the cheap chunk-divergence diagnostic.",
    )
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    eval_path = args.eval_config or (
        repo_root() / "configs" / "eval" / "libero_eval.yaml"
    )
    eval_payload = OmegaConf.to_container(OmegaConf.load(eval_path), resolve=True)
    eval_payload["n_episodes"] = args.n_episodes

    run_language_check(
        spec=load_embodiment_spec(args.embodiment),
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
        task_keys=tuple(args.tasks),
        seed=args.seed,
        n_demos=args.n_demos,
        bootstrap=eval_payload.get("bootstrap"),
        pretrained=not args.no_pretrained,
        out_dir=args.out_dir,
        run_rollouts=not args.divergence_only,
    )


if __name__ == "__main__":
    main()
