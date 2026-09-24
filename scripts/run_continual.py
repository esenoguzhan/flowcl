"""Thin CLI: run one sequential continual-learning experiment.

Spec §1: no logic in scripts/. See :mod:`flowcl.train.continual`.

Examples::

    uv run python scripts/run_continual.py --curriculum seq_hetero --method seq_ft
    uv run python scripts/run_continual.py --curriculum seq_hetero --reverse --seed 1
    uv run python scripts/run_continual.py --curriculum seq_hetero --joint-reference
"""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from flowcl.data.config import load_embodiment_spec, load_method_config
from flowcl.data.curriculum import load_curriculum
from flowcl.envs.evaluation import eval_config_from_dict
from flowcl.train.continual import run_continual
from flowcl.train.trainer import TrainConfig
from flowcl.utils.libero_paths import repo_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curriculum", required=True)
    parser.add_argument(
        "--reverse",
        action="store_true",
        help=(
            "Derive the reverse-order curriculum (§5). Note this also changes which "
            "task §3.3 fits normalization statistics on."
        ),
    )
    parser.add_argument("--method", default="seq_ft", help="configs/method/<name>.yaml")
    parser.add_argument("--embodiment", default="libero_franka")
    parser.add_argument("--policy", default="flowpolicy_base")
    parser.add_argument("--eval-config", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")

    parser.add_argument(
        "--steps-per-task",
        type=int,
        default=30000,
        help="Optimiser steps per curriculum stage.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--n-episodes", type=int, default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help=(
            "Train without rollouts. Exercises the training path where no GL context "
            "exists; produces no retention matrix and therefore no metrics."
        ),
    )
    parser.add_argument(
        "--joint-reference",
        action="store_true",
        help=(
            "Train the §10.4 joint multi-task reference on the union instead of "
            "running the curriculum sequentially."
        ),
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Run on an uncommitted tree (recorded). Final Stage A runs must not use this.",
    )
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Write into an existing run directory. Off by default: never overwrite a run.",
    )
    parser.add_argument(
        "--single-task-baseline",
        action="store_true",
        help="Write FWT against the Gate 0 single-task references (must exist).",
    )
    parser.add_argument(
        "--t1-reference-run",
        type=Path,
        default=None,
        help=(
            "Run directory to compare the stage-0 model against (fail fast on a broken "
            "pairing). Default for non-seq_ft methods: the seed-namespace (seq_ft) run, "
            "if it exists."
        ),
    )
    parser.add_argument("--no-t1-check", action="store_true")
    parser.add_argument(
        "--identity-reference-run",
        type=Path,
        default=None,
        help="Run whose checkpoints this run must equal bitwise at --identity-stages.",
    )
    parser.add_argument("--identity-stages", type=int, nargs="*", default=[])
    parser.add_argument("--t1-pairing-max-rel-diff", type=float, default=None)
    args = parser.parse_args()

    spec = load_embodiment_spec(args.embodiment)
    curriculum = load_curriculum(args.curriculum)
    if args.reverse:
        curriculum = curriculum.reversed()

    eval_path = args.eval_config or (
        repo_root() / "configs" / "eval" / "libero_eval.yaml"
    )
    eval_payload = OmegaConf.to_container(OmegaConf.load(eval_path), resolve=True)
    if args.n_episodes is not None:
        eval_payload["n_episodes"] = args.n_episodes
    eval_cfg = eval_config_from_dict(eval_payload)

    train_cfg = TrainConfig(
        steps=args.steps_per_task,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        device=args.device,
        amp=args.amp,
    )

    if args.joint_reference:
        from flowcl.experiments.references import run_joint_reference

        # Scale the budget with the task count so each task gets the same number of
        # gradient steps as one sequential stage.
        train_cfg.steps = args.steps_per_task * len(curriculum)
        reference = run_joint_reference(
            curriculum,
            spec=spec,
            policy_config=args.policy,
            train_cfg=train_cfg,
            eval_cfg=eval_cfg,
            seed=args.seed,
            bootstrap=eval_payload.get("bootstrap"),
            pretrained=not args.no_pretrained,
        )
        print(f"[flowcl] joint reference mean success {reference.mean_rate():.3f}")
        return

    from flowcl.train.continual import T1_PAIRING_MAX_REL_DIFF, seed_namespace_run_id

    method_name, method_kwargs = load_method_config(args.method)
    t1_reference = args.t1_reference_run
    if t1_reference is None and not args.no_t1_check and method_name != "seq_ft":
        candidate = repo_root() / "results" / seed_namespace_run_id(curriculum.name, args.seed)
        if (candidate / "checkpoints" / "stage0.pt").is_file():
            t1_reference = candidate
    if args.no_t1_check:
        t1_reference = None
    run_continual(
        curriculum,
        method_name=method_name,
        spec=spec,
        policy_config=args.policy,
        train_cfg=train_cfg,
        eval_cfg=eval_cfg,
        method_kwargs=method_kwargs,
        seed=args.seed,
        bootstrap=eval_payload.get("bootstrap"),
        pretrained=not args.no_pretrained,
        evaluate=not args.no_eval,
        exist_ok=args.exist_ok,
        single_task_baseline=args.single_task_baseline,
        require_clean_tree=not args.allow_dirty,
        t1_reference_run=t1_reference,
        t1_pairing_max_rel_diff=(
            args.t1_pairing_max_rel_diff
            if args.t1_pairing_max_rel_diff is not None
            else T1_PAIRING_MAX_REL_DIFF
        ),
        identity_reference_run=args.identity_reference_run,
        identity_stages=tuple(args.identity_stages),
    )


if __name__ == "__main__":
    main()
