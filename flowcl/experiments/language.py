"""The §4.1 language-discriminability experiment, which gates ``seq_correlated``.

Joint-train one policy on two ``libero_object`` tasks, then ask whether the instruction
actually selects the behaviour (:mod:`flowcl.analysis.language_check`). Joint training
rather than sequential is the right setup: it removes forgetting from the picture, so a
failure can only mean the policy never learned to use language in the first place.
"""

from __future__ import annotations

import json
from pathlib import Path

from flowcl.analysis.language_check import (
    LanguageCheckReport,
    instruction_sensitivity,
    instruction_swap_rollouts,
)
from flowcl.data.spec import EmbodimentSpec
from flowcl.data.tasks import TaskRef, resolve_tasks
from flowcl.envs.libero_env import EvalConfig
from flowcl.train.pipeline import build_dataset, train_on_tasks
from flowcl.train.trainer import TrainConfig
from flowcl.utils.libero_paths import repo_root

# Two libero_object tasks: identical scene and identical motion, different target
# object. That is the hardest case for language and the exact structure seq_correlated
# is built from, so it is the case worth testing.
DEFAULT_TASK_PAIR = (
    "libero_object/pick_up_the_milk_and_place_it_in_the_basket",
    "libero_object/pick_up_the_tomato_sauce_and_place_it_in_the_basket",
)


def run_language_check(
    spec: EmbodimentSpec,
    policy_config: str | Path | dict,
    train_cfg: TrainConfig,
    eval_cfg: EvalConfig,
    task_keys: tuple[str, ...] = DEFAULT_TASK_PAIR,
    seed: int = 0,
    n_demos: int | None = None,
    bootstrap: dict | None = None,
    dataset_dir: Path | None = None,
    results_root: Path | None = None,
    pretrained: bool = True,
    out_dir: Path | None = None,
    run_rollouts: bool = True,
) -> LanguageCheckReport:
    """Joint-train on ``task_keys`` and test instruction discriminability.

    Args:
        run_rollouts: When False, only the cheap forward-pass divergence diagnostic
            runs. Useful to find a dead language pathway in seconds, but it cannot
            produce a verdict -- :attr:`LanguageCheckReport.passed` is False without
            swap rollouts, by design.
    """
    refs: tuple[TaskRef, ...] = resolve_tasks(list(task_keys))
    if len(refs) < 2:
        raise ValueError("the language check needs at least two tasks")

    run_id = f"langcheck__{'__'.join(r.name for r in refs)}__seed{seed}"
    trained = train_on_tasks(
        refs,
        spec=spec,
        policy_config=policy_config,
        train_cfg=train_cfg,
        run_id=run_id,
        seed=seed,
        n_demos=n_demos,
        dataset_dir=dataset_dir,
        results_root=results_root,
        pretrained=pretrained,
        extra_config={
            "role": "language_discriminability_check",
            "spec_sections": ["4.1", "5 seq_correlated"],
        },
        exist_ok=True,
    )

    report = LanguageCheckReport(task_keys=tuple(r.task_key for r in refs))

    # Diagnostic: from one task's own observations, do two instructions yield
    # different chunks at all?
    from flowcl.data.dataset import collate_chunks

    probe_dataset = build_dataset(
        [refs[0]], spec, trained.stats, n_demos=2, dataset_dir=dataset_dir
    )
    probe_batch = collate_chunks(
        [probe_dataset[i] for i in range(0, min(len(probe_dataset), 256), 8)]
    )
    probe_batch["images"] = {
        k: v.to(next(trained.policy.parameters()).device)
        for k, v in probe_batch["images"].items()
    }
    probe_batch["state"] = probe_batch["state"].to(
        next(trained.policy.parameters()).device
    )
    report.sensitivity.append(
        instruction_sensitivity(
            trained.policy,
            probe_batch,
            instruction_a=refs[0].language,
            instruction_b=refs[1].language,
            seed=seed,
        )
    )

    if run_rollouts:
        for i, ref in enumerate(refs):
            other = refs[(i + 1) % len(refs)]
            report.swaps.append(
                instruction_swap_rollouts(
                    trained.policy,
                    ref,
                    swapped_instruction=other.language,
                    spec=spec,
                    stats=trained.stats,
                    run_id=trained.run.run_id,
                    cfg=eval_cfg,
                    bootstrap=bootstrap,
                )
            )

    print("\n" + report.describe(), flush=True)

    out_dir = Path(out_dir) if out_dir else (repo_root() / "results" / "gate0")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "language_check.json"
    out_path.write_text(json.dumps(report.as_dict(), indent=2) + "\n")
    print(f"[flowcl] wrote {out_path}")
    return report
