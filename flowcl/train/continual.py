"""The sequential continual-learning runner (§6, §8.1, §8.2).

One policy is carried across the curriculum's stages. After *every* stage the policy is
evaluated on *every* task — not only the ones seen so far — because §8.2's FWT needs the
above-diagonal entries ``R[j-1][j]``, performance on a task before it was ever trained.
Evaluating only the seen tasks is the cheaper thing to do and it silently makes FWT
uncomputable, which is why :meth:`RetentionMatrix.assert_complete_through` exists.

The runner is method-agnostic: it never branches on which method is active. It calls
the five §6 hooks in a fixed order via :func:`flowcl.train.trainer.train_one_task`, so
two methods' results differ only by their mechanism.

The §3.3 invariant is enforced at every stage boundary, as the spec demands: statistics
come from task 1 only and :func:`flowcl.data.stats.assert_frozen` is called before each
stage against the fingerprint recorded at stage 1. "Recomputing stats per task silently
changes the target distribution and corrupts forgetting measurements" — so the check is
a call site, not a comment.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from flowcl.analysis.metrics import CLSummary, Estimate, RetentionMatrix, summarize
from flowcl.data.curriculum import Curriculum
from flowcl.data.spec import EmbodimentSpec
from flowcl.data.stats import NormalizationStats, assert_frozen
from flowcl.envs.evaluation import EvaluationReport, evaluate_tasks
from flowcl.envs.libero_env import EvalConfig
from flowcl.models.build import build_policy, load_policy_config
from flowcl.train.checkpoint import save_checkpoint
from flowcl.train.pipeline import build_dataset, fit_stats
from flowcl.train.trainer import TrainConfig, TrainLog, train_one_task
from flowcl.utils.run import RunHandle, create_run
from flowcl.utils.seeding import derive_seed


@dataclass
class StageRecord:
    """What one curriculum stage produced."""

    stage: int
    task_key: str
    n_demos: int
    dataset_size: int
    train_log: TrainLog
    checkpoint: Path
    evaluation: EvaluationReport

    def as_dict(self) -> dict:
        return {
            "stage": self.stage,
            "task_key": self.task_key,
            "n_demos": self.n_demos,
            "dataset_size": self.dataset_size,
            "final_loss": self.train_log.final_loss,
            "mean_last_50_loss": self.train_log.mean_last(50),
            "steps": self.train_log.steps,
            "train_wall_clock_s": self.train_log.wall_clock_s,
            "checkpoint": str(self.checkpoint),
            "evaluation": self.evaluation.as_dict(),
        }


@dataclass
class ContinualResult:
    """A complete sequential run: the retention matrix plus the §8.2 systems numbers."""

    run_id: str
    method: str
    curriculum: str
    seed: int
    task_keys: tuple[str, ...]
    matrix: RetentionMatrix
    stages: list[StageRecord] = field(default_factory=list)
    systems: dict = field(default_factory=dict)
    run: RunHandle | None = None

    def estimates(self) -> dict[tuple[int, str], Estimate]:
        """``(stage, task_key) -> Estimate``, so no rate travels without its CI."""
        out = {}
        for record in self.stages:
            for evaluation in record.evaluation.tasks:
                out[(record.stage, evaluation.task_key)] = evaluation.estimate
        return out

    def summary(self, baseline: dict[str, float] | None = None) -> CLSummary:
        return summarize(self.matrix, baseline=baseline)

    def as_dict(self, baseline: dict[str, float] | None = None) -> dict:
        return {
            "run_id": self.run_id,
            "method": self.method,
            "curriculum": self.curriculum,
            "seed": self.seed,
            "task_keys": list(self.task_keys),
            "retention_matrix": {
                "task_keys": list(self.matrix.task_keys),
                # NaN is not valid JSON, so unevaluated cells are explicit nulls.
                "values": [
                    [None if np.isnan(v) else float(v) for v in row]
                    for row in self.matrix.values
                ],
                "n_rollouts": self.matrix.n_rollouts.tolist(),
            },
            "metrics": self.summary(baseline).as_dict(),
            "stages": [record.as_dict() for record in self.stages],
            "systems": self.systems,
        }

    def save(self, path: str | Path, baseline: dict[str, float] | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(baseline), indent=2) + "\n")
        return path


def continual_run_id(method: str, curriculum: str, seed: int) -> str:
    """Stable run id. Every field that changes the result is in the name."""
    return f"{curriculum}__{method}__seed{seed}"


def run_continual(
    curriculum: Curriculum,
    method_name: str,
    spec: EmbodimentSpec,
    policy_config: str | Path | dict,
    train_cfg: TrainConfig,
    eval_cfg: EvalConfig,
    method_kwargs: dict | None = None,
    seed: int = 0,
    bootstrap: dict | None = None,
    dataset_dir: Path | None = None,
    results_root: Path | None = None,
    pretrained: bool = True,
    exist_ok: bool = True,
    evaluate: bool = True,
) -> ContinualResult:
    """Train one policy through ``curriculum`` under ``method_name``.

    Args:
        curriculum: Ordered task sequence (§5).
        method_name: Key into :data:`flowcl.methods.base.METHOD_REGISTRY`.
        method_kwargs: Method hyperparameters, from ``configs/method/<name>.yaml``.
        seed: Resolved seed. Note it does **not** feed the evaluation initial states,
            which derive from ``(run_id, task_key, episode_idx)`` (§8.3).
        evaluate: Set False to exercise the training path without a GL context. The
            retention matrix is then empty and no metric can be computed, which is
            correct: there is no such thing as a rollout-free success rate.

    Returns:
        A :class:`ContinualResult` whose artifacts are already on disk.
    """
    from flowcl.methods.base import build_method

    method = build_method(method_name, **(method_kwargs or {}))
    run_id = continual_run_id(method_name, curriculum.name, seed)
    raw_policy_cfg = load_policy_config(policy_config)

    run = create_run(
        run_id=run_id,
        cfg={
            "run_id": run_id,
            "seed": seed,
            "method": {"name": method_name, **(method_kwargs or {})},
            "curriculum": {
                "name": curriculum.name,
                "tasks": [
                    {"task_key": s.task_key, "n_demos": s.n_demos}
                    for s in curriculum.stages
                ],
                "description": curriculum.description,
                "expectation": curriculum.expectation,
            },
            "embodiment": spec.to_dict(),
            "policy": raw_policy_cfg,
            "train": {k: v for k, v in vars(train_cfg).items()},
            "eval": {k: v for k, v in vars(eval_cfg).items()},
        },
        seed=seed,
        results_root=results_root,
        exist_ok=exist_ok,
    )

    curriculum.assert_consistent(dataset_dir)
    torch.manual_seed(seed)

    # §3.3: fit once on task 1, then freeze for the whole curriculum.
    first = curriculum.stages[0]
    stats = fit_stats(
        first.ref, spec, n_demos=first.n_demos, dataset_dir=dataset_dir
    )
    stats.save(run.artifact("stats.json"))
    stats_fingerprint = stats.fingerprint()
    print(
        f"[flowcl] {run_id}: stats fitted on {stats.fitted_on_task_id} "
        f"(fingerprint {stats_fingerprint})",
        flush=True,
    )

    policy = build_policy(raw_policy_cfg, spec, pretrained=pretrained)
    policy.to(torch.device(train_cfg.device))
    parameter_report = policy.parameter_report()

    matrix = RetentionMatrix.empty(curriculum.task_keys)
    result = ContinualResult(
        run_id=run_id,
        method=method_name,
        curriculum=curriculum.name,
        seed=seed,
        task_keys=curriculum.task_keys,
        matrix=matrix,
        run=run,
    )

    total_started = time.perf_counter()
    for stage_idx, stage in enumerate(curriculum.stages):
        # The §3.3 stage-boundary assertion. Deliberately before the dataset is built,
        # so a method that refitted stats cannot get as far as training on them.
        assert_frozen(
            stats,
            embodiment=spec.name,
            first_task_id=curriculum.first_task_key,
            expected_fingerprint=stats_fingerprint,
        )

        print(
            f"\n[flowcl] === {run_id} stage {stage_idx}: {stage.task_key} ===",
            flush=True,
        )
        dataset = build_dataset(
            [stage.ref],
            spec,
            stats,
            n_demos=stage.n_demos,
            dataset_dir=dataset_dir,
        )

        generator = torch.Generator(device="cpu").manual_seed(
            derive_seed(run_id, stage.task_key, stage_idx)
        )
        train_log = train_one_task(
            policy,
            dataset,
            train_cfg,
            method=method,
            task_idx=stage_idx,
            generator=generator,
        )

        checkpoint = save_checkpoint(
            run.subdir("checkpoints") / f"stage{stage_idx}.pt",
            policy=policy,
            policy_config=raw_policy_cfg,
            spec=spec,
            stats=stats,
            run_id=run_id,
            stage=stage_idx,
            task_key=stage.task_key,
            extra={
                "method": method_name,
                "curriculum": curriculum.name,
                "method_state": method.state_dict(),
                "final_loss": train_log.final_loss,
            },
        )

        # §8.2: evaluate on every task, including the unseen ones, or FWT is lost.
        evaluation = (
            evaluate_tasks(
                policy,
                curriculum.refs,
                spec,
                stats,
                run_id=run_id,
                cfg=eval_cfg,
                bootstrap=bootstrap,
                stage=stage_idx,
            )
            if evaluate
            else EvaluationReport(run_id=run_id, stage=stage_idx)
        )
        evaluation.save(run.subdir("eval") / f"stage{stage_idx}.json")

        for task_position, task_key in enumerate(curriculum.task_keys):
            entry = evaluation.by_task().get(task_key)
            if entry is not None:
                matrix.set(
                    stage_idx, task_position, entry.estimate.value, entry.n_rollouts
                )

        result.stages.append(
            StageRecord(
                stage=stage_idx,
                task_key=stage.task_key,
                n_demos=stage.n_demos,
                dataset_size=len(dataset),
                train_log=train_log,
                checkpoint=checkpoint,
                evaluation=evaluation,
            )
        )
        # Episodes hold every demo's pixels; releasing the stage's dataset keeps peak
        # host memory proportional to one task rather than the whole curriculum.
        del dataset

    result.systems = {
        "trainable_params": parameter_report["trainable"],
        "frozen_params": parameter_report["frozen"],
        "registry_layers": parameter_report["registry_layers"],
        "method_stored_mb": getattr(method, "stored_bytes", lambda: 0)() / 1e6,
        "is_exemplar_free": getattr(method, "is_exemplar_free", True),
        "total_wall_clock_s": time.perf_counter() - total_started,
        "train_wall_clock_s": sum(r.train_log.wall_clock_s for r in result.stages),
    }

    if evaluate:
        result.save(run.artifact("result.json"))
        print(
            f"\n[flowcl] {run_id} F_1 = "
            f"{result.summary().final_average_success:.3f}",
            flush=True,
        )
    return result


def baseline_from_single_task_runs(
    task_keys: tuple[str, ...],
    seed: int = 0,
    results_root: Path | None = None,
) -> dict[str, float]:
    """Collect the §10.4 independent single-task success rates, for FWT.

    Reads the ``eval.json`` that :func:`flowcl.experiments.gate0.run_gate0` wrote, so
    FWT reuses Gate 0's runs instead of retraining them.
    """
    from flowcl.experiments.gate0 import single_task_run_id
    from flowcl.utils.libero_paths import repo_root

    root = Path(results_root) if results_root else (repo_root() / "results")
    baseline = {}
    for task_key in task_keys:
        path = root / single_task_run_id(task_key, seed) / "eval.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"No single-task reference for {task_key} at {path}. FWT needs the "
                "§10.4 independent references; run scripts/gate0.py first."
            )
        report = EvaluationReport.load(path)
        baseline[task_key] = report.by_task()[task_key].estimate.value
    return baseline
