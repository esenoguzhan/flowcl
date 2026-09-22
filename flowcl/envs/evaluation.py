"""Evaluate a policy on a set of LIBERO tasks, per the §8.1 protocol.

Kept separate from ``scripts/evaluate.py`` so the continual runner (§6, §10.4) can
fill retention-matrix rows by calling :func:`evaluate_tasks` directly instead of
shelling out.

Everything here is protocol, not policy: the rollouts use the shared fixed
initial-state set, the seed derives from ``(run_id, task_key, episode_idx)`` and never
from the method name (§8.3), and every reported rate arrives as an
:class:`~flowcl.analysis.metrics.Estimate` with a bootstrap CI (§8.2, §11).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from flowcl.analysis.hooks import assert_no_active_capture
from flowcl.analysis.metrics import Estimate, success_estimate
from flowcl.data.spec import EmbodimentSpec
from flowcl.data.stats import NormalizationStats
from flowcl.data.tasks import TaskRef
from flowcl.envs.libero_env import EvalConfig, LiberoTaskEnv, RolloutResult


@dataclass
class TaskEvaluation:
    """Per-task rollout outcomes plus the §8.2 estimate."""

    task_key: str
    successes: list[bool]
    n_steps: list[int]
    seeds: list[int]
    estimate: Estimate
    wall_clock_s: float = 0.0

    @property
    def n_rollouts(self) -> int:
        return len(self.successes)

    def as_dict(self) -> dict:
        return {
            "task_key": self.task_key,
            # Per-rollout successes are stored, not just the rate, so a CI can be
            # recomputed later with different bootstrap settings without re-running
            # 50 rollouts.
            "successes": [bool(s) for s in self.successes],
            "n_steps": list(self.n_steps),
            "seeds": list(self.seeds),
            "success_rate": self.estimate.value,
            "ci_low": self.estimate.low,
            "ci_high": self.estimate.high,
            "confidence": self.estimate.confidence,
            "n_rollouts": self.n_rollouts,
            "wall_clock_s": self.wall_clock_s,
        }


@dataclass
class EvaluationReport:
    """All tasks evaluated for one checkpoint."""

    run_id: str
    stage: int | None
    tasks: list[TaskEvaluation] = field(default_factory=list)

    def by_task(self) -> dict[str, TaskEvaluation]:
        return {t.task_key: t for t in self.tasks}

    def as_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "stage": self.stage,
            "tasks": [t.as_dict() for t in self.tasks],
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2) + "\n")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "EvaluationReport":
        payload = json.loads(Path(path).read_text())
        return cls(
            run_id=payload["run_id"],
            stage=payload.get("stage"),
            tasks=[
                TaskEvaluation(
                    task_key=t["task_key"],
                    successes=t["successes"],
                    n_steps=t["n_steps"],
                    seeds=t["seeds"],
                    estimate=Estimate(
                        value=t["success_rate"],
                        low=t["ci_low"],
                        high=t["ci_high"],
                        n=t["n_rollouts"],
                        confidence=t.get("confidence", 0.95),
                    ),
                    wall_clock_s=t.get("wall_clock_s", 0.0),
                )
                for t in payload["tasks"]
            ],
        )


def eval_config_from_dict(payload: dict) -> EvalConfig:
    """Build an :class:`EvalConfig` from ``configs/eval/*.yaml``.

    Unknown keys are an error: a typo'd protocol key that was silently ignored would
    mean the run used different settings than its own config claims.
    """
    known = {
        "n_episodes",
        "max_steps",
        "execute_k",
        "euler_steps",
        "temporal_ensembling",
        "temporal_ensemble_coef",
        "image_size",
        "record_video",
    }
    # `bootstrap` is consumed by the metrics layer, not the rollout loop.
    unknown = sorted(set(payload) - known - {"bootstrap"})
    if unknown:
        raise ValueError(
            f"unknown keys in eval config: {unknown}; known keys are {sorted(known)}"
        )
    return EvalConfig(**{k: v for k, v in payload.items() if k in known})


def evaluate_task(
    policy,
    ref: TaskRef,
    spec: EmbodimentSpec,
    stats: NormalizationStats,
    run_id: str,
    cfg: EvalConfig,
    bootstrap: dict | None = None,
    progress: bool = True,
) -> TaskEvaluation:
    """Run ``cfg.n_episodes`` rollouts on one task and summarise them.

    The env is constructed and closed inside this call. That is deliberate: MuJoCo
    contexts are expensive but holding many open across a 4-task curriculum has been
    a reliable source of GPU memory exhaustion, and rollout cost dominates setup.
    """
    bootstrap = bootstrap or {}
    started = time.perf_counter()

    results: list[RolloutResult] = []
    with LiberoTaskEnv(
        suite=ref.suite,
        task_idx=ref.task_idx,
        spec=spec,
        image_size=cfg.image_size,
    ) as env:
        for episode_idx in range(cfg.n_episodes):
            results.append(
                env.rollout(policy, episode_idx, stats, run_id, cfg)
            )
            if progress:
                so_far = sum(r.success for r in results)
                print(
                    f"[flowcl] {ref.task_key} rollout {episode_idx + 1}/"
                    f"{cfg.n_episodes}: {'success' if results[-1].success else 'fail'} "
                    f"({so_far}/{len(results)} so far)",
                    flush=True,
                )

    successes = [r.success for r in results]
    return TaskEvaluation(
        task_key=ref.task_key,
        successes=successes,
        n_steps=[r.n_steps for r in results],
        seeds=[r.seed for r in results],
        estimate=success_estimate(
            successes,
            seed=bootstrap.get("seed", 0),
            n_bootstrap=bootstrap.get("n_resamples", 10000),
            confidence=bootstrap.get("confidence", 0.95),
        ),
        wall_clock_s=time.perf_counter() - started,
    )


def evaluate_tasks(
    policy,
    refs: list[TaskRef] | tuple[TaskRef, ...],
    spec: EmbodimentSpec,
    stats: NormalizationStats,
    run_id: str,
    cfg: EvalConfig,
    bootstrap: dict | None = None,
    stage: int | None = None,
    progress: bool = True,
) -> EvaluationReport:
    """Evaluate every task in ``refs``, in the given order."""
    if not refs:
        raise ValueError("evaluate_tasks received no tasks")
    # §7.1: analysis hooks must not be active during evaluation rollouts.
    assert_no_active_capture(policy)

    report = EvaluationReport(run_id=run_id, stage=stage)
    for ref in refs:
        evaluation = evaluate_task(
            policy, ref, spec, stats, run_id, cfg, bootstrap, progress=progress
        )
        print(
            f"[flowcl] {ref.task_key}: {evaluation.estimate.format_pp()} "
            f"over {evaluation.n_rollouts} rollouts "
            f"({evaluation.wall_clock_s:.0f}s)",
            flush=True,
        )
        report.tasks.append(evaluation)
    return report
