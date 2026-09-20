"""The §10.4 reference runs every continual number is compared against.

Two references, and they answer different questions:

* **Independent single-task** — one policy per task, trained alone. Upper bound on what
  the architecture can do on a task, and the baseline FWT subtracts (§8.2). Produced by
  :mod:`flowcl.experiments.gate0`, since Gate 0 needs exactly these runs.
* **Joint multi-task** — one policy trained on the union of the curriculum's tasks,
  i.i.d. Upper bound on what *one set of weights* can hold, so it bounds what any
  continual method could achieve without extra capacity.

The gap between the two is itself informative: if joint training already falls well
short of the single-task references, the tasks interfere at the representation level and
no continual method can close that part of the gap.

The joint reference deliberately uses the curriculum's own frozen §3.3 statistics,
fitted on the curriculum's first task. Refitting them over the union would give the
reference a normalisation advantage the sequential runs never had, and the comparison
would partly measure that.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from flowcl.analysis.metrics import Estimate
from flowcl.data.curriculum import Curriculum
from flowcl.data.spec import EmbodimentSpec
from flowcl.envs.evaluation import EvaluationReport, evaluate_tasks
from flowcl.envs.libero_env import EvalConfig
from flowcl.train.pipeline import train_on_tasks
from flowcl.train.trainer import TrainConfig


def joint_run_id(curriculum_name: str, seed: int) -> str:
    return f"joint__{curriculum_name}__seed{seed}"


@dataclass
class ReferenceReport:
    """Per-task estimates from one reference run."""

    kind: str
    run_id: str
    estimates: dict = field(default_factory=dict)

    def rates(self) -> dict[str, float]:
        return {k: v.value for k, v in self.estimates.items()}

    def mean_rate(self) -> float:
        if not self.estimates:
            raise ValueError(f"{self.kind} reference has no estimates")
        return sum(self.rates().values()) / len(self.estimates)

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "run_id": self.run_id,
            "per_task": {
                k: {
                    "success_rate": v.value,
                    "ci_low": v.low,
                    "ci_high": v.high,
                    "n_rollouts": v.n,
                    "formatted": v.format_pp(),
                }
                for k, v in self.estimates.items()
            },
            "mean_rate": self.mean_rate() if self.estimates else None,
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2) + "\n")
        return path


def run_joint_reference(
    curriculum: Curriculum,
    spec: EmbodimentSpec,
    policy_config: str | Path | dict,
    train_cfg: TrainConfig,
    eval_cfg: EvalConfig,
    seed: int = 0,
    bootstrap: dict | None = None,
    dataset_dir: Path | None = None,
    results_root: Path | None = None,
    pretrained: bool = True,
) -> ReferenceReport:
    """Train one policy on the union of the curriculum's tasks and evaluate all of them.

    ``train_cfg.steps`` should be scaled by the number of tasks relative to a single
    stage, or the joint reference sees fewer gradient steps per task than the sequential
    runs and understates the achievable ceiling. The caller owns that choice because it
    is a budget decision, but the run's ``config.yaml`` records ``steps`` either way.
    """
    trained = train_on_tasks(
        curriculum.refs,
        spec=spec,
        policy_config=policy_config,
        train_cfg=train_cfg,
        run_id=joint_run_id(curriculum.name, seed),
        seed=seed,
        n_demos=curriculum.stages[0].n_demos,
        dataset_dir=dataset_dir,
        results_root=results_root,
        pretrained=pretrained,
        extra_config={
            "role": "joint_multitask_reference",
            "spec_sections": ["10.4 references"],
            "curriculum": curriculum.name,
        },
        exist_ok=True,
    )

    report = evaluate_tasks(
        trained.policy,
        curriculum.refs,
        spec,
        trained.stats,
        run_id=trained.run.run_id,
        cfg=eval_cfg,
        bootstrap=bootstrap,
        stage=None,
    )
    report.save(trained.run.artifact("eval.json"))

    reference = ReferenceReport(
        kind="joint_multitask",
        run_id=trained.run.run_id,
        estimates={t.task_key: t.estimate for t in report.tasks},
    )
    reference.save(trained.run.artifact("reference.json"))
    return reference


def load_single_task_reference(
    task_keys: tuple[str, ...],
    seed: int = 0,
    results_root: Path | None = None,
) -> ReferenceReport:
    """Load the §10.4 independent single-task references Gate 0 produced."""
    from flowcl.experiments.gate0 import single_task_run_id
    from flowcl.utils.libero_paths import repo_root

    root = Path(results_root) if results_root else (repo_root() / "results")
    estimates: dict[str, Estimate] = {}
    for task_key in task_keys:
        path = root / single_task_run_id(task_key, seed) / "eval.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing single-task reference for {task_key} at {path}; run "
                "scripts/gate0.py first."
            )
        estimates[task_key] = (
            EvaluationReport.load(path).by_task()[task_key].estimate
        )
    return ReferenceReport(
        kind="single_task", run_id=f"single__seed{seed}", estimates=estimates
    )
