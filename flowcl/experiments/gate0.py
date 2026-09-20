"""Gate 0 (§10.3): is single-task success adequate?

For each chosen task, train an *independent* single-task policy and evaluate it under
the §8.1 protocol. These runs do double duty: they are Gate 0's evidence, and they are
the §10.4 independent single-task references that FWT needs, so they are written to
``results/`` under stable run ids rather than being thrown away.

The verdict itself lives in :func:`flowcl.analysis.gates.gate0`; this module only
produces the numbers it consumes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch

from flowcl.analysis.gates import GateResult, gate0
from flowcl.data.spec import EmbodimentSpec
from flowcl.data.tasks import TaskRef
from flowcl.envs.evaluation import EvaluationReport, evaluate_tasks
from flowcl.envs.libero_env import EvalConfig
from flowcl.train.pipeline import TrainedPolicy, train_on_tasks
from flowcl.train.trainer import TrainConfig
from flowcl.utils.libero_paths import repo_root


def single_task_run_id(task_key: str, seed: int) -> str:
    """Stable run id for a single-task reference.

    ``/`` becomes ``__`` so the id is a legal single directory name, and the seed is in
    the id so three seeds cannot overwrite each other.
    """
    return f"single__{task_key.replace('/', '__')}__seed{seed}"


@dataclass
class Gate0Report:
    """Everything Gate 0 produced, so the verdict is auditable."""

    result: GateResult
    evaluations: dict = field(default_factory=dict)
    checkpoints: dict = field(default_factory=dict)
    final_losses: dict = field(default_factory=dict)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.result.as_dict()
        payload["checkpoints"] = {k: str(v) for k, v in self.checkpoints.items()}
        payload["final_losses"] = self.final_losses
        payload["evaluations"] = {
            key: report.as_dict() for key, report in self.evaluations.items()
        }
        path.write_text(json.dumps(payload, indent=2) + "\n")
        return path


def train_single_task_reference(
    ref: TaskRef,
    spec: EmbodimentSpec,
    policy_config: str | Path | dict,
    train_cfg: TrainConfig,
    seed: int = 0,
    n_demos: int | None = None,
    dataset_dir: Path | None = None,
    results_root: Path | None = None,
    pretrained: bool = True,
    exist_ok: bool = True,
) -> TrainedPolicy:
    """Train one independent single-task policy (§10.4 reference, Gate 0 evidence)."""
    return train_on_tasks(
        [ref],
        spec=spec,
        policy_config=policy_config,
        train_cfg=train_cfg,
        run_id=single_task_run_id(ref.task_key, seed),
        seed=seed,
        n_demos=n_demos,
        dataset_dir=dataset_dir,
        results_root=results_root,
        pretrained=pretrained,
        extra_config={
            "role": "single_task_reference",
            "spec_sections": ["10.3 Gate 0", "10.4 references"],
        },
        exist_ok=exist_ok,
    )


def run_gate0(
    refs: list[TaskRef] | tuple[TaskRef, ...],
    spec: EmbodimentSpec,
    policy_config: str | Path | dict,
    train_cfg: TrainConfig,
    eval_cfg: EvalConfig,
    seed: int = 0,
    n_demos: int | None = None,
    bootstrap: dict | None = None,
    dataset_dir: Path | None = None,
    results_root: Path | None = None,
    pretrained: bool = True,
    out_dir: Path | None = None,
) -> Gate0Report:
    """Train and evaluate one policy per task, then record the Gate 0 verdict.

    Each task gets its own policy: Gate 0 asks whether the architecture can learn a
    single task, so a jointly-trained policy would answer a different question and
    could mask a task the model cannot fit at all.
    """
    if not refs:
        raise ValueError("run_gate0 received no tasks")

    out_dir = Path(out_dir) if out_dir else (repo_root() / "results" / "gate0")
    evaluations: dict[str, EvaluationReport] = {}
    estimates = {}
    checkpoints = {}
    losses = {}

    for ref in refs:
        print(f"\n[flowcl] === Gate 0: {ref.task_key} ===", flush=True)
        trained = train_single_task_reference(
            ref,
            spec=spec,
            policy_config=policy_config,
            train_cfg=train_cfg,
            seed=seed,
            n_demos=n_demos,
            dataset_dir=dataset_dir,
            results_root=results_root,
            pretrained=pretrained,
        )
        checkpoints[ref.task_key] = trained.checkpoint
        losses[ref.task_key] = {
            "final": trained.log.final_loss,
            "mean_last_50": trained.log.mean_last(50),
            "steps": trained.log.steps,
            "wall_clock_s": trained.log.wall_clock_s,
        }

        report = evaluate_tasks(
            trained.policy,
            [ref],
            spec,
            trained.stats,
            run_id=trained.run.run_id,
            cfg=eval_cfg,
            bootstrap=bootstrap,
            stage=0,
        )
        report.save(trained.run.artifact("eval.json"))
        evaluations[ref.task_key] = report
        estimates[ref.task_key] = report.by_task()[ref.task_key].estimate

        # A single-task policy is ~100M parameters of encoder plus its optimiser state;
        # holding four of them alive across the loop is what exhausts a 24 GB card.
        del trained
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result = gate0(estimates)
    print("\n" + result.describe(), flush=True)

    gate_report = Gate0Report(
        result=result,
        evaluations=evaluations,
        checkpoints=checkpoints,
        final_losses=losses,
    )
    print(f"[flowcl] wrote {gate_report.save(out_dir / 'gate0.json')}")
    return gate_report
