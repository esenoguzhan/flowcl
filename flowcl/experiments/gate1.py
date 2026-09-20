"""Gate 1 (§10.3): does forgetting exist?

Run ``seq_ft`` through each candidate sequence and measure how much is forgotten. If
nothing is forgotten, there is no continual-learning effect to study and §10.3 says
change the curriculum rather than proceed — every method's ``F_1`` would then be equal
and the whole Stage A comparison would be measuring noise.

Forgetting is reported two ways, because they can disagree and the disagreement is
informative:

* **NBT** (§8.2) — the drop from each task's own just-after-training performance. This is
  what Gate 1 is judged on, since it isolates forgetting from the policy's raw ability.
* **gap to the single-task references** (§10.4) — how far the final sequential policy is
  from independently trained policies. Larger than NBT when the sequential run never
  learned a task well in the first place, which is a Gate 0 problem masquerading as
  forgetting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from flowcl.analysis.gates import GateResult, gate1
from flowcl.analysis.metrics import (
    final_average_success,
    negative_backward_transfer,
    per_task_forgetting,
)
from flowcl.data.curriculum import Curriculum
from flowcl.data.spec import EmbodimentSpec
from flowcl.envs.libero_env import EvalConfig
from flowcl.train.continual import ContinualResult, run_continual
from flowcl.train.trainer import TrainConfig
from flowcl.utils.libero_paths import repo_root


@dataclass
class Gate1Report:
    """Gate 1's verdict plus the per-sequence evidence."""

    result: GateResult
    runs: dict = field(default_factory=dict)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.result.as_dict()
        payload["runs"] = self.runs
        path.write_text(json.dumps(payload, indent=2) + "\n")
        return path


def sequence_evidence(
    result: ContinualResult,
    single_task_baseline: dict[str, float] | None = None,
) -> dict:
    """Forgetting numbers for one sequential run."""
    matrix = result.matrix
    forgetting = per_task_forgetting(matrix)
    evidence = {
        "run_id": result.run_id,
        "F_1": final_average_success(matrix),
        "NBT": negative_backward_transfer(matrix),
        "per_task_forgetting": forgetting,
        "worst_task_forgetting": max(forgetting.values()),
        "diagonal": {
            key: float(matrix.values[i, i])
            for i, key in enumerate(matrix.task_keys)
        },
    }
    if single_task_baseline:
        final = {
            key: float(matrix.values[matrix.n_tasks - 1, i])
            for i, key in enumerate(matrix.task_keys)
        }
        gaps = {
            key: single_task_baseline[key] - final[key]
            for key in matrix.task_keys
            if key in single_task_baseline
        }
        evidence["gap_to_single_task"] = gaps
        if gaps:
            evidence["mean_gap_to_single_task"] = sum(gaps.values()) / len(gaps)
    return evidence


def run_gate1(
    curricula: list[Curriculum],
    spec: EmbodimentSpec,
    policy_config: str | Path | dict,
    train_cfg: TrainConfig,
    eval_cfg: EvalConfig,
    seed: int = 0,
    bootstrap: dict | None = None,
    dataset_dir: Path | None = None,
    results_root: Path | None = None,
    pretrained: bool = True,
    use_single_task_baseline: bool = True,
    out_dir: Path | None = None,
) -> Gate1Report:
    """Run ``seq_ft`` through each curriculum and record the Gate 1 verdict."""
    if not curricula:
        raise ValueError("run_gate1 received no curricula")

    runs: dict[str, dict] = {}
    forgetting_by_sequence: dict[str, float] = {}

    for curriculum in curricula:
        print(f"\n[flowcl] === Gate 1: {curriculum.name} (seq_ft) ===", flush=True)
        result = run_continual(
            curriculum,
            method_name="seq_ft",
            spec=spec,
            policy_config=policy_config,
            train_cfg=train_cfg,
            eval_cfg=eval_cfg,
            seed=seed,
            bootstrap=bootstrap,
            dataset_dir=dataset_dir,
            results_root=results_root,
            pretrained=pretrained,
        )

        baseline = None
        if use_single_task_baseline:
            from flowcl.experiments.references import load_single_task_reference

            try:
                baseline = load_single_task_reference(
                    curriculum.task_keys, seed=seed, results_root=results_root
                ).rates()
            except FileNotFoundError as exc:
                # Not fatal: NBT alone decides the gate. Say so rather than silently
                # omitting a column from the evidence.
                print(f"[flowcl] no single-task baseline ({exc}); reporting NBT only")

        evidence = sequence_evidence(result, baseline)
        runs[curriculum.name] = evidence
        forgetting_by_sequence[curriculum.name] = evidence["NBT"]

    verdict = gate1(forgetting_by_sequence)
    print("\n" + verdict.describe(), flush=True)
    for name, evidence in runs.items():
        print(
            f"  {name}: NBT {100 * evidence['NBT']:.1f} pp, "
            f"F_1 {100 * evidence['F_1']:.1f}%",
            flush=True,
        )

    report = Gate1Report(result=verdict, runs=runs)
    out_dir = Path(out_dir) if out_dir else (repo_root() / "results" / "gate1")
    print(f"[flowcl] wrote {report.save(out_dir / 'gate1.json')}")
    return report
