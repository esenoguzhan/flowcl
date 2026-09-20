"""Gate records: the §10.3 decision points, written down rather than remembered.

Spec §10.3 defines five gates, each a go/no-go question with a threshold. The point of
this module is that a gate verdict becomes an artifact on disk with its evidence
attached, so "did Gate 2 pass?" is answerable from ``results/`` months later instead of
from memory. §11 also requires that a failing gate is a reportable outcome, not
something to quietly work around — so :class:`GateResult` has no "warning" state, only
a boolean verdict plus the numbers behind it.

The §4.1 encoder escape hatch gets its own record type, because the spec is emphatic
that it is a *one-time* decision: "Decide this in Month 1, once, and record it. Do not
silently vary the encoder between runs." :func:`record_escape_hatch` therefore refuses
to overwrite an existing decision.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from flowcl.utils.libero_paths import repo_root

# §10.3 Gate 0: >= 80% success on each chosen LIBERO task.
GATE0_SUCCESS_THRESHOLD = 0.80

# §4.1: the escape hatch triggers only *below* ~70%. Deliberately different from the
# gate threshold: 70-80% means the policy needs fixing (more steps, better recipe),
# not a different encoder. Swapping the encoder inside that band would burn the
# one-time decision on a problem it does not solve.
ESCAPE_HATCH_THRESHOLD = 0.70


@dataclass
class GateResult:
    """One gate's verdict plus the evidence for it.

    Attributes:
        gate: Gate number, 0-4.
        question: The §10.3 question, verbatim, so the artifact is self-describing.
        criterion: The threshold in words.
        passed: The verdict. No third state by design.
        evidence: Numbers behind the verdict. Anything a reader would need to
            disagree with it.
        notes: Free text, e.g. which escape hatch was invoked.
    """

    gate: int
    question: str
    criterion: str
    passed: bool
    evidence: dict = field(default_factory=dict)
    notes: str = ""
    run_id: str | None = None

    def as_dict(self) -> dict:
        return {
            "gate": self.gate,
            "question": self.question,
            "criterion": self.criterion,
            "passed": self.passed,
            "evidence": self.evidence,
            "notes": self.notes,
            "run_id": self.run_id,
            "recorded": datetime.now(timezone.utc).isoformat(),
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2) + "\n")
        return path

    def describe(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        lines = [
            f"[Gate {self.gate}] {verdict}",
            f"  question:  {self.question}",
            f"  criterion: {self.criterion}",
        ]
        if self.notes:
            lines.append(f"  notes:     {self.notes}")
        return "\n".join(lines)


def gate0(
    per_task_estimates: dict,
    threshold: float = GATE0_SUCCESS_THRESHOLD,
    run_id: str | None = None,
) -> GateResult:
    """Evaluate §10.3 Gate 0 from per-task success estimates.

    Args:
        per_task_estimates: ``task_key -> Estimate`` from
            :mod:`flowcl.envs.evaluation`.
        threshold: :data:`GATE0_SUCCESS_THRESHOLD`.

    The verdict uses the *point estimate*, not the CI lower bound. With 50 rollouts a
    true 80% success rate has a lower bound near 68%, so requiring the interval to
    clear the threshold would fail policies that meet the spec. The interval is
    recorded either way, and tasks whose CI straddles the threshold are flagged in the
    evidence as ``borderline`` rather than being silently counted as pass or fail.
    """
    if not per_task_estimates:
        raise ValueError("gate0 received no task estimates")

    per_task = {}
    for task_key, estimate in per_task_estimates.items():
        per_task[task_key] = {
            "success_rate": estimate.value,
            "ci_low": estimate.low,
            "ci_high": estimate.high,
            "n_rollouts": estimate.n,
            "formatted": estimate.format_pp(),
            "meets_threshold": estimate.value >= threshold,
            "borderline": estimate.low < threshold <= estimate.high,
            "below_escape_hatch": estimate.value < ESCAPE_HATCH_THRESHOLD,
        }

    failing = sorted(k for k, v in per_task.items() if not v["meets_threshold"])
    hatch_candidates = sorted(
        k for k, v in per_task.items() if v["below_escape_hatch"]
    )

    if failing:
        if hatch_candidates:
            notes = (
                f"{len(hatch_candidates)} task(s) below the §4.1 escape-hatch "
                f"threshold of {ESCAPE_HATCH_THRESHOLD:.0%}: {hatch_candidates}. §4.1 "
                "permits switching to a from-scratch ResNet-18 for the single-task "
                "capability study, ONCE, recorded via record_escape_hatch()."
            )
        else:
            notes = (
                f"task(s) {failing} are between the §4.1 escape-hatch threshold of "
                f"{ESCAPE_HATCH_THRESHOLD:.0%} and Gate 0's {threshold:.0%}. §4.1's "
                "encoder swap is NOT indicated here; fix the training recipe (steps, "
                "lr, demo count) instead and re-run the gate."
            )
    else:
        notes = ""

    return GateResult(
        gate=0,
        question="Is single-task success adequate?",
        criterion=(
            f">= {threshold:.0%} success on each chosen LIBERO task, else fix the "
            "policy before anything else"
        ),
        passed=not failing,
        evidence={
            "threshold": threshold,
            "escape_hatch_threshold": ESCAPE_HATCH_THRESHOLD,
            "per_task": per_task,
            "failing_tasks": failing,
            "escape_hatch_candidates": hatch_candidates,
        },
        notes=notes,
        run_id=run_id,
    )


def escape_hatch_path(results_root: Path | None = None) -> Path:
    """Where the one-time §4.1 encoder decision lives.

    Repo-level, not run-level: the decision applies to every subsequent run, so it
    must not be buried inside the directory of the run that happened to trigger it.
    """
    root = results_root or (repo_root() / "results")
    return Path(root) / "encoder_decision.json"


def record_escape_hatch(
    reason: str,
    evidence: dict,
    encoder: str = "resnet18_scratch",
    results_root: Path | None = None,
) -> Path:
    """Record the §4.1 encoder swap, refusing to overwrite a previous decision.

    Raises:
        FileExistsError: If a decision was already recorded. §4.1 says decide once;
            a second swap would mean the encoder varied between runs, which makes
            every cross-run comparison in the thesis invalid.
    """
    path = escape_hatch_path(results_root)
    if path.exists():
        existing = json.loads(path.read_text())
        raise FileExistsError(
            f"A §4.1 encoder decision was already recorded in {path} on "
            f"{existing.get('recorded')} ({existing.get('encoder')!r}). §4.1 allows "
            "this once. Do not silently vary the encoder between runs."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "encoder": encoder,
                "reason": reason,
                "evidence": evidence,
                "spec_section": "4.1",
                "recorded": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
        + "\n"
    )
    return path


def active_encoder_decision(results_root: Path | None = None) -> dict | None:
    """The recorded §4.1 decision, or ``None`` if the default encoder still stands."""
    path = escape_hatch_path(results_root)
    if not path.is_file():
        return None
    return json.loads(path.read_text())
