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

# §10.3 Gate 1: forgetting of at least 15 percentage points on at least one sequence.
GATE1_FORGETTING_THRESHOLD = 0.15

# §10.3 Gate 2: "rho_l after T1 not already ~ 1 on most registry layers". Made concrete
# (decision recorded in docs/runs/): a layer is saturated when rho_l >= 0.90 at the
# §7.2 default eps = 0.95, and the gate fails when MORE than half the registry layers
# are saturated. Exactly half passes: "most" means a strict majority.
GATE2_DEFAULT_EPS = 0.95
GATE2_SATURATION_RHO = 0.90
GATE2_MAX_SATURATED_FRACTION = 0.50

# Registry groups whose input is a raw low-dimensional signal (state d=8, action d=7).
# Their rho_l moves in steps of 1/d and saturates trivially; they are counted, but
# flagged so a reader does not mistake them for evidence about the trunk/decoder.
GATE2_SMALL_D_GROUPS = ("trunk_input", "decoder_input")

# §10.3 Gate 3: "c_l after T2; c_l ~ 1 => hard projection incompatible with plasticity".
# Made concrete (decision recorded in docs/runs/): a layer is blocked when the mean
# per-batch c_l >= 0.95 at eps = 0.95, i.e. hard projection would leave <= 31% of its
# gradient norm. The gate fails when a strict majority of EITHER half (trunk, decoder)
# is blocked: judged per half so a blocked decoder cannot be outvoted by the trunk.
GATE3_DEFAULT_EPS = 0.95
GATE3_BLOCKED_C = 0.95
GATE3_MAX_BLOCKED_FRACTION = 0.50


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


def gate1(
    per_sequence_forgetting: dict,
    threshold: float = GATE1_FORGETTING_THRESHOLD,
    run_id: str | None = None,
) -> GateResult:
    """Evaluate §10.3 Gate 1: does forgetting exist?

    §10.3's threshold is ``F_1 >= 15 pp on at least one sequence, else change
    curriculum``. Read literally that is ambiguous, since §8.2 defines ``F_1`` as *final
    average success*, and a final average success of 15% would be a catastrophically bad
    policy rather than evidence of forgetting. The quantity the gate is about is the
    forgetting *gap* measured in percentage points, so this function takes per-sequence
    forgetting and the docstring records the reading. The deviation is noted in
    ``docs/`` as §11 requires.

    Args:
        per_sequence_forgetting: ``sequence_name -> forgetting in [0, 1]``, i.e. NBT from
            :func:`flowcl.analysis.metrics.negative_backward_transfer`, or equivalently
            the drop from the joint/single-task reference.
        threshold: Fraction, not percentage points. 0.15 == 15 pp.

    "At least one sequence" is the right quantifier here, unlike Gate 0's "each task":
    the gate asks whether the *benchmark* can exhibit forgetting at all, and one
    sequence that does is enough to proceed.
    """
    if not per_sequence_forgetting:
        raise ValueError("gate1 received no sequences")

    per_sequence = {
        name: {
            "forgetting": float(value),
            "forgetting_pp": 100 * float(value),
            "meets_threshold": float(value) >= threshold,
        }
        for name, value in per_sequence_forgetting.items()
    }
    passing = sorted(k for k, v in per_sequence.items() if v["meets_threshold"])

    return GateResult(
        gate=1,
        question="Does forgetting exist?",
        criterion=(
            f"forgetting >= {100 * threshold:.0f} pp on at least one sequence, else "
            "change curriculum"
        ),
        passed=bool(passing),
        evidence={
            "threshold": threshold,
            "per_sequence": per_sequence,
            "passing_sequences": passing,
        },
        notes=(
            ""
            if passing
            else (
                "No sequence forgets enough to measure a continual-learning effect. "
                "§10.3 says change the curriculum rather than proceeding: with nothing "
                "to forget, every method's F_1 would be equal and the comparison would "
                "measure noise."
            )
        ),
        run_id=run_id,
    )


def gate2(
    per_layer_rho: dict[str, float],
    groups: dict[str, str],
    d_in: dict[str, int],
    alternative_rhos: dict[str, dict[str, float]] | None = None,
    eps: float = GATE2_DEFAULT_EPS,
    saturation_rho: float = GATE2_SATURATION_RHO,
    max_saturated_fraction: float = GATE2_MAX_SATURATED_FRACTION,
    run_id: str | None = None,
) -> GateResult:
    """Evaluate §10.3 Gate 2: is projection geometrically plausible?

    Args:
        per_layer_rho: ``layer -> rho_l`` at ``eps`` from the *primary* basis, i.e. the
            gradient-reachable tokens (:mod:`flowcl.analysis.hooks`). Decides the gate.
            Registry order is preserved in the evidence.
        groups: ``layer -> registry group``.
        d_in: ``layer -> input width``.
        alternative_rhos: Robustness views, e.g. ``{"valid": {...}, "all": {...}}``.
            Reported, and layers whose saturation classification differs between views
            are listed; they never change the verdict.
    """
    if not per_layer_rho:
        raise ValueError("gate2 received no layers")
    missing = sorted(set(per_layer_rho) - set(groups)) + sorted(
        set(per_layer_rho) - set(d_in)
    )
    if missing:
        raise ValueError(f"gate2: layers without group/d_in: {missing}")
    bad = {k: v for k, v in per_layer_rho.items() if not 0.0 < v <= 1.0}
    if bad:
        raise ValueError(f"gate2: rho_l must lie in (0, 1], got {bad}")

    saturated = [k for k, v in per_layer_rho.items() if v >= saturation_rho]
    fraction = len(saturated) / len(per_layer_rho)
    passed = fraction <= max_saturated_fraction

    per_group: dict[str, dict] = {}
    for group in dict.fromkeys(groups[k] for k in per_layer_rho):
        values = sorted(v for k, v in per_layer_rho.items() if groups[k] == group)
        n = len(values)
        median = (
            values[n // 2] if n % 2 else 0.5 * (values[n // 2 - 1] + values[n // 2])
        )
        per_group[group] = {
            "n_layers": n,
            "median_rho": median,
            "min_rho": values[0],
            "max_rho": values[-1],
            "n_saturated": sum(1 for k in saturated if groups[k] == group),
        }

    per_layer = {
        name: {
            "group": groups[name],
            "d_in": d_in[name],
            "rho": rho,
            "saturated": rho >= saturation_rho,
            "small_d": groups[name] in GATE2_SMALL_D_GROUPS,
        }
        for name, rho in per_layer_rho.items()
    }

    classification_changes: dict[str, list[str]] = {}
    for view, rhos in (alternative_rhos or {}).items():
        for name, rho in rhos.items():
            if name in per_layer:
                per_layer[name][f"rho_{view}"] = rho
    if alternative_rhos and len(alternative_rhos) >= 2:
        names = list(alternative_rhos)
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                common = set(alternative_rhos[a]) & set(alternative_rhos[b])
                changed = [
                    k
                    for k in per_layer_rho
                    if k in common
                    and (alternative_rhos[a][k] >= saturation_rho)
                    != (alternative_rhos[b][k] >= saturation_rho)
                ]
                classification_changes[f"{a}_vs_{b}"] = changed

    notes = ""
    if not passed:
        notes = (
            f"{len(saturated)}/{len(per_layer_rho)} registry layers already have "
            f"rho_l >= {saturation_rho:.2f} after Task 1. Hard projection (GPM) would "
            "leave those layers almost no free directions, so a GPM/SGP comparison "
            "would mostly measure frozen layers rather than protected ones. Report "
            "this as the Gate 2 outcome; do not lower eps to make it pass."
        )

    return GateResult(
        gate=2,
        question="Is projection geometrically plausible?",
        criterion=(
            f"rho_l after T1 not already ~1 on most registry layers: at eps={eps}, "
            f"at most {max_saturated_fraction:.0%} of layers with rho_l >= "
            f"{saturation_rho:.2f}"
        ),
        passed=passed,
        evidence={
            "eps": eps,
            "saturation_rho": saturation_rho,
            "max_saturated_fraction": max_saturated_fraction,
            "n_layers": len(per_layer_rho),
            "n_saturated": len(saturated),
            "saturated_fraction": fraction,
            "saturated_layers": saturated,
            "small_d_groups": list(GATE2_SMALL_D_GROUPS),
            "per_group": per_group,
            "per_layer": per_layer,
            "classification_changes": classification_changes,
        },
        notes=notes,
        run_id=run_id,
    )


def gate3(
    per_layer_c: dict[str, float],
    groups: dict[str, str],
    ci: dict[str, tuple[float, float]] | None = None,
    extra_layer_evidence: dict[str, dict] | None = None,
    aggregate_evidence: dict | None = None,
    eps: float = GATE3_DEFAULT_EPS,
    blocked_c: float = GATE3_BLOCKED_C,
    max_blocked_fraction: float = GATE3_MAX_BLOCKED_FRACTION,
    run_id: str | None = None,
) -> GateResult:
    """Evaluate §10.3 Gate 3: does the new task need protected directions?

    Args:
        per_layer_c: ``layer -> c_l`` at ``eps``: the mean per-batch interference of
            Task-2 gradients at the start of T2 against the Task-1 basis. Decides the
            gate. Registry order is preserved in the evidence.
        groups: ``layer -> registry group``.
        ci: ``layer -> (low, high)`` bootstrap interval over batches. A layer whose
            interval straddles ``blocked_c`` is flagged ``borderline``; the point
            estimate still decides, as in Gate 0.
        extra_layer_evidence: Per-layer numbers reported alongside (energy-weighted and
            full-dataset ``c_l``, the ``sqrt(rho_l)`` baseline, ...). Never decide.
        aggregate_evidence: Whole-network numbers, e.g. ``c_global``.

    Halves come from registry names: ``trunk.*`` and ``flow_head.*`` (the decoder).
    """
    if not per_layer_c:
        raise ValueError("gate3 received no layers")
    missing = sorted(set(per_layer_c) - set(groups))
    if missing:
        raise ValueError(f"gate3: layers without a group: {missing}")
    bad = {k: v for k, v in per_layer_c.items() if not 0.0 <= v <= 1.0}
    if bad:
        raise ValueError(f"gate3: c_l must lie in [0, 1], got {bad}")

    def half_of(name: str) -> str:
        if name.startswith("trunk."):
            return "trunk"
        if name.startswith("flow_head."):
            return "decoder"
        raise ValueError(f"gate3: cannot assign {name!r} to the trunk or the decoder")

    per_layer: dict[str, dict] = {}
    for name, c in per_layer_c.items():
        entry = {
            "group": groups[name],
            "half": half_of(name),
            "c": c,
            "blocked": c >= blocked_c,
            "small_d": groups[name] in GATE2_SMALL_D_GROUPS,
        }
        if ci is not None:
            low, high = ci[name]
            entry["ci_low"], entry["ci_high"] = low, high
            entry["borderline"] = low < blocked_c <= high
        entry.update((extra_layer_evidence or {}).get(name, {}))
        per_layer[name] = entry

    per_half: dict[str, dict] = {}
    for half in ("trunk", "decoder"):
        members = [n for n, e in per_layer.items() if e["half"] == half]
        if not members:
            continue
        blocked = [n for n in members if per_layer[n]["blocked"]]
        fraction = len(blocked) / len(members)
        per_half[half] = {
            "n_layers": len(members),
            "n_blocked": len(blocked),
            "blocked_fraction": fraction,
            "blocked_layers": blocked,
            "median_c": _median([per_layer_c[n] for n in members]),
            "failing": fraction > max_blocked_fraction,
        }
    failing_halves = [h for h, v in per_half.items() if v["failing"]]

    per_group: dict[str, dict] = {}
    for group in dict.fromkeys(groups[k] for k in per_layer_c):
        values = [v for k, v in per_layer_c.items() if groups[k] == group]
        per_group[group] = {
            "n_layers": len(values),
            "median_c": _median(values),
            "min_c": min(values),
            "max_c": max(values),
            "n_blocked": sum(1 for v in values if v >= blocked_c),
        }

    notes = ""
    if failing_halves:
        notes = (
            f"A strict majority of {' and '.join(failing_halves)} layers have "
            f"c_l >= {blocked_c:.2f}: Task 2's gradients lie almost entirely in Task 1's "
            "protected input subspace there, so hard projection (GPM) would block "
            "learning Task 2 in that half. The method comparison must rely on soft "
            "projection (SGP alpha_l) and report GPM as expected to fail on plasticity. "
            "Do not change eps to pass."
        )

    return GateResult(
        gate=3,
        question="Does the new task need protected directions?",
        criterion=(
            f"Task-2 gradients at the start of T2 vs the Task-1 basis at eps={eps}: a "
            f"layer is blocked when mean per-batch c_l >= {blocked_c:.2f}; fail when "
            f"more than {max_blocked_fraction:.0%} of the trunk OR of the decoder "
            "layers are blocked"
        ),
        passed=not failing_halves,
        evidence={
            "eps": eps,
            "blocked_c": blocked_c,
            "max_blocked_fraction": max_blocked_fraction,
            "n_layers": len(per_layer_c),
            "failing_halves": failing_halves,
            "per_half": per_half,
            "per_group": per_group,
            "borderline_layers": [n for n, e in per_layer.items() if e.get("borderline")],
            "small_d_groups": list(GATE2_SMALL_D_GROUPS),
            "per_layer": per_layer,
            **(aggregate_evidence or {}),
        },
        notes=notes,
        run_id=run_id,
    )


GATE4_CRITERIA = ("rho", "angles", "c")


def gate4(criteria: dict[str, dict], rule: dict, evidence: dict | None = None,
          run_id: str | None = None) -> GateResult:
    """Evaluate §9 Gate 4: is ``s``-conditioning justified?

    Unlike Gates 2-3, the rule's thresholds are pre-registered in
    ``configs/analysis/flowtime.yaml`` (committed before the run) and passed in as
    ``rule``; :mod:`flowcl.analysis.flowtime` applies them per layer and per seed.

    Args:
        criteria: ``"rho" | "angles" | "c"`` -> :func:`flowcl.analysis.flowtime.reproducible`
            output: the fraction of the *same* s-dependent layers passing in every seed.
        rule: the pre-registered rule block, recorded verbatim.
        evidence: everything else a reader needs (per-seed results, controls).
    """
    missing = [q for q in GATE4_CRITERIA if q not in criteria]
    if missing:
        raise ValueError(f"Gate 4 needs all three criteria; missing {missing}")
    failed = [q for q in GATE4_CRITERIA if not criteria[q]["passed"]]
    return GateResult(
        gate=4,
        question="Is s-conditioning justified?",
        criterion=(
            "Reproducible variation in rho_l(s), c_l(s) and non-trivial principal angles "
            "across s-bins over >= 3 seeds: for each criterion, at least "
            f"{rule['min_layer_fraction']:.0%} of the same s-dependent layers pass in every "
            "seed; all three criteria must pass"
        ),
        passed=not failed,
        evidence={
            "criteria": {q: criteria[q] for q in GATE4_CRITERIA},
            "failed_criteria": failed,
            "rule": rule,
            **(evidence or {}),
        },
        notes=(
            "" if not failed else
            "Failure does not establish that no layer contains flow-time-dependent geometry. "
            "It means that broad, layerwise-reproducible evidence sufficient to justify a "
            "general s-binned method was not obtained."
        ),
        run_id=run_id,
    )


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        raise ValueError("median of an empty list")
    mid = n // 2
    return ordered[mid] if n % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])


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
