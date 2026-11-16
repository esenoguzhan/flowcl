"""Checks of the 8-task resume smoke (``scripts/queue_t8_smoke.sh``; pre-E3 plan, B).

The E3 path resumes 4-task runs into ``seq_hetero_t8`` at stage 4, seen-only. It had never run on GPU.
The smoke (seed 2, 200 steps, 1 episode) makes these runs under one root, each a ``seq_hetero_t8``
resume at stage 4 unless noted:

* ``from_new/``: seq_ft, plain GPM and replay resumed from the new-code 4-task smokes, and f = 0.7
  resumed from its own 4-task smoke (``ne70/``);
* ``from_ref/``: seq_ft, plain GPM and replay resumed from the old-code 4-task smokes. Those, like the
  production sources, carry no loss, peak or stage-record files, and are bitwise equal to the new ones;
* ``probe/``: plain GPM and replay with probes every 50 steps;
* ``cut/``: plain GPM killed during stage 6, once its stage-5 record exists, as a reboot would leave it;
* ``again/``: that run resumed at stage 6 from its own folder.

:func:`check_smoke` then requires:

* ``from_ref`` equal to ``from_new`` at stages 4-7, ``probe`` equal to ``from_new``, ``again`` equal to
  ``from_new`` at stages 6-7 (checkpoint tensors, method artifacts, rollouts;
  :mod:`flowcl.experiments.smoke_compare`);
* every resumed run's matrix: rows 0-3 imported with T1-T4 and no T5-T8, rows 4-7 seen-only;
* stage records for all 8 stages, equal to result.json; loss files for 8 stages (``from_new``,
  ``probe``, ``again``) or 4 (``from_ref``); peak memory set for trained stages and null for stages
  imported from ``from_ref``'s old-code sources; probe files with T1..T(k) at steps 0, 50, ..., 200;
* the cut run: no result.json, records 0-5 and no record 6;
* strict sequence reports (no ``--allow-dirty``) for every resumed method run against the ``from_new``
  seq_ft resume, passing, with the checks by method: ``resume_source``, ``seed_namespace`` and an
  inherited ``t1_pairing`` for all; inherited ``identity`` at stages 0-1 for f = 0.7 and at stage 0 for
  replay; ``memory_chained`` across 3->4 for the GPM runs (not applicable for replay).
"""

from __future__ import annotations

import json
from pathlib import Path

from flowcl.experiments import sequence_report as sr
from flowcl.experiments.smoke_compare import compare_runs

CURRICULUM = "seq_hetero_t8"
SEED = 2
LABELS = {"seq_ft": "seq_ft", "gpm": "gpm_projected_adam", "replay": "replay", "ne70": "gpm_projected_adam_ne70"}
RESUMED = {"from_new": ("seq_ft", "gpm", "replay", "ne70"), "from_ref": ("seq_ft", "gpm", "replay"),
           "probe": ("gpm", "replay"), "again": ("gpm",)}
REPORTED = [("from_new", "gpm"), ("from_new", "ne70"), ("from_new", "replay"), ("from_ref", "gpm"),
            ("from_ref", "replay"), ("probe", "gpm"), ("probe", "replay"), ("again", "gpm")]
IDENTITY = {"ne70": [0, 1], "replay": [0]}
N_STAGES, START = 8, 4


def run_dir(root: Path, sub: str, method: str) -> Path:
    return Path(root) / sub / f"{CURRICULUM}__{LABELS[method]}__seed{SEED}"


def report_config(reference: str = f"{CURRICULUM}__seq_ft__seed{SEED}") -> dict:
    cfg = sr.load_report_config()
    cfg["criteria"]["expected_thresholds_by_reference"] = {reference: {"mode": "derived", "margin": 0.15}}
    return cfg


def matrix_problems(result: dict) -> list[str]:
    values, problems = result["retention_matrix"]["values"], []
    for i, row in enumerate(values):
        for j, v in enumerate(row):
            expected = (j < START) if i < START else (j <= i)
            if (v is not None) != expected:
                problems.append(f"cell ({i},{j}) is {'set' if v is not None else 'null'}")
    return problems


def run_fields(run: Path, sub: str) -> dict:
    """The record fields of one resumed run."""
    result = json.loads((run / "result.json").read_text())
    problems = matrix_problems(result)
    stages = result["stages"]
    for k in range(N_STAGES):
        path = run / "train" / f"stage{k}_record.json"
        if not path.is_file():
            problems.append(f"no record for stage {k}")
        elif json.loads(path.read_text()) != stages[k]:
            problems.append(f"stage {k}'s record differs from result.json")
    with_loss = [k for k, s in enumerate(stages) if s.get("loss_file")]
    expected_loss = list(range(START, N_STAGES)) if sub == "from_ref" else list(range(N_STAGES))
    if with_loss != expected_loss:
        problems.append(f"loss files for stages {with_loss}, expected {expected_loss}")
    for k in with_loss:
        if not (run / stages[k]["loss_file"]["path"]).is_file():
            problems.append(f"stage {k}'s loss file is missing")
    peaks = result["systems"]["peak_gpu"]["by_stage"]
    first_own = 6 if sub == "again" else START
    for k in range(N_STAGES):
        value = (peaks.get(str(k)) or {}).get("train_allocated_bytes")
        if k >= first_own and not value:
            problems.append(f"stage {k} (trained here) has no peak memory")
        if sub == "from_ref" and k < START and value is not None:
            problems.append(f"stage {k} (imported from an old-code source) has peak memory {value}")
    if sub == "probe":
        for k in range(START, N_STAGES):
            entry = stages[k].get("probe_file")
            if not entry:
                problems.append(f"stage {k} has no probe file")
                continue
            probes = json.loads((run / entry["path"]).read_text())
            if len(probes["tasks"]) != k or [r["step"] for r in probes["records"]] != [0, 50, 100, 150, 200]:
                problems.append(f"stage {k}'s probes: {len(probes['tasks'])} tasks, steps "
                                f"{[r['step'] for r in probes['records']]}")
    if result.get("eval_unseen") is not False:
        problems.append("eval_unseen is not recorded as false")
    cache = (result.get("resume") or {}).get("text_cache") or {}
    if len(cache) != N_STAGES:
        problems.append(f"the text-cache record covers {len(cache)} tasks")
    return {"passed": not problems, "problems": problems, "text_cache": cache}


def cut_state(run: Path) -> dict:
    records = sorted(int(p.stem.split("_")[0].removeprefix("stage")) for p in (run / "train").glob("stage*_record.json"))
    problems = []
    if (run / "result.json").exists():
        problems.append("the interrupted run has a result.json")
    if records != list(range(6)):
        problems.append(f"records for stages {records}, expected 0-5")
    return {"passed": not problems, "problems": problems, "records": records}


def report_check(report: dict, method: str) -> dict:
    checks, problems = report["provenance_checks"], []
    failures = sr.strict_failures(report)
    problems += [f"strict: {f}" for f in failures]
    for name in ("resume_source", "seed_namespace", "t1_pairing"):
        if (checks.get(name) or {}).get("passed") is not True:
            problems.append(f"{name} did not pass")
    if not (checks.get("t1_pairing") or {}).get("inherited_from"):
        problems.append("t1_pairing is not inherited")
    if method in IDENTITY:
        identity = checks.get("identity") or {}
        if identity.get("passed") is not True or identity.get("expected_stages") != IDENTITY[method]:
            problems.append(f"identity: {identity.get('expected_stages')} passed={identity.get('passed')}")
    elif "identity" in checks:
        problems.append("an identity check where none was configured")
    chained = checks.get("memory_chained") or {}
    if method in ("gpm", "ne70"):
        if chained.get("passed") is not True or "3->4" not in (chained.get("steps") or {}):
            problems.append("memory_chained did not pass across 3->4")
    elif chained.get("passed") is not None:
        problems.append("memory_chained should be not applicable")
    return {"passed": not problems, "problems": problems}


def check_smoke(root: Path) -> dict:
    root = Path(root)
    out: dict = {"comparisons": {}, "fields": {}, "reports": {}}
    pairs = {f"from_ref_{m}": (run_dir(root, "from_new", m), run_dir(root, "from_ref", m), None)
             for m in RESUMED["from_ref"]}
    pairs.update({f"probe_{m}": (run_dir(root, "from_new", m), run_dir(root, "probe", m), None)
                  for m in RESUMED["probe"]})
    pairs["again_gpm"] = (run_dir(root, "from_new", "gpm"), run_dir(root, "again", "gpm"), [6, 7])
    for name, (a, b, stages) in pairs.items():
        full = compare_runs(a, b, stages=stages)
        if stages is None and set(full.get("stages", {})) != {str(k) for k in range(START, N_STAGES)}:
            full = {**full, "passed": False, "reason": f"compared stages {sorted(full.get('stages', {}))}"}
        out["comparisons"][name] = {"passed": full["passed"], "detail": full}
    for sub, methods in RESUMED.items():
        for m in methods:
            out["fields"][f"{sub}_{m}"] = run_fields(run_dir(root, sub, m), sub)
    out["cut"] = cut_state(run_dir(root, "cut", "gpm"))
    reference = run_dir(root, "from_new", "seq_ft")
    (root / "reports").mkdir(exist_ok=True)
    for sub, m in REPORTED:
        report = sr.build_report(run_dir(root, sub, m), reference, config=report_config())
        (root / "reports" / f"{sub}_{m}.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
        out["reports"][f"{sub}_{m}"] = report_check(report, m)
    sections = [*out["comparisons"].values(), *out["fields"].values(), out["cut"], *out["reports"].values()]
    out["passed"] = all(s["passed"] for s in sections)
    return out
