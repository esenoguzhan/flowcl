"""The allocation-rules pilot: three allocation rules against flat 0.99 at matched capacity.

The rule is ``configs/analysis/allocation_rules.yaml`` (pre-registered; its header states the
question, the predictions and every label). Each arm (``gpm_alloc_{novelty,size,room}``) reaches,
after T2, T3 and T4, the integer protected count ``Σ d_out k`` that ``gpm_e99`` reached on the same
seed, and differs from it only in where the capacity goes (:mod:`flowcl.analysis.allocation`).

* :func:`predict` — the post-T2 extension of every rule, offline. The T2 Grams are identical in all
  arms (shared stage-1 weights and capture seeds), so ne90's T2 artifact (its residual spectrum and
  history; tr K reconstructed as in :func:`flowcl.experiments.high_protection.predicted_t2_ranks`)
  determines every rule's T2 allocation. It first reproduces e99's (flat 0.99) and ne90's
  (novelty 0.9) T2 ranks per layer, and e99's counts against the method configs' schedules.
* :func:`check_arm` — one finished arm: identity and T1 memory against plain GPM, matching after
  the post-T2 and post-T3 extensions (post-T4 reported for E3), the post-T2 ranks against the
  prediction (jump layers may differ by one: ``prediction_borderline``), distinctness from e99
  (capacity reallocated across layers after post-T2 or post-T3), the strict sequence report.
* :func:`summary` — the evaluation gates (provenance, rep-0 reproduction, pairing of every task x
  repetition across arms, fresh noise across repetitions), ΔACC = ACC(arm) − ACC(e99) after T4 over
  tasks x states x repetitions with a state-clustered CI per task column, the labels, and the
  descriptive measures (A5 rollout bootstrap at rep 0, capacity per stage and half, R and the
  subspace overlap per extension).
* :func:`smoke_configs` — method configs for a smoke, with the schedule read off a smoke e99 run.

Every number names the files it came from; the summary pins the rule's SHA-256.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from flowcl.utils.libero_paths import repo_root
from flowcl.utils.run import atomic_write_text, file_sha256, git_sha

ARMS = ("novelty", "size", "room")
COMPARATOR = "e99"
LABELS = {"plain": "gpm_projected_adam", "e99": "gpm_projected_adam_e99", "ne90": "gpm_projected_adam_ne90",
          "seq_ft": "seq_ft", **{a: f"gpm_projected_adam_alloc_{a}" for a in ARMS}}
CURRICULUM = "seq_hetero"
STAGE = 3  # the checkpoint after T4


def rule_path(path: str | Path | None = None) -> Path:
    return Path(path) if path else repo_root() / "configs" / "analysis" / "allocation_rules.yaml"


def load_rule(path: str | Path | None = None) -> dict:
    return OmegaConf.to_container(OmegaConf.load(rule_path(path)), resolve=True)


def namespace(seed: int) -> str:
    return f"{CURRICULUM}__seq_ft__seed{seed}"


def rep_namespace(seed: int, rep: int) -> str:
    """Repetition 0 is the runs' own namespace; ``r >= 1`` the same states with fresh noise."""
    return namespace(seed) if rep == 0 else f"{namespace(seed)}__rep{rep}"


def run_dir(results: Path, role: str, seed: int) -> Path:
    return Path(results) / f"{CURRICULUM}__{LABELS[role]}__seed{seed}"


def out_dir(results: Path, seed: int) -> Path:
    return Path(results) / "allocation_rules" / f"seed{seed}"


def eval_path(results: Path, seed: int, arm: str, rep: int) -> Path:
    return out_dir(results, seed) / f"{arm}.rep{rep}.eval.json"


def report_path(results: Path, arm: str, seed: int) -> Path:
    return Path(results) / f"gpm_seq_alloc_{arm}_seed{seed}" / "report.json"


def _k_hash(k: dict[str, int]) -> str:
    return hashlib.sha256(json.dumps(sorted(k.items())).encode()).hexdigest()


def _dims(run: Path, layers: list[str]) -> dict[str, tuple[int, int]]:
    from flowcl.experiments.high_protection import layer_dims

    return layer_dims(Path(run) / "checkpoints" / "stage0.pt", layers)


def _memory(run: Path, task: int):
    from flowcl.analysis.subspace import load_bases

    return load_bases(Path(run) / "method" / f"memory_task{task}.pt")


def counts_from_history(history: dict, d_out: dict[str, int]) -> dict[int, int]:
    """``task -> Σ d_out k_after`` from a memory artifact's ``memory_history``."""
    return {int(t): sum(d_out[n] * h[n]["k_after"] for n in d_out) for t, h in history.items()}


def halves(k_after: dict[str, int], dims: dict[str, tuple[int, int]]) -> dict[str, float]:
    """Parameter-weighted protected fraction of all registry weights, the trunk and the decoder."""
    out = {}
    for half, pick in (("all", lambda n: True), ("trunk", lambda n: n.startswith("trunk.")),
                       ("decoder", lambda n: not n.startswith("trunk."))):
        names = [n for n in dims if pick(n)]
        out[half] = sum(dims[n][0] * k_after[n] for n in names) / sum(dims[n][0] * dims[n][1] for n in names)
    return out


# ---- the offline prediction --------------------------------------------------------------------------


def artifact_stats(bases: dict, info: dict, d_out: dict[str, int], rank_tol: float | None) -> list:
    """:class:`LayerStats` of an extension from its artifact: the stored residual spectrum
    (``singular_values**2``) and history, tr K reconstructed as ``Σ_{i<=k_added} λ_i / (captured − p)``.

    A layer the artifact did not extend has no stored spectrum; it must have no energy outside the
    memory (or a full memory), and then no rule can extend it (``cap = 0``).
    """
    from flowcl.analysis.allocation import LayerStats
    from flowcl.analysis.subspace import NEW_ENERGY_ATOL

    out = []
    for name, h in info.items():
        p, ka, d, kb = h["proj_energy_fraction"], h["k_added"], h["d_in"], h["k_before"]
        if ka == 0:
            if 1.0 - p > NEW_ENERGY_ATOL and kb < d:
                raise ValueError(f"{name}: the artifact added nothing, yet {1 - p:.3e} of the energy is "
                                 "outside the memory; its spectrum is not stored, so it cannot be predicted")
            out.append(LayerStats(name=name, d_in=d, d_out=d_out[name], k_before=kb, proj=p, total=1.0,
                                  eigenvalues=torch.zeros(0, dtype=torch.float64), cap=0))
            continue
        lam = bases[name].singular_values.to(torch.float64) ** 2
        total = float(lam[:ka].sum()) / (h["captured_energy_fraction"] - p)
        tol = rank_tol if rank_tol is not None else d * torch.finfo(torch.float64).eps
        rank = int((lam > tol * float(lam[0])).sum())
        out.append(LayerStats(name=name, d_in=d, d_out=d_out[name], k_before=kb, proj=p, total=total,
                              eigenvalues=lam, cap=min(rank, d - kb)))
    return out


def predict(seed: int, *, results: Path | None = None, rule: dict | None = None) -> dict:
    """Every rule's post-T2 allocation for ``seed``, with the reproduction checks (module docstring)."""
    from flowcl.analysis.allocation import allocate, match_capacity, reallocated
    from flowcl.data.config import load_method_config
    from flowcl.experiments.gate2 import load_subspace_config

    results = Path(results) if results else repo_root() / "results"
    ne90, e99 = run_dir(results, "ne90", seed), run_dir(results, "e99", seed)
    bases, meta = _memory(ne90, 1)
    info = meta["memory_history"]["1"]
    _, e99_meta = _memory(e99, 3)
    he = e99_meta["memory_history"]
    names = list(info)
    dims = _dims(ne90, names)
    d_out = {n: dims[n][0] for n in names}
    cap_cfg = load_subspace_config(repo_root() / "configs" / "analysis" / "subspace.yaml")
    stats = artifact_stats(bases, info, d_out, cap_cfg.rank_tol)
    del bases
    counts = counts_from_history(he, d_out)
    k_before = {s.name: s.k_before for s in stats}
    k_e99 = {n: he["1"][n]["k_after"] for n in names}

    checks = {}
    t1_e99 = {n: he["0"][n]["k_after"] for n in names}
    t1_ne90 = {n: meta["memory_history"]["0"][n]["k_after"] for n in names}
    checks["t1_ranks_equal"] = {"passed": t1_e99 == t1_ne90 and t1_e99 == k_before}
    flat = {n: k_before[n] + k for n, k in allocate("flat", stats, 0.99).items()}
    nov = {n: k_before[n] + k for n, k in allocate("novelty", stats, 0.9).items()}
    k_ne90 = {n: info[n]["k_after"] for n in names}
    checks["flat_0.99_reproduces_e99"] = {"passed": flat == k_e99,
                                          "layers_equal": sum(flat[n] == k_e99[n] for n in names),
                                          "n_layers": len(names)}
    checks["novelty_0.9_reproduces_ne90"] = {"passed": nov == k_ne90,
                                             "layers_equal": sum(nov[n] == k_ne90[n] for n in names),
                                             "n_layers": len(names)}
    wanted = [counts[1], counts[2], counts[3]]
    config_problems = []
    for arm in ARMS:
        _, kwargs = load_method_config(f"gpm_alloc_{arm}")
        got = (kwargs.get("capacity_schedule") or {}).get(namespace(seed))
        if got != wanted:
            config_problems.append(f"gpm_alloc_{arm}: {got} != e99's {wanted}")
    checks["schedule_matches_e99"] = {"passed": not config_problems, "problems": config_problems,
                                      "e99_counts": wanted}

    arms = {}
    for arm in ARMS:
        m = match_capacity(arm, stats, counts[1])
        k_after = {n: k_before[n] + m["k_added"][n] for n in names}
        arms[arm] = {**{k: v for k, v in m.items() if k != "k_added"}, "halves": halves(k_after, dims),
                     "reallocated_vs_e99": reallocated(k_after, k_e99, dims), "k_after": k_after,
                     "k_hash": _k_hash(k_after)}
    return {"seed": seed, "namespace": namespace(seed), "target_t2": counts[1], "e99_counts": counts,
            "full": sum(do * di for do, di in dims.values()), "e99_halves_t2": halves(k_e99, dims),
            "checks": checks, "passed": all(c["passed"] for c in checks.values()), "arms": arms,
            "inputs": {"ne90_memory1": {"path": str(ne90 / "method" / "memory_task1.pt"),
                                        "sha256": file_sha256(ne90 / "method" / "memory_task1.pt")},
                       "e99_memory3": {"path": str(e99 / "method" / "memory_task3.pt"),
                                       "sha256": file_sha256(e99 / "method" / "memory_task3.pt")}}}


# ---- smoke ---------------------------------------------------------------------------------------------


def smoke_configs(e99_run: Path, out: Path) -> dict:
    """Write ``gpm_alloc_<arm>_smoke.yaml`` into ``out``: the production configs with the schedule of
    ``e99_run``'s own counts (a smoke e99), keyed by its seed namespace."""
    from flowcl.data.config import load_method_config

    e99_run, out = Path(e99_run), Path(out)
    _, meta = _memory(e99_run, 3)
    history = meta["memory_history"]
    names = list(history["3"])
    d_out = {n: d[0] for n, d in _dims(e99_run, names).items()}
    counts = counts_from_history(history, d_out)
    ns = meta["seed_namespace_run_id"]
    out.mkdir(parents=True, exist_ok=True)
    written = {}
    for arm in ARMS:
        name, kwargs = load_method_config(f"gpm_alloc_{arm}")
        kwargs = {"name": name, **kwargs, "capacity_schedule": {ns: [counts[1], counts[2], counts[3]]}}
        path = out / f"gpm_alloc_{arm}_smoke.yaml"
        OmegaConf.save(OmegaConf.create(kwargs), path)
        written[arm] = str(path)
    return {"passed": True, "namespace": ns, "counts": counts, "configs": written}


# ---- per-arm gates ---------------------------------------------------------------------------------------

# Run-config keys added after the archived e99 runs (7 Oct), with the neutral value a newer run records.
NEUTRAL_NEW_KEYS = {"train.augment_shift_px": 0, "train.augment_color": 0.0, "train.augment_daylight": 0.0,
                    "train.checkpoint_every": 0, "provenance.identity_reference_checkpoints": {}}


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and v:
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def config_diff(new: Path, old: Path) -> dict:
    """Two runs' ``config.yaml`` agree on everything but the run ids and the method block; keys the
    older run lacks are allowed only from :data:`NEUTRAL_NEW_KEYS` at their neutral value."""
    a = _flatten(OmegaConf.to_container(OmegaConf.load(Path(new) / "config.yaml"), resolve=True))
    b = _flatten(OmegaConf.to_container(OmegaConf.load(Path(old) / "config.yaml"), resolve=True))
    skip = lambda k: k in ("run_id", "method_run_id") or k.startswith("method.")  # noqa: E731
    problems = []
    for k in sorted(set(a) | set(b)):
        if skip(k):
            continue
        if k in a and k not in b:
            if not (k in NEUTRAL_NEW_KEYS and a[k] == NEUTRAL_NEW_KEYS[k]):
                problems.append(f"{k}: {a[k]!r} only in the newer run")
        elif k not in a:
            problems.append(f"{k}: {b[k]!r} only in the older run")
        elif a[k] != b[k]:
            problems.append(f"{k}: {a[k]!r} != {b[k]!r}")
    return {"passed": not problems, "problems": problems, "new": str(new), "old": str(old)}


def required_report_check(path: Path, expected_run: str, smoke: bool) -> dict:
    """The strict sequence report exists, is for the run, and every check it requires passed
    (``clean_git_sha`` excused in smoke)."""
    if not Path(path).is_file():
        return {"passed": False, "missing": str(path)}
    report = json.loads(Path(path).read_text())
    checks = report.get("provenance_checks") or {}
    problems = []
    if report.get("method_run_id") != expected_run:
        problems.append(f"report is for {report.get('method_run_id')!r}, expected {expected_run!r}")
    for name in report.get("provenance_required") or []:
        check = checks.get(name)
        if not isinstance(check, dict) or "passed" not in check:
            problems.append(f"{name}: absent")
        elif not check["passed"] and not (smoke and name == "clean_git_sha"):
            problems.append(f"{name}: failed")
    if not report.get("provenance_required"):
        problems.append("the report names no required checks")
    return {"passed": not problems, "problems": problems, "sha256": file_sha256(path)}


def _as_recorded(run: Path, plain: Path) -> Path:
    """``plain`` spelled as the run's provenance recorded it (the runner keeps its CLI path, e.g. the
    relative ``results/...``), when both name the same directory; :func:`ne70_queue.check_run` compares
    the spelling."""
    try:
        prov = OmegaConf.to_container(OmegaConf.load(Path(run) / "config.yaml"), resolve=True).get("provenance") or {}
    except (OSError, ValueError):
        return Path(plain)
    recorded = prov.get("identity_reference_run")
    if recorded and Path(recorded).resolve() == Path(plain).resolve():
        return Path(recorded)
    return Path(plain)


def check_arm(arm: str, seed: int, *, results: Path | None = None, rule: dict | None = None,
              comparator: Path | None = None, plain: Path | None = None, smoke: bool = False,
              prediction: dict | None = None, run_only: bool = False) -> dict:
    """The gates of one finished arm (module docstring); ``passed`` is their conjunction.

    ``run_only`` (the queue's check after each run) stops before distinctness and the strict report,
    which the summary adds; a run that fails these run gates is not evaluated.
    """
    from flowcl.analysis.allocation import reallocated, subspace_overlap
    from flowcl.experiments.ne70_queue import check_run

    rule = rule or load_rule()
    g = rule["gates"]
    results = Path(results) if results else repo_root() / "results"
    run = run_dir(results, arm, seed)
    e99 = Path(comparator) if comparator else run_dir(results, "e99", seed)
    plain = Path(plain) if plain else run_dir(results, "plain", seed)
    gates: dict = {}
    descriptive: dict = {}
    gates["comparison"] = check_run(run, _as_recorded(run, plain))
    gates["comparator_config"] = config_diff(run, e99) if (run / "config.yaml").is_file() else {
        "passed": False, "missing": str(run / "config.yaml")}
    try:
        _, meta = _memory(run, 3)
        _, e99_meta = _memory(e99, 3)
    except (OSError, ValueError, KeyError) as exc:
        gates["artifacts"] = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
        return {"arm": arm, "seed": seed, "passed": False, "gates": gates, "descriptive": descriptive}
    hist, alloc = meta["memory_history"], meta.get("allocation_history") or {}
    he = e99_meta["memory_history"]
    names = list(hist["3"])
    dims = _dims(run, names)
    gates["rule_config"] = {"passed": meta["config"].get("allocation_rule") == arm,
                            "allocation_rule": meta["config"].get("allocation_rule")}

    tol = g["matching_tol_pp"]
    matching = {}
    for t in [*g["matching_tasks"], g["e3_matching_task"]]:
        a = alloc.get(str(t))
        if a is None:
            matching[t] = {"passed": False, "missing": True}
            continue
        ok = abs(a["error_pp"]) <= tol
        matching[t] = {"passed": ok, "status": a["status"], "error_pp": a["error_pp"], "count": a["count"],
                       "target": a["target"], "lower": a["lower"], "upper": a["upper"],
                       "floor_count": a["floor_count"], "ceiling_count": a["ceiling_count"],
                       "knob": a["knob"], "jump_layers": a["jump_layers"]}
    gates["matching"] = {"passed": all(matching[t]["passed"] for t in g["matching_tasks"]),
                         "tasks": g["matching_tasks"], "tol_pp": tol, "per_task": matching}
    descriptive["e3_matching"] = matching[g["e3_matching_task"]]

    if smoke:
        gates["implementation"] = {"passed": True, "skipped": "smoke: no registered prediction"}
    else:
        prediction = prediction or predict(seed, results=results, rule=rule)
        pred = prediction["arms"][arm]
        actual = {n: hist["1"][n]["k_after"] for n in names}
        differing = {n: actual[n] - pred["k_after"][n] for n in names if actual[n] != pred["k_after"][n]}
        borderline = {n: d for n, d in differing.items()
                      if n in pred["jump_layers"] and abs(d) <= g["borderline_max_abs"]}
        bad = {n: d for n, d in differing.items() if n not in borderline}
        registered = (rule.get("predictions") or {}).get(str(seed), {}).get(arm, {}).get("k_hash")
        gates["implementation"] = {"passed": not bad and (registered is None or registered == pred["k_hash"]),
                                   "differing": differing, "prediction_borderline": sorted(borderline),
                                   "prediction_hash": pred["k_hash"], "registered_hash": registered}

    if run_only:
        return {"arm": arm, "seed": seed, "run": str(run), "passed": all(x["passed"] for x in gates.values()),
                "gates": gates, "descriptive": descriptive, "run_only": True}
    realloc, overlap = {}, {}
    for t in g["distinct_tasks"]:
        ka = {n: hist[str(t)][n]["k_after"] for n in names}
        kb = {n: he[str(t)][n]["k_after"] for n in names}
        realloc[t] = reallocated(ka, kb, dims)
        bases_a, _ = _memory(run, t)
        bases_b, _ = _memory(e99, t)
        overlap[t] = subspace_overlap({n: bases_a[n].vectors for n in names},
                                      {n: bases_b[n].vectors for n in names}, {n: dims[n][0] for n in names})
        del bases_a, bases_b
    gates["distinct"] = {"passed": max(realloc.values()) * 100 >= g["distinct_min_pp"],
                         "reallocated_pp": {t: 100 * r for t, r in realloc.items()},
                         "min_pp": g["distinct_min_pp"]}
    descriptive["subspace_overlap_vs_e99"] = overlap
    descriptive["capacity"] = {t: halves({n: hist[str(t)][n]["k_after"] for n in names}, dims)
                               for t in sorted(int(k) for k in hist)}
    descriptive["capacity_e99"] = {t: halves({n: he[str(t)][n]["k_after"] for n in names}, dims)
                                   for t in sorted(int(k) for k in he)}
    gates["strict_report"] = required_report_check(report_path(results, arm, seed), run.name, smoke)
    return {"arm": arm, "seed": seed, "run": str(run), "passed": all(x["passed"] for x in gates.values()),
            "gates": gates, "descriptive": descriptive}


# ---- the primary measure (pure) ------------------------------------------------------------------------


def acc_state_paired_ci(a: dict[str, np.ndarray], b: dict[str, np.ndarray], bootstrap: dict) -> dict:
    """``ACC(a) − ACC(b)`` over tasks x (repetitions, states) outcome arrays, paired.

    ACC is the mean over task columns of each column's mean success. A state with its repetitions is
    one observation; the CI resamples states independently per task column (percentile bootstrap).
    """
    if sorted(a) != sorted(b):
        raise ValueError("both arms need the same task columns")
    diffs = []
    for key in sorted(a):
        x, y = np.asarray(a[key], dtype=np.float64), np.asarray(b[key], dtype=np.float64)
        if x.shape != y.shape or x.ndim != 2:
            raise ValueError(f"{key}: expected equal (repetitions, states) arrays, got {x.shape} and {y.shape}")
        diffs.append((x - y).mean(axis=0))
    point = float(np.mean([d.mean() for d in diffs]))
    rng = np.random.default_rng(bootstrap["seed"])
    n = bootstrap["n_resamples"]
    draws = np.zeros(n)
    for d in diffs:
        idx = rng.integers(0, d.size, size=(n, d.size))
        draws += d[idx].mean(axis=1)
    draws /= len(diffs)
    alpha = 1.0 - bootstrap["confidence"]
    low, high = (float(v) for v in np.quantile(draws, [alpha / 2, 1 - alpha / 2]))
    return {"diff": point, "low": min(low, point), "high": max(high, point),
            "n_states": int(diffs[0].size), "repetitions": int(np.asarray(next(iter(a.values()))).shape[0]),
            "n_tasks": len(diffs)}


def label(ci: dict) -> str:
    if ci["low"] > 0:
        return "allocation_helps"
    if ci["high"] < 0:
        return "allocation_hurts"
    return "unresolved"


def overall(arms: dict[str, dict]) -> dict:
    """The across-arm sentence is allowed only when all three arms are valid and distinct and null."""
    labels = {a: r["label"] for a, r in arms.items()}
    if sorted(labels) == sorted(ARMS) and all(v == "unresolved" for v in labels.values()):
        return {"outcome": "no_resolvable_difference", "labels": labels,
                "text": "The tested allocations showed no resolvable ACC difference from flat at matched "
                        "capacity (seed 0, exploratory)."}
    return {"outcome": "per_arm", "labels": labels, "text": "Reported arm by arm."}


# ---- the summary ---------------------------------------------------------------------------------------


def _first_failed(gates: dict) -> str:
    order = [("comparison", "invalid_comparison"), ("comparator_config", "invalid_comparison"),
             ("artifacts", "invalid_run"), ("rule_config", "invalid_run"),
             ("matching", "invalid_matching"), ("implementation", "invalid_implementation"),
             ("distinct", "not_distinct"), ("strict_report", "invalid_run")]
    for key, name in order:
        if key in gates and not gates[key]["passed"]:
            return name
    return "invalid_run"


def summary(seed: int, *, results: Path | None = None, rule_file: Path | None = None, smoke: bool = False,
            comparator: Path | None = None, plain: Path | None = None) -> dict:
    """Gates, the primary measure and the labels of one seed (module docstring); written to
    ``<results>/allocation_rules/seed<S>/summary.json``."""
    from flowcl.analysis.rollout_bootstrap import RunRollouts, paired_rollout_bootstrap
    from flowcl.envs.evaluation import EvaluationReport
    from flowcl.experiments.high_protection_followup import provenance_check, reproduction_check

    rule = load_rule(rule_file)
    results = Path(results) if results else repo_root() / "results"
    ev = rule["evaluation"]
    n_episodes = rule["smoke"]["n_episodes"] if smoke else ev["n_episodes"]
    reps = list(ev["repetitions"])
    bootstrap = rule["bootstrap"]
    runs = {a: run_dir(results, a, seed) for a in ARMS}
    runs[COMPARATOR] = Path(comparator) if comparator else run_dir(results, COMPARATOR, seed)
    out: dict = {"git_sha": git_sha(), "seed": seed, "smoke": smoke,
                 "rule": {"path": str(rule_path(rule_file)), "sha256": file_sha256(rule_path(rule_file))},
                 "runs": {a: str(p) for a, p in runs.items()}, "arms": {}, "evaluation_gates": {}}
    prediction = None if smoke else predict(seed, results=results, rule=rule)
    if prediction is not None:
        out["prediction_checks"] = prediction["checks"]
    arm_checks = {}
    for arm in ARMS:
        try:
            arm_checks[arm] = check_arm(arm, seed, results=results, rule=rule, comparator=runs[COMPARATOR],
                                        plain=plain, smoke=smoke, prediction=prediction)
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            arm_checks[arm] = {"passed": False, "gates": {"artifacts": {"passed": False,
                                                                        "error": f"{type(exc).__name__}: {exc}"}}}

    # ---- evaluation cells: rep 0 from each run's stored stage-3 evaluation, reps >= 1 from evaluate.py
    task_keys = json.loads((runs[COMPARATOR] / "result.json").read_text())["task_keys"]
    cells: dict[str, dict[int, dict[str, object]]] = {a: {} for a in runs}
    eg = out["evaluation_gates"]
    for arm, run in runs.items():
        checkpoint = run / "checkpoints" / f"stage{STAGE}.pt"
        try:
            stored = EvaluationReport.load(run / "eval" / f"stage{STAGE}.json").by_task()
            cells[arm][0] = {k: stored[k] for k in task_keys}
            rerun_positions = ev["rep0_rerun"][arm]
            rerun_keys = task_keys if rerun_positions == "all" else [task_keys[j] for j in rerun_positions]
            new0 = EvaluationReport.load(eval_path(results, seed, arm, 0))
            eg[f"provenance_{arm}_rep0"] = _provenance(provenance_check, checkpoint, run.name, new0,
                                                       rep_namespace(seed, 0), n_episodes, rerun_keys)
            for key in rerun_keys:
                eg[f"reproduction_{arm}_{key}"] = reproduction_check(new0.by_task()[key], stored[key], n_episodes)
            for r in [r for r in reps if r > 0]:
                report = EvaluationReport.load(eval_path(results, seed, arm, r))
                eg[f"provenance_{arm}_rep{r}"] = _provenance(provenance_check, checkpoint, run.name, report,
                                                             rep_namespace(seed, r), n_episodes, task_keys)
                cells[arm][r] = {k: report.by_task()[k] for k in task_keys}
        except (OSError, ValueError, KeyError) as exc:
            eg[f"inputs_{arm}"] = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
    complete = all(sorted(cells[a]) == reps for a in runs)
    if complete:
        unpaired = [f"{k} rep {r}" for r in reps for k in task_keys
                    if len({tuple(cells[a][r][k].seeds) for a in runs}) != 1]
        eg["pairing"] = {"passed": not unpaired, "unpaired": unpaired,
                         "compared": len(reps) * len(task_keys) * len(runs)}
        stale = [f"{a} {k}" for a in runs for k in task_keys
                 if len({tuple(cells[a][r][k].seeds) for r in reps}) != len(reps)]
        eg["fresh_noise"] = {"passed": not stale, "repeated": stale}
    eval_ok = complete and all(x.get("passed") for x in eg.values())

    payloads = {}
    for arm, run in runs.items():
        try:
            payloads[arm] = json.loads((run / "result.json").read_text())
        except (OSError, ValueError):
            payloads[arm] = None
    for arm in ARMS:
        c = arm_checks[arm]
        entry = {"gates": c.get("gates"), "descriptive": c.get("descriptive"), "valid": bool(c["passed"])}
        if not c["passed"]:
            entry["label"] = _first_failed(c.get("gates") or {})
        elif not eval_ok:
            entry["label"] = "invalid_evaluation"
        else:
            a_out = {k: np.array([cells[arm][r][k].successes for r in reps], dtype=np.float64) for k in task_keys}
            b_out = {k: np.array([cells[COMPARATOR][r][k].successes for r in reps], dtype=np.float64)
                     for k in task_keys}
            ci = acc_state_paired_ci(a_out, b_out, bootstrap)
            entry["primary"] = {"delta_acc": ci,
                                "acc_arm": float(np.mean([v.mean() for v in a_out.values()])),
                                "acc_e99": float(np.mean([v.mean() for v in b_out.values()])),
                                "per_task": {k: float((a_out[k] - b_out[k]).mean()) for k in task_keys}}
            entry["label"] = label(ci)
        if payloads.get(arm) and payloads.get(COMPARATOR):
            try:
                est = paired_rollout_bootstrap(RunRollouts.from_payload(payloads[arm], label=arm),
                                               RunRollouts.from_payload(payloads[COMPARATOR], label=COMPARATOR),
                                               n_bootstrap=bootstrap["n_resamples"],
                                               confidence=bootstrap["confidence"], seed=bootstrap["seed"])
                entry["rep0_paired"] = {m: {"value": e.value, "low": e.low, "high": e.high} for m, e in est.items()}
                entry["retention"] = {"arm": payloads[arm]["retention_matrix"],
                                      "e99": payloads[COMPARATOR]["retention_matrix"]}
            except (ValueError, KeyError) as exc:
                entry["rep0_paired"] = {"error": f"{type(exc).__name__}: {exc}"}
        out["arms"][arm] = entry
    out["evaluation_valid"] = eval_ok
    valid = {a: r for a, r in out["arms"].items() if r.get("label") in ("allocation_helps", "allocation_hurts",
                                                                       "unresolved")}
    out["overall"] = overall(valid) if len(valid) == len(ARMS) else {
        "outcome": "per_arm", "labels": {a: r.get("label") for a, r in out["arms"].items()},
        "text": "Reported arm by arm (not every arm is valid and distinct)."}
    out["promote_to_seeds_1_2"] = sorted(a for a, r in out["arms"].items() if r.get("label") == "allocation_helps")
    path = out_dir(results, seed) / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(out, indent=2, default=str) + "\n")
    out["path"] = str(path)
    return out


def _provenance(check, checkpoint: Path, run_name: str, report, ns: str, n_episodes: int, keys: list[str]) -> dict:
    problems = []
    for key in keys:
        r = check(checkpoint, run_name, report, ns, STAGE, n_episodes, key)
        problems += [f"{key}: {p}" for p in r["problems"]]
    return {"passed": not problems, "problems": problems, "checkpoint_sha256": file_sha256(checkpoint),
            "tasks": keys}
