"""T5 saturation stress test: an SGP alpha sweep from adaptive GPM's post-T4 state (seed 0).

Exploratory, one seed (``configs/analysis/t5_sweep.yaml``). Every arm starts from the same
weights (adaptive GPM's ``stage3.pt``) and the same memory and trains one new task, T5, so the
sweep tests a local, saturation-triggered GPM -> SGP switch:

1. :func:`run_replay` rebuilds adaptive GPM's memory with SGP's update rule from its own stage
   checkpoints, capture seeds and text-cache state, recording every task's ``sigma'``;
   :func:`compare_layer` is the validity gate (bitwise reported; subspace, spectrum, projector
   and the protected operator ``Q = M Lambda M^T`` decide); :func:`lambda_for_alpha` recomputes
   Eq. 2/10 for any alpha; :func:`nominal_passthrough` reads T5's start Gram (reported only);
2. :func:`run_arm`: free, hard and SGP at each alpha, through
   :func:`flowcl.experiments.low_update.train_stage` (stage 4, the runner's text-cache state);
3. :func:`run_summary`, always: :func:`judge_alpha`, :func:`select_alpha`, :func:`classify`.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Callable

import torch
from omegaconf import OmegaConf

from flowcl.experiments.hparam_study import _free, default_evaluator, paired_against
from flowcl.experiments.low_update import (
    _root,
    abandoned_attempts,
    load_low_update_config,
    load_published,
    refuse_dirty,
    seed_context,
    train_stage,
    warm_runner_cache,
)
from flowcl.utils.libero_paths import repo_root
from flowcl.utils.run import atomic_write_text, file_sha256, git_sha

TOL = 1e-9  # success rates are k/50: allow float rounding only
REPLAY_INVALID, INCOMPLETE, ALPHA_FOUND = "replay_invalid", "incomplete", "alpha_found"
NO_COST, ACCUMULATION, NONE_PASSES = ("no_detected_saturation_cost", "accumulation_limited",
                                      "none_passes")
VERDICT_TEXT = {
    REPLAY_INVALID: "The replay did not reproduce adaptive GPM's memory: no arm is interpreted.",
    INCOMPLETE: "An arm the verdict needs is missing (a failed run, not a finding).",
    ALPHA_FOUND: "An alpha beats hard projection on T5 in rollouts and probe loss while keeping "
                 "T1-T4 within tolerance: a local, saturation-triggered GPM -> SGP switch helped.",
    NO_COST: "Hard projection's T5 cost against no projection is not detected on this transition "
             "(not proof that there is none): SGP can only be judged on a longer sequence.",
    ACCUMULATION: "No alpha passes and every alpha >= the rule's minimum is effectively hard: "
                  "Eq. 10's accumulation leaves no soft headroom; a fix would be a new SGP variant.",
    NONE_PASSES: "No tested alpha meets the plasticity-retention criteria on this T5 transition; "
                 "this alpha sweep stops (one task and one seed do not settle SGP in general).",
}


def load_sweep_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else repo_root() / "configs" / "analysis" / "t5_sweep.yaml"
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def _half(name: str) -> str:
    return "trunk" if name.startswith("trunk.") else "decoder"


def arm_run_id(name: str, seed: int) -> str:
    return f"t5__{name}__seed{seed}"


def arm_dir(cfg: dict, root: Path, name: str) -> Path:
    return root / cfg["out"] / "arms" / arm_run_id(name, cfg["seed"])


def replay_dir(cfg: dict, root: Path) -> Path:
    return root / cfg["out"] / "replay"


def alphas(cfg: dict) -> list[float]:
    return [float(a["alpha"]) for a in cfg["arms"].values() if a["projection"] == "scaled"]


# ---- the source run (adaptive GPM) -----------------------------------------------------------------


def source_paths(cfg: dict, root: Path) -> dict:
    run = root / cfg["source"]["run"]
    paths = {"run": run, "result": run / "result.json", "gate0": root / cfg["t5"]["gate0"]["report"]}
    for t in range(4):
        paths[f"stage{t}"] = run / "checkpoints" / f"stage{t}.pt"
        paths[f"memory{t}"] = run / "method" / f"memory_task{t}.pt"
    return paths


def check_start(cfg: dict, ctx, root: Path) -> dict:
    """The pinned references, the source's method and recipe, the extended curriculum, and
    T5's Gate 0 pass, all before anything trains."""
    src, t5 = cfg["source"], cfg["t5"]
    paths = source_paths(cfg, root)
    problems = {}
    for key, expected in src["sha256"].items():
        actual = file_sha256(paths[key])
        if actual != expected:
            problems[f"sha256.{key}"] = (actual, expected)
    conf = OmegaConf.to_container(OmegaConf.load(paths["run"] / "config.yaml"), resolve=True)
    m = conf.get("method", {})
    if (m.get("name") != "gpm" or m.get("eps") != src["eps"]
            or m.get("new_energy_fraction") != src["new_energy_fraction"]
            or m.get("capture_config") != Path(str(src["capture_config"])).stem):
        problems["method"] = m
    own, ref = {"schedule": "cosine", **conf["train"]}, {"schedule": "cosine", **ctx.ref_config["train"]}
    if own != ref:
        problems["train"] = {k: (own.get(k), ref.get(k)) for k in set(own) | set(ref) if own.get(k) != ref.get(k)}
    result = json.loads(paths["result"].read_text())
    if result.get("seed_namespace_run_id") != ctx.namespace:
        problems["namespace"] = result.get("seed_namespace_run_id")
    for t in range(4):
        arts = {a["path"]: a["sha256"] for a in result["stages"][t]["method_artifacts"]}
        if arts.get(f"method/memory_task{t}.pt") != src["sha256"][f"memory{t}"]:
            problems[f"memory_artifact{t}"] = arts
    if ctx.runs["adaptive_gpm"] != src["run"]:
        problems["context"] = f"the seed context's adaptive GPM run is {ctx.runs['adaptive_gpm']!r}"
    stages = ctx.curriculum.stages
    if len(stages) != int(t5["stage"]) + 1 or stages[int(t5["stage"])].task_key != t5["task_key"]:
        problems["curriculum"] = [st.task_key for st in stages]
    g = t5["gate0"]
    if len(str(g["sha256"])) != 64:
        problems["gate0"] = f"T5's Gate 0 report is not pinned ({g['sha256']!r})"
    elif file_sha256(paths["gate0"]) != g["sha256"]:
        problems["gate0"] = "T5's Gate 0 report differs from the pinned one"
    else:
        per_task = json.loads(paths["gate0"].read_text())["evidence"]["per_task"]
        got = per_task.get(t5["task_key"], {}).get("success_rate")
        if got is None or got < float(g["threshold"]) - TOL:
            problems["gate0"] = f"T5 did not pass Gate 0 ({got})"
    if problems:
        raise ValueError(f"the T5 sweep's start does not match its rule: {problems}")
    return {"passed": True, "namespace": ctx.namespace}


def _context(cfg: dict, root: Path, low_update_cfg: dict | None):
    ctx = seed_context(low_update_cfg or load_low_update_config(), int(cfg["seed"]), root,
                       extend_curriculum=cfg["curriculum"])
    check_start(cfg, ctx, root)
    return ctx, source_paths(cfg, root)


def _make_datasets(ctx, spec, stats, tasks, dataset_dir, build_datasets) -> dict:
    from flowcl.train.pipeline import build_dataset

    stages = ctx.curriculum.stages
    make = build_datasets or (lambda st: build_dataset([st.ref], spec, stats, n_demos=st.n_demos,
                                                       dataset_dir=dataset_dir))
    return {j: make(stages[j]) for j in tasks}


# ---- pure pieces --------------------------------------------------------------------------------------


def lambda_for_alpha(records: dict, alpha: float) -> dict[str, torch.Tensor]:
    """Eq. 2 and Eq. 10 over the replay's per-task ``sigma'`` (``records[task][layer]``),
    with the same functions SGP uses, so the replay's own alpha reproduces it exactly."""
    from flowcl.analysis.importance import accumulate_importance, sgp_importance

    lam: dict[str, torch.Tensor] = {}
    for task in sorted(records, key=int):
        for name, upd in records[task].items():
            prev = lam.get(name, torch.zeros(0, dtype=torch.float64))
            k_old = upd["sigma_old"].numel()
            if prev.numel() != k_old:
                raise ValueError(f"{name}, task {task}: {k_old} old sigmas for {prev.numel()} importances")
            lam_task = sgp_importance(torch.cat([upd["sigma_old"], upd["sigma_new"]]).to(torch.float64),
                                      alpha)
            lam[name] = accumulate_importance(prev, lam_task[:k_old], lam_task[k_old:])
    return lam


def _sigma_vector(spectrum: torch.Tensor, d_in: int) -> torch.Tensor:
    """As the memory artifact stores it: sqrt of the residual spectrum, zero-padded to ``d_in``."""
    sigma = torch.zeros(d_in, dtype=torch.float64)
    sigma[: spectrum.numel()] = spectrum.to(torch.float64).clamp(min=0).sqrt()
    return sigma


def _rel(a: torch.Tensor, b: torch.Tensor, floor: float) -> float:
    return float((a - b).norm()) / max(float(b.norm()), floor)


def compare_layer(M_rep: torch.Tensor, sigma_rep: torch.Tensor, k_added_rep: int,
                  M_ref: torch.Tensor, sigma_ref: torch.Tensor, k_added_ref: int,
                  lambdas: dict, tol: dict) -> dict:
    """The replay's memory against the reference's, one layer and task.

    ``lambdas``: label -> the replay's importances for this layer (``"hard"`` is all ones). The
    gate is the subspace, spectrum, projector and protected-operator agreement; bitwise is
    reported. ``Q_N`` takes the replay's importances by a one-to-one column assignment."""
    from scipy.optimize import linear_sum_assignment

    M_rep, M_ref = M_rep.to(torch.float64), M_ref.to(torch.float64)
    out = {"bitwise": bool(M_rep.shape == M_ref.shape and torch.equal(M_rep, M_ref)
                           and torch.equal(sigma_rep, sigma_ref.to(sigma_rep.dtype))),
           "k": int(M_rep.shape[1]), "k_ref": int(M_ref.shape[1]),
           "k_added": int(k_added_rep), "k_added_ref": int(k_added_ref)}
    if M_rep.shape != M_ref.shape or k_added_rep != k_added_ref:
        return {**out, "passed": False, "reason": "dimension, rank or k_added differs"}
    scale = max(float(sigma_ref.abs().max()) if sigma_ref.numel() else 0.0, 1e-300)
    spectrum = float((sigma_rep - sigma_ref.to(torch.float64)).abs().max()) / scale if sigma_ref.numel() else 0.0
    if out["bitwise"]:
        checks = {"spectrum": 0.0, "projector": 0.0, "min_cosine": 1.0,
                  "operator": {label: 0.0 for label in lambdas}}
    else:
        k = M_rep.shape[1]
        projector = _rel(M_rep @ M_rep.T, M_ref @ M_ref.T, tol["operator_floor"])
        cross = M_rep.T @ M_ref
        min_cos = float(torch.linalg.svdvals(cross).min()) if k else 1.0
        rows, cols = linear_sum_assignment(-cross.abs().numpy()) if k else ([], [])
        operator = {}
        for label, lam in lambdas.items():
            lam = lam.to(torch.float64)
            lam_ref = torch.zeros(k, dtype=torch.float64)
            lam_ref[torch.as_tensor(cols, dtype=torch.long)] = lam[torch.as_tensor(rows, dtype=torch.long)]
            operator[label] = _rel((M_rep * lam) @ M_rep.T, (M_ref * lam_ref) @ M_ref.T, tol["operator_floor"])
        checks = {"spectrum": spectrum, "projector": projector, "min_cosine": min_cos, "operator": operator}
    passed = (checks["spectrum"] <= tol["spectrum_rtol"] and checks["projector"] <= tol["projector_rtol"]
              and checks["min_cosine"] >= 1.0 - tol["cosine_atol"]
              and all(v <= tol["operator_rtol"] for v in checks["operator"].values()))
    return {**out, **checks, "passed": bool(passed)}


def nominal_passthrough(memory: dict, lambdas: dict, grams: dict, ratio: float) -> dict:
    """Per half: T5's free-energy fraction ``1 - sum m^T K m / tr K`` and, per label, the scaled
    pass-through ``sum (1 - lambda) m^T K m / tr K``; effectively hard when the pass-through is
    below ``ratio`` x the free fraction in both halves. Nominal only: not the realised update."""
    from flowcl.analysis.importance import basis_energy

    acc: dict = {}
    for name, M in memory.items():
        K = grams[name].to(torch.float64)
        energy = basis_energy(M.to(torch.float64), K)
        a = acc.setdefault(_half(name), {"trace": 0.0, "memory": 0.0, "pass": {l: 0.0 for l in lambdas}})
        a["trace"] += float(torch.trace(0.5 * (K + K.T)))
        a["memory"] += float(energy.sum())
        for label, lam in lambdas.items():
            a["pass"][label] += float(((1.0 - lam[name].to(torch.float64)) * energy).sum())
    out = {"halves": {}, "effectively_hard": {}}
    for half, a in acc.items():
        free = 1.0 - a["memory"] / a["trace"]
        out["halves"][half] = {"free_fraction": free,
                               "passthrough": {l: v / a["trace"] for l, v in a["pass"].items()}}
    for label in lambdas:
        out["effectively_hard"][label] = all(h["passthrough"][label] < ratio * h["free_fraction"]
                                             for h in out["halves"].values())
    return out


def lambda_stats(lam: dict) -> dict:
    out = {}
    for half in ("trunk", "decoder"):
        vals = [v for n, v in lam.items() if _half(n) == half]
        if vals:
            allv = torch.cat(vals)
            out[half] = {"n_bases": int(allv.numel()), "hard_fraction": float((allv == 1.0).float().mean()),
                         "median": float(allv.median()), "mean_unprotected": float((1.0 - allv).mean())}
    return out


def judge_alpha(gain: dict, ratio: dict, old: dict, rule: dict) -> dict:
    """Against hard: the T5 gain, the T5 probe log-ratio, and the old-task guard (point
    estimates); ``retention_borderline`` where an old task's paired CI contains -tolerance."""
    tol = rule["retention_tolerance"]
    retention = {j: {"alpha": o["alpha"], "hard": o["hard"], "passed": o["alpha"] >= o["hard"] - tol - TOL,
                     "borderline": o["paired"]["low"] < -tol < o["paired"]["high"]}
                 for j, o in old.items()}
    checks = {"t5_gain": gain["diff"] >= rule["t5_gain"] - TOL and gain["low"] > 0,
              "t5_probe": ratio["high"] < 0,
              "retention": all(r["passed"] for r in retention.values())}
    return {"promising": all(checks.values()), "checks": checks, "retention": retention,
            "retention_borderline": sorted(j for j, r in retention.items() if r["borderline"])}


def select_alpha(judged: dict) -> float | None:
    """The largest T5 gain; ties: the lower T5 probe loss, then the larger alpha."""
    cands = [(j["gain"], -j["l5"], a) for a, j in judged.items() if j["promising"]]
    return max(cands)[2] if cands else None


def classify(replay_ok: bool, complete: bool, selected: float | None, gap: dict | None,
             effectively_hard: dict, hard_alpha_min: float) -> dict:
    """First match wins (see the config's header)."""
    high = [a for a in effectively_hard if a >= hard_alpha_min]
    if not replay_ok:
        verdict = REPLAY_INVALID
    elif not complete or gap is None:
        verdict = INCOMPLETE
    elif selected is not None:
        verdict = ALPHA_FOUND
    elif not gap["low"] > 0:
        verdict = NO_COST
    elif high and all(effectively_hard[a] for a in high):
        verdict = ACCUMULATION
    else:
        verdict = NONE_PASSES
    return {"verdict": verdict, "text": VERDICT_TEXT[verdict], "selected_alpha": selected,
            "premise_detected": None if gap is None else gap["low"] > 0}


# ---- step 0: the replay ---------------------------------------------------------------------------------


def _publish_dir(attempt: Path, canonical: Path) -> None:
    if canonical.exists():
        raise FileExistsError(f"{canonical} already exists; a published replay is never overwritten")
    os.rename(attempt, canonical)


def run_replay(cfg: dict, results_root=None, device="cuda", allow_dirty=False, dataset_dir=None,
               build_datasets=None, low_update_cfg: dict | None = None, n_tasks: int | None = None) -> dict:
    """Rebuild the source memory with SGP's update rule, check it, record sigma', read T5's
    start Gram; publish ``replay/`` atomically. ``n_tasks < 4`` is a smoke run (no gate)."""
    from flowcl.analysis.subspace import load_bases
    from flowcl.experiments.gate2 import capture_task_grams, load_subspace_config
    from flowcl.methods.base import TaskContext
    from flowcl.methods.sgp import SGP
    from flowcl.train.checkpoint import load_checkpoint
    from flowcl.utils.seeding import derive_seed

    refuse_dirty(allow_dirty)
    root = _root(results_root)
    ctx, paths = _context(cfg, root, low_update_cfg)
    canonical = replay_dir(cfg, root)
    if (canonical / "replay.json").is_file():
        print("[flowcl] replay: published, reused", flush=True)
        return json.loads((canonical / "replay.json").read_text())
    src, tol = cfg["source"], cfg["replay"]
    n = 4 if n_tasks is None else int(n_tasks)
    amp = bool(ctx.ref_config["train"].get("amp", False))
    stages = ctx.curriculum.stages
    capture_config = src["capture_config"]
    sgp = SGP(alpha=float(tol["alpha"]), eps=src["eps"], new_energy_fraction=src["new_energy_fraction"],
              update_memory=True, capture_config=capture_config)
    records, checks, cache_state = {}, {}, {}
    started = time.perf_counter()
    for t in range(n):
        loaded = load_checkpoint(paths[f"stage{t}"], device=device)
        ds = _make_datasets(ctx, loaded.spec, loaded.stats, [t], dataset_dir, build_datasets)
        # The runner's cache when it captured task t: t's instruction encoded under AMP if t == 0
        # (its training step), in fp32 otherwise (the previous stage's evaluation).
        cache_state[t] = warm_runner_cache(loaded.policy, {t: ds[t]}, device, amp=amp and t == 0)[t]
        context = TaskContext(task_key=stages[t].task_key, dataset=ds[t], device=device,
                              seed_namespace_run_id=ctx.namespace, method_run_id=src["run"])
        sgp._update_memory(loaded.policy, t, context)
        records[t] = {name: {k: v.detach().clone().cpu() for k, v in upd.items()}
                      for name, upd in sgp._importance_updates.items()}
        ref_bases, ref_meta = load_bases(paths[f"memory{t}"])
        ref_hist = ref_meta["memory_history"][str(t)]
        lam = {"hard": None, **{f"{a:g}": lambda_for_alpha(records, a) for a in alphas(cfg)}}
        per_layer = {}
        for name, M in sgp._memory.items():
            labels = {"hard": torch.ones(M.shape[1], dtype=torch.float64),
                      **{k: v[name] for k, v in lam.items() if k != "hard"}}
            ref = ref_bases[name]
            per_layer[name] = compare_layer(
                M, _sigma_vector(sgp._memory_spectra[name], M.shape[0]),
                int(sgp.memory_history[t][name]["k_added"]), ref.vectors,
                ref.singular_values.to(torch.float64), int(ref_hist[name]["k_added"]), labels, tol)
        checks[t] = {"layers": per_layer, "n_layers": len(per_layer),
                     "passed": all(c["passed"] for c in per_layer.values()),
                     "bitwise": all(c["bitwise"] for c in per_layer.values()),
                     "n_failed": sum(not c["passed"] for c in per_layer.values())}
        print(f"[flowcl] replay task {t}: gate {'pass' if checks[t]['passed'] else 'FAIL'}, "
              f"bitwise {checks[t]['bitwise']} ({time.perf_counter() - started:.0f}s)", flush=True)
        del loaded, ds
        _free(device)
    identity = {name: torch.equal(lambda_for_alpha(records, float(tol["alpha"]))[name], sgp._importance[name])
                for name in sgp._importance}
    complete = n == 4
    gate = complete and all(c["passed"] for c in checks.values()) and all(identity.values())
    lam_by_alpha = {f"{a:g}": lambda_for_alpha(records, a) for a in alphas(cfg)}
    nominal = None
    if complete:
        k5 = int(cfg["t5"]["stage"])
        loaded = load_checkpoint(paths["stage3"], device=device)
        ds = _make_datasets(ctx, loaded.spec, loaded.stats, [k5], dataset_dir, build_datasets)
        warm_runner_cache(loaded.policy, {k5: ds[k5]}, device, amp=False)  # T5: fp32, as the runner
        cfg_path = Path(str(capture_config))
        if cfg_path.suffix != ".yaml":
            cfg_path = repo_root() / "configs" / "analysis" / f"{capture_config}.yaml"
        tag, key = cfg["nominal"]["capture_tag"], stages[k5].task_key
        cap = capture_task_grams(loaded.policy, ds[k5], load_subspace_config(cfg_path), device,
                                 probe_seed=derive_seed(ctx.namespace, f"{tag}_probe::{key}", k5),
                                 capture_seed=derive_seed(ctx.namespace, f"{tag}::{key}", k5))
        grams = {name: cap.accumulators[name].gram[cap.primary_view(name)] for name in sgp._memory}
        labels = {"hard": {n: torch.ones(M.shape[1], dtype=torch.float64) for n, M in sgp._memory.items()},
                  **lam_by_alpha}
        nominal = nominal_passthrough(sgp._memory, labels, grams, cfg["nominal"]["effectively_hard_ratio"])
        del loaded, ds, cap, grams
        _free(device)

    out_root = root / cfg["out"]
    out_root.mkdir(parents=True, exist_ok=True)
    attempt = out_root / f".replay_attempt_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    attempt.mkdir()
    sgp.save_artifacts(attempt, n - 1, context=TaskContext(
        task_key=stages[n - 1].task_key, dataset=None, device=device,
        seed_namespace_run_id=ctx.namespace, method_run_id=src["run"]))
    torch.save(records, attempt / "records.pt")
    summary = {
        "git_sha": git_sha(), "source": src["run"], "n_tasks": n, "complete": complete,
        "gate": {"passed": bool(gate), "bitwise": complete and all(c["bitwise"] for c in checks.values()),
                 "identity": all(identity.values()),
                 "per_task": {str(t): {k: v for k, v in c.items() if k != "layers"} for t, c in checks.items()},
                 "worst": {str(t): {"spectrum": max(l.get("spectrum", math.inf) for l in c["layers"].values()),
                                    "projector": max(l.get("projector", math.inf) for l in c["layers"].values()),
                                    "min_cosine": min(l.get("min_cosine", -math.inf) for l in c["layers"].values()),
                                    "operator": max(max(l["operator"].values(), default=0.0)
                                                    if "operator" in l else math.inf
                                                    for l in c["layers"].values())}
                           for t, c in checks.items()},
                 "failed_layers": {str(t): sorted(n for n, l in c["layers"].items() if not l["passed"])[:20]
                                   for t, c in checks.items()}},
        "cache_state": {str(t): v for t, v in cache_state.items()},
        "lambda_stats": {a: lambda_stats(v) for a, v in lam_by_alpha.items()},
        "nominal": nominal,
        "memory_file": f"memory_task{n - 1}.pt",
        "memory_sha256": file_sha256(attempt / f"memory_task{n - 1}.pt"),
        "records_sha256": file_sha256(attempt / "records.pt"),
        "wall_clock_s": time.perf_counter() - started,
    }
    atomic_write_text(attempt / "replay.json", json.dumps(summary, indent=2) + "\n")
    _publish_dir(attempt, canonical)
    print(f"[flowcl] replay {'PASSED' if gate else 'FAILED'} (bitwise {summary['gate']['bitwise']})",
          flush=True)
    return summary


def load_replay(cfg: dict, root: Path) -> tuple[dict, dict, dict]:
    """The published replay (verified), its records, and its memory bases."""
    from flowcl.analysis.subspace import load_bases

    d = replay_dir(cfg, root)
    summary = json.loads((d / "replay.json").read_text())
    if file_sha256(d / "records.pt") != summary["records_sha256"] or \
            file_sha256(d / summary["memory_file"]) != summary["memory_sha256"]:
        raise RuntimeError(f"{d}: the replay's artifacts changed after publication")
    records = torch.load(d / "records.pt", weights_only=False)
    bases, _ = load_bases(d / summary["memory_file"])
    return summary, records, bases


def replay_open(cfg: dict, results_root=None) -> bool:
    d = replay_dir(cfg, _root(results_root))
    if not (d / "replay.json").is_file():
        return False
    s = json.loads((d / "replay.json").read_text())
    return bool(s["complete"] and s["gate"]["passed"])


# ---- the arms ---------------------------------------------------------------------------------------------


def make_setup(arm: dict, cfg: dict, bases: dict, records: dict) -> Callable:
    """``setup_method``: the §7.4 freeze only; hard projection; or SGP at the arm's alpha, all
    on the replayed memory."""
    from flowcl.methods.gpm import GPM, freeze_to_allowlist
    from flowcl.methods.seq_ft import SeqFT
    from flowcl.methods.sgp import SGP

    src = cfg["source"]

    def setup(policy):
        projection = arm["projection"]
        if projection == "none":
            freeze_to_allowlist(policy)
            return SeqFT()
        common = {"eps": src["eps"], "new_energy_fraction": src["new_energy_fraction"], "update_memory": False}
        if projection == "hard":
            method = GPM(**common)
            method.set_memory(bases)
        elif projection == "scaled":
            method = SGP(alpha=float(arm["alpha"]), **common)
            method.set_memory(bases, importance=lambda_for_alpha(records, float(arm["alpha"])))
        else:
            raise ValueError(f"unknown projection {projection!r}")
        return method

    return setup


def arm_extra(arm: dict, bases: dict, records: dict, stage: int) -> Callable:
    """``extra_row``: the realised T5 update split against the replayed memory (hard / scaled
    by the arm's own lambda; all-hard for the free and hard arms), and the projection logs."""
    from flowcl.experiments.sequence_report import c_trajectories
    from flowcl.experiments.sgp_coupling import importance_split

    memory = {n: b.vectors.to(torch.float64) for n, b in bases.items()}
    lam = (lambda_for_alpha(records, float(arm["alpha"])) if arm["projection"] == "scaled"
           else {n: torch.ones(M.shape[1], dtype=torch.float64) for n, M in memory.items()})

    def extra(policy, method, start_state) -> dict:
        end = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
        out = {"update_split": importance_split(start_state, end, memory, lam)}
        logs = getattr(method, "task_logs", {}).get(stage) if method is not None else None
        if logs:
            dyn = c_trajectories(logs)
            residuals = [r["max_residual_over_bound"] for r in logs.get("residuals", {}).values()]
            out["projection_logs"] = {"gradient_c_mean": dyn["gradient_c"]["mean_of_medians"],
                                      "update_c_mean": dyn["update_c"]["mean_of_medians"],
                                      "worst_residual_over_bound": max(residuals, default=None)}
        return out

    return extra


def run_arm(cfg: dict, name: str, results_root=None, device="cuda", allow_dirty=False, evaluator=None,
            bootstrap=None, dataset_dir=None, build_datasets=None, low_update_cfg: dict | None = None,
            stop_after: int | None = None, n_episodes: int | None = None) -> dict:
    """One arm through :func:`train_stage` at T5. ``stop_after`` / ``n_episodes``: smoke only."""
    refuse_dirty(allow_dirty)
    root = _root(results_root)
    ctx, paths = _context(cfg, root, low_update_cfg)
    if name not in cfg["arms"]:
        raise ValueError(f"unknown arm {name!r}")
    if not replay_open(cfg, root):
        raise RuntimeError("the replay gate has not passed: no arm is trained")
    summary, records, bases = load_replay(cfg, root)
    arm = cfg["arms"][name]
    if evaluator is None:
        evaluator, bootstrap = default_evaluator(n_episodes)
    stage = int(cfg["t5"]["stage"])
    run_id = arm_run_id(name, cfg["seed"])
    spec = {"arm": name, "projection": arm["projection"], "replay_memory_sha256": summary["memory_sha256"],
            **({"alpha": float(arm["alpha"])} if "alpha" in arm else {})}
    return train_stage(
        ctx, stage, int(cfg["recipe"]["steps"]), float(cfg["recipe"]["lr"]), paths["stage3"],
        arm_dir(cfg, root, name), run_id, method_spec=spec,
        train_overrides=None if stop_after is None else {"stop_after": int(stop_after)},
        setup_method=make_setup(arm, cfg, bases, records), extra_row=arm_extra(arm, bases, records, stage),
        evaluator=evaluator, bootstrap=bootstrap, device=device, dataset_dir=dataset_dir,
        build_datasets=build_datasets, warm_cache="runner",
    )


# ---- the summary (always) ------------------------------------------------------------------------------


def run_summary(cfg: dict, results_root=None, device="cuda", dataset_dir=None, build_datasets=None,
                low_update_cfg: dict | None = None, bootstrap: dict | None = None) -> dict:
    from flowcl.envs.evaluation import EvaluationReport
    from flowcl.experiments.optimizer_study import pareto, probe_units, update_size
    from flowcl.experiments.sgp_coupling import log_ratio_ci
    from flowcl.experiments.sgp_sensitivity import publish_selection
    from flowcl.train.checkpoint import load_checkpoint

    root = _root(results_root)
    out = root / cfg["out"]
    ctx, paths = _context(cfg, root, low_update_cfg)
    if bootstrap is None:
        _, bootstrap = default_evaluator()
    k = int(cfg["t5"]["stage"])
    stages = ctx.curriculum.stages
    keys = {j: stages[j].task_key for j in range(k + 1)}
    replay = json.loads((replay_dir(cfg, root) / "replay.json").read_text()) \
        if (replay_dir(cfg, root) / "replay.json").is_file() else None
    replay_ok = replay_open(cfg, root)

    rows = {}
    for name in cfg["arms"]:
        d = arm_dir(cfg, root, name)
        if (d / "row.json").is_file():
            rows[name] = load_published(d, {"run_id": arm_run_id(name, cfg["seed"])})
    start = load_checkpoint(paths["stage3"])
    datasets = _make_datasets(ctx, start.spec, start.stats, range(k + 1), dataset_dir, build_datasets)
    start_params = {n: p.detach().cpu() for n, p in start.policy.named_parameters()}
    registry = set(start.policy.projectable_parameters())
    del start
    start_eval = ctx.evals["adaptive_gpm"][k - 1].by_task()

    units, evals, arms = {}, {}, {}
    for name, row in rows.items():
        d = arm_dir(cfg, root, name)
        loaded = load_checkpoint(d / row["checkpoint"], device=device)
        stored = {j: row["cells"][str(j)]["probe_loss"] for j in range(k + 1)}
        units[name] = probe_units(loaded.policy, datasets, ctx.probe, device, stored,
                                  cfg["probe_tolerance"], name)
        end = {n: p.detach().cpu() for n, p in loaded.policy.named_parameters()}
        size = update_size(start_params, end, registry, registry)
        del loaded, end
        _free(device)
        evals[name] = EvaluationReport.load(d / "eval.json").by_task()
        success = {str(j): row["cells"][str(j)]["success"] for j in range(k + 1)}
        probe = {str(j): units[name][str(j)]["mean"] for j in range(k + 1)}
        arms[name] = {
            "run_id": row["run_id"], "method": row["method"], "git_sha": row["git_sha"],
            "success": success, "probe_loss": probe,
            "mean_old_success": sum(success[str(j)] for j in range(k)) / k,
            "mean_old_probe": sum(probe[str(j)] for j in range(k)) / k,
            "training": row["training"], "update_split": row.get("update_split"),
            "update_size": size, "projection_logs": row.get("projection_logs"),
            "change_from_start": {str(j): paired_against(evals[name][keys[j]], start_eval[keys[j]], bootstrap)
                                  for j in range(k)},
        }

    t5 = str(k)
    complete = set(rows) == set(cfg["arms"])
    gap = (log_ratio_ci({a: units[a][t5] for a in ("hard", "free")}, [(1, "hard", "free")],
                        cfg["batch_bootstrap"]) if {"hard", "free"} <= set(units) else None)
    judged = {}
    if "hard" in rows:
        for name, arm in cfg["arms"].items():
            if arm["projection"] != "scaled" or name not in rows:
                continue
            a = float(arm["alpha"])
            gain = paired_against(evals[name][keys[k]], evals["hard"][keys[k]], bootstrap)
            ratio = log_ratio_ci({x: units[x][t5] for x in (name, "hard")}, [(1, name, "hard")],
                                 cfg["batch_bootstrap"])
            old = {str(j): {"alpha": arms[name]["success"][str(j)], "hard": arms["hard"]["success"][str(j)],
                            "paired": paired_against(evals[name][keys[j]], evals["hard"][keys[j]], bootstrap)}
                   for j in range(k)}
            judged[a] = {"arm": name, "gain": gain["diff"], "t5_gain": gain, "t5_log_ratio": ratio,
                         "l5": arms[name]["probe_loss"][t5],
                         **judge_alpha(gain, ratio, old, cfg["rule"])}
    selected = select_alpha(judged) if complete else None
    effectively_hard = ({float(a): v for a, v in replay["nominal"]["effectively_hard"].items() if a != "hard"}
                        if replay and replay.get("nominal") else {})
    verdict = classify(replay_ok, complete, selected, gap, effectively_hard, float(cfg["rule"]["hard_alpha_min"]))
    selection = None
    if verdict["verdict"] == ALPHA_FOUND:
        selection, sha = publish_selection(out, {"git_sha": git_sha(), "selected_alpha": selected,
                                                 "rule": cfg["rule"], "judged": {f"{a:g}": j for a, j in judged.items()}})
        if selection["selected_alpha"] != selected:
            raise RuntimeError(f"the published selection ({selection['selected_alpha']}) differs from "
                               f"this summary's ({selected})")
        selection = {**selection, "sha256": sha}
    points = {n: (a["success"][t5], a["mean_old_success"]) for n, a in arms.items()}
    losses = {n: (a["probe_loss"][t5], a["mean_old_probe"]) for n, a in arms.items()}
    high = {f"{a:g}": {"l5_above_hard": j["l5"] > arms["hard"]["probe_loss"][t5]}
            for a, j in judged.items() if a >= 800}
    summary = {
        "git_sha": git_sha(), "config": cfg,
        "framing": "exploratory, one seed; a local, saturation-triggered GPM -> SGP switch from adaptive "
                   "GPM's post-T4 state (not an SGP sequence from T1)",
        "verdict": verdict, "premise_gap": gap, "replay": replay and {"gate": replay["gate"],
                                                                      "lambda_stats": replay["lambda_stats"],
                                                                      "nominal": replay["nominal"]},
        "judged": {f"{a:g}": j for a, j in judged.items()}, "selection": selection,
        "t5_threshold": None, "arms": arms,
        "pareto": {"success": pareto(points, (True, True)), "probe": pareto(losses, (False, False))},
        "coupling_prediction": high, "skipped_arms": [n for n in cfg["arms"] if n not in rows],
        "abandoned_attempts": sorted(abandoned_attempts(out / "arms")
                                     + [p.name for p in out.glob(".replay_attempt_*")]),
    }
    g0 = json.loads(paths["gate0"].read_text())["evidence"]["per_task"][cfg["t5"]["task_key"]]
    summary["t5_threshold"] = {"gate0_success": g0["success_rate"],
                               "threshold": g0["success_rate"] - cfg["t5"]["gate0"]["margin"]}
    atomic_write_text(out / "summary.json", json.dumps(summary, indent=2, default=str) + "\n")
    print(f"[flowcl] T5 sweep: {verdict['verdict']} (selected {selected}); arms {sorted(rows)}", flush=True)
    return summary
