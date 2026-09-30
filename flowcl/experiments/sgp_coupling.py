"""SGP update-rule coupling test at T4 (alpha 800, seed 0): exploratory, one seed.

Adaptive SGP at alpha 800 (A800) fitted T4 worse than adaptive GPM
(``docs/runs/2026-09-30_sgp_a800_seed0.md``). Every arm here trains T4 only, from A800's stage-2
state with A800's T3 memory and importances (``configs/analysis/sgp_coupling.yaml``): AdamW (the
recipe) or vanilla SGD (no momentum, no weight decay, no clipping), crossed with no projection
(the §7.4 freeze only), hard projection (lambda = 1) and the scaled projection. Under vanilla SGD
our adaptation is exactly the scaled projection ``dW = -eta G (I - M Lambda M^T)``; the SGD arms
also drop clipping, so this is an update-rule coupling test, not a test of AdamW alone.

1. :func:`run_arm` ``aw_soft`` reproduces A800's T4, and :func:`reproduction_gate` must pass;
2. :func:`run_pilot`: the AdamW pilots record step norms (:class:`StepNormRecorder`) that
   :func:`matched_rate` turns into SGD rates; the SGD pilots run at scaled rates, loss only;
3. :func:`run_select`: :func:`pilot_verdict` and :func:`select_rate`, frozen in a bundle;
4. :func:`run_arm` for the other five arms (published through
   :func:`flowcl.experiments.low_update.train_stage`);
5. :func:`run_summary`, always: :func:`log_ratio_ci`, :func:`classify`, :func:`importance_split`.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from omegaconf import OmegaConf

from flowcl.experiments.hparam_study import default_evaluator, paired_against
from flowcl.experiments.low_update import (
    _root,
    abandoned_attempts,
    load_low_update_config,
    load_published,
    pilot_stage,
    refuse_dirty,
    seed_context,
    train_stage,
)
from flowcl.experiments.sgp_report import _quantiles, _resample
from flowcl.experiments.sgp_sensitivity import load_selection, publish_selection
from flowcl.utils.libero_paths import repo_root
from flowcl.utils.run import atomic_write_text, file_sha256, git_sha

ARMS = ("aw_soft", "aw_hard", "aw_free", "sgd_free", "sgd_hard", "sgd_soft")
GATE_ARM = "aw_soft"
GATE_FILE = "reproduction.json"
TOL = 1e-9  # success rates are k/50: allow float rounding only

INVALID, NO_PROJECTION_COST, NO_COUPLING_COST = ("invalid_reproduction", "no_projection_cost",
                                                 "no_t4_coupling_cost")
INCONCLUSIVE, SUPPORTED, MIXED = "inconclusive_sgd", "coupling_supported", "mixed"
INCOMPLETE = "incomplete"
VERDICT_TEXT = {
    INVALID: "The reproduction gate failed: aw_soft does not reproduce A800's T4, so no coupling "
             "contrast is interpreted.",
    NO_PROJECTION_COST: "Hard projection does not measurably cost T4 plasticity from this "
                        "checkpoint: there is no local plasticity gap for scaled projection to recover.",
    NO_COUPLING_COST: "Under AdamW the scaled projection costs no T4 plasticity against hard "
                      "projection with the same memory: A800's deficit lies in its earlier stages.",
    INCONCLUSIVE: "Vanilla SGD is not viable here (no rate selected, or unprojected SGD misses the "
                  "T4 threshold): the update-rule question is untested.",
    SUPPORTED: "The exact SGD update rule removes the cost of the scaled projection seen under our "
               "AdamW adaptation (optimizer / update-rule coupling; not AdamW alone).",
    MIXED: "None of the pre-registered patterns.",
    INCOMPLETE: "An arm the verdict needs is missing (a failed run, not a finding).",
}


def load_coupling_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else repo_root() / "configs" / "analysis" / "sgp_coupling.yaml"
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def _half(name: str) -> str:
    return "trunk" if name.startswith("trunk.") else "decoder"


def arm_run_id(name: str, seed: int) -> str:
    return f"coup__{name}__seed{seed}"


def pilot_run_id(name: str, seed: int, scale: float | None = None) -> str:
    return f"pilot__{name}{'' if scale is None else f'_x{scale:g}'}__seed{seed}"


def arm_dir(cfg: dict, root: Path, name: str) -> Path:
    return root / cfg["out"] / "arms" / arm_run_id(name, cfg["seed"])


def pilot_dir(cfg: dict, root: Path, name: str, scale: float | None = None) -> Path:
    return root / cfg["out"] / "pilots" / pilot_run_id(name, cfg["seed"], scale)


# ---- the source run (A800) ---------------------------------------------------------------------


def source_paths(cfg: dict, root: Path) -> dict:
    src, k = cfg["source"], int(cfg["source"]["stage"])
    run = root / src["run"]
    return {"run": run, "start": run / "checkpoints" / f"stage{k - 1}.pt",
            "memory": run / "method" / f"memory_task{k - 1}.pt",
            "end": run / "checkpoints" / f"stage{k}.pt",
            "gpm_logs": run / "method" / f"gpm_logs_task{k}.json",
            "result": run / "result.json", "diagnostics": root / src["diagnostics"],
            "sequence_report": root / src["sequence_report"], "step_log": root / src["step_log"]}


def check_start(cfg: dict, ctx, root: Path) -> dict:
    """A800's references are the committed ones, its method and recipe are the rule's, its T3
    memory is the one its stage-2 checkpoint recorded, and its diagnostics share our probe."""
    src, k = cfg["source"], int(cfg["source"]["stage"])
    paths = source_paths(cfg, root)
    problems = {}
    for key, expected in src["sha256"].items():
        actual = file_sha256(paths[key])
        if actual != expected:
            problems[f"sha256.{key}"] = (actual, expected)
    conf = OmegaConf.to_container(OmegaConf.load(paths["run"] / "config.yaml"), resolve=True)
    m = conf.get("method", {})
    if (m.get("name") != "sgp" or not math.isclose(float(m.get("alpha", -1)), float(src["alpha"]))
            or m.get("eps") != src["eps"] or m.get("new_energy_fraction") != src["new_energy_fraction"]):
        problems["method"] = m
    own = {"schedule": "cosine", **conf["train"]}  # the field postdates these runs
    ref = {"schedule": "cosine", **ctx.ref_config["train"]}
    if own != ref:
        problems["train"] = {key: (own.get(key), ref.get(key)) for key in set(own) | set(ref)
                             if own.get(key) != ref.get(key)}
    result = json.loads(paths["result"].read_text())
    if result.get("seed_namespace_run_id") != ctx.namespace:
        problems["namespace"] = result.get("seed_namespace_run_id")
    artifacts = {a["path"]: a["sha256"] for a in result["stages"][k - 1]["method_artifacts"]}
    if artifacts.get(f"method/memory_task{k - 1}.pt") != src["sha256"]["memory"]:
        problems["memory_artifact"] = artifacts
    diag = json.loads(paths["diagnostics"].read_text())
    if diag["method_run_id"] != src["run"] or diag["reference_run_id"] != ctx.ref_dir.name:
        problems["diagnostics"] = (diag["method_run_id"], diag["reference_run_id"])
    if diag["config"]["probe"] != ctx.probe:
        problems["probe"] = "A800's diagnostics used another probe"
    thresholds = json.loads(paths["sequence_report"].read_text())["criteria"]["thresholds"]
    if abs(thresholds[k] - cfg["verdict"]["sgd_t4_threshold"]) > TOL:
        problems["t4_threshold"] = (thresholds[k], cfg["verdict"]["sgd_t4_threshold"])
    if problems:
        raise ValueError(f"source run {src['run']} does not match the rule: {problems}")
    return {"passed": True, "namespace": ctx.namespace, "sha256": dict(src["sha256"])}


def load_memory(paths: dict) -> tuple[dict, dict]:
    """A800's T3 memory ``M`` (float64) and importances ``lambda``, per registry layer."""
    from flowcl.analysis.subspace import load_bases

    bases, _ = load_bases(paths["memory"])
    return ({n: b.vectors.to(torch.float64) for n, b in bases.items()},
            {n: b.meta["importance"].to(torch.float64) for n, b in bases.items()})


# ---- methods --------------------------------------------------------------------------------------


class StepNormRecorder:
    """Wraps a method; at every step ``t`` in ``window`` records the gradient SGD would apply,
    ``||G_t||`` (AMP-unscaled, after the method's gradient projection, before clipping, over the
    trainable tensors), the realised step ``||dW_t||`` (after the method's post-step
    projection), and the schedule factor. Reads only: training is bitwise unchanged."""

    def __init__(self, inner, window, factor: Callable[[int], float]) -> None:
        self.inner = inner
        self.window = (int(window[0]), int(window[1]))
        self.factor = factor
        self.records: list[dict] = []
        self._layers: dict[str, list] = {}
        self._params = None
        self._grad = None
        self._before = None

    def __getattr__(self, name):  # logs, state and the rest are the wrapped method's
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)

    def on_task_start(self, policy, task_idx, *, context) -> None:
        self.inner.on_task_start(policy, task_idx, context=context)
        self._params = [(n, p) for n, p in policy.named_parameters() if p.requires_grad]

    def build_batch(self, dataset, task_idx):
        return self.inner.build_batch(dataset, task_idx)

    def modify_loss(self, loss, batch, policy, *, outputs=None):
        return self.inner.modify_loss(loss, batch, policy, outputs=outputs)

    def modify_gradients(self, policy, batch_meta) -> None:
        self.inner.modify_gradients(policy, batch_meta)
        if not self.window[0] <= batch_meta["step"] < self.window[1]:
            return
        with torch.no_grad():
            missing = [n for n, p in self._params if p.grad is None]
            if missing:
                raise RuntimeError(f"no gradient for trainable tensors {missing[:3]}")
            self._grad = torch.stack([p.grad.detach().float().norm() for _, p in self._params]).double().cpu()
            self._before = [p.detach().clone() for _, p in self._params]

    def after_step(self, policy, step_meta) -> None:
        self.inner.after_step(policy, step_meta)
        if self._before is None:
            return
        step = step_meta["step"]
        with torch.no_grad():
            u = torch.stack([(p.detach() - b).float().norm()
                             for (_, p), b in zip(self._params, self._before)]).double().cpu()
        g = self._grad
        self.records.append({"step": step, "f": self.factor(step),
                             "grad_norm": float(g.pow(2).sum().sqrt()),
                             "update_norm": float(u.pow(2).sum().sqrt())})
        for (name, _), gi, ui in zip(self._params, g.tolist(), u.tolist()):
            self._layers.setdefault(name, []).append((step, gi, ui))
        self._grad = self._before = None

    def on_task_end(self, policy, task_idx, *, context) -> None:
        self.inner.on_task_end(policy, task_idx, context=context)

    def summary(self) -> dict:
        layer_rates = {}
        for name, rows in self._layers.items():
            rates = [u / (self.factor(s) * g) for s, g, u in rows if _usable(g, u, self.factor(s))]
            layer_rates[name] = statistics.median(rates) if rates else None
        return {"window": list(self.window), "records": self.records, "layer_rates": layer_rates}


def _usable(g: float, u: float, f: float) -> bool:
    return math.isfinite(g) and math.isfinite(u) and g > 0 and u > 0 and f > 0


def make_arm_method(arm: dict, cfg: dict, paths: dict, recorder: tuple | None = None) -> Callable:
    """``setup_method``: the §7.4 freeze only, hard projection or the scaled projection, all with
    A800's T3 memory (restored with its SHA and identity checks); optionally recorded."""
    from flowcl.methods.gpm import GPM, freeze_to_allowlist
    from flowcl.methods.seq_ft import SeqFT
    from flowcl.methods.sgp import SGP

    src = cfg["source"]

    def setup(policy):
        projection = arm["projection"]
        if projection == "none":
            freeze_to_allowlist(policy)  # before the optimiser is built, as for GPM
            method = SeqFT()
        else:
            common = {"eps": src["eps"], "new_energy_fraction": src["new_energy_fraction"],
                      "update_memory": False}
            if projection == "hard":
                method = GPM(**common)
            elif projection == "scaled":
                method = SGP(alpha=float(src["alpha"]), **common)
            else:
                raise ValueError(f"unknown projection {projection!r}")
            method.restore_memory(paths["memory"], src["sha256"]["memory"], method_run_id=src["run"],
                                  task_idx=int(src["stage"]) - 1)
        return method if recorder is None else StepNormRecorder(method, *recorder)

    return setup


# ---- pure pieces ----------------------------------------------------------------------------------


def matched_rate(records: list[dict], window) -> dict:
    """``eta = median_t ||dW_t|| / (f_t ||G_t||)`` over the window: the SGD peak rate whose step
    has AdamW's norm at the same point of the schedule. AMP-skipped steps are excluded."""
    lo, hi = window
    ratios, skipped = [], 0
    for r in records:
        if not lo <= r["step"] < hi:
            continue
        if not _usable(r["grad_norm"], r["update_norm"], r["f"]):
            skipped += 1
            continue
        ratios.append(r["update_norm"] / (r["f"] * r["grad_norm"]))
    if not ratios:
        raise ValueError(f"no usable step in the calibration window {list(window)}")
    q10, q90 = np.quantile(ratios, [0.1, 0.9])
    return {"eta": statistics.median(ratios), "n_steps": len(ratios), "n_skipped": skipped,
            "q10": float(q10), "q90": float(q90)}


def layer_rates(per_layer: dict) -> dict:
    """The per-layer implied rates, as quantiles per half (a diagnostic only)."""
    out = {}
    for half in ("trunk", "decoder"):
        vals = [v for n, v in per_layer.items() if _half(n) == half and v is not None]
        if vals:
            q = np.quantile(vals, [0.0, 0.1, 0.5, 0.9, 1.0])
            out[half] = {"n_layers": len(vals), **dict(zip(("min", "q10", "median", "q90", "max"),
                                                           map(float, q)))}
    return out


def pilot_verdict(row: dict, adamw_end: float, slow_factor: float) -> dict:
    """``diverged`` (non-finite loss, or no T4 probe improvement), ``slow`` (T4 probe more than
    ``slow_factor`` x the AdamW counterpart's), else ``viable``. T4 quantities only."""
    start, end = row["probe"]["start"], row["probe"]["end"]
    if not row["losses_finite"] or not math.isfinite(end) or end >= start:
        verdict = "diverged"
    elif end > slow_factor * adamw_end:
        verdict = "slow"
    else:
        verdict = "viable"
    return {"verdict": verdict, "probe_start": start, "probe_end": end, "adamw_end": adamw_end}


def select_rate(verdicts: dict) -> float | None:
    """1x if viable, else the viable scale nearest to 1x on a log scale (ties: the smaller)."""
    viable = [float(s) for s, v in verdicts.items() if v == "viable"]
    if not viable:
        return None
    return min(viable, key=lambda s: (abs(math.log(s)), s))


def importance_split(start: dict, end: dict, memory: dict, importance: dict) -> dict:
    """Norms of the realised update per half: in total, inside the memory, along its hard
    (lambda = 1) and scaled (lambda < 1) directions, and outside it (free)."""
    acc = {h: dict.fromkeys(("total", "memory", "hard", "scaled"), 0.0) for h in ("trunk", "decoder")}
    for name, M in memory.items():
        key = f"{name}.weight"
        delta = end[key].to(torch.float64) - start[key].to(torch.float64)
        hard = importance[name].to(torch.float64) == 1.0
        col = ((delta @ M.to(torch.float64)) ** 2).sum(dim=0)
        a = acc[_half(name)]
        a["total"] += float((delta ** 2).sum())
        a["memory"] += float(col.sum())
        a["hard"] += float(col[hard].sum())
        a["scaled"] += float(col[~hard].sum())
    out = {}
    for half, a in acc.items():
        if a["total"] == 0.0 and a["memory"] == 0.0:
            continue
        out[half] = {k: math.sqrt(v) for k, v in a.items()}
        out[half]["free"] = math.sqrt(max(0.0, a["total"] - a["memory"]))
    return out


def log_ratio_ci(units: dict, terms: list, boot: dict) -> dict:
    """``sum sign * ln(L_num / L_den)`` of weighted probe means, with a CI from resampling the
    matched probe batches jointly for every arm involved."""
    arms = sorted({a for _, num, den in terms for a in (num, den)})
    w = np.asarray(units[arms[0]]["weights"], dtype=np.float64)
    if any(not np.array_equal(np.asarray(units[a]["weights"], dtype=np.float64), w) for a in arms):
        raise ValueError(f"probe batches of {arms} are not matched")
    L = {a: np.asarray(units[a]["losses"], dtype=np.float64) for a in arms}
    full = {a: float((w * L[a]).sum() / w.sum()) for a in arms}
    point = sum(s * (math.log(full[n]) - math.log(full[d])) for s, n, d in terms)
    idx = _resample(w.size, boot)
    means = {a: (w[idx] * L[a][idx]).sum(axis=1) / w[idx].sum(axis=1) for a in arms}
    values = sum(s * (np.log(means[n]) - np.log(means[d])) for s, n, d in terms)
    low, high = _quantiles(values, point, boot)
    return {"value": point, "low": low, "high": high, "n": int(w.size),
            "terms": [list(t) for t in terms], "means": full, "unit": "matched probe batch"}


def sign_label(ci: dict | None, above: str, below: str) -> str | None:
    if ci is None:
        return None
    if ci["low"] > 0:
        return above
    if ci["high"] < 0:
        return below
    return "none"


def retention_guard(soft_cells: dict, hard_cells: dict, tolerance: float, tasks=(0, 1, 2)) -> dict:
    per_task = {}
    for j in tasks:
        soft, hard = soft_cells[str(j)]["success"], hard_cells[str(j)]["success"]
        per_task[str(j)] = {"soft": soft, "hard": hard, "passed": soft >= hard - tolerance - TOL}
    return {"passed": all(t["passed"] for t in per_task.values()), "tolerance": tolerance,
            "per_task": per_task}


def classify(gate_passed: bool, gap: dict | None, c_aw: dict | None, delta: dict | None,
             sgd_viable: bool, guard: dict | None) -> dict:
    """The composite label, first match wins (see the config's header). A quantity whose arms
    are missing gives ``incomplete``, never a finding."""
    aw = sign_label(c_aw, "cost", "benefit")
    if not gate_passed:
        verdict = INVALID
    elif gap is None or c_aw is None:
        verdict = INCOMPLETE
    elif not gap["low"] > 0:
        verdict = NO_PROJECTION_COST
    elif aw in ("none", "benefit"):
        verdict = NO_COUPLING_COST
    elif not sgd_viable:
        verdict = INCONCLUSIVE
    elif delta is None or guard is None:
        verdict = INCOMPLETE
    elif delta["high"] < 0 and guard["passed"]:
        verdict = SUPPORTED
    else:
        verdict = MIXED
    return {"verdict": verdict, "text": VERDICT_TEXT[verdict], "c_aw": aw,
            "premise": None if gap is None else gap["low"] > 0}


# ---- the reproduction gate -------------------------------------------------------------------------


def parse_step_log(path: Path, run_id: str, stage: int, steps: int) -> dict[int, str]:
    """The ``[flowcl] step N/steps loss X`` lines of one stage's section of a runner log."""
    lines = Path(path).read_text().splitlines()
    header = f"=== {run_id} stage {stage}:"
    start = next((i for i, line in enumerate(lines) if header in line), None)
    if start is None:
        raise ValueError(f"{path}: no section {header!r}")
    pattern = re.compile(rf"^\[flowcl\] step (\d+)/{steps} loss (\S+) lr ")
    out = {}
    for line in lines[start + 1:]:
        if "=== " in line and " stage " in line:
            break
        m = pattern.match(line)
        if m:
            out[int(m.group(1))] = m.group(2)
    return out


def logged_steps(steps: int, log_every: int) -> list[int]:
    """The 1-based step numbers the trainer prints."""
    if not log_every:
        return []
    return sorted({s + 1 for s in range(steps) if s % log_every == 0} | {steps})


def c_logs_equal(mine: dict, reference: dict) -> dict:
    """Every logged ``gradient_c`` / ``update_c`` value equal (NaN, a skipped step, equals NaN)."""
    n, bad = 0, []
    for key in ("gradient_c", "update_c"):
        a = {str(k): v for k, v in (mine or {}).get(key, {}).items()}
        b = reference.get(key, {})
        if set(a) != set(b):
            bad.append(f"{key}: logged steps differ ({len(a)} vs {len(b)})")
            continue
        for step, layers in b.items():
            if set(a[step]) != set(layers):
                bad.append(f"{key} step {step}: layers differ")
                continue
            for layer, v in layers.items():
                n += 1
                w = a[step][layer]
                if not (v == w or (v != v and w != w)):
                    bad.append(f"{key} step {step} {layer}: {w!r} vs {v!r}")
    return {"n_compared": n, "n_mismatch": len(bad), "first_mismatches": bad[:5],
            "equal": n > 0 and not bad}


def _within(a: float, b: float, tol: dict) -> bool:
    return abs(a - b) <= tol["atol"] + tol["rtol"] * abs(b)


def reproduction_gate(cfg: dict, root: Path, row: dict, out_dir: Path) -> dict:
    """``aw_soft`` against A800's T4: tensors, probes, and the training trajectory."""
    from flowcl.envs.evaluation import EvaluationReport
    from flowcl.train.continual import stage_identity_check

    src, k = cfg["source"], int(cfg["source"]["stage"])
    paths = source_paths(cfg, root)
    tensors = stage_identity_check(out_dir / row["checkpoint"], paths["run"], k)
    diag = json.loads(paths["diagnostics"].read_text())["loss_matrix"]["method"][k]
    tol = cfg["probe_tolerance"]
    probes = {str(j): {"got": row["cells"][str(j)]["probe_loss"], "expected": diag[j],
                       "passed": _within(row["cells"][str(j)]["probe_loss"], diag[j], tol)}
              for j in range(k + 1)}
    result = json.loads(paths["result"].read_text())["stages"][k]
    losses = json.loads((out_dir / "losses.json").read_text())
    tail = losses[-50:]
    conf = OmegaConf.to_container(OmegaConf.load(out_dir / "config.yaml"), resolve=True)
    steps, log_every = int(conf["train"]["steps"]), int(conf["train"]["log_every"])
    printed = parse_step_log(paths["step_log"], src["run"], k, steps)
    expected_steps = logged_steps(steps, log_every)
    mismatched = [s for s in expected_steps if s not in printed or f"{losses[s - 1]:.6f}" != printed[s]]
    training = {
        "final_loss": {"got": losses[-1], "expected": result["final_loss"],
                       "passed": losses[-1] == result["final_loss"]},
        "mean_last_50_loss": {"got": sum(tail) / len(tail), "expected": result["mean_last_50_loss"],
                              "passed": sum(tail) / len(tail) == result["mean_last_50_loss"]},
        "step_log": {"n_expected": len(expected_steps), "n_printed": len(printed),
                     "n_mismatch": len(mismatched), "first_mismatches": mismatched[:5],
                     "passed": len(printed) == len(expected_steps) and not mismatched},
        "c_logs": row.get("c_log_reproduction", {"equal": False, "note": "missing"}),
    }
    training["c_logs"]["passed"] = bool(training["c_logs"].get("equal"))
    # Descriptive: the paired episodes of the two evaluations.
    mine = EvaluationReport.load(out_dir / "eval.json").by_task()
    rollouts = {}
    for t in result["evaluation"]["tasks"]:
        te = mine[t["task_key"]]
        rollouts[t["task_key"]] = {
            "same_seeds": list(te.seeds) == list(t["seeds"]),
            "n_differing_episodes": sum(a != b for a, b in zip(te.successes, t["successes"]))}
    checks = {"tensors": tensors["passed"], "probes": all(p["passed"] for p in probes.values()),
              **{key: v["passed"] for key, v in training.items()}}
    return {"passed": all(checks.values()), "checks": checks,
            "tensors": {k2: tensors[k2] for k2 in ("n_tensors", "n_different", "different",
                                                    "missing", "extra", "reference_sha256")},
            "probes": probes, "training": training, "rollouts_descriptive": rollouts,
            "row_sha256": file_sha256(out_dir / "row.json")}


def gate_status(cfg: dict, root: Path) -> dict | None:
    path = root / cfg["out"] / GATE_FILE
    return json.loads(path.read_text()) if path.is_file() else None


def gate_open(cfg: dict, results_root=None) -> bool:
    status = gate_status(cfg, _root(results_root))
    return bool(status and status["passed"])


def _require_gate(cfg: dict, root: Path) -> None:
    if not gate_open(cfg, root):
        raise RuntimeError("the reproduction gate has not passed: aw_soft must reproduce A800's "
                           "T4 before any other arm or pilot is trained")


# ---- arms and pilots --------------------------------------------------------------------------------


def arm_extra(cfg: dict, paths: dict, reference_logs: dict | None = None) -> Callable:
    """``extra_row``: the realised update split, the projection logs, and (for the gate arm)
    the per-step c-log comparison against A800's."""
    from flowcl.experiments.sequence_report import c_trajectories

    stage = int(cfg["source"]["stage"])
    memory, importance = load_memory(paths)

    def extra(policy, method, start_state) -> dict:
        end = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
        out = {"update_split": importance_split(start_state, end, memory, importance)}
        logs = getattr(method, "task_logs", {}).get(stage) if method is not None else None
        if logs:
            dyn = c_trajectories(logs)
            residuals = [r["max_residual_over_bound"] for r in logs.get("residuals", {}).values()]
            out["projection_logs"] = {
                "gradient_c_mean": dyn["gradient_c"]["mean_of_medians"],
                "update_c_mean": dyn["update_c"]["mean_of_medians"],
                "worst_residual_over_bound": max(residuals, default=None)}
        if reference_logs is not None:
            out["c_log_reproduction"] = c_logs_equal(logs, reference_logs)
        return out

    return extra


def _context(cfg: dict, root: Path, low_update_cfg: dict | None):
    ctx = seed_context(low_update_cfg or load_low_update_config(), int(cfg["seed"]), root)
    check_start(cfg, ctx, root)
    return ctx, source_paths(cfg, root)


def run_arm(cfg: dict, name: str, results_root=None, device="cuda", allow_dirty=False, evaluator=None,
            bootstrap=None, dataset_dir=None, build_datasets=None,
            low_update_cfg: dict | None = None) -> dict:
    """One arm through :func:`train_stage`; ``aw_soft`` also writes the reproduction gate."""
    refuse_dirty(allow_dirty)
    root = _root(results_root)
    ctx, paths = _context(cfg, root, low_update_cfg)
    if name not in cfg["arms"]:
        raise ValueError(f"unknown arm {name!r}")
    arm = cfg["arms"][name]
    if name != GATE_ARM:
        _require_gate(cfg, root)
    lr, overrides, spec = float(cfg["recipe"]["lr"]), None, {}
    if arm["rule"] == "sgd":
        sel, sha = load_selection(root / cfg["out"])
        rate = sel["rates"][arm["rate"]]
        if rate["selected_scale"] is None:
            raise RuntimeError(f"rate {arm['rate']!r} is not viable: {name} is not trained")
        lr = rate["lr"]
        overrides = dict(cfg["sgd_overrides"])
        spec = {"rate": arm["rate"], "eta": rate["eta"], "scale": rate["selected_scale"],
                "selection_sha256": sha}
    elif arm["rule"] != "adamw":
        raise ValueError(f"unknown update rule {arm['rule']!r}")
    reference_logs = json.loads(paths["gpm_logs"].read_text()) if name == GATE_ARM else None
    out_dir = arm_dir(cfg, root, name)
    run_id = arm_run_id(name, cfg["seed"])
    row = train_stage(
        ctx, int(cfg["source"]["stage"]), int(cfg["recipe"]["steps"]), lr, paths["start"], out_dir,
        run_id, method_spec={"arm": name, "rule": arm["rule"], "projection": arm["projection"],
                             "lr": lr, **spec},
        train_overrides=overrides, setup_method=make_arm_method(arm, cfg, paths),
        extra_row=arm_extra(cfg, paths, reference_logs), evaluator=evaluator, bootstrap=bootstrap,
        device=device, dataset_dir=dataset_dir, build_datasets=build_datasets,
    )
    if name == GATE_ARM:
        gate = reproduction_gate(cfg, root, row, out_dir)
        atomic_write_text(root / cfg["out"] / GATE_FILE, json.dumps(gate, indent=2) + "\n")
        print(f"[flowcl] reproduction gate {'PASSED' if gate['passed'] else 'FAILED'}: "
              f"{gate['checks']}", flush=True)
    return row


def run_pilot(cfg: dict, name: str, scale: float | None = None, results_root=None, device="cuda",
              allow_dirty=False, dataset_dir=None, build_datasets=None,
              low_update_cfg: dict | None = None, stop_after: int | None = None) -> dict:
    """An AdamW calibration pilot (recorded), or an SGD pilot at ``scale`` x its matched rate.

    ``stop_after`` replaces ``pilot.steps`` (a smoke run only)."""
    from flowcl.train.trainer import TrainConfig, schedule_factor

    refuse_dirty(allow_dirty)
    root = _root(results_root)
    ctx, paths = _context(cfg, root, low_update_cfg)
    _require_gate(cfg, root)
    arm = cfg["arms"][name]
    steps, stop = int(cfg["recipe"]["steps"]), int(stop_after or cfg["pilot"]["steps"])
    if arm["rule"] == "adamw":
        if scale is not None or name not in cfg["calibration"]["rates"].values():
            raise ValueError(f"{name}: AdamW pilots are the calibration arms, run without a scale")
        lr, overrides, spec = float(cfg["recipe"]["lr"]), None, {}
        tc = TrainConfig(**{**ctx.ref_config["train"], "steps": steps, "lr": lr, "device": device})
        setup = make_arm_method(arm, cfg, paths, recorder=(cfg["calibration"]["window"],
                                                           lambda t: schedule_factor(tc, t)))

        def extra(policy, method, _):
            return {"recorder": method.summary()}
    else:
        if scale is None:
            raise ValueError(f"{name}: an SGD pilot needs a scale")
        cal = calibration(cfg, root)[arm["rate"]]
        lr = cal["eta"] * float(scale)
        overrides = dict(cfg["sgd_overrides"])
        spec = {"rate": arm["rate"], "eta": cal["eta"], "scale": float(scale)}
        setup, extra = make_arm_method(arm, cfg, paths), None
    return pilot_stage(
        ctx, int(cfg["source"]["stage"]), steps, lr, paths["start"], pilot_dir(cfg, root, name, scale),
        pilot_run_id(name, cfg["seed"], scale), stop_after=stop,
        method_spec={"arm": name, "rule": arm["rule"], "projection": arm["projection"], "lr": lr, **spec},
        train_overrides=overrides, setup_method=setup, extra_row=extra, device=device,
        dataset_dir=dataset_dir, build_datasets=build_datasets,
    )


def _pilot_row(cfg: dict, root: Path, name: str, scale: float | None = None) -> dict:
    run_id = pilot_run_id(name, cfg["seed"], scale)
    row = load_published(pilot_dir(cfg, root, name, scale), {"run_id": run_id, "kind": "pilot"})
    if row is None:
        raise RuntimeError(f"pilot {run_id} is not published")
    return row


def calibration(cfg: dict, root: Path) -> dict:
    """The matched SGD rates from the published AdamW pilots, with the per-layer diagnostic."""
    out = {}
    for rate, arm in cfg["calibration"]["rates"].items():
        rec = _pilot_row(cfg, root, arm)["recorder"]
        out[rate] = {"arm": arm, **matched_rate(rec["records"], cfg["calibration"]["window"]),
                     "layer_rates": layer_rates(rec["layer_rates"])}
    return out


def run_select(cfg: dict, results_root=None) -> dict:
    """Verdicts for every SGD pilot, the selected scale per rate, frozen in a bundle."""
    root = _root(results_root)
    out = root / cfg["out"]
    if (out / "selection").exists():
        return load_selection(out)[0]
    cal = calibration(cfg, root)
    rates, starts = {}, []
    for rate, adamw_arm in cfg["calibration"]["rates"].items():
        adamw = _pilot_row(cfg, root, adamw_arm)
        starts.append(adamw["probe"]["start"])
        sgd_arm = cfg["pilot"]["sgd_arms"][rate]
        verdicts = {}
        for s in cfg["pilot"]["scales"]:
            row = _pilot_row(cfg, root, sgd_arm, float(s))
            starts.append(row["probe"]["start"])
            verdicts[f"{float(s):g}"] = {"run_id": row["run_id"], "lr": row["lr"],
                                         **pilot_verdict(row, adamw["probe"]["end"],
                                                         float(cfg["pilot"]["slow_factor"]))}
        chosen = select_rate({s: v["verdict"] for s, v in verdicts.items()})
        rates[rate] = {"sgd_arm": sgd_arm, "adamw_arm": adamw_arm, "eta": cal[rate]["eta"],
                       "calibration": cal[rate], "verdicts": verdicts, "selected_scale": chosen,
                       "lr": None if chosen is None else cal[rate]["eta"] * chosen,
                       "status": "not_viable" if chosen is None else "selected"}
    if len(set(starts)) != 1:
        raise RuntimeError(f"the pilots' T4 start probes differ: {sorted(set(starts))}")
    k = int(cfg["source"]["stage"])
    stored = json.loads(source_paths(cfg, root)["diagnostics"].read_text())["loss_matrix"]["method"][k - 1][k]
    if not _within(starts[0], stored, cfg["probe_tolerance"]):
        raise RuntimeError(f"T4 start probe {starts[0]!r} does not reproduce A800's L[{k - 1}][{k}] "
                           f"{stored!r}")
    selection = {"git_sha": git_sha(), "rule": {"calibration": cfg["calibration"], "pilot": cfg["pilot"]},
                 "t4_start_probe": starts[0], "rates": rates}
    sel, sha = publish_selection(out, selection)
    for rate, r in sel["rates"].items():
        print(f"[flowcl] rate {rate}: eta {r['eta']:.4g}, verdicts "
              f"{ {s: v['verdict'] for s, v in r['verdicts'].items()} } -> {r['status']}", flush=True)
    return sel


def rate_open(cfg: dict, rate: str, results_root=None) -> bool:
    out = _root(results_root) / cfg["out"]
    if not (out / "selection").is_dir():
        return False
    sel, _ = load_selection(out)
    return sel["rates"][rate]["selected_scale"] is not None


# ---- the summary (always) ----------------------------------------------------------------------------


def run_summary(cfg: dict, results_root=None, device="cuda", dataset_dir=None, build_datasets=None,
                low_update_cfg: dict | None = None, bootstrap: dict | None = None) -> dict:
    from flowcl.envs.evaluation import EvaluationReport
    from flowcl.experiments.optimizer_study import probe_units, update_size
    from flowcl.train.checkpoint import load_checkpoint
    from flowcl.train.pipeline import build_dataset

    root = _root(results_root)
    out = root / cfg["out"]
    ctx, paths = _context(cfg, root, low_update_cfg)
    if bootstrap is None:
        _, bootstrap = default_evaluator()
    k = int(cfg["source"]["stage"])
    t4 = str(k)
    gate = gate_status(cfg, root)
    selection, selection_sha = load_selection(out) if (out / "selection").is_dir() else (None, None)

    rows = {}
    for name in ARMS:
        d = arm_dir(cfg, root, name)
        if (d / "row.json").is_file():
            rows[name] = load_published(d, {"run_id": arm_run_id(name, cfg["seed"])})
    start = load_checkpoint(paths["start"])
    stages = ctx.curriculum.stages
    make = build_datasets or (lambda st: build_dataset([st.ref], start.spec, start.stats,
                                                       n_demos=st.n_demos, dataset_dir=dataset_dir))
    datasets = {j: make(stages[j]) for j in range(k + 1)}
    start_params = {n: p.detach().cpu() for n, p in start.policy.named_parameters()}
    registry = set(start.policy.projectable_parameters())
    del start

    units, arms = {}, {}
    source_evals = {"a800_stage2": EvaluationReport.load(paths["run"] / "eval" / f"stage{k - 1}.json").by_task(),
                    "a800_stage3": EvaluationReport.load(paths["run"] / "eval" / f"stage{k}.json").by_task(),
                    "adaptive_gpm_stage3": ctx.evals["adaptive_gpm"][k].by_task()}
    evals = {}
    for name, row in rows.items():
        d = arm_dir(cfg, root, name)
        loaded = load_checkpoint(d / row["checkpoint"], device=device)
        stored = {j: row["cells"][str(j)]["probe_loss"] for j in range(k + 1)}
        units[name] = probe_units(loaded.policy, datasets, ctx.probe, device, stored,
                                  cfg["probe_tolerance"], name)
        end = {n: p.detach().cpu() for n, p in loaded.policy.named_parameters()}
        size = update_size(start_params, end, registry, registry)
        del loaded, end
        evals[name] = EvaluationReport.load(d / "eval.json").by_task()
        keys = {j: stages[j].task_key for j in range(k + 1)}
        arms[name] = {
            "run_id": row["run_id"], "method": row["method"], "git_sha": row["git_sha"],
            "success": {str(j): row["cells"][str(j)]["success"] for j in range(k + 1)},
            "probe_loss": {str(j): units[name][str(j)]["mean"] for j in range(k + 1)},
            "training": row["training"], "update_split": row.get("update_split"),
            "update_size": size, "projection_logs": row.get("projection_logs"),
            "paired_vs": {ref: {str(j): paired_against(evals[name][keys[j]], source_evals[ref][keys[j]], bootstrap)
                                for j in range(k + 1)}
                          for ref in ("a800_stage3", "adaptive_gpm_stage3")},
            "change_from_stage2": {str(j): paired_against(evals[name][keys[j]],
                                                          source_evals["a800_stage2"][keys[j]], bootstrap)
                                   for j in range(k)},
        }

    def ci(terms):
        needed = {a for _, n, dd in terms for a in (n, dd)}
        if not needed <= set(units):
            return None
        return log_ratio_ci({a: units[a][t4] for a in needed}, terms, cfg["batch_bootstrap"])

    gap = ci([(1, "aw_hard", "aw_free")])
    c_aw = ci([(1, "aw_soft", "aw_hard")])
    c_sgd = ci([(1, "sgd_soft", "sgd_hard")])
    delta = ci([(1, "sgd_soft", "sgd_hard"), (-1, "aw_soft", "aw_hard")])
    direct = ci([(1, "sgd_soft", "aw_soft")])
    both_rates = selection is not None and all(r["selected_scale"] is not None
                                               for r in selection["rates"].values())
    sgd_viable = (both_rates and "sgd_free" in rows
                  and rows["sgd_free"]["cells"][t4]["success"] >= cfg["verdict"]["sgd_t4_threshold"] - TOL)
    guard = (retention_guard(rows["sgd_soft"]["cells"], rows["sgd_hard"]["cells"],
                             cfg["verdict"]["retention_tolerance"], tasks=range(k))
             if {"sgd_soft", "sgd_hard"} <= set(rows) else None)
    verdict = classify(bool(gate and gate["passed"]), gap, c_aw, delta, sgd_viable, guard)
    q2 = sign_label(c_sgd, "cost", "gain")
    soft_vs_hard = {}
    for rule, (soft, hard) in {"adamw": ("aw_soft", "aw_hard"), "sgd": ("sgd_soft", "sgd_hard")}.items():
        if soft in evals and hard in evals:
            soft_vs_hard[rule] = {str(j): paired_against(evals[soft][stages[j].task_key],
                                                         evals[hard][stages[j].task_key], bootstrap)
                                  for j in range(k + 1)}
    summary = {
        "git_sha": git_sha(), "config": cfg, "framing": "exploratory, one seed; an optimizer / "
        "update-rule coupling test (the SGD arms change AdamW -> SGD and clipping -> none)",
        "verdict": {**verdict, "q2_scaled_under_sgd": q2,
                    "flags": [] if guard is None or guard["passed"] else ["gain_with_forgetting"]},
        "reproduction_gate": gate, "selection": selection, "selection_sha256": selection_sha,
        "quantities": {"gap_aw": gap, "c_aw": c_aw, "c_sgd": c_sgd, "delta": delta,
                       "sgd_soft_vs_aw_soft_descriptive": direct},
        "sgd_viable": {"viable": sgd_viable, "both_rates_selected": both_rates,
                       "sgd_free_t4": rows.get("sgd_free", {}).get("cells", {}).get(t4, {}).get("success"),
                       "threshold": cfg["verdict"]["sgd_t4_threshold"]},
        "retention_guard": guard, "arms": arms, "soft_vs_hard_rollouts": soft_vs_hard,
        "skipped_arms": [a for a in ARMS if a not in rows],
        "abandoned_attempts": sorted(abandoned_attempts(out / "arms") + abandoned_attempts(out / "pilots")
                                     + [p.name for p in out.glob(".selection_attempt_*")]),
    }
    atomic_write_text(out / "summary.json", json.dumps(summary, indent=2, default=str) + "\n")
    print(f"[flowcl] SGP coupling: {verdict['verdict']} (c_AW {verdict['c_aw']}, Q2 {q2}); "
          f"arms {sorted(rows)}", flush=True)
    return summary
