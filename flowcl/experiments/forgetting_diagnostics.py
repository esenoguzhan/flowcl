"""No-training forgetting diagnostics after a four-task ``gpm_projected_adam`` run.

The seed-0 run (``docs/runs/2026-09-24_gpm_seq_hetero.md``) learned every task but kept
only T1: Object fell from 78% to 2% while Goal trained, Goal from 94% to 56% while
LIBERO-10 trained. Fixed total-energy GPM protected only 22–49% of each later task's
*new* input energy (95% for T1). Whether that under-protection explains the forgetting is
open; these diagnostics **localize the likely mechanism** on the existing checkpoints.
They do not establish causality — the adaptive-GPM intervention is the causal test.

Two measurements:

* **Fixed-batch loss matrix** ``L[i][j]`` for the method and the ``seq_ft`` reference
  (:func:`flowcl.analysis.probes.probe_loss`, the pilot's exact probe). seq_ft's stage-0/1
  cells must reproduce the pilot's recorded values (instrument check, fails loud).
* **Activation interference** per registry layer for a transition ``a -> b``:
  ``r_l = ||ΔW X|| / ||W X||`` with ``W = W^(a)``, ``ΔW = W^(b) - W^(a)`` and ``X`` a task's
  inputs to the layer, all from input Grams (:func:`activation_interference`). *Direct*
  uses the stage-``a`` activations (the unprotected residual); *after drift* the stage-``b``
  activations (only ``X`` changes). Capture seeds are independent of the stage, so both
  see identical observations, ``s`` and noise.

The pre-registered decision (``configs/analysis/forgetting_diagnostics.yaml``) compares a
forgotten target task with a retained control task under the **same** update: the loss
ratios ``R_O``, ``R_S`` partition the outcome first, then ``Q`` = median of the per-layer
``q_l = r_l(target) / r_l(control)`` decides among the selective cases
(:func:`classify_forgetting`).
"""

from __future__ import annotations

import json
import math
import statistics
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from flowcl.analysis.interference import (
    activation_interference,
    energy_outside,
    relative_interference,
)
from flowcl.analysis.probes import probe_loss
from flowcl.utils.libero_paths import repo_root
from flowcl.utils.run import atomic_write_text, file_sha256, git_sha
from flowcl.utils.seeding import derive_seed

CASES = {
    "A": "Direct unprotected interference: Object-specific loss damage and larger "
         "interference than the control already on stage-a activations. Next: test the "
         "adaptive-GPM variant (the causal test).",
    "B": "Compounded representation drift / stale bases: selective loss damage, small "
         "direct interference, large interference after drift.",
    "C": "Rollout / probe-loss mismatch: rollouts collapsed but the fixed-batch loss did "
         "not at least double. Closed-loop fragility is one candidate, not a conclusion. "
         "Stronger projection is not yet justified.",
    "D": "Sparse-layer or downstream interference: selective loss damage, but both median "
         "interference ratios small. Inspect the tail statistics and top layers.",
    "E": "Non-selective probe-loss degradation: the target's loss at least doubled but not "
         "at least twice as much as the control's. The control comparison is inconclusive; "
         "Q is reported, not interpreted.",
}
HALVES = ("trunk", "decoder")


# ---- configuration -------------------------------------------------------------

_REQUIRED = {
    "method_run", "reference_run", "probe", "instrument_check", "capture_config",
    "capture_seed_tags", "comparisons", "decision", "reporting", "out",
}


def load_diag_config(path: str | Path | None = None) -> dict:
    """Load and validate the pre-registered config; raises on anything missing or odd."""
    path = Path(path) if path else repo_root() / "configs" / "analysis" / "forgetting_diagnostics.yaml"
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    missing = _REQUIRED - set(cfg)
    if missing:
        raise ValueError(f"{path}: missing keys {sorted(missing)}")
    if "primary" not in cfg["comparisons"]:
        raise ValueError(f"{path}: comparisons need a 'primary' entry (it decides)")
    for name, comp in cfg["comparisons"].items():
        a, b = comp["transition"]
        if not (0 <= a < b) or comp["target"] == comp["control"]:
            raise ValueError(f"{path}: comparison {name!r} is malformed: {comp}")
        if comp["target"] > a or comp["control"] > a:
            raise ValueError(
                f"{path}: comparison {name!r}: target and control must be trained by stage "
                f"{a} (the start of the transition), got {comp}"
            )
    for key in ("loss_ratio_min", "selectivity_min", "q_large"):
        if not cfg["decision"][key] > 0:
            raise ValueError(f"{path}: decision.{key} must be > 0")
    if set(cfg["capture_seed_tags"]) != {"probe", "capture"}:
        raise ValueError(f"{path}: capture_seed_tags must be exactly probe and capture")
    return cfg


def capture_seeds(seed_namespace: str, task_key: str, tags: dict) -> tuple[int, int]:
    """``(probe_seed, capture_seed)`` for a task — deliberately **not** stage-dependent.

    Both stages of a transition therefore see identical observations, ``s`` and noise, so
    ``K^(a)`` and ``K^(b)`` differ only through the weights.
    """
    return (
        derive_seed(seed_namespace, f"{tags['probe']}::{task_key}", 0),
        derive_seed(seed_namespace, f"{tags['capture']}::{task_key}", 0),
    )


# ---- pure statistics -----------------------------------------------------------


def percentile(values: list[float], p: float) -> float | None:
    """Linear-interpolation percentile that tolerates ``inf`` (no ``inf * 0 = nan``)."""
    vals = sorted(values)
    if not vals:
        return None
    pos = p * (len(vals) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    a, b = vals[lo], vals[hi]
    frac = pos - lo
    if frac == 0 or a == b:
        return a
    if math.isinf(b):
        return math.inf
    return a + (b - a) * frac


def _half(name: str) -> str:
    return "trunk" if name.startswith("trunk.") else "decoder"


def loss_ratio(L: list[list[float]], a: int, b: int, task: int) -> float:
    if L[a][task] <= 0.0:
        raise ValueError(f"L[{a}][{task}] = {L[a][task]} is not positive; no ratio")
    return L[b][task] / L[a][task]


def loss_state(R_O: float, R_S: float, decision: dict) -> str:
    """``stable`` / ``nonselective`` / ``selective`` (inclusive thresholds)."""
    if R_O < decision["loss_ratio_min"]:
        return "stable"
    selectivity = math.inf if R_S == 0 else R_O / R_S
    return "selective" if selectivity >= decision["selectivity_min"] else "nonselective"


def layer_ratios(r_target: dict[str, float], r_control: dict[str, float]) -> dict:
    """``q_l = r_l(target) / r_l(control)`` per layer; ``None`` marks an excluded layer."""
    if sorted(r_target) != sorted(r_control):
        raise ValueError("target and control cover different layers")
    return {n: relative_interference(r_target[n], r_control[n]) for n in r_target}


def ratio_summary(
    q: dict[str, float | None],
    reporting: dict,
    groups: dict[str, str] | None = None,
    r_target: dict[str, float] | None = None,
    r_control: dict[str, float] | None = None,
) -> dict:
    """Median (decisive), tail percentiles, max, share ``>= threshold`` and top-k.

    Scopes: each half and ``all``; per §4.5 group only the median. Excluded layers
    (``None``) are counted, never silently dropped.
    """
    scopes: dict[str, list[str]] = {"trunk": [], "decoder": [], "all": []}
    for name in q:
        scopes[_half(name)].append(name)
        scopes["all"].append(name)
    out = {}
    for scope, names in scopes.items():
        kept = [n for n in names if q[n] is not None]
        vals = [q[n] for n in kept]
        top = sorted(kept, key=lambda n: q[n], reverse=True)[: reporting["top_k"]]
        out[scope] = {
            "n_layers": len(kept),
            "n_excluded": len(names) - len(kept),
            "median": percentile(vals, 0.5),
            **{f"p{round(100 * p)}": percentile(vals, p) for p in reporting["percentiles"]},
            "max": max(vals) if vals else None,
            "share_ge_threshold": (
                sum(1 for v in vals if v >= reporting["q_share_threshold"]) / len(vals)
                if vals else None
            ),
            "top": [
                {
                    "layer": n,
                    "group": (groups or {}).get(n),
                    "q": q[n],
                    "r_target": (r_target or {}).get(n),
                    "r_control": (r_control or {}).get(n),
                }
                for n in top
            ],
        }
    if groups:
        by_group: dict[str, list[float]] = {}
        for name, v in q.items():
            if v is not None:
                by_group.setdefault(groups[name], []).append(v)
        out["by_group_median"] = {g: percentile(v, 0.5) for g, v in sorted(by_group.items())}
    return out


def half_medians(values: dict[str, float | None]) -> dict[str, float | None]:
    """Median over layers per half, skipping ``None``."""
    return {
        half: percentile([v for n, v in values.items() if _half(n) == half and v is not None], 0.5)
        for half in HALVES
    }


def classify_forgetting(stats: dict, decision: dict) -> dict:
    """The pre-registered case A–E. Exactly one applies.

    ``stats``: ``R_O``, ``R_S``, and ``Q_direct`` / ``Q_drift`` as ``{half: median or None}``.
    The loss state partitions first (C: stable, E: non-selective); only then does ``Q``
    decide among the selective cases (A, B, D). "Large" means ``Q >= q_large`` in either
    half.
    """
    R_O, R_S = stats["R_O"], stats["R_S"]
    state = loss_state(R_O, R_S, decision)

    def large(Q: dict) -> list[str]:
        return [h for h in HALVES if Q.get(h) is not None and Q[h] >= decision["q_large"]]

    direct_large, drift_large = large(stats["Q_direct"]), large(stats["Q_drift"])
    if state == "stable":
        case = "C"
    elif state == "nonselective":
        case = "E"
    elif direct_large:
        case = "A"
    elif drift_large:
        case = "B"
    else:
        case = "D"
    return {
        "case": case,
        "interpretation": CASES[case],
        "loss_state": state,
        "R_O": R_O,
        "R_S": R_S,
        "selectivity": math.inf if R_S == 0 else R_O / R_S,
        "F_loss": math.log(R_O / R_S) if R_S > 0 and R_O > 0 else None,
        "Q_direct": stats["Q_direct"],
        "Q_drift": stats["Q_drift"],
        "Q_direct_large_in": direct_large,
        "Q_drift_large_in": drift_large,
    }


def instrument_check(
    L_reference: list[list[float]],
    task_keys: list[str],
    pilot_references: dict[int, dict[str, float]],
    rel_tol: float,
) -> dict:
    """Reference-run probe losses must reproduce the pilot's recorded ones; raises if not.

    ``pilot_references``: stage -> task key -> the pilot's recorded probe loss for the
    same seq_ft checkpoint and probe.
    """
    rows, bad = [], []
    for stage, probes in pilot_references.items():
        for key, expected in probes.items():
            got = L_reference[stage][task_keys.index(key)]
            rel = abs(got - expected) / abs(expected)
            rows.append({"stage": stage, "task_key": key, "expected": expected, "got": got,
                         "rel_diff": rel})
            if rel > rel_tol:
                bad.append(rows[-1])
    if not rows:
        raise ValueError("instrument check compared nothing; the pilot references are empty")
    if bad:
        raise RuntimeError(
            "probe instrument check failed: the reference run's fixed-batch losses do not "
            f"reproduce the pilot's (rel_tol {rel_tol}): {bad}. The probe, the data or the "
            "checkpoint changed; the loss matrix would not be comparable."
        )
    return {"passed": True, "rel_tol": rel_tol, "rows": rows}


def pilot_probe_references(pilot_json: Path) -> dict[int, dict[str, float]]:
    """The pilot's seq_ft probe losses: stage 0 (``stage0``) and stage 1 (``seq_ft_stage1``)."""
    refs = json.loads(Path(pilot_json).read_text())["references"]
    return {0: refs["stage0"]["probes"], 1: refs["seq_ft_stage1"]["probes"]}


def jsonable(obj):
    """Recursively replace non-finite floats (``inf`` -> ``"inf"``) for strict JSON."""
    if isinstance(obj, float):
        if math.isnan(obj):
            raise ValueError("NaN in the diagnostics report")
        if math.isinf(obj):
            return "inf" if obj > 0 else "-inf"
        return obj
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    return obj


# ---- measurement ---------------------------------------------------------------


def registry_weights(checkpoint: Path, names: list[str]) -> dict[str, torch.Tensor]:
    """Registry weights ``(d_out, d_in)`` from a checkpoint, on the CPU."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    return {n: state[f"{n}.weight"].clone() for n in names}


def layer_stats(
    grams: dict[str, torch.Tensor],
    W_a: dict[str, torch.Tensor],
    W_b: dict[str, torch.Tensor],
    memory: dict[str, torch.Tensor] | None,
) -> dict[str, dict]:
    """Per layer: interference ``r`` of the update ``W_b - W_a`` on these inputs, and the
    inputs' energy outside ``memory`` (``None`` without a memory)."""
    out = {}
    for name, K in grams.items():
        Wa = W_a[name].to(torch.float64)
        dW = W_b[name].to(torch.float64) - Wa
        e = None
        if memory is not None and float(torch.trace(K)) > 0.0:
            e = energy_outside(memory[name], K)
        out[name] = {"r": activation_interference(dW, Wa, K, name), "e": e}
    return out


def capture_grams(policy, dataset, subspace_cfg, device, probe_seed: int, capture_seed: int) -> dict:
    """Each registry layer's input Gram on its primary view (float64, CPU)."""
    from flowcl.experiments.gate2 import capture_task_grams

    capture = capture_task_grams(
        policy, dataset, subspace_cfg, device, probe_seed=probe_seed, capture_seed=capture_seed
    )
    return {
        name: acc.gram[capture.primary_view(name)].to(torch.float64)
        for name, acc in capture.accumulators.items()
    }


def _capture_plan(comparisons: dict) -> dict:
    """``run -> stage -> task -> [(comparison, role, kind)]``: who needs which capture.

    The method run needs target and control at both ends of each transition; the
    reference run only the target (for the anchor ratio).
    """
    plan: dict = {"method": {}, "reference": {}}
    for comp_name, comp in comparisons.items():
        a, b = comp["transition"]
        for stage, kind in ((a, "direct"), (b, "drift")):
            for role in ("target", "control"):
                plan["method"].setdefault(stage, {}).setdefault(comp[role], []).append(
                    (comp_name, role, kind)
                )
            plan["reference"].setdefault(stage, {}).setdefault(comp["target"], []).append(
                (comp_name, "target", kind)
            )
    return plan


# ---- orchestration ---------------------------------------------------------------


def run_forgetting_diagnostics(
    config: dict | None = None,
    results_root: Path | None = None,
    dataset_dir: Path | None = None,
    device: str = "cuda",
    allow_dirty: bool = False,
    out: Path | None = None,
) -> dict:
    """Loss matrix + activation interference + the pre-registered classification."""
    from flowcl.analysis.subspace import load_bases
    from flowcl.data.curriculum import load_curriculum
    from flowcl.experiments.gate2 import load_subspace_config
    from flowcl.experiments.sequence_report import memory_matrix
    from flowcl.train.checkpoint import load_checkpoint
    from flowcl.train.pipeline import build_dataset

    started = time.time()
    cfg = config or load_diag_config()
    sha = git_sha()
    if sha.endswith("-dirty") and not allow_dirty:
        raise RuntimeError(
            f"working tree is dirty ({sha}); the decision rule must be committed before the "
            "diagnostics run. Commit first, or pass allow_dirty (recorded)."
        )
    root = Path(results_root) if results_root else repo_root() / "results"
    dirs = {"method": root / cfg["method_run"], "reference": root / cfg["reference_run"]}
    run_cfgs = {
        k: OmegaConf.to_container(OmegaConf.load(d / "config.yaml"), resolve=True)
        for k, d in dirs.items()
    }
    # Data exactly as training built it: the run's curriculum n_demos, stats frozen at T1.
    curriculum = load_curriculum(run_cfgs["method"]["curriculum"])
    task_keys = list(curriculum.task_keys)
    if list(load_curriculum(run_cfgs["reference"]["curriculum"]).task_keys) != task_keys:
        raise ValueError("method and reference runs cover different task sequences")
    seed_ns = run_cfgs["method"]["seed_namespace_run_id"]
    if seed_ns != dirs["reference"].name:
        raise ValueError(
            f"method seed namespace {seed_ns!r} is not the reference run "
            f"{dirs['reference'].name!r}; the runs are not paired"
        )
    n = len(task_keys)
    base = load_checkpoint(dirs["method"] / "checkpoints" / "stage0.pt")
    ref_base = load_checkpoint(dirs["reference"] / "checkpoints" / "stage0.pt")
    if base.stats.fingerprint() != ref_base.stats.fingerprint():
        raise ValueError("method and reference runs normalise with different stats")
    names = [e.name for e in base.policy.projectable_layers()]
    groups = {e.name: e.group for e in base.policy.projectable_layers()}
    datasets = [
        build_dataset([stage.ref], base.spec, base.stats, n_demos=stage.n_demos,
                      dataset_dir=dataset_dir)
        for stage in curriculum.stages
    ]
    del base, ref_base

    capture_cfg_path = Path(cfg["capture_config"])
    if capture_cfg_path.suffix != ".yaml":
        capture_cfg_path = repo_root() / "configs" / "analysis" / f"{cfg['capture_config']}.yaml"
    subspace_cfg = load_subspace_config(capture_cfg_path)

    plan = _capture_plan(cfg["comparisons"])
    stages_needed = sorted({s for c in cfg["comparisons"].values() for s in c["transition"]})
    weights = {
        run: {s: registry_weights(dirs[run] / "checkpoints" / f"stage{s}.pt", names)
              for s in stages_needed}
        for run in dirs
    }
    memories = {}
    for comp in cfg["comparisons"].values():
        a = comp["transition"][0]
        if a not in memories:
            bases, _ = load_bases(dirs["method"] / "method" / f"memory_task{a}.pt")
            memories[a] = {name: memory_matrix(b) for name, b in bases.items()}

    L: dict[str, list[list[float]]] = {}
    stats: dict = {run: {} for run in dirs}
    seeds_used: dict = {run: {} for run in dirs}
    timings: dict = {}
    for run, run_dir in dirs.items():
        L[run] = []
        for stage in range(n):
            t0 = time.time()
            loaded = load_checkpoint(run_dir / "checkpoints" / f"stage{stage}.pt", device=device)
            policy = loaded.policy
            L[run].append([probe_loss(policy, ds, cfg["probe"], device) for ds in datasets])
            for task, uses in plan[run].get(stage, {}).items():
                probe_seed, capture_seed = capture_seeds(seed_ns, task_keys[task], cfg["capture_seed_tags"])
                seeds_used[run].setdefault(task, set()).add((probe_seed, capture_seed))
                grams = capture_grams(policy, datasets[task], subspace_cfg, device,
                                      probe_seed, capture_seed)
                for comp_name, role, kind in uses:
                    a, b = cfg["comparisons"][comp_name]["transition"]
                    per_layer = layer_stats(
                        grams, weights[run][a], weights[run][b],
                        memories[a] if run == "method" else None,
                    )
                    stats[run].setdefault(comp_name, {}).setdefault(role, {})[kind] = per_layer
                del grams
            timings[f"{run}_stage{stage}_s"] = time.time() - t0
            del policy, loaded
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[flowcl] {run} stage {stage}: losses "
                  + ", ".join(f"{v:.5f}" for v in L[run][-1]), flush=True)

    for run, per_task in seeds_used.items():
        unpaired = {t: s for t, s in per_task.items() if len(s) != 1}
        if unpaired:
            raise RuntimeError(f"{run}: capture seeds differ between stages for tasks {unpaired}")

    check = None
    if cfg["instrument_check"]:
        check = instrument_check(
            L["reference"], task_keys,
            pilot_probe_references(root / cfg["instrument_check"]["pilot_json"]),
            cfg["instrument_check"]["rel_tol"],
        )

    comparisons = {}
    for comp_name, comp in cfg["comparisons"].items():
        a, b = comp["transition"]
        m = stats["method"][comp_name]
        r = {role: {kind: {n_: v["r"] for n_, v in m[role][kind].items()} for kind in m[role]}
             for role in m}
        e = {role: {kind: {n_: v["e"] for n_, v in m[role][kind].items()} for kind in m[role]}
             for role in m}
        ref_r = {kind: {n_: v["r"] for n_, v in stats["reference"][comp_name]["target"][kind].items()}
                 for kind in ("direct", "drift")}
        q = {kind: layer_ratios(r["target"][kind], r["control"][kind]) for kind in ("direct", "drift")}
        summaries = {
            kind: ratio_summary(q[kind], cfg["reporting"], groups, r["target"][kind], r["control"][kind])
            for kind in ("direct", "drift")
        }
        decision = classify_forgetting(
            {
                "R_O": loss_ratio(L["method"], a, b, comp["target"]),
                "R_S": loss_ratio(L["method"], a, b, comp["control"]),
                "Q_direct": {h: summaries["direct"][h]["median"] for h in HALVES},
                "Q_drift": {h: summaries["drift"][h]["median"] for h in HALVES},
            },
            cfg["decision"],
        )
        comparisons[comp_name] = {
            "transition": [a, b],
            "target": task_keys[comp["target"]],
            "control": task_keys[comp["control"]],
            "classification": decision,
            "q_summary": summaries,
            "reported": {
                "anchor_r_method_over_reference": {
                    kind: half_medians(
                        {n_: relative_interference(r["target"][kind][n_], ref_r[kind][n_])
                         for n_ in names}
                    )
                    for kind in ("direct", "drift")
                },
                "median_r": {
                    role: {kind: half_medians(r[role][kind]) for kind in ("direct", "drift")}
                    for role in ("target", "control")
                },
                "reference_median_r_target": {kind: half_medians(ref_r[kind]) for kind in ("direct", "drift")},
                "energy_outside_memory": {
                    role: {kind: half_medians(e[role][kind]) for kind in ("direct", "drift")}
                    for role in ("target", "control")
                },
                "amplification_drift_over_direct": {
                    role: half_medians(
                        {n_: relative_interference(r[role]["drift"][n_], r[role]["direct"][n_])
                         for n_ in names}
                    )
                    for role in ("target", "control")
                },
            },
            "per_layer": {
                n_: {
                    "group": groups[n_],
                    "q_direct": q["direct"][n_], "q_drift": q["drift"][n_],
                    "r_target_direct": r["target"]["direct"][n_],
                    "r_target_drift": r["target"]["drift"][n_],
                    "r_control_direct": r["control"]["direct"][n_],
                    "r_control_drift": r["control"]["drift"][n_],
                    "r_reference_target_direct": ref_r["direct"][n_],
                    "r_reference_target_drift": ref_r["drift"][n_],
                    "e_target_direct": e["target"]["direct"][n_],
                    "e_target_drift": e["target"]["drift"][n_],
                    "e_control_direct": e["control"]["direct"][n_],
                    "e_control_drift": e["control"]["drift"][n_],
                }
                for n_ in names
            },
        }

    loss_forgetting = {}
    for a in range(n - 1):
        for j in range(a + 1):
            d_ref = L["reference"][a + 1][j] - L["reference"][a][j]
            d_m = L["method"][a + 1][j] - L["method"][a][j]
            loss_forgetting[f"{a}->{a + 1}|{j}"] = d_m / d_ref if d_ref > 0 else None

    primary = comparisons["primary"]["classification"]["case"]
    others = {k: v["classification"]["case"] for k, v in comparisons.items() if k != "primary"}
    inputs = {
        f"{run}/checkpoints/stage{s}.pt": file_sha256(dirs[run] / "checkpoints" / f"stage{s}.pt")
        for run in dirs for s in range(n)
    }
    inputs.update({
        f"method/method/memory_task{a}.pt": file_sha256(dirs["method"] / "method" / f"memory_task{a}.pt")
        for a in memories
    })
    report = {
        "git_sha": sha,
        "allow_dirty": allow_dirty,
        "method_run_id": dirs["method"].name,
        "reference_run_id": dirs["reference"].name,
        "seed_namespace_run_id": seed_ns,
        "task_keys": task_keys,
        "config": cfg,
        "inputs_sha256": inputs,
        "loss_matrix": L,
        "instrument_check": check,
        "loss_forgetting_fraction": loss_forgetting,
        "comparisons": comparisons,
        "decision": {
            "primary_case": primary,
            "primary_interpretation": CASES[primary],
            "other_cases": others,
            "disagreements": [k for k, c in others.items() if c != primary],
            "note": "The diagnostics localize a likely mechanism; they do not establish "
                    "causality. The adaptive-GPM intervention is the causal test.",
        },
        "timings_s": {**timings, "total": time.time() - started},
    }
    out = Path(out) if out else root / cfg["out"]
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out, json.dumps(jsonable(report), indent=2) + "\n")
    print_summary(report)
    print(f"[flowcl] wrote {out}", flush=True)
    return report


def print_summary(report: dict) -> None:
    keys = [k.split("/")[0].replace("libero_", "") for k in report["task_keys"]]
    for run in ("method", "reference"):
        print(f"\n[flowcl] loss matrix ({run}):  " + " | ".join(keys))
        for i, row in enumerate(report["loss_matrix"][run]):
            print(f"    after {keys[i]:8s} " + " | ".join(f"{v:.5f}" for v in row))
    for name, comp in report["comparisons"].items():
        c = comp["classification"]
        print(f"  {name}: transition {comp['transition']}, target {comp['target'].split('/')[0]}, "
              f"control {comp['control'].split('/')[0]}")
        print(f"    R_O {c['R_O']:.3f}, R_S {c['R_S']:.3f}, selectivity {c['selectivity']:.3f} "
              f"-> loss {c['loss_state']}")
        for kind in ("direct", "drift"):
            s = comp["q_summary"][kind]
            print(f"    Q_{kind}: " + ", ".join(
                f"{h} median {s[h]['median']} p90 {s[h].get('p90')} max {s[h]['max']} "
                f"share>=2 {s[h]['share_ge_threshold']}" for h in HALVES))
        print(f"    CASE {c['case']}: {c['interpretation']}")
    d = report["decision"]
    print(f"  decision: primary case {d['primary_case']}; others {d['other_cases']}; "
          f"disagreements {d['disagreements']}")