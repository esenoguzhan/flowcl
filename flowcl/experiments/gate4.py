"""Gate 4: flow-time (``s``) characterization over three seeds (README §9, §7.5). No training.

For each seed's post-T1 checkpoint (seq_ft ``stage0.pt``, bitwise equal to the GPM runs'):

1. **Per-bin T1 bases.** ``policy.s_sampler`` is swapped for ``BinnedSSampler(b)`` and the
   whole T1 dataset is captured (:func:`flowcl.experiments.gate2.capture_task_grams`), in
   two replicates A/B that differ only in their capture seed (``s`` and ``A_0``). Dataset
   order and token-position subsampling are shared by every capture. One pooled capture
   (the policy's own sampler) is the reference.
2. **Negative control.** The s-independent layers' sufficient statistics must equal the
   pooled capture's in every bin/replicate — else the instrument is broken and this raises.
3. **Per-bin ``c_l``.** T2 gradients at the same weights with ``s ~ U(bin)``, via
   :func:`flowcl.experiments.gate3.measure_gradient_interference`, whose generators are
   rebuilt identically per bin: identical batches, ``A_0`` and base uniforms (checked from
   the recorded ``s``). Each batch gradient is projected against ``M_{b,A}``, ``M_{b,B}``
   and the pooled basis in the same pass.
4. **Rules** (:mod:`flowcl.analysis.flowtime`, pre-registered in
   ``configs/analysis/flowtime.yaml``) per s-dependent layer and seed; a criterion passes
   when at least half of the *same* layers pass in every seed (:func:`reproducible`).
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path

import torch
from omegaconf import OmegaConf

from flowcl.analysis.flowtime import (
    angle_rule,
    assert_s_independent,
    c_rule,
    cosine_summary,
    overlap,
    principal_cosines,
    reproducible,
    rho_rule,
)
from flowcl.analysis.gates import gate4
from flowcl.analysis.hooks import S_DEPENDENT_KINDS, layer_kind
from flowcl.analysis.subspace import basis_from_gram
from flowcl.models.flow_head import S_BINS, BinnedSSampler
from flowcl.utils.libero_paths import repo_root
from flowcl.utils.run import atomic_write_text, file_sha256, git_sha
from flowcl.utils.seeding import derive_seed

_REQUIRED = {
    "runs", "stage", "basis_task", "gradient_task", "capture_config", "interference_config",
    "bins", "replicates", "seed_tags", "eps", "expected_s_dependent_layers", "small_d_in",
    "rule", "negative_control", "out",
}


def _analysis_config_path(value: str) -> Path:
    path = Path(value)
    return path if path.suffix == ".yaml" else repo_root() / "configs" / "analysis" / f"{value}.yaml"


def load_flowtime_config(path: str | Path | None = None) -> dict:
    """Load and validate the pre-registered Gate 4 config."""
    path = Path(path) if path else repo_root() / "configs" / "analysis" / "flowtime.yaml"
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    missing = _REQUIRED - set(cfg)
    if missing:
        raise ValueError(f"{path}: missing keys {sorted(missing)}")
    if list(cfg["bins"]) != list(range(len(S_BINS))):
        raise ValueError(f"{path}: bins {cfg['bins']} must be every S_BINS index")
    if len(cfg["replicates"]) != 2 or len(set(cfg["replicates"])) != 2:
        raise ValueError(f"{path}: exactly two distinct replicates are required")
    for key in ("angles", "c"):
        low, high = cfg["rule"][key]["extreme_bins"]
        if not (low in cfg["bins"] and high in cfg["bins"] and low < high):
            raise ValueError(f"{path}: rule.{key}.extreme_bins {cfg['rule'][key]['extreme_bins']}")
    if len(cfg["runs"]) < 3:
        raise ValueError(f"{path}: Gate 4 needs >= 3 seeds, got {len(cfg['runs'])}")
    return cfg


@contextmanager
def s_sampler_as(policy, sampler):
    """Temporarily replace ``policy.s_sampler`` (restored even on error)."""
    original = policy.s_sampler
    policy.s_sampler = sampler
    try:
        yield policy
    finally:
        policy.s_sampler = original


def s_dependent_layers(policy) -> list[str]:
    """Registry layers whose activation inputs depend on ``s`` (registry order)."""
    return [e.name for e in policy.projectable_layers() if layer_kind(e.name) in S_DEPENDENT_KINDS]


def normalized_uniforms(s: torch.Tensor, bin_idx: int) -> torch.Tensor:
    low, high = S_BINS[bin_idx]
    return (s.to(torch.float64) - low) / (high - low)


# ---- one seed ----------------------------------------------------------------------------


def _bases(capture, names: list[str], scfg) -> dict:
    out = {}
    for name in names:
        acc = capture.accumulators[name]
        view = capture.primary_view(name)
        out[name] = basis_from_gram(acc.gram[view], layer=name, n_samples=acc.n[view],
                                    thresholds=scfg.energy_thresholds, neg_tol=scfg.neg_tol,
                                    rank_tol=scfg.rank_tol)
    return out


def _primary_stats(capture, names: list[str]) -> dict:
    return {n: (capture.accumulators[n].n[capture.primary_view(n)],
                capture.accumulators[n].gram[capture.primary_view(n)].clone())
            for n in names}


def run_seed(run_dir: Path, cfg: dict, device, dataset_dir: Path | None = None) -> dict:
    """Every Gate 4 measurement and per-layer rule for one seed's post-T1 checkpoint."""
    from flowcl.data.curriculum import load_curriculum
    from flowcl.data.tasks import TaskRef
    from flowcl.experiments.gate2 import capture_task_grams, load_subspace_config
    from flowcl.experiments.gate3 import load_interference_config, measure_gradient_interference
    from flowcl.train.checkpoint import load_checkpoint
    from flowcl.train.pipeline import build_dataset

    started = time.perf_counter()
    scfg = load_subspace_config(_analysis_config_path(cfg["capture_config"]))
    icfg = load_interference_config(_analysis_config_path(cfg["interference_config"]))
    eps, bins, reps, tags = cfg["eps"], cfg["bins"], cfg["replicates"], cfg["seed_tags"]
    if eps not in scfg.energy_thresholds or eps not in icfg.energy_thresholds:
        raise ValueError(f"eps {eps} must be swept by both the capture and gradient configs")

    ckpt = run_dir / "checkpoints" / f"stage{cfg['stage']}.pt"
    loaded = load_checkpoint(ckpt)
    run_cfg = OmegaConf.to_container(OmegaConf.load(run_dir / "config.yaml"), resolve=True)
    keys = list(load_curriculum(run_cfg["curriculum"]).task_keys)
    t1, t2 = keys[cfg["basis_task"]], keys[cfg["gradient_task"]]
    if loaded.task_key != t1:
        raise ValueError(f"{ckpt} was trained on {loaded.task_key!r}, expected {t1!r}")
    ns = run_dir.name  # the seq_ft run id is the seed namespace

    def dataset(key: str, n_demos):
        return build_dataset([TaskRef.from_key(key)], loaded.spec, loaded.stats,
                             n_demos=n_demos, dataset_dir=dataset_dir)

    ds1, ds2 = dataset(t1, scfg.n_demos), dataset(t2, icfg.n_demos)
    policy = loaded.policy.to(device)
    policy.eval()
    registry = list(policy.projectable_layers())
    names = [e.name for e in registry]
    groups = {e.name: e.group for e in registry}
    d_in = {e.name: e.d_in for e in registry}
    s_names = s_dependent_layers(policy)
    if len(s_names) != cfg["expected_s_dependent_layers"]:
        raise RuntimeError(f"S_DEPENDENT_KINDS resolves to {len(s_names)} layers, expected "
                           f"{cfg['expected_s_dependent_layers']}: {s_names}")
    other = [n for n in names if n not in s_names]
    probe_seed = derive_seed(ns, f"{tags['probe']}::{t1}", 0)

    # Pooled reference capture (the policy's own sampler).
    pooled_cap = capture_task_grams(policy, ds1, scfg, device, probe_seed=probe_seed,
                                    capture_seed=derive_seed(ns, f"{tags['capture']}::pooled::{t1}", 0))
    pooled = _bases(pooled_cap, names, scfg)
    reference = _primary_stats(pooled_cap, other)
    del pooled_cap

    # Per-bin, per-replicate captures of T1; negative control on the s-independent layers.
    tol = cfg["negative_control"]
    bases: dict[tuple[int, str], dict] = {}
    control = {"max_rel_gram_diff": 0.0, "bitwise_equal": 0, "compared": 0}
    for b in bins:
        for r in reps:
            capture_seed = derive_seed(ns, f"{tags['capture']}::b{b}::{r}::{t1}", 0)
            with s_sampler_as(policy, BinnedSSampler(b)):
                cap = capture_task_grams(policy, ds1, scfg, device, probe_seed=probe_seed,
                                         capture_seed=capture_seed)
            bases[(b, r)] = _bases(cap, s_names, scfg)
            stats = _primary_stats(cap, other)
            for name in other:
                diff = assert_s_independent(name, reference[name], stats[name], f"b{b}{r}",
                                            tol["gram_rel_tol"])
                control["compared"] += 1
                control["max_rel_gram_diff"] = max(control["max_rel_gram_diff"], diff)
                if diff == 0.0 and torch.equal(reference[name][1], stats[name][1]):
                    control["bitwise_equal"] += 1
                    continue  # identical input -> identical basis; nothing more to check
                basis = _bases(cap, [name], scfg)[name]
                if basis.ranks != pooled[name].ranks:
                    raise RuntimeError(f"negative control failed: {name} (b{b}{r}) ranks "
                                       f"{basis.ranks} vs pooled {pooled[name].ranks}")
                ov = overlap(basis.basis(eps), pooled[name].basis(eps))
                if ov < 1.0 - tol["overlap_tol"]:
                    raise RuntimeError(f"negative control failed: {name} (b{b}{r}) overlap "
                                       f"{ov} with the pooled basis")
            del cap, stats

    # Per-bin T2 gradients with common random numbers, projected against A, B and pooled.
    meta = {"run_id": loaded.run_id, "task_idx": loaded.stage,
            "stats_fingerprint": loaded.stats.fingerprint(), "task_key": t1}
    per_batch: dict[int, dict[str, dict[str, list[float]]]] = {}
    u_reference, u_max_diff = None, 0.0
    for b in bins:
        def basis_set(rep: str) -> dict:
            return {n: (bases[(b, rep)][n] if n in s_names else pooled[n]) for n in names}

        with s_sampler_as(policy, BinnedSSampler(b)):
            gi = measure_gradient_interference(
                loaded, basis_set(reps[0]), meta, ds2, icfg, device=device, label=f"b{b}",
                checkpoint_path=ckpt, extra_bases={reps[1]: basis_set(reps[1]), "pooled": pooled},
            )
        u = torch.cat([normalized_uniforms(s, b) for s in gi.s_trace])
        if u_reference is None:
            u_reference = u
        elif u.shape != u_reference.shape:
            raise RuntimeError(f"bin {b} measured {u.numel()} draws, bin {bins[0]} {u_reference.numel()}")
        else:
            u_max_diff = max(u_max_diff, float((u - u_reference).abs().max()))
        per_batch[b] = {
            n: {reps[0]: gi.layers[n].per_batch_c(eps),
                reps[1]: gi.layers[n].per_batch_c(eps, reps[1]),
                "pooled": gi.layers[n].per_batch_c(eps, "pooled")}
            for n in names
        }
    if u_max_diff > 1e-5:
        raise RuntimeError(f"flow-time draws are not paired across bins (max |du| {u_max_diff:.2e})")

    # Per-layer rules and descriptive numbers.
    rule = cfg["rule"]
    lo_a, hi_a = rule["angles"]["extreme_bins"]
    lo_c, hi_c = rule["c"]["extreme_bins"]
    labels = [(b, r) for b in bins for r in reps]
    layers, pass_sets = {}, {"rho": set(), "angles": set(), "c": set()}
    for n in s_names:
        rho = {r: [bases[(b, r)][n].rhos[eps] for b in bins] for r in reps}
        M = {lab: bases[lab][n].basis(eps) for lab in labels}
        rho_res = rho_rule(rho[reps[0]], rho[reps[1]], rule["rho"]["min_delta"],
                           rule["rho"]["noise_multiple"])
        across = [overlap(M[(lo_a, r)], M[(hi_a, q)]) for r in reps for q in reps]
        within = [overlap(M[(lo_a, reps[0])], M[(lo_a, reps[1])]),
                  overlap(M[(hi_a, reps[0])], M[(hi_a, reps[1])])]
        ang_res = angle_rule(across, within, rule["angles"]["margin"])
        c_res = c_rule(
            {r: (per_batch[lo_c][n][r], per_batch[hi_c][n][r]) for r in reps},
            rule["c"]["min_delta"], rule["c"]["n_bootstrap"], rule["c"]["confidence"],
            rule["c"]["bootstrap_seed"],
        )
        for q, res in (("rho", rho_res), ("angles", ang_res), ("c", c_res)):
            if res["passed"]:
                pass_sets[q].add(n)
        layers[n] = {
            "group": groups[n], "d_in": d_in[n], "small_d": d_in[n] < cfg["small_d_in"],
            "rho": {"per_replicate": rho, "pooled": pooled[n].rhos[eps]},
            "overlap_matrix": {
                "labels": [f"b{b}{r}" for b, r in labels],
                "values": [[overlap(M[x], M[y]) for y in labels] for x in labels],
            },
            "cosines_extreme": cosine_summary(principal_cosines(M[(lo_a, reps[0])], M[(hi_a, reps[0])])),
            "c_mean": {b: {k: sum(v) / len(v) for k, v in per_batch[b][n].items()} for b in bins},
            "c_per_batch": {b: per_batch[b][n] for b in (lo_c, hi_c)},
            "rules": {"rho": rho_res, "angles": ang_res, "c": c_res},
        }
    non_s_c = {n: {b: {k: sum(v) / len(v) for k, v in per_batch[b][n].items()} for b in bins}
               for n in other}
    return {
        "run": ns,
        "checkpoint": str(ckpt),
        "checkpoint_sha256": file_sha256(ckpt),
        "stats_fingerprint": loaded.stats.fingerprint(),
        "basis_task": t1,
        "gradient_task": t2,
        "s_dependent_layers": s_names,
        "small_d_layers": [n for n in s_names if d_in[n] < cfg["small_d_in"]],
        "negative_control": control,
        "crn_max_abs_du": u_max_diff,
        "pass_sets": {q: sorted(v) for q, v in pass_sets.items()},
        "layers": layers,
        "non_s_dependent_c_mean": non_s_c,
        "wall_clock_s": time.perf_counter() - started,
    }


# ---- the gate ------------------------------------------------------------------------------


def run_gate4(cfg: dict | None = None, results_root: Path | None = None,
              dataset_dir: Path | None = None, device: str = "cuda",
              allow_dirty: bool = False, out: Path | None = None) -> dict:
    from flowcl.experiments.forgetting_diagnostics import jsonable

    cfg = cfg or load_flowtime_config()
    sha = git_sha()
    if sha.endswith("-dirty") and not allow_dirty:
        raise RuntimeError(f"working tree is dirty ({sha}); Gate 4's rule must be committed "
                           "before it runs. Commit first, or pass allow_dirty (recorded).")
    root = Path(results_root) if results_root else repo_root() / "results"
    started = time.perf_counter()
    seeds = {}
    for run in cfg["runs"]:
        print(f"[flowcl] Gate 4: {run}", flush=True)
        seeds[run] = run_seed(root / run, cfg, device, dataset_dir)
        print(f"[flowcl]   pass fractions: " + ", ".join(
            f"{q} {len(v) / len(seeds[run]['s_dependent_layers']):.2f}"
            for q, v in seeds[run]["pass_sets"].items()), flush=True)
    layer_sets = {tuple(s["s_dependent_layers"]) for s in seeds.values()}
    if len(layer_sets) != 1:
        raise RuntimeError("seeds disagree on the s-dependent layer set")
    s_names = list(next(iter(layer_sets)))
    criteria = {
        q: reproducible({run: set(s["pass_sets"][q]) for run, s in seeds.items()}, s_names,
                        cfg["rule"]["min_layer_fraction"])
        for q in ("rho", "angles", "c")
    }
    result = gate4(criteria, cfg["rule"], evidence={"n_s_dependent_layers": len(s_names)})
    report = {
        "git_sha": sha,
        "allow_dirty": allow_dirty,
        "config": cfg,
        "gate": result.as_dict(),
        "seeds": seeds,
        "wall_clock_s": time.perf_counter() - started,
    }
    out = Path(out) if out else root / cfg["out"]
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out, json.dumps(jsonable(report), indent=2) + "\n")
    print_summary(report)
    print(f"[flowcl] wrote {out}", flush=True)
    return report


def print_summary(report: dict) -> None:
    gate = report["gate"]
    print(f"\n[flowcl] Gate 4: {'PASS' if gate['passed'] else 'FAIL'} — {gate['criterion']}")
    for q, c in gate["evidence"]["criteria"].items():
        per_seed = ", ".join(f"{s.split('__')[-1]} {v:.2f}" for s, v in c["per_seed_fraction"].items())
        print(f"  {q}: same-layer fraction {c['fraction']:.2f} (>= {c['min_fraction']:.2f}) "
              f"{'PASS' if c['passed'] else 'FAIL'}; per seed {per_seed}; jaccard {c['jaccard']}")
    for run, s in report["seeds"].items():
        nc = s["negative_control"]
        print(f"  {run}: negative control max rel Gram diff {nc['max_rel_gram_diff']:.2e} "
              f"({nc['bitwise_equal']}/{nc['compared']} bitwise equal); CRN max |du| "
              f"{s['crn_max_abs_du']:.1e}; small-d {s['small_d_layers']}")
    if gate["notes"]:
        print(f"  {gate['notes']}")
