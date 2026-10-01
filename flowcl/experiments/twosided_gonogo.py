"""E1: the two-sided protection go/no-go measurement (``configs/analysis/twosided_gonogo.yaml``).

No training. From adaptive GPM seed 0's own stage checkpoints and text-cache state:

1. :func:`run_capture` (one task index ``i`` = T(i+1), on ``stage{i}.pt``): the production
   forward-only capture (GPM's seeds) gives the input Gram ``K_i``; :func:`replay_memory` checks
   it rebuilds ``memory_task{i}``; :func:`capture_task_sensitivity` then runs the same data and
   s/noise stream grad-enabled, accumulating ``R`` Hutchinson output Grams per layer
   (:class:`flowcl.analysis.sensitivity.OutputGradientCapture`) while an ``ActivationCapture``
   rides the same forward (the forward-identity check). Published atomically.
2. :func:`run_summary`, always: the ``U`` chains (``extend_basis`` on the output Grams, the rule
   ``M`` uses), the parameter-weighted protected fractions, the forecast, the estimator stability,
   and :func:`classify`.
"""

from __future__ import annotations

import dataclasses
import json
import os
import statistics
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from flowcl.experiments.hparam_study import _free
from flowcl.experiments.low_update import (
    _root,
    load_low_update_config,
    refuse_dirty,
    seed_context,
    warm_runner_cache,
)
from flowcl.utils.libero_paths import repo_root
from flowcl.utils.run import atomic_write_text, file_sha256, git_sha

INVALID, INCOMPLETE, PASS, FAIL = "invalid", "incomplete", "pass", "fail"
VERDICT_TEXT = {
    INVALID: "A capture failed a validity check: no fraction is interpreted.",
    INCOMPLETE: "Fewer than four production captures (a failed or smoke run, not a finding).",
    PASS: "Two-sided protection would protect at most the registered fraction of the trunk's "
          "weight directions at the T4 pilot's start: potential capacity savings (E2 tests "
          "whether retention survives).",
    FAIL: "Two-sided protection would still protect more than the registered fraction of the "
          "trunk's weight directions: the output side does not free enough capacity.",
}
EXIT_CODES = {PASS: 0, FAIL: 0, INVALID: 3, INCOMPLETE: 4}
PROBES = "ABCDEFGH"
N_TASKS = 4


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else repo_root() / "configs" / "analysis" / "twosided_gonogo.yaml"
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def probe_labels(cfg: dict) -> list[str]:
    return list(PROBES[: int(cfg["capture"]["n_probes"])])


def capture_dir(cfg: dict, root: Path, i: int) -> Path:
    return root / cfg["out"] / "capture" / f"task{i}"


def bases_dir(cfg: dict, root: Path) -> Path:
    return root / cfg["out"] / "bases"


def _eps_key(eps: float) -> str:
    return f"{float(eps):g}"


def decisive_bases_name(cfg: dict, i: int) -> str:
    d = cfg["u_rule"]["decisive"]
    return f"U_{d['rule']}_e{_eps_key(d['eps_out'])}_task{i}.pt"


def _source_root(source_root) -> Path:
    return Path(source_root) if source_root else repo_root() / "results"


def _capture_config(cfg: dict):
    from flowcl.experiments.gate2 import load_subspace_config

    path = Path(str(cfg["source"]["capture_config"]))
    if path.suffix != ".yaml":
        path = repo_root() / "configs" / "analysis" / f"{cfg['source']['capture_config']}.yaml"
    return load_subspace_config(path)


# ---- the source run (adaptive GPM) -----------------------------------------------------------------


def source_paths(cfg: dict, root: Path) -> dict:
    run = root / cfg["source"]["run"]
    paths = {"run": run, "result": run / "result.json"}
    for t in range(N_TASKS):
        paths[f"stage{t}"] = run / "checkpoints" / f"stage{t}.pt"
        paths[f"memory{t}"] = run / "method" / f"memory_task{t}.pt"
    return paths


def check_source(cfg: dict, ctx, root: Path) -> dict:
    """The pinned files, the source's method, namespace and memory artifacts, and the curriculum."""
    src = cfg["source"]
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
    result = json.loads(paths["result"].read_text())
    if result.get("seed_namespace_run_id") != ctx.namespace:
        problems["namespace"] = result.get("seed_namespace_run_id")
    for t in range(N_TASKS):
        arts = {a["path"]: a["sha256"] for a in result["stages"][t]["method_artifacts"]}
        if arts.get(f"method/memory_task{t}.pt") != src["sha256"][f"memory{t}"]:
            problems[f"memory_artifact{t}"] = arts
    if ctx.runs["adaptive_gpm"] != src["run"]:
        problems["context"] = f"the seed context's adaptive GPM run is {ctx.runs['adaptive_gpm']!r}"
    if len(ctx.curriculum.stages) < N_TASKS:
        problems["curriculum"] = [st.task_key for st in ctx.curriculum.stages]
    if problems:
        raise ValueError(f"E1's source does not match its rule: {problems}")
    return {"passed": True, "namespace": ctx.namespace}


# ---- capture ------------------------------------------------------------------------------------------


class LimitedDataset:
    """The first ``n`` samples of a dataset (smoke runs only); everything else delegates."""

    def __init__(self, base, n: int) -> None:
        self.base = base
        self.n = min(int(n), len(base))

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i):
        if not 0 <= i < self.n:
            raise IndexError(i)
        return self.base[i]

    def __getattr__(self, name):
        if name == "base":
            raise AttributeError(name)
        return getattr(self.base, name)


def compare_gram(K_ref: torch.Tensor, K_new: torch.Tensor) -> dict:
    """``||K_new - K_ref||_F / ||K_ref||_F``, finiteness, and bitwise equality."""
    finite = bool(torch.isfinite(K_ref).all() and torch.isfinite(K_new).all())
    if K_ref.shape != K_new.shape:
        return {"bitwise": False, "finite": finite, "rel": float("inf")}
    bitwise = bool(torch.equal(K_ref, K_new))
    rel = 0.0 if bitwise else float((K_new - K_ref).norm()) / max(float(K_ref.norm()), 1e-300)
    return {"bitwise": bitwise, "finite": finite, "rel": rel if finite else float("inf")}


def capture_task_sensitivity(policy, dataset, sub_cfg, device, *, probe_seed: int, capture_seed: int,
                             u_seed: int, output_subsample_seed: int, n_probes: int) -> dict:
    """The production input capture, then the grad-enabled output pass over the same stream.

    Returns the production ``TaskCapture``, the riding input accumulators, the output
    accumulators, and the weight/gradient invariance checks.
    """
    from flowcl.analysis.hooks import ActivationCapture
    from flowcl.analysis.sensitivity import OutputGradientCapture
    from flowcl.experiments.gate2 import capture_task_grams
    from flowcl.models.flow_head import draw_with_generator
    from flowcl.train.trainer import build_dataloader, move_batch

    if sub_cfg.tag_s_bins:
        raise ValueError("E1 captures pooled Grams; the capture config must not tag s bins")
    device = torch.device(device)
    weights = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    grads = {n: None if p.grad is None else p.grad.detach().clone() for n, p in policy.named_parameters()}

    started = time.perf_counter()
    production = capture_task_grams(policy, dataset, sub_cfg, device, probe_seed=probe_seed,
                                    capture_seed=capture_seed)
    t_production = time.perf_counter() - started
    views = {name: production.primary_view(name) for name in production.accumulators}

    riding = ActivationCapture.for_policy(policy, tokens_per_sample=sub_cfg.tokens_per_sample,
                                          subsample_seed=sub_cfg.subsample_seed)
    outputs = OutputGradientCapture.for_policy(policy, views=views, n_probes=n_probes,
                                               tokens_per_sample=sub_cfg.tokens_per_sample,
                                               subsample_seed=output_subsample_seed)
    generator = torch.Generator(device="cpu").manual_seed(capture_seed)
    u_generator = torch.Generator(device="cpu").manual_seed(u_seed)
    loader = build_dataloader(dataset, batch_size=sub_cfg.batch_size, num_workers=sub_cfg.num_workers,
                              shuffle=False)
    policy.eval()
    n_batches = 0
    started = time.perf_counter()
    with riding, outputs:
        for batch in loader:
            batch = move_batch(batch, device)
            size = batch["actions"].shape[0]
            s = policy.s_sampler.sample(size, device, generator=generator)
            noise = draw_with_generator(tuple(batch["actions"].shape), device=device, generator=generator,
                                        dtype=torch.float32, normal=True)
            mask = batch["action_mask"]
            with riding.batch_context(s, mask), outputs.recording(mask), torch.enable_grad(), \
                    torch.autocast(device_type=device.type, enabled=False):
                out = policy(batch, s=s, noise=noise)
            velocity = out["velocity"]
            if velocity.ndim != 3 or tuple(velocity.shape[:2]) != tuple(mask.shape):
                raise RuntimeError(f"velocity {tuple(velocity.shape)} does not match the mask "
                                   f"{tuple(mask.shape)}")
            outputs.backward(velocity, mask.unsqueeze(-1), u_generator)
            del out, velocity
            n_batches += 1
    t_outputs = time.perf_counter() - started

    weights_unchanged = all(torch.equal(v, weights[k]) for k, v in policy.state_dict().items())
    grads_unchanged = all(
        (p.grad is None and grads[n] is None)
        or (p.grad is not None and grads[n] is not None and torch.equal(p.grad, grads[n]))
        for n, p in policy.named_parameters()
    )
    return {"production": production, "views": views, "riding": riding.accumulators,
            "outputs": outputs.accumulators, "weights_unchanged": weights_unchanged,
            "grads_unchanged": grads_unchanged, "n_batches": n_batches,
            "wall_clock_s": {"production": t_production, "outputs": t_outputs}}


def replay_memory(K: dict, prev: dict | None, ref_bases: dict, ref_hist: dict, src: dict, sub_cfg,
                  tol: dict) -> dict:
    """GPM's memory update from ``K`` (the rule ``memory_task{i}`` was built with), per layer
    against the reference (t5_sweep's :func:`compare_layer`, hard labels)."""
    from flowcl.analysis.subspace import extend_basis
    from flowcl.experiments.t5_sweep import _sigma_vector, compare_layer

    per_layer = {}
    for name, gram in K.items():
        M_old = None if prev is None else prev[name].vectors.to(torch.float64)
        M_new, info = extend_basis(M_old, gram, src["eps"], name, neg_tol=sub_cfg.neg_tol,
                                   rank_tol=sub_cfg.rank_tol,
                                   new_energy_fraction=src["new_energy_fraction"])
        spectrum = info.pop("residual_spectrum")
        ref = ref_bases[name]
        per_layer[name] = compare_layer(
            M_new, _sigma_vector(spectrum, M_new.shape[0]), int(info["k_added"]), ref.vectors,
            ref.singular_values.to(torch.float64), int(ref_hist[name]["k_added"]),
            {"hard": torch.ones(M_new.shape[1], dtype=torch.float64)}, tol)
    return {
        "passed": all(c["passed"] for c in per_layer.values()),
        "bitwise": all(c["bitwise"] for c in per_layer.values()),
        "n_layers": len(per_layer),
        "n_failed": sum(not c["passed"] for c in per_layer.values()),
        "failed_layers": sorted(n for n, c in per_layer.items() if not c["passed"])[:20],
        "worst": {
            "spectrum": max(c.get("spectrum", float("inf")) for c in per_layer.values()),
            "projector": max(c.get("projector", float("inf")) for c in per_layer.values()),
            "min_cosine": min(c.get("min_cosine", -float("inf")) for c in per_layer.values()),
        },
    }


def run_capture(cfg: dict, i: int, *, results_root=None, source_root=None, device="cuda",
                allow_dirty=False, dataset_dir=None, build_datasets=None,
                low_update_cfg: dict | None = None, smoke_batches: int | None = None) -> dict:
    """Capture task index ``i`` and publish ``capture/task{i}/`` atomically (reused if published)."""
    from flowcl.analysis.subspace import load_bases
    from flowcl.experiments.t5_sweep import _make_datasets
    from flowcl.train.checkpoint import load_checkpoint
    from flowcl.utils.seeding import derive_seed

    if not 0 <= i < N_TASKS:
        raise ValueError(f"task index {i} is not in 0..{N_TASKS - 1}")
    sha = refuse_dirty(allow_dirty)
    root, src_root = _root(results_root), _source_root(source_root)
    canonical = capture_dir(cfg, root, i)
    if (canonical / "capture.json").is_file():
        print(f"[flowcl] capture task {i}: published, reused", flush=True)
        return load_capture(cfg, root, i)
    ctx = seed_context(low_update_cfg or load_low_update_config(), int(cfg["seed"]), src_root)
    check_source(cfg, ctx, src_root)
    paths = source_paths(cfg, src_root)
    src, cap = cfg["source"], cfg["capture"]
    smoke = smoke_batches is not None
    sub_cfg = _capture_config(cfg)
    min_samples = float(sub_cfg.min_samples_per_dim)
    if smoke:  # three batches cannot meet N >= 10 d: recorded as unverified, not enforced
        sub_cfg = dataclasses.replace(sub_cfg, min_samples_per_dim=0.0)
    stage = ctx.curriculum.stages[i]
    key = stage.task_key
    ns = ctx.namespace
    seeds = {
        "probe": derive_seed(ns, f"gpm_memory_probe::{key}", i),
        "capture": derive_seed(ns, f"gpm_memory::{key}", i),
        "u": derive_seed(ns, f"{cap['u_tag']}::{key}", i),
        "output_subsample": int(cap["output_subsample_seed"]),
        "input_subsample": int(sub_cfg.subsample_seed),
    }
    amp = bool(ctx.ref_config["train"].get("amp", False))

    started = time.perf_counter()
    loaded = load_checkpoint(paths[f"stage{i}"], device=device)
    if loaded.payload.get("stage") != i or loaded.payload.get("task_key") != key:
        raise ValueError(f"stage{i}.pt is stage {loaded.payload.get('stage')} of "
                         f"{loaded.payload.get('task_key')!r}, expected {key!r}")
    ds = _make_datasets(ctx, loaded.spec, loaded.stats, [i], dataset_dir, build_datasets)[i]
    n_full = len(ds)
    if smoke:
        ds = LimitedDataset(ds, int(smoke_batches) * sub_cfg.batch_size)
    # The runner's cache when it captured task i: T1 under AMP (its training step), later tasks
    # in fp32 (the previous stage's evaluation).
    cache_state = warm_runner_cache(loaded.policy, {i: ds}, device, amp=amp and i == 0)[i]
    res = capture_task_sensitivity(
        loaded.policy, ds, sub_cfg, device, probe_seed=seeds["probe"], capture_seed=seeds["capture"],
        u_seed=seeds["u"], output_subsample_seed=seeds["output_subsample"], n_probes=int(cap["n_probes"]))
    groups = {e.name: e.group for e in loaded.policy.projectable_layers()}
    layers = list(groups)
    del loaded
    _free(device)

    production = res["production"]
    K = {n: production.accumulators[n].gram[res["views"][n]] for n in layers}
    K_riding = {n: res["riding"][n].gram[res["views"][n]] for n in layers}

    # memory-replay consistency (unverified in smoke runs)
    ref_bases, ref_meta = load_bases(paths[f"memory{i}"])
    prev = load_bases(paths[f"memory{i - 1}"])[0] if i > 0 else None
    try:
        replay = replay_memory(K, prev, ref_bases, ref_meta["memory_history"][str(i)], src, sub_cfg,
                               cfg["replay"])
    except (RuntimeError, ValueError) as exc:
        if not smoke:
            raise
        replay = {"passed": False, "bitwise": False, "error": str(exc)}
    replay["status"] = "unverified" if smoke else "verified"
    del ref_bases, prev

    # forward identity (enforced, smoke included)
    per_layer = {n: compare_gram(K[n], K_riding[n]) for n in layers}
    rtol = float(cfg["identity"]["gram_rtol"])
    identity = {
        "passed": all(c["finite"] and c["rel"] <= rtol for c in per_layer.values()),
        "bitwise": all(c["bitwise"] for c in per_layer.values()),
        "gram_rtol": rtol,
        "worst_rel": max(c["rel"] for c in per_layer.values()),
        "failed_layers": sorted(n for n, c in per_layer.items() if not (c["finite"] and c["rel"] <= rtol)),
    }

    labels = probe_labels(cfg)
    outs = res["outputs"]
    finite = {n: all(bool(torch.isfinite(g).all()) for g in outs[n].gram) for n in layers}
    ratios = {n: min(outs[n].n) / outs[n].d_out for n in layers}
    samples = {
        "passed": all(r >= min_samples for r in ratios.values()),
        "min_samples_per_dim": min_samples,
        "worst_ratio": min(ratios.values()),
        "worst_layers": sorted(layers, key=lambda n: ratios[n])[:5],
        "status": "unverified" if smoke else "verified",
    }
    checks = {
        "replay": replay,
        "identity": identity,
        "outputs_finite": all(finite.values()),
        "samples": samples,
        "weights_unchanged": res["weights_unchanged"],
        "grads_unchanged": res["grads_unchanged"],
    }
    enforced = (identity["passed"] and checks["outputs_finite"] and res["weights_unchanged"]
                and res["grads_unchanged"])
    valid = bool(enforced and (smoke or (replay["passed"] and samples["passed"])))

    out_root = canonical.parent
    out_root.mkdir(parents=True, exist_ok=True)
    attempt = out_root / f".attempt_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_task{i}"
    attempt.mkdir()
    files = {}
    for r, label in enumerate(labels):
        name = f"grams_{label}.pt"
        torch.save({"probe": label, "task": i, "layers": layers,
                    "grams": {n: outs[n].gram[r] for n in layers},
                    "n": {n: outs[n].n[r] for n in layers}}, attempt / name)
        files[name] = file_sha256(attempt / name)
    record = {
        "git_sha": sha, "source": src["run"], "task": i, "task_key": key,
        "stage_checkpoint_sha256": src["sha256"][f"stage{i}"],
        "smoke": smoke, "smoke_batches": smoke_batches,
        "n_dataset_samples": n_full, "n_captured_samples": len(ds), "n_batches": res["n_batches"],
        "seeds": seeds, "n_probes": int(cap["n_probes"]), "probe_labels": labels,
        "tokens_per_sample": sub_cfg.tokens_per_sample,
        "cache_state": cache_state,
        "reachability": production.reachability,
        "layers": [{"name": n, "group": groups[n], "kind": outs[n].kind, "d_out": outs[n].d_out,
                    "d_in": production.accumulators[n].d_in, "view": res["views"][n],
                    "n_out": outs[n].n, "n_in": production.accumulators[n].n[res["views"][n]],
                    "identity_rel": per_layer[n]["rel"], "identity_bitwise": per_layer[n]["bitwise"]}
                   for n in layers],
        "checks": checks, "valid": valid, "files": files,
        "wall_clock_s": {**res["wall_clock_s"], "total": time.perf_counter() - started},
    }
    atomic_write_text(attempt / "capture.json", json.dumps(record, indent=2) + "\n")
    if canonical.exists():
        raise FileExistsError(f"{canonical} already exists; a published capture is never overwritten")
    os.rename(attempt, canonical)
    print(f"[flowcl] capture task {i}: {'VALID' if valid else 'INVALID'} (replay "
          f"{replay['status']} {'pass' if replay['passed'] else 'FAIL'}, bitwise {replay['bitwise']}; "
          f"identity {'pass' if identity['passed'] else 'FAIL'}, bitwise {identity['bitwise']}, "
          f"worst rel {identity['worst_rel']:.2e}; samples {samples['status']} worst "
          f"{samples['worst_ratio']:.1f}/dim; {record['wall_clock_s']['total']:.0f}s)", flush=True)
    return record


def load_capture(cfg: dict, root: Path, i: int) -> dict:
    """A published capture's record, its Gram files re-verified."""
    d = capture_dir(cfg, root, i)
    record = json.loads((d / "capture.json").read_text())
    for name, sha in record["files"].items():
        if file_sha256(d / name) != sha:
            raise RuntimeError(f"{d / name}: changed after publication")
    return record


def load_grams(cfg: dict, root: Path, i: int, label: str) -> dict[str, torch.Tensor]:
    payload = torch.load(capture_dir(cfg, root, i) / f"grams_{label}.pt", map_location="cpu",
                         weights_only=False)
    return payload["grams"]


# ---- the summary --------------------------------------------------------------------------------------


def chain(grams: list[dict], eps: float, f: float | None, sub_cfg, keep: bool = False):
    """``U`` accumulated task by task with ``extend_basis`` (the call that builds ``M``).

    Returns ``(k, bases)``: ``k[i][layer]`` the union's rank after task index ``i``, and (if
    ``keep``) ``bases[i][layer]`` the union itself."""
    from flowcl.analysis.subspace import extend_basis

    U: dict[str, torch.Tensor | None] = {n: None for n in grams[0]}
    ks, kept = [], []
    for G in grams:
        for name, gram in G.items():
            U[name], _ = extend_basis(U[name], gram, eps, name, neg_tol=sub_cfg.neg_tol,
                                      rank_tol=sub_cfg.rank_tol, new_energy_fraction=f)
        ks.append({n: int(u.shape[1]) for n, u in U.items()})
        if keep:
            kept.append(dict(U))
    return ks, kept


def classify(valid: bool, complete: bool, decisive: float | None, threshold: float) -> dict:
    """First match wins (see the config's header)."""
    if not valid:
        verdict = INVALID
    elif not complete or decisive is None:
        verdict = INCOMPLETE
    elif decisive <= threshold:
        verdict = PASS
    else:
        verdict = FAIL
    return {"verdict": verdict, "text": VERDICT_TEXT[verdict], "exit_code": EXIT_CODES[verdict]}


def run_summary(cfg: dict, *, results_root=None, source_root=None) -> dict:
    """Always runs; writes ``summary.json`` and the decisive bases. CPU only."""
    from flowcl.analysis.sensitivity import fraction_table, forecast_k, subspace_overlap
    from flowcl.analysis.subspace import load_bases

    started = time.perf_counter()
    root, src_root = _root(results_root), _source_root(source_root)
    paths = source_paths(cfg, src_root)
    sub_cfg = _capture_config(cfg)
    rule, labels = cfg["rule"], probe_labels(cfg)

    captures = {}
    for i in range(N_TASKS):
        if (capture_dir(cfg, root, i) / "capture.json").is_file():
            captures[i] = load_capture(cfg, root, i)
    available = []
    for i in range(N_TASKS):
        if i not in captures:
            break
        available.append(i)
    smoke = sorted(i for i, c in captures.items() if c["smoke"])
    valid = all(c["valid"] for c in captures.values())
    complete = len(available) == N_TASKS and not smoke

    out = {"git_sha": git_sha(), "config": cfg,
           "framing": "E1: potential capacity savings only (no training); adaptive GPM seed 0, the "
                      "development seed.",
           "available_tasks": available, "smoke_tasks": smoke,
           "captures": {str(i): {"valid": c["valid"], "smoke": c["smoke"], "checks": c["checks"],
                                 "git_sha": c["git_sha"], "files": c["files"]}
                        for i, c in captures.items()}}
    decisive = None
    tables, forecast, stability, bases_files = {}, None, None, {}
    if available:
        layer_rows = captures[available[0]]["layers"]
        for i in available[1:]:
            if [(r["name"], r["d_out"], r["d_in"]) for r in captures[i]["layers"]] != \
                    [(r["name"], r["d_out"], r["d_in"]) for r in layer_rows]:
                raise RuntimeError(f"capture {i}'s layers differ from capture {available[0]}'s")
        dims = {r["name"]: (r["d_out"], r["d_in"]) for r in layer_rows}
        groups = {r["name"]: r["group"] for r in layer_rows}
        k_in = []
        for i in available:
            if file_sha256(paths[f"memory{i}"]) != cfg["source"]["sha256"][f"memory{i}"]:
                raise RuntimeError(f"memory_task{i}.pt differs from the pinned file")
            mem, _ = load_bases(paths[f"memory{i}"])
            k_in.append({n: int(mem[n].vectors.shape[1]) for n in dims})
            del mem
        per_probe = {label: [load_grams(cfg, root, i, label) for i in available] for label in labels}
        summed = [{n: sum(per_probe[label][j][n] for label in labels) for n in dims}
                  for j in range(len(available))]

        dec = cfg["u_rule"]["decisive"]
        rules = cfg["u_rule"]["rules"]
        k_out = {}
        for rname, f in rules.items():
            for eps in cfg["u_rule"]["eps_out"]:
                keep = rname == dec["rule"] and float(eps) == float(dec["eps_out"])
                ks, kept = chain(summed, float(eps), f, sub_cfg, keep=keep)
                k_out[(rname, _eps_key(eps))] = ks
                tables.setdefault(rname, {})[_eps_key(eps)] = {
                    str(i): {"k_out": ks[j], "fractions": fraction_table(ks[j], k_in[j], dims, groups)}
                    for j, i in enumerate(available)}
                if keep:
                    decisive_bases = kept
        dkey = (dec["rule"], _eps_key(dec["eps_out"]))
        d_idx = int(rule["decisive_task"])

        def frac(i):
            return tables[dkey[0]][dkey[1]][str(i)]["fractions"][rule["half"]]

        if d_idx in available:
            decisive = frac(d_idx)["two_sided"]

        # stability: the decisive rule on each probe alone
        f_dec = rules[dec["rule"]]
        per_label, kept_label = {}, {}
        for label in labels:
            ks, kept = chain(per_probe[label], float(dec["eps_out"]), f_dec, sub_cfg, keep=True)
            per_label[label] = {str(i): fraction_table(ks[j], k_in[j], dims, groups)[rule["half"]]
                                for j, i in enumerate(available)}
            kept_label[label] = kept
        stability = {"fractions": per_label}
        if d_idx in available and len(labels) >= 2:
            j = available.index(d_idx)
            ov = {n: subspace_overlap(kept_label[labels[0]][j][n], kept_label[labels[1]][j][n])
                  for n in dims}
            stability["overlap_median"] = {
                half: statistics.median(v for n, v in ov.items() if n.startswith(prefix))
                for half, prefix in (("trunk", "trunk."), ("decoder", "flow_head."))}
            stability["overlap_per_layer"] = ov
            sides = {label: per_label[label][str(d_idx)]["two_sided"] <= float(rule["threshold"])
                     for label in labels}
            stability["estimator_sensitive"] = len(set(sides.values())) > 1
        del kept_label

        # forecast to T8 (labelled as a forecast)
        a, b = (int(x) for x in cfg["forecast"]["from_tasks"])
        if a in available and b in available:
            ja, jb = available.index(a), available.index(b)
            ks = k_out[dkey]
            fk_out = {n: forecast_k(ks[ja][n], ks[jb][n], dims[n][0], b + 1, b - a, cfg["forecast"]["to"])
                      for n in dims}
            fk_in = {n: forecast_k(k_in[ja][n], k_in[jb][n], dims[n][1], b + 1, b - a, cfg["forecast"]["to"])
                     for n in dims}
            forecast = {"label": "FORECAST (linear extrapolation; not a measurement)",
                        "tasks": {str(t): fraction_table({n: fk_out[n][t] for n in dims},
                                                         {n: fk_in[n][t] for n in dims}, dims, groups)
                                  for t in cfg["forecast"]["to"]}}

        # the decisive bases, for E2
        target = bases_dir(cfg, root)
        target.mkdir(parents=True, exist_ok=True)
        for j, i in enumerate(available):
            path = target / decisive_bases_name(cfg, i)
            payload = {"meta": {"kind": "twosided_output_union", "rule": dec["rule"],
                                "eps_out": float(dec["eps_out"]), "new_energy_fraction": f_dec,
                                "task_idx": i, "tasks_in_union": available[: j + 1],
                                "source": cfg["source"]["run"], "probes": labels,
                                "orientation": "(d_out, k_out) orthonormal columns in each layer's "
                                               "output space"},
                       "layers": list(dims), "bases": decisive_bases[j]}
            if path.is_file():
                old = torch.load(path, map_location="cpu", weights_only=False)
                if old["layers"] != payload["layers"] or not all(
                        torch.equal(old["bases"][n], payload["bases"][n]) for n in dims):
                    raise RuntimeError(f"{path} exists and differs from the recomputed union")
            else:
                tmp = path.with_name(path.name + ".tmp")
                torch.save(payload, tmp)
                os.replace(tmp, path)
            bases_files[path.name] = file_sha256(path)

    verdict = classify(valid, complete, decisive, float(rule["threshold"]))
    labels_out = []
    if stability and stability.get("estimator_sensitive"):
        labels_out.append("estimator_sensitive")
    near = None
    if decisive is not None:
        near = frac(int(rule["decisive_task"]))["rho_out"] >= float(rule["near_full_rank"])
        if near:
            labels_out.append("near_full_rank")
    gpm_at = frac(int(rule["decisive_task"]))["gpm"] if decisive is not None else None
    out.update({
        "verdict": {**verdict, "labels": labels_out, "decisive": decisive, "gpm_at_decisive": gpm_at,
                    "threshold": float(rule["threshold"]), "near_full_rank": near},
        "tables": tables, "forecast": forecast, "stability": stability, "bases": bases_files,
        "wall_clock_s": time.perf_counter() - started,
    })
    path = root / cfg["out"] / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(out, indent=2) + "\n")
    d = "n/a" if decisive is None else f"{decisive:.4f}"
    g = "n/a" if gpm_at is None else f"{gpm_at:.4f}"
    print(f"[flowcl] E1 verdict: {verdict['verdict']} {labels_out} (trunk two-sided {d}, GPM {g}, "
          f"threshold {rule['threshold']})", flush=True)
    return out
