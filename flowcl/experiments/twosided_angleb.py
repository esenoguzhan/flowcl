"""E1-B: the angle-B capacity gate (``configs/analysis/twosided_angleb.yaml``).

No training. From E1's source (adaptive GPM seed 0) and the recovery study's published map:

1. :func:`run_capture` (one task index ``i`` = T(i+1), on ``stage{i}.pt``): the integrated executed
   prefix's exact coherent output Grams per (phase bin, group) and the independent-step Hutchinson
   Grams (:class:`flowcl.analysis.integrated_sensitivity.IntegratedOutputCapture`), with the
   integration identity checked bitwise on every batch. Published atomically.
2. :func:`run_summary`, always (CPU): the variants' Grams from the stored ones, E1's U chains and
   protected fractions, the verdict and labels, E1's numbers alongside, the alignment check.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
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
from flowcl.experiments.twosided_gonogo import (
    N_TASKS,
    LimitedDataset,
    _capture_config,
    _eps_key,
    _source_root,
    chain,
    check_source,
    source_paths,
)
from flowcl.utils.libero_paths import repo_root
from flowcl.utils.run import atomic_write_text, file_sha256, git_sha

INVALID, INCOMPLETE, PASS, FAIL = "invalid", "incomplete", "pass", "fail"
VERDICT_TEXT = {
    INVALID: "A capture failed a validity check: no fraction is interpreted.",
    INCOMPLETE: "Fewer than four production captures (a failed or smoke run, not a finding).",
    PASS: "Under the injury-weighted continuous executed-prefix surrogate, two-sided protection would "
          "protect at most the registered fraction of the trunk's weight directions at the T4 pilot's "
          "start: potential capacity (E2-B tests retention).",
    FAIL: "The tested surrogate did not meet the preregistered 50% capacity budget under this capture "
          "and accumulation rule.",
}
EXIT_CODES = {PASS: 0, FAIL: 0, INVALID: 3, INCOMPLETE: 4}
PROBES = "ABCDEFGH"


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else repo_root() / "configs" / "analysis" / "twosided_angleb.yaml"
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def probe_labels(cfg: dict) -> list[str]:
    return list(PROBES[: int(cfg["capture"]["n_probes"])])


def capture_dir(cfg: dict, root: Path, i: int) -> Path:
    return root / cfg["out"] / "capture" / f"task{i}"


def bases_dir(cfg: dict, root: Path) -> Path:
    return root / cfg["out"] / "bases"


def bases_name(cfg: dict, variant: str, i: int) -> str:
    dec = cfg["u_rule"]["decisive"]
    return f"{variant}_{dec['rule']}_e{float(dec['eps_out'])}_task{i}.pt"


def _group_ranges(cfg: dict) -> list[tuple[int, int]]:
    return [tuple(int(x) for x in v) for v in cfg["capture"]["groups"].values()]


# ---- the recovery inputs ---------------------------------------------------------------------------


def _rel(path: str | Path, base: Path) -> Path:
    path = Path(str(path))
    return path if path.is_absolute() else base / path


def recovery_inputs(cfg: dict, src_root: Path, task_key: str, i: int) -> dict:
    """Recovery task ``i``'s rows, sigma and weights, with every pin checked."""
    from flowcl.analysis.recovery import phase_group_weights

    rec = cfg["recovery"]
    rule_file = _rel(rec["rule"], repo_root())
    if file_sha256(rule_file) != rec["rule_sha256"]:
        raise ValueError(f"{rule_file} differs from the pinned recovery rule")
    analysis = src_root / rec["analysis"]
    if file_sha256(analysis) != rec["analysis_sha256"]:
        raise ValueError(f"{analysis} differs from the pinned recovery analysis")
    rrule = OmegaConf.to_container(OmegaConf.load(rule_file), resolve=True)
    if rrule["tasks"][i]["key"] != task_key:
        raise ValueError(f"recovery task {i} is {rrule['tasks'][i]['key']!r}, the curriculum's is {task_key!r}")
    work = src_root / rec["work"] / task_key.replace("/", "__")
    nominal = json.loads((work / "nominal.json").read_text())
    rows, files = [], {"nominal.json": file_sha256(work / "nominal.json")}
    for f in sorted(work.glob("rollout_*.json")):
        payload = json.loads(f.read_text())
        if payload["binding"]["rule_sha256"] != rec["rule_sha256"] or payload["binding"].get("smoke"):
            raise ValueError(f"{f}: not a production work file of the pinned recovery rule")
        rows.extend(payload["rows"])
        files[f.name] = file_sha256(f)
    if nominal["binding"]["rule_sha256"] != rec["rule_sha256"] or nominal["fallback"]:
        raise ValueError(f"{work}/nominal.json: not the pinned rule's main regime")
    weights = phase_group_weights(rows, int(rec["n_bins"]), float(rec["magnitude"]))
    lengths = [int(r["n_steps"]) for r in nominal["rollouts"].values() if r["success"]]
    return {"key": task_key, "namespace": rrule["namespace"], "rule": rrule, "rows": rows,
            "sigma": [float(x) for x in nominal["sigma"]], "weights": weights, "files": files,
            "rollout_lengths": lengths}


def variant_weights(cfg: dict, weights: dict, namespace: str, key: str, i: int) -> dict[str, list]:
    """``w_v[bin][group]`` for every variant (the config's header)."""
    from flowcl.utils.seeding import derive_seed

    w = np.asarray(weights["w"], dtype=np.float64)
    pooled = np.asarray(weights["pooled"], dtype=np.float64)
    perm = np.random.default_rng(derive_seed(namespace, f"{cfg['capture']['shuffle_tag']}::{key}", i)).permutation(w.size)
    grip = list(cfg["capture"]["groups"]).index("gripper")
    r0 = w.copy()
    r0[:, grip] = 0.0
    out = {"R": w, "I": np.ones_like(w), "S": w.reshape(-1)[perm].reshape(w.shape),
           "P": np.broadcast_to(pooled, w.shape).copy(), "R0": r0}
    unknown = set(cfg["variants"]) - set(out)
    if unknown:
        raise ValueError(f"unknown variants {sorted(unknown)}")
    return {v: out[v].tolist() for v in cfg["variants"]}


def phase_bins(t: torch.Tensor, lengths: torch.Tensor, n_bins: int) -> torch.Tensor:
    """``min(n_bins - 1, floor(n_bins * t / L))``."""
    return torch.clamp((t.long() * n_bins) // lengths.long(), max=n_bins - 1)


# ---- capture ------------------------------------------------------------------------------------------


def capture_integrated(policy, dataset, sub_cfg, device, *, cfg: dict, scale: list[float], w_R: list,
                       seeds: dict) -> dict:
    """The integrated prefix's Grams over ``dataset`` in a fixed order (the config's header)."""
    from flowcl.analysis.integrated_sensitivity import IntegratedOutputCapture, integrate
    from flowcl.models.flow_head import draw_with_generator
    from flowcl.train.trainer import build_dataloader, move_batch

    device = torch.device(device)
    weights = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    grads = {n: None if p.grad is None else p.grad.detach().clone() for n, p in policy.named_parameters()}
    cap_cfg = cfg["capture"]
    n_bins, k = int(cfg["recovery"]["n_bins"]), int(cap_cfg["execute_k"])
    ranges = _group_ranges(cfg)
    d_action = int(policy.flow_head.d_action)
    if ranges[-1][1] != d_action or len(scale) != d_action:
        raise ValueError("the groups and the scale must cover the action dimensions")
    wd = torch.zeros(n_bins, d_action, dtype=torch.float32)   # w_R per (bin, dimension)
    for g, (lo, hi) in enumerate(ranges):
        wd[:, lo:hi] = torch.tensor([row[g] for row in w_R], dtype=torch.float32).unsqueeze(1)
    c = torch.tensor(scale, dtype=torch.float32)
    lengths = torch.tensor([ep.length for ep in dataset.episodes], dtype=torch.long)
    cap = IntegratedOutputCapture.for_policy(policy, groups=ranges, n_bins=n_bins,
                                             n_probes=int(cap_cfg["n_probes"]),
                                             tokens_per_sample=sub_cfg.tokens_per_sample,
                                             subsample_seed=seeds["rows"])
    noise_gen = torch.Generator(device="cpu").manual_seed(seeds["noise"])
    u_gen = torch.Generator(device="cpu").manual_seed(seeds["u"])
    loader = build_dataloader(dataset, batch_size=sub_cfg.batch_size, num_workers=sub_cfg.num_workers,
                              shuffle=False)
    policy.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    n_batches, mismatched, worst = 0, 0, 0.0
    t_vjp = 0.0
    started = time.perf_counter()
    with cap:
        for batch in loader:
            batch = move_batch(batch, device)
            size = batch["actions"].shape[0]
            noise = draw_with_generator((size, policy.horizon, d_action), device=device, generator=noise_gen,
                                        dtype=torch.float32, normal=True)
            bins = phase_bins(batch["t"].cpu(), lengths[batch["episode_idx"].cpu()], n_bins)
            probes = [torch.randn((size, k, d_action), generator=u_gen, dtype=torch.float32)
                      * wd[bins].sqrt().unsqueeze(1) * c for _ in range(int(cap_cfg["n_probes"]))]
            with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
                reference = policy.sample(batch, noise=noise)
            with cap.recording(), torch.enable_grad(), torch.autocast(device_type=device.type, enabled=False):
                actions = integrate(policy, batch, noise, policy.euler_steps)
            if not torch.equal(actions.detach(), reference):
                mismatched += 1
                worst = max(worst, float((actions.detach() - reference).abs().max()))
            t0 = time.perf_counter()
            cap.backward(actions[:, :k], scale, bins, probes)
            t_vjp += time.perf_counter() - t0
            del actions, reference
            n_batches += 1
    weights_unchanged = all(torch.equal(v, weights[n]) for n, v in policy.state_dict().items())
    grads_unchanged = all(
        (p.grad is None and grads[n] is None)
        or (p.grad is not None and grads[n] is not None and torch.equal(p.grad, grads[n]))
        for n, p in policy.named_parameters())
    return {"accumulators": cap.accumulators, "n_batches": n_batches,
            "identity": {"passed": mismatched == 0, "bitwise_batches": n_batches - mismatched,
                         "mismatched_batches": mismatched, "worst_abs": worst},
            "weights_unchanged": weights_unchanged, "grads_unchanged": grads_unchanged,
            "wall_clock_s": {"capture": time.perf_counter() - started, "vjp_and_accumulation": t_vjp},
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}


def run_capture(cfg: dict, i: int, *, results_root=None, source_root=None, device="cuda",
                allow_dirty=False, dataset_dir=None, build_datasets=None,
                low_update_cfg: dict | None = None, smoke_batches: int | None = None) -> dict:
    """Capture task index ``i`` and publish ``capture/task{i}/`` atomically (reused if published)."""
    from flowcl.analysis.recovery import demo_gripper_closed_by_bin
    from flowcl.experiments.t5_sweep import _make_datasets
    from flowcl.train.checkpoint import load_checkpoint
    from flowcl.utils.seeding import derive_seed

    if not 0 <= i < N_TASKS:
        raise ValueError(f"task index {i} is not in 0..{N_TASKS - 1}")
    sha = refuse_dirty(allow_dirty)
    root, src_root = _root(results_root), _source_root(source_root)
    canonical = capture_dir(cfg, root, i)
    if (canonical / "capture.json").is_file():
        print(f"[flowcl] E1-B capture task {i}: published, reused", flush=True)
        return load_capture(cfg, root, i)
    ctx = seed_context(low_update_cfg or load_low_update_config(), int(cfg["seed"]), src_root)
    check_source(cfg, ctx, src_root)
    paths = source_paths(cfg, src_root)
    cap = cfg["capture"]
    smoke = smoke_batches is not None
    sub_cfg = _capture_config(cfg)
    min_samples = float(sub_cfg.min_samples_per_dim)
    key = ctx.curriculum.stages[i].task_key
    ns = ctx.namespace
    rec = recovery_inputs(cfg, src_root, key, i)
    seeds = {"noise": derive_seed(ns, f"{cap['noise_tag']}::{key}", i),
             "u": derive_seed(ns, f"{cap['u_tag']}::{key}", i),
             "shuffle": derive_seed(ns, f"{cap['shuffle_tag']}::{key}", i),
             "rows": int(cap["row_subsample_seed"])}
    variants = variant_weights(cfg, rec["weights"], ns, key, i)
    amp = bool(ctx.ref_config["train"].get("amp", False))

    started = time.perf_counter()
    loaded = load_checkpoint(paths[f"stage{i}"], device=device)
    if loaded.payload.get("stage") != i or loaded.payload.get("task_key") != key:
        raise ValueError(f"stage{i}.pt is stage {loaded.payload.get('stage')} of "
                         f"{loaded.payload.get('task_key')!r}, expected {key!r}")
    std = [float(x) for x in loaded.stats.action.std_array()]
    scale = [s / sg for s, sg in zip(std, rec["sigma"])]
    ds = _make_datasets(ctx, loaded.spec, loaded.stats, [i], dataset_dir, build_datasets)[i]
    n_full = len(ds)
    demo = {"lengths": [int(ep.length) for ep in ds.episodes],
            "gripper": demo_gripper_closed_by_bin([ep.action[:, -1] for ep in ds.episodes],
                                                  int(cfg["recovery"]["n_bins"]))}
    if smoke:
        ds = LimitedDataset(ds, int(smoke_batches) * sub_cfg.batch_size)
    cache_state = warm_runner_cache(loaded.policy, {i: ds}, device, amp=amp and i == 0)[i]
    res = capture_integrated(loaded.policy, ds, sub_cfg, device, cfg=cfg, scale=scale, w_R=rec["weights"]["w"],
                             seeds=seeds)
    meta = {e.name: (e.group, e.module.in_features) for e in loaded.policy.projectable_layers()}
    layers = [n for n in meta if n in res["accumulators"]]
    del loaded
    _free(device)

    accs = res["accumulators"]
    finite = {n: bool(torch.isfinite(accs[n].coherent).all()) and all(bool(torch.isfinite(h).all())
                                                                      for h in accs[n].hutch) for n in layers}
    ratios = {n: accs[n].n / accs[n].d_out for n in layers}
    samples = {"passed": all(r >= min_samples for r in ratios.values()), "min_samples_per_dim": min_samples,
               "worst_ratio": min(ratios.values()), "worst_layers": sorted(layers, key=lambda n: ratios[n])[:5],
               "status": "unverified" if smoke else "verified"}
    checks = {"identity": res["identity"], "outputs_finite": all(finite.values()), "samples": samples,
              "weights_unchanged": res["weights_unchanged"], "grads_unchanged": res["grads_unchanged"]}
    enforced = (res["identity"]["passed"] and checks["outputs_finite"] and res["weights_unchanged"]
                and res["grads_unchanged"])
    valid = bool(enforced and (smoke or samples["passed"]))

    out_root = canonical.parent
    out_root.mkdir(parents=True, exist_ok=True)
    attempt = out_root / f".attempt_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_task{i}"
    attempt.mkdir()
    files = {}
    torch.save({"task": i, "layers": layers, "groups": list(cap["groups"]),
                "n_bins": int(cfg["recovery"]["n_bins"]),
                "grams": {n: accs[n].coherent for n in layers}, "n": {n: accs[n].n for n in layers},
                "n_bin": {n: accs[n].n_bin for n in layers}}, attempt / "grams_coherent.pt")
    files["grams_coherent.pt"] = file_sha256(attempt / "grams_coherent.pt")
    for r, label in enumerate(probe_labels(cfg)):
        name = f"grams_hutch_{label}.pt"
        torch.save({"probe": label, "task": i, "layers": layers,
                    "grams": {n: accs[n].hutch[r] for n in layers}, "n": {n: accs[n].n for n in layers}},
                   attempt / name)
        files[name] = file_sha256(attempt / name)
    record = {
        "git_sha": sha, "source": cfg["source"]["run"], "task": i, "task_key": key,
        "stage_checkpoint_sha256": cfg["source"]["sha256"][f"stage{i}"],
        "smoke": smoke, "smoke_batches": smoke_batches,
        "n_dataset_samples": n_full, "n_captured_samples": len(ds), "n_batches": res["n_batches"],
        "seeds": seeds, "probe_labels": probe_labels(cfg), "tokens_per_sample": sub_cfg.tokens_per_sample,
        "execute_k": int(cap["execute_k"]), "batch_size": sub_cfg.batch_size, "cache_state": cache_state,
        "recovery": {"rule_sha256": cfg["recovery"]["rule_sha256"],
                     "analysis_sha256": cfg["recovery"]["analysis_sha256"], "files": rec["files"],
                     "sigma": rec["sigma"], "weights": rec["weights"], "rollout_lengths": rec["rollout_lengths"]},
        "variants": variants, "std": std, "scale": scale, "demo": demo,
        "layers": [{"name": n, "group": meta[n][0], "kind": accs[n].kind, "d_out": accs[n].d_out,
                    "d_in": meta[n][1], "n_rows": accs[n].n, "n_bin": accs[n].n_bin, "calls": accs[n].calls}
                   for n in layers],
        "checks": checks, "valid": valid, "files": files, "peak_gpu_bytes": res["peak_gpu_bytes"],
        "wall_clock_s": {**res["wall_clock_s"], "total": time.perf_counter() - started},
    }
    atomic_write_text(attempt / "capture.json", json.dumps(record, indent=2) + "\n")
    if canonical.exists():
        raise FileExistsError(f"{canonical} already exists; a published capture is never overwritten")
    os.rename(attempt, canonical)
    ident = res["identity"]
    print(f"[flowcl] E1-B capture task {i}: {'VALID' if valid else 'INVALID'} (identity "
          f"{ident['bitwise_batches']}/{res['n_batches']} bitwise, worst {ident['worst_abs']:.2e}; samples "
          f"{samples['status']} worst {samples['worst_ratio']:.1f}/dim; {res['n_batches']} batches, "
          f"{record['wall_clock_s']['total']:.0f}s, VJPs {res['wall_clock_s']['vjp_and_accumulation']:.0f}s)",
          flush=True)
    return record


def load_capture(cfg: dict, root: Path, i: int) -> dict:
    """A published capture's record, its Gram files re-verified."""
    d = capture_dir(cfg, root, i)
    record = json.loads((d / "capture.json").read_text())
    for name, sha in record["files"].items():
        if file_sha256(d / name) != sha:
            raise RuntimeError(f"{d / name}: changed after publication")
    return record


def _load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


# ---- the summary --------------------------------------------------------------------------------------


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


def alignment(cfg: dict, rec: dict, record: dict) -> dict:
    """Command-level phase evidence: per-bin gripper-closed fractions, demos (``record``, a capture)
    against the recovery rollouts (``rec``, :func:`recovery_inputs`), and the median rollout / demo
    length ratio."""
    from flowcl.analysis.recovery import gripper_closed_by_bin, perturbation_bank
    from flowcl.utils.seeding import derive_seed

    rr = rec["rule"]
    key, ns = rec["key"], rec["namespace"]
    sigma = np.asarray(rec["sigma"])
    banks: dict = {}

    def direction(rollout, query, bank_index):
        if (rollout, query) not in banks:
            banks[(rollout, query)] = perturbation_bank(
                sigma, int(rr["eval"]["execute_k"]), rr["perturbations"]["magnitudes"],
                int(rr["perturbations"]["n_isotropic_per_magnitude"]),
                derive_seed(ns, f"recovery::bank::{key}", int(rollout) * 100 + int(query)))
        return float(np.sign(banks[(rollout, query)][int(bank_index)]["delta"][0, -1]))

    n_bins, mag = int(cfg["recovery"]["n_bins"]), float(cfg["recovery"]["magnitude"])
    rollouts = gripper_closed_by_bin(rec["rows"], direction, n_bins, mag)
    demo = record["demo"]
    ratio = statistics.median(rec["rollout_lengths"]) / statistics.median(demo["lengths"])
    return {"label": "command-level evidence of phase correspondence (not evidence that an object is "
                     "carried)",
            "demo_closed_fraction": demo["gripper"]["closed_fraction"],
            "rollout_closed_fraction": rollouts["closed_fraction"], "rollout_determined": rollouts["determined"],
            "rollout_undetermined": rollouts["undetermined"],
            "median_rollout_steps": statistics.median(rec["rollout_lengths"]),
            "median_demo_steps": statistics.median(demo["lengths"]), "length_ratio": ratio}


def run_summary(cfg: dict, *, results_root=None, source_root=None) -> dict:
    """Always runs; writes ``summary.json`` and the variants' decisive bases. CPU only."""
    from flowcl.analysis.integrated_sensitivity import combine_grams
    from flowcl.analysis.sensitivity import fraction_table, forecast_k, subspace_overlap
    from flowcl.analysis.subspace import load_bases

    started = time.perf_counter()
    root, src_root = _root(results_root), _source_root(source_root)
    paths = source_paths(cfg, src_root)
    sub_cfg = _capture_config(cfg)
    rule, labels = cfg["rule"], probe_labels(cfg)
    variants, dvar = list(cfg["variants"]), cfg["decisive_variant"]

    captures = {i: load_capture(cfg, root, i) for i in range(N_TASKS)
                if (capture_dir(cfg, root, i) / "capture.json").is_file()}
    available = []
    for i in range(N_TASKS):
        if i not in captures:
            break
        available.append(i)
    smoke = sorted(i for i, c in captures.items() if c["smoke"])
    valid = all(c["valid"] for c in captures.values())
    complete = len(available) == N_TASKS and not smoke

    e1_path = src_root / cfg["e1"]["summary"]
    e1 = None
    if e1_path.is_file():
        if file_sha256(e1_path) != cfg["e1"]["sha256"]:
            raise ValueError(f"{e1_path} differs from the pinned E1 summary")
        e1s = json.loads(e1_path.read_text())
        e1 = {"decisive": e1s["verdict"]["decisive"], "labels": e1s["verdict"]["labels"],
              "adaptive_0.95_by_task": {i: e1s["tables"]["adaptive"]["0.95"][i]["fractions"][rule["half"]]
                                        for i in e1s["tables"]["adaptive"]["0.95"]}}

    out = {"git_sha": git_sha(), "config": cfg,
           "framing": "E1-B: potential capacity under the injury-weighted continuous executed-prefix "
                      "surrogate (no training); adaptive GPM seed 0, the development seed.",
           "available_tasks": available, "smoke_tasks": smoke,
           "captures": {str(i): {"valid": c["valid"], "smoke": c["smoke"], "checks": c["checks"],
                                 "git_sha": c["git_sha"], "files": c["files"], "seeds": c["seeds"],
                                 "wall_clock_s": c["wall_clock_s"], "peak_gpu_bytes": c["peak_gpu_bytes"]}
                        for i, c in captures.items()},
           "weights": {str(i): {"recovery": c["recovery"]["weights"], "variants": c["variants"],
                                "scale": c["scale"]} for i, c in captures.items()},
           "e1": e1}
    decisive, tables, stability, forecast, bases_files, align = {}, {}, None, None, {}, {}
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
        # the weights recomputed from the pinned work files must be the captured ones
        for i in available:
            rec = recovery_inputs(cfg, src_root, captures[i]["task_key"], i)
            if rec["weights"] != captures[i]["recovery"]["weights"]:
                raise RuntimeError(f"capture {i}: the recovery weights changed since the capture")
            align[str(i)] = alignment(cfg, rec, captures[i])

        grams: dict[str, list[dict]] = {v: [] for v in variants}
        hutch: dict[str, list[dict]] = {label: [] for label in labels}
        for i in available:
            d = capture_dir(cfg, root, i)
            coh = _load(d / "grams_coherent.pt")
            for v in variants:
                w = captures[i]["variants"][v]
                grams[v].append({n: combine_grams(coh["grams"][n], w) for n in dims})
            del coh
            for label in labels:
                hutch[label].append(dict(_load(d / f"grams_hutch_{label}.pt")["grams"]))
        grams["Ri"] = [{n: sum(hutch[label][j][n] for label in labels) for n in dims}
                       for j in range(len(available))]

        dec = cfg["u_rule"]["decisive"]
        rules = cfg["u_rule"]["rules"]
        dkey = (dec["rule"], _eps_key(dec["eps_out"]))
        d_idx = int(rule["decisive_task"])
        k_out, kept_bases = {}, {}
        for v, gl in grams.items():
            for rname, f in rules.items():
                for eps in cfg["u_rule"]["eps_out"]:
                    keep = (rname, _eps_key(eps)) == dkey and v != "Ri" and v != "R0"
                    ks, kept = chain(gl, float(eps), f, sub_cfg, keep=keep)
                    k_out[(v, rname, _eps_key(eps))] = ks
                    tables.setdefault(v, {}).setdefault(rname, {})[_eps_key(eps)] = {
                        str(i): {"k_out": ks[j], "fractions": fraction_table(ks[j], k_in[j], dims, groups)}
                        for j, i in enumerate(available)}
                    if keep:
                        kept_bases[v] = kept
            if d_idx in available:
                decisive[v] = tables[v][dkey[0]][dkey[1]][str(d_idx)]["fractions"][rule["half"]]

        # Ri's stability: the decisive rule on each probe alone
        f_dec = rules[dec["rule"]]
        per_label, kept_label = {}, {}
        for label in labels:
            ks, kept = chain(hutch[label], float(dec["eps_out"]), f_dec, sub_cfg, keep=True)
            per_label[label] = {str(i): fraction_table(ks[j], k_in[j], dims, groups)[rule["half"]]
                                for j, i in enumerate(available)}
            kept_label[label] = kept
        stability = {"variant": "Ri", "fractions": per_label}
        if d_idx in available and len(labels) >= 2:
            j = available.index(d_idx)
            ov = {n: subspace_overlap(kept_label[labels[0]][j][n], kept_label[labels[1]][j][n]) for n in dims}
            stability["overlap_median"] = {
                half: statistics.median(x for n, x in ov.items() if n.startswith(prefix))
                for half, prefix in (("trunk", "trunk."), ("decoder", "flow_head."))}
            sides = {label: per_label[label][str(d_idx)]["two_sided"] <= float(rule["threshold"])
                     for label in labels}
            stability["estimator_sensitive"] = len(set(sides.values())) > 1
        del kept_label, hutch

        # the decisive variant's forecast to T8 (labelled as a forecast)
        a, b = (int(x) for x in cfg["forecast"]["from_tasks"])
        if a in available and b in available:
            ja, jb = available.index(a), available.index(b)
            ks = k_out[(dvar, *dkey)]
            fk_out = {n: forecast_k(ks[ja][n], ks[jb][n], dims[n][0], b + 1, b - a, cfg["forecast"]["to"])
                      for n in dims}
            fk_in = {n: forecast_k(k_in[ja][n], k_in[jb][n], dims[n][1], b + 1, b - a, cfg["forecast"]["to"])
                     for n in dims}
            forecast = {"label": "FORECAST (linear extrapolation; not a measurement)", "variant": dvar,
                        "tasks": {str(t): fraction_table({n: fk_out[n][t] for n in dims},
                                                         {n: fk_in[n][t] for n in dims}, dims, groups)
                                  for t in cfg["forecast"]["to"]}}

        # the variants' decisive-rule bases, for E2-B
        target = bases_dir(cfg, root)
        target.mkdir(parents=True, exist_ok=True)
        for v, kept in kept_bases.items():
            for j, i in enumerate(available):
                path = target / bases_name(cfg, v, i)
                payload = {"meta": {"kind": "twosided_angleb_output_union", "variant": v, "rule": dec["rule"],
                                    "eps_out": float(dec["eps_out"]), "new_energy_fraction": f_dec,
                                    "task_idx": i, "tasks_in_union": available[: j + 1],
                                    "source": cfg["source"]["run"],
                                    "orientation": "(d_out, k_out) orthonormal columns in each layer's output "
                                                   "space"},
                           "layers": list(dims), "bases": kept[j]}
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

    thr = float(rule["threshold"])
    dec_frac = decisive.get(dvar)
    verdict = classify(valid, complete, None if dec_frac is None else dec_frac["two_sided"], thr)
    labels_out = []
    if verdict["verdict"] in (PASS, FAIL):
        if decisive["I"]["two_sided"] <= thr:
            labels_out.append("uniform_integrated_suffices")
        if decisive["S"]["two_sided"] <= thr:
            labels_out.append("concentration_suffices")
        if (decisive["P"]["two_sided"] <= thr) != (dec_frac["two_sided"] <= thr):
            labels_out.append("phase_mapping_sensitive")
        if dec_frac["rho_out"] >= float(rule["near_full_rank"]):
            labels_out.append("near_full_rank")
    out.update({
        "verdict": {**verdict, "labels": labels_out, "variant": dvar,
                    "decisive": None if dec_frac is None else dec_frac["two_sided"],
                    "gpm_at_decisive": None if dec_frac is None else dec_frac["gpm"],
                    "rho_out_at_decisive": None if dec_frac is None else dec_frac["rho_out"],
                    "threshold": thr,
                    "by_variant": {v: {"two_sided": x["two_sided"], "rho_out": x["rho_out"]}
                                   for v, x in decisive.items()},
                    "e1_velocity": None if e1 is None else e1["decisive"]},
        "tables": tables, "forecast": forecast, "stability": stability, "alignment": align,
        "bases": bases_files, "wall_clock_s": time.perf_counter() - started,
    })
    path = root / cfg["out"] / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(out, indent=2) + "\n")
    by = ", ".join(f"{v} {x['two_sided']:.4f}" for v, x in decisive.items()) or "n/a"
    print(f"[flowcl] E1-B verdict: {verdict['verdict']} {labels_out} (trunk two-sided: {by}; threshold {thr})",
          flush=True)
    return out
