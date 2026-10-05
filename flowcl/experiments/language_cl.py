"""Language-conditioned base study (``configs/analysis/language_cl.yaml``).

Stage 0 trains A jointly with S2 (same scene and object, another goal), so the base has a training
contrast that requires reading the instruction; stage 1 learns S (another similar goal) under
seq_ft and plain GPM. The training runs are ``scripts/run_continual.py`` on the
``langbase_cabinet_stove`` curriculum; this module adds what the runner does not record, and judges
the rule:

1. :func:`run_swaps`: instruction-swap rollouts on a run's stage checkpoint, with the requested
   task's goal scored along the way (stage 0: A <-> S2 on seq_ft; stage 1: A <-> S on each arm);
2. :func:`run_probe`: the descriptive probes on four checkpoints (the A-only base of similar_cl,
   this base, and both stage-1 checkpoints);
3. :func:`run_summary`: the validity checks, the cells, L1a / L1b / L3 / L4, and ``summary.json``
   (written atomically in every case).

Before any rollout or probe the text cache is rebuilt as the checkpoint's runner built it
(:func:`flowcl.train.continual.warm_runner_text_cache`) and its precision recorded.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from flowcl.experiments import similar_tasks as st
from flowcl.utils.libero_paths import repo_root
from flowcl.utils.run import atomic_write_text, file_sha256, git_sha

VALID, INVALID, INCOMPLETE = "valid", "invalid", "incomplete"
EXIT_CODES = {VALID: 0, INVALID: 3, INCOMPLETE: 4}
ARMS = ("seq_ft", "gpm")
PROBE_LABELS = ("a_only_stage0", "base_stage0", "seq_ft_stage1", "gpm_stage1")


def config_path() -> Path:
    return repo_root() / "configs" / "analysis" / "language_cl.yaml"


def load_config(path: str | Path | None = None) -> dict:
    return OmegaConf.to_container(OmegaConf.load(Path(path) if path else config_path()), resolve=True)


def _root(results_root) -> Path:
    return Path(results_root) if results_root else repo_root() / "results"


def out_dir(cfg: dict, root: Path) -> Path:
    return root / cfg["out"]


def method_spec(cfg: dict, arm: str) -> tuple[str, dict, str]:
    """``(registry name, kwargs, display label)`` of an arm's method config file."""
    from flowcl.experiments.reverse_order import method_spec as spec

    return spec({"methods": cfg["arms"]}, arm)


def run_name(cfg: dict, arm: str) -> str:
    from flowcl.train.continual import continual_run_id

    return continual_run_id(method_spec(cfg, arm)[2], cfg["curriculum"], int(cfg["seed"]))


def namespace(cfg: dict) -> str:
    from flowcl.train.continual import seed_namespace_run_id

    return seed_namespace_run_id(cfg["curriculum"], int(cfg["seed"]))


def tasks(cfg: dict) -> dict[str, str]:
    return {"A": cfg["task_a"], "S2": cfg["task_co"], "S": cfg["task_b"]}


def _conf(run_dir: Path) -> dict:
    return OmegaConf.to_container(OmegaConf.load(run_dir / "config.yaml"), resolve=True)


def _bootstrap() -> dict:
    return OmegaConf.to_container(
        OmegaConf.load(repo_root() / "configs" / "eval" / "libero_eval.yaml"), resolve=True)["bootstrap"]


def instruction_of(task_key: str, dataset_dir=None) -> str:
    """The instruction a task was trained under: its demo file's own."""
    from flowcl.data.libero_adapter import read_task_metadata
    from flowcl.data.tasks import TaskRef

    return read_task_metadata(TaskRef.from_key(task_key).demo_path(dataset_dir)).language


def _attempt_publish(canonical: Path, filename: str, record: dict) -> None:
    attempt = st._attempt(canonical.parent, canonical.name)
    atomic_write_text(attempt / filename, json.dumps(record, indent=2, default=str) + "\n")
    st._publish(attempt, canonical)


def rebuild_cache(policy, curriculum_name: str, extra_keys: list[str], dataset_dir, device, amp: bool) -> dict:
    """The runner's text cache for ``curriculum_name`` (its stage-0 train instructions under AMP
    when the run used AMP on CUDA, every other evaluated task's in fp32), then ``extra_keys`` (tasks
    outside that curriculum) in fp32. Returns ``{task_key: {"text", "precision"}}``."""
    from flowcl.data.curriculum import load_curriculum
    from flowcl.train.continual import warm_runner_text_cache

    if policy.text_encoder._cache:
        raise RuntimeError("the text cache is not fresh")
    state = warm_runner_text_cache(policy, load_curriculum(curriculum_name), dataset_dir, device, amp=amp)
    for key in extra_keys:
        if key in state:
            continue
        text = instruction_of(key, dataset_dir)
        with torch.no_grad(), torch.autocast(device_type=torch.device(device).type, enabled=False):
            policy.text_encoder([text])
        state[key] = {"text": text, "precision": "fp32"}
    return state


def expected_precision(curriculum_name: str, keys: list[str], amp: bool, device_type: str) -> dict[str, str]:
    """What :func:`rebuild_cache` must have produced: a stage-0 train task ``amp`` when the run used
    AMP on CUDA, everything else ``fp32``."""
    from flowcl.data.curriculum import load_curriculum

    first = {r.task_key for r in load_curriculum(curriculum_name).stages[0].train_refs}
    return {k: ("amp" if (k in first and amp and device_type == "cuda") else "fp32") for k in keys}


# ---- step: swap rollouts ----------------------------------------------------------------------------------


def swap_pairs(cfg: dict, stage: int) -> list[tuple[str, str]]:
    """``(scene task, requested task)``: stage 0 A <-> S2, stage 1 A <-> S."""
    t = tasks(cfg)
    other = t["S2"] if stage == 0 else t["S"]
    return [(t["A"], other), (other, t["A"])]


def swap_path(cfg: dict, root: Path, stage: int, arm: str) -> Path:
    return out_dir(cfg, root) / "swaps" / f"stage{stage}_{arm}" / "swaps.json"


def model_signatures(task_keys: list[str], spec, image_size: int) -> dict[str, dict]:
    from flowcl.data.tasks import TaskRef
    from flowcl.envs.libero_env import LiberoTaskEnv

    out = {}
    for key in task_keys:
        ref = TaskRef.from_key(key)
        with LiberoTaskEnv(suite=ref.suite, task_idx=ref.task_idx, spec=spec, image_size=image_size) as env:
            out[key] = env.model_signature()
    return out


def run_swaps(cfg: dict, stage: int, arm: str, *, results_root=None, device="cuda", allow_dirty=False,
              dataset_dir=None, swap_fn=None, signature_fn=None) -> dict:
    """Swap rollouts on ``arm``'s ``stage{stage}.pt``; the correct arm is rerun at stage 0 (compared
    with the eval) and reused from the eval at stage 1."""
    from flowcl.analysis.language_check import MIN_SWAP_SUCCESS_DROP, instruction_swap_rollouts
    from flowcl.data.tasks import TaskRef
    from flowcl.envs.evaluation import EvaluationReport, eval_config_from_dict
    from flowcl.experiments.low_update import refuse_dirty
    from flowcl.train.checkpoint import load_checkpoint
    from flowcl.utils.seeding import derive_seed

    if stage not in (0, 1) or arm not in ARMS or (stage == 0 and arm != "seq_ft"):
        raise ValueError(f"swaps are registered for stage 0 seq_ft and stage 1 of each arm, not "
                         f"stage {stage} {arm}")
    if float(cfg["swap_drop_min"]) != MIN_SWAP_SUCCESS_DROP:
        raise ValueError(f"swap_drop_min {cfg['swap_drop_min']} != language_check's {MIN_SWAP_SUCCESS_DROP}")
    sha = refuse_dirty(allow_dirty)
    root = _root(results_root)
    canonical = swap_path(cfg, root, stage, arm).parent
    if (canonical / "swaps.json").is_file():
        print(f"[flowcl] swaps stage {stage} {arm}: published, reused", flush=True)
        return json.loads((canonical / "swaps.json").read_text())
    run = root / run_name(cfg, arm)
    ckpt = run / "checkpoints" / f"stage{stage}.pt"
    ckpt_sha = file_sha256(ckpt)
    conf = _conf(run)
    ns = namespace(cfg)
    if conf.get("seed_namespace_run_id") != ns:
        raise ValueError(f"{run.name} has namespace {conf.get('seed_namespace_run_id')!r}, expected {ns!r}")
    eval_cfg = eval_config_from_dict(conf["eval"])
    report = EvaluationReport.load(run / "eval" / f"stage{stage}.json")
    loaded = load_checkpoint(ckpt, device=device)
    if loaded.stage != stage:
        raise ValueError(f"{ckpt} is stage {loaded.stage}, expected {stage}")
    policy = loaded.policy
    policy.eval()
    amp = bool(conf["train"].get("amp"))
    pairs = swap_pairs(cfg, stage)
    keys = sorted({k for p in pairs for k in p})
    cache = rebuild_cache(policy, cfg["curriculum"], [], dataset_dir, device, amp)
    weights = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    signatures = (signature_fn or model_signatures)(keys, loaded.spec, eval_cfg.image_size)
    same_scene = all(signatures[k] == signatures[keys[0]] for k in keys)
    started = time.perf_counter()
    swap_fn = swap_fn or instruction_swap_rollouts
    results = {}
    for task, requested in pairs:
        ref = TaskRef.from_key(task)
        own_text, requested_text = instruction_of(task, dataset_dir), instruction_of(requested, dataset_dir)
        entry = report.by_task()[task]
        reuse = stage == 1
        swap = swap_fn(policy, ref, requested_text, loaded.spec, loaded.stats, ns, eval_cfg, bootstrap=_bootstrap(),
                       requested_task=requested, correct_successes=list(entry.successes) if reuse else None)
        expected_seeds = [derive_seed(ns, task, ep) for ep in range(eval_cfg.n_episodes)]
        results[f"{task}->{requested}"] = {
            **swap.as_dict(),
            "own_instruction_matches_env": own_text.strip().lower() == ref.language.strip().lower(),
            "eval_seeds_match": list(swap.seeds) == list(entry.seeds) == expected_seeds,
            "determinism": (None if reuse else {
                "equal": [bool(x) for x in swap.correct_successes] == [bool(x) for x in entry.successes],
                "n_different": sum(bool(a) != bool(b) for a, b in zip(swap.correct_successes, entry.successes)),
            }),
        }
    unchanged = all(torch.equal(v, weights[k]) for k, v in policy.state_dict().items())
    record = {"git_sha": sha, "stage": stage, "arm": arm, "run": run.name, "checkpoint": str(ckpt),
              "checkpoint_sha256": ckpt_sha, "namespace": ns, "n_episodes": eval_cfg.n_episodes,
              "amp": amp, "device_type": torch.device(device).type, "text_cache": cache,
              "model_signatures_equal": same_scene, "model_signature": signatures[keys[0]],
              "swaps": results, "weights_unchanged": unchanged, "wall_clock_s": time.perf_counter() - started}
    _attempt_publish(canonical, "swaps.json", record)
    print(f"[flowcl] swaps stage {stage} {arm}: " + "; ".join(
        f"{k.split('/')[1][:24]}: drop {v['drop_estimate']['value']:+.2f}, requested "
        f"{(v['requested_estimate'] or {}).get('value')}" for k, v in results.items()), flush=True)
    return record


# ---- step: probes ----------------------------------------------------------------------------------------


def probe_path(cfg: dict, root: Path, label: str) -> Path:
    return out_dir(cfg, root) / "probes" / label / "probe.json"


def probe_target(cfg: dict, root: Path, label: str, a_only_root: Path | None = None) -> dict:
    """The checkpoint, its run, the curriculum its runner used, and (for the stage-0 bases) the GPM
    memory to measure the instruction signal against, with the SHA-256 its checkpoint names."""
    a_root = a_only_root or root
    if label == "a_only_stage0":
        run, mem_run, curriculum = a_root / cfg["a_only"]["seq_ft_run"], a_root / cfg["a_only"]["gpm_run"], \
            cfg["a_only"]["curriculum"]
        stage = 0
    elif label == "base_stage0":
        run, mem_run, curriculum, stage = root / run_name(cfg, "seq_ft"), root / run_name(cfg, "gpm"), \
            cfg["curriculum"], 0
    elif label in ("seq_ft_stage1", "gpm_stage1"):
        arm = label.split("_stage")[0]
        run, mem_run, curriculum, stage = root / run_name(cfg, arm), None, cfg["curriculum"], 1
    else:
        raise ValueError(f"unknown probe {label!r}; registered: {PROBE_LABELS}")
    memory = None
    if mem_run is not None:
        mem_ckpt = torch.load(mem_run / "checkpoints" / "stage0.pt", map_location="cpu", weights_only=False)
        arts = {a["path"]: a["sha256"] for a in (mem_ckpt.get("extra") or {}).get("method_artifacts") or []}
        memory = {"path": str(mem_run / "method" / "memory_task0.pt"),
                  "expected_sha256": arts.get("method/memory_task0.pt"),
                  "logs": str(mem_run / "method" / "gpm_logs_task0.json")}
    return {"run": run, "checkpoint": run / "checkpoints" / f"stage{stage}.pt", "stage": stage,
            "curriculum": curriculum, "memory": memory}


def load_memory(memory: dict) -> tuple[dict[str, torch.Tensor], dict]:
    """The GPM memory ``{layer: M}`` after verifying the SHA-256 its stage checkpoint names."""
    from flowcl.analysis.subspace import load_bases

    sha = file_sha256(memory["path"])
    if sha != memory["expected_sha256"]:
        raise ValueError(f"{memory['path']}: SHA-256 {sha} != the checkpoint's {memory['expected_sha256']}")
    bases, meta = load_bases(memory["path"])
    return {n: b.vectors.to(torch.float64) for n, b in bases.items()}, {"sha256": sha, "eps": meta.get("eps"),
                                                                          "task_key": meta.get("task_key")}


def memory_ranks(M: dict[str, torch.Tensor], dims: dict[str, tuple[int, int]]) -> dict:
    """Per-layer ``k`` and ``k / d_in``, and the parameter-weighted protected fraction per half."""
    layers = {n: {"k": int(m.shape[1]), "d_in": int(m.shape[0]), "fraction": m.shape[1] / m.shape[0]}
              for n, m in M.items()}
    halves = {}
    for half in ("trunk", "decoder"):
        names = [n for n in M if st._in_scope(n, half)]
        num = sum(dims[n][0] * layers[n]["k"] for n in names)
        den = sum(dims[n][0] * layers[n]["d_in"] for n in names)
        halves[half] = {"parameter_weighted": num / den if den else None,
                        "median_fraction": (sorted(layers[n]["fraction"] for n in names)[len(names) // 2]
                                            if names else None),
                        "total_k": sum(layers[n]["k"] for n in names)}
    return {"layers": layers, "halves": halves}


def instruction_probe(policy, ds_a, ds_b, text_a: str, text_b: str, cfg: dict, sub, device, label: str,
                      n_steps: int, smoke_batches: int | None, M: dict | None) -> dict:
    """similar_cl's instruction probe (identical batches, s, noise and positions for ``label``), plus
    each layer's instruction-signal share outside ``M`` when a memory is given."""
    from flowcl.experiments.similar_cl import chunk_sensitivity
    from flowcl.utils.seeding import derive_seed

    dims, _ = st._dims(policy)
    out = {}
    for direction, ds, ref, swap in (("A_data_B_instruction", ds_a, text_a, text_b),
                                     ("B_data_A_instruction", ds_b, text_b, text_a)):
        tag = f"{label}::{direction}"
        cap, velocity = st.paired_swap(policy, ds, swap, sub, device, derive_seed(cfg["swap"]["seed_tag"], tag, 0),
                                       n_batches=smoke_batches or int(cfg["swap"]["n_batches"]))
        size = {n: (float(torch.trace(cap.D[n])) / float(torch.trace(cap.K[n]))
                    if float(torch.trace(cap.K[n])) > 0 else None) for n in cap.D}
        entry = {"reference": ref, "swap": swap, "velocity": velocity, "size_halves": st.half_summary(size, dims),
                 "chunks": chunk_sensitivity(policy, ds, swap, cfg["chunks"], device, tag, n_steps, smoke_batches)}
        if M is not None:
            share = {n: st.outside_share(M[n], cap.D[n]) for n in cap.D}
            entry["outside_share_halves"] = st.half_summary(share, dims)
            entry["size_layers"], entry["outside_share_layers"] = size, share
        out[direction] = entry
        del cap
    return out


def token_projection_gradient(policy, dataset, cfg: dict, key: str, device) -> dict:
    """``||d loss / d token_projection.weight||`` on one fixed batch (fp32; weights untouched)."""
    from flowcl.data.dataset import collate_chunks
    from flowcl.models.flow_head import draw_with_generator
    from flowcl.train.trainer import move_batch
    from flowcl.utils.seeding import derive_seed

    size = min(int(cfg["gradient"]["batch_size"]), len(dataset))
    gen = torch.Generator(device="cpu").manual_seed(derive_seed(cfg["gradient"]["seed_tag"], key, 0))
    idx = torch.randperm(len(dataset), generator=gen)[:size].tolist()
    batch = move_batch(collate_chunks([dataset[i] for i in idx]), torch.device(device))
    s = policy.s_sampler.sample(size, torch.device(device), generator=gen)
    noise = draw_with_generator(tuple(batch["actions"].shape), device=device, generator=gen, dtype=torch.float32,
                                normal=True)
    weight = policy.text_encoder.token_projection.weight
    policy.eval()
    policy.zero_grad(set_to_none=True)
    with torch.autocast(device_type=torch.device(device).type, enabled=False):
        loss = policy(batch, s=s, noise=noise)["loss"]
    (grad,) = torch.autograd.grad(loss, [weight])
    policy.zero_grad(set_to_none=True)
    return {"loss": float(loss), "grad_norm": float(grad.norm()), "weight_norm": float(weight.norm()),
            "n_samples": size}


def memory_coverage(policy, dataset, M: dict, sub, device, key: str, cfg: dict) -> dict:
    """The share of ``dataset``'s own activation energy (primary view) inside ``M``, per layer."""
    from flowcl.analysis.subspace import captured_energy_fraction
    from flowcl.experiments.gate2 import capture_task_grams
    from flowcl.utils.seeding import derive_seed

    dims, _ = st._dims(policy)
    tag = cfg["coverage"]["seed_tag"]
    cap = capture_task_grams(policy, dataset, sub, device, probe_seed=derive_seed(f"{tag}::probe", key, 0),
                             capture_seed=derive_seed(f"{tag}::capture", key, 0))
    layers = {n: captured_energy_fraction(M[n].to(torch.float64),
                                          cap.accumulators[n].gram[cap.primary_view(n)].to(torch.float64))
              for n in M}
    del cap
    return {"layers": layers, "halves": st.half_summary(layers, dims)}


def run_probe(cfg: dict, label: str, *, results_root=None, a_only_root=None, device="cuda", allow_dirty=False,
              dataset_dir=None, build_datasets=None, smoke_batches: int | None = None) -> dict:
    from flowcl.analysis.language_check import instruction_token_contrast
    from flowcl.analysis.probes import probe_loss
    from flowcl.experiments.low_update import refuse_dirty
    from flowcl.experiments.similar_cl import _probe_cfg
    from flowcl.train.checkpoint import load_checkpoint

    sha = refuse_dirty(allow_dirty)
    root = _root(results_root)
    canonical = probe_path(cfg, root, label).parent
    if (canonical / "probe.json").is_file():
        print(f"[flowcl] probe {label}: published, reused", flush=True)
        return json.loads((canonical / "probe.json").read_text())
    target = probe_target(cfg, root, label, Path(a_only_root) if a_only_root else None)
    ckpt = target["checkpoint"]
    ckpt_sha = file_sha256(ckpt)
    conf = _conf(target["run"])
    loaded = load_checkpoint(ckpt, device=device)
    if loaded.stage != target["stage"]:
        raise ValueError(f"{ckpt} is stage {loaded.stage}, expected {target['stage']}")
    policy = loaded.policy
    policy.eval()
    t = tasks(cfg)
    a, pairs = t["A"], [t["S2"], t["S"]]
    amp = bool(conf["train"].get("amp"))
    n_steps = int(conf["eval"].get("euler_steps", 10))
    smoke = smoke_batches is not None
    sub = st._capture_config({"capture_config": cfg["swap"]["capture_config"]}, smoke)
    cov_sub = st._capture_config({"capture_config": cfg["coverage"]["capture_config"]}, smoke)
    size = max(int(cfg["probe"]["batch_size"]), int(cfg["chunks"]["batch_size"]), int(sub.batch_size),
               int(cfg["gradient"]["batch_size"]))
    probe = _probe_cfg(cfg, smoke_batches)
    cache = rebuild_cache(policy, target["curriculum"], [a, *pairs], dataset_dir, device, amp)
    weights = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    started = time.perf_counter()
    M, memory = None, None
    if target["memory"] is not None:
        M, memory = load_memory(target["memory"])
        memory["ranks"] = memory_ranks(M, st._dims(policy)[0])
    ds_a = st._dataset(a, loaded, dataset_dir, build_datasets, smoke_batches, size)
    text = {k: cache[k]["text"] for k in (a, *pairs)}
    losses = {a: probe_loss(policy, ds_a, probe, device)}
    grads = {a: token_projection_gradient(policy, ds_a, cfg, a, device)}
    coverage = {}
    if label == "base_stage0":
        coverage[a] = memory_coverage(policy, ds_a, M, cov_sub, device, a, cfg)
    instr, contrast = {}, {}
    for b in pairs:   # one B dataset in memory at a time
        ds_b = st._dataset(b, loaded, dataset_dir, build_datasets, smoke_batches, size)
        losses[b] = probe_loss(policy, ds_b, probe, device)
        grads[b] = token_projection_gradient(policy, ds_b, cfg, b, device)
        if label == "base_stage0" and b == t["S2"]:
            coverage[b] = memory_coverage(policy, ds_b, M, cov_sub, device, b, cfg)
        instr[b] = instruction_probe(policy, ds_a, ds_b, text[a], text[b], cfg, sub, device, f"{a}->{b}", n_steps,
                                     smoke_batches, M)
        contrast[b] = instruction_token_contrast(policy, text[a], text[b])
        del ds_b
    unchanged = all(torch.equal(v, weights[k]) for k, v in policy.state_dict().items())
    record = {"git_sha": sha, "smoke": smoke, "label": label, "run": target["run"].name,
              "curriculum": target["curriculum"], "checkpoint": str(ckpt), "checkpoint_sha256": ckpt_sha,
              "device_type": torch.device(device).type, "amp": amp, "text_cache": cache,
              "probe_loss": losses, "instruction": instr, "token_contrast": contrast,
              "token_projection_gradient": grads, "memory": memory, "memory_coverage": coverage or None,
              "weights_unchanged": unchanged, "wall_clock_s": time.perf_counter() - started}
    _attempt_publish(canonical, "probe.json", record)
    print(f"[flowcl] probe {label}: probe loss {losses}; chunk A<->S2 "
          f"{instr[t['S2']]['A_data_B_instruction']['chunks']['relative']}; weights unchanged {unchanged}", flush=True)
    return record


def run_probes(cfg: dict, labels: list[str] | None = None, **kw) -> list[str]:
    """Every registered probe whose checkpoint exists; returns the problems (missing or failed)."""
    root = _root(kw.get("results_root"))
    a_root = kw.get("a_only_root")
    problems = []
    for label in labels or PROBE_LABELS:
        try:
            ckpt = probe_target(cfg, root, label, Path(a_root) if a_root else None)["checkpoint"]
        except (OSError, KeyError) as exc:
            problems.append(f"{label}: {type(exc).__name__}: {exc}")
            print(f"[flowcl] probe {label}: target unavailable ({exc}), skipped", flush=True)
            continue
        if not ckpt.is_file():
            problems.append(f"{ckpt}: missing")
            print(f"[flowcl] probe {label}: {ckpt} missing, skipped", flush=True)
            continue
        try:
            run_probe(cfg, label, **kw)
        except Exception as exc:  # one probe's failure must not block the others
            problems.append(f"{label}: {type(exc).__name__}: {exc}")
            print(f"[flowcl] probe {label} FAILED: {exc}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return problems


# ---- the judgment (pure) --------------------------------------------------------------------------------------


def _mean(x) -> float:
    x = list(x)
    if not x:
        raise ValueError("empty success list")
    return sum(bool(v) for v in x) / len(x)


def _paired(a, b, bootstrap: dict) -> dict:
    from flowcl.experiments.reverse_order import _paired as paired

    return paired([float(bool(v)) for v in a], [float(bool(v)) for v in b], bootstrap)


def judge_l1a(r0: dict[str, list], swaps0: dict[str, dict], cfg: dict) -> dict:
    """``r0``: stage-0 successes of A and S2; ``swaps0``: the two stage-0 swap records (``drop_estimate``)."""
    learned = {k: _mean(v) for k, v in r0.items()}
    out = {"learned": learned, "learned_min": float(cfg["learned_min"]), "swap_drop_min": float(cfg["swap_drop_min"])}
    out["discriminates"] = {k: bool(v["drop_estimate"]["value"] >= float(cfg["swap_drop_min"])
                                    and v["drop_estimate"]["low"] > 0) for k, v in swaps0.items()}
    out["drops"] = {k: v["drop_estimate"] for k, v in swaps0.items()}
    if any(v < float(cfg["learned_min"]) for v in learned.values()):
        return {**out, "verdict": "constituent_not_learned"}
    if len(swaps0) == 2 and all(out["discriminates"].values()):
        return {**out, "verdict": "instruction_dependence_demonstrated"}
    return {**out, "verdict": "instruction_dependence_not_demonstrated"}


def judge_l1b(swaps0: dict[str, dict], cfg: dict, l1a: dict) -> dict:
    rates = {k: v["requested_estimate"] for k, v in swaps0.items()}
    out = {"requested": rates, "selection_min": float(cfg["selection_min"])}
    if l1a["verdict"] != "instruction_dependence_demonstrated":
        return {**out, "verdict": "not_judged", "reason": f"L1a is {l1a['verdict']}"}
    met = len(rates) == 2 and all(r is not None and r["value"] >= float(cfg["selection_min"]) for r in rates.values())
    if met:
        return {**out, "verdict": "task_selection_threshold_met",
                "wording": f"the requested goal was reached in >= {cfg['selection_min']} of swapped rollouts in both "
                           "directions (meeting the threshold, not reliable selection)"}
    return {**out, "verdict": "task_selection_not_demonstrated",
            "wording": "instruction dependence demonstrated; task selection not demonstrated under the chosen criterion"}


def judge_l3(task: str, r0: list, r1: dict[str, list], r0_seq: list, s1: dict[str, list], cfg: dict,
             bootstrap: dict, l1a: dict) -> dict:
    """Retention of ``task`` (A or S2) while S is learned. ``r0``: GPM's stage-0 successes on
    ``task``; ``r0_seq``: seq_ft's; ``r1``: each arm's stage-1 successes on ``task``; ``s1``: each
    arm's stage-1 successes on S."""
    fmax, pmax = float(cfg["forgetting_max"]), float(cfg["plasticity_cost_max"])
    q = {"dR": _paired(r1["gpm"], r1["seq_ft"], bootstrap),
         "forgetting": {"seq_ft": _paired(r0_seq, r1["seq_ft"], bootstrap),
                        "gpm": _paired(r0, r1["gpm"], bootstrap)},
         "dP": _paired(s1["gpm"], s1["seq_ft"], bootstrap),
         "acquired": {arm: _mean(v) for arm, v in s1.items()}}
    out = {"task": task, "quantities": q, "forgetting_max": fmax, "plasticity_cost_max": pmax,
           "acquired_min": float(cfg["acquired_min"]), "interpretable": l1a["verdict"] ==
           "instruction_dependence_demonstrated"}
    if any(v < float(cfg["acquired_min"]) for v in q["acquired"].values()):
        return {**out, "outcome": "inconclusive_acquisition"}
    if q["forgetting"]["seq_ft"]["diff"] <= fmax:
        return {**out, "outcome": "no_interference"}
    if q["dR"]["low"] > 0:
        if q["forgetting"]["gpm"]["diff"] <= fmax:
            return {**out, "outcome": "protects" if q["dP"]["diff"] >= -pmax else "protects_at_plasticity_cost"}
        return {**out, "outcome": "partial"}
    return {**out, "outcome": "no_demonstrated_advantage",
            "wording": "no demonstrated advantage for this GPM configuration, after a language-conditioned base"}


# ---- checks ---------------------------------------------------------------------------------------------------


def _cells(run_dir: Path, keys: list[str], n_episodes: int):
    """``{stage: {task: TaskEvaluation}}`` for stages 0-1, or ``(None, problems)``."""
    from flowcl.envs.evaluation import EvaluationReport

    out, problems = {}, []
    for stage in (0, 1):
        path = run_dir / "eval" / f"stage{stage}.json"
        if not path.is_file():
            problems.append(f"{run_dir.name}: no eval/stage{stage}.json")
            continue
        by = EvaluationReport.load(path).by_task()
        out[stage] = {}
        for k in keys:
            e = by.get(k)
            if e is None or e.n_rollouts != n_episodes or len(e.seeds) != n_episodes:
                problems.append(f"{run_dir.name}: stage {stage} {k} has "
                                f"{None if e is None else e.n_rollouts} rollouts, expected {n_episodes}")
            else:
                out[stage][k] = e
    if not (run_dir / "result.json").is_file():
        problems.append(f"{run_dir.name}: no result.json")
    return (None if problems else out), problems


def production_check(run_dir: Path, cfg: dict, arm: str, smoke: bool) -> dict:
    """Curriculum (with the joint stage as declared), seed, namespace, recipe, episodes, method, clean
    tree; the stage-0 checkpoint's train keys and the steps actually trained."""
    conf = _conf(run_dir)
    name, kwargs, _ = method_spec(cfg, arm)
    want = cfg["smoke"] if smoke else cfg["production"]
    t = tasks(cfg)
    problems = {}
    cur = conf["curriculum"]
    if cur["name"] != cfg["curriculum"]:
        problems["curriculum"] = cur["name"]
    stage0 = cur["tasks"][0] if cur.get("tasks") else {}
    if (stage0.get("task_key"), stage0.get("co_train"), stage0.get("steps_factor")) != \
            (t["A"], [t["S2"]], int(want["steps_factor"])):
        problems["joint_stage"] = stage0
    if [x["task_key"] for x in cur.get("tasks", [])] != [t["A"], t["S"]]:
        problems["stages"] = [x.get("task_key") for x in cur.get("tasks", [])]
    if conf.get("seed") != int(cfg["seed"]):
        problems["seed"] = conf.get("seed")
    if conf.get("seed_namespace_run_id") != namespace(cfg):
        problems["namespace"] = conf.get("seed_namespace_run_id")
    for k, v in want["train"].items():
        if conf["train"].get(k) != v:
            problems[f"train.{k}"] = (conf["train"].get(k), v)
    if conf["eval"].get("n_episodes") != want["n_episodes"]:
        problems["eval.n_episodes"] = (conf["eval"].get("n_episodes"), want["n_episodes"])
    method = {k: v for k, v in conf["method"].items() if k != "display_name"}
    if method != {"name": name, **kwargs}:
        problems["method"] = method
    result = json.loads((run_dir / "result.json").read_text())
    steps = [s.get("steps") for s in result.get("stages", [])]
    want_steps = [int(want["train"]["steps"]) * int(want["steps_factor"]), int(want["train"]["steps"])]
    if steps != want_steps:
        problems["steps_trained"] = (steps, want_steps)
    joint = (result["stages"][0].get("joint") if result.get("stages") else None) or {}
    if joint.get("train_task_keys") != [t["A"], t["S2"]]:
        problems["joint_train_keys"] = joint.get("train_task_keys")
    s0 = torch.load(run_dir / "checkpoints" / "stage0.pt", map_location="cpu", weights_only=False)
    if ((s0.get("extra") or {}).get("joint") or {}).get("train_task_keys") != [t["A"], t["S2"]]:
        problems["stage0_checkpoint_joint"] = (s0.get("extra") or {}).get("joint")
    sha = (run_dir / "git_sha").read_text().strip() if (run_dir / "git_sha").is_file() else None
    clean = sha is not None and not sha.endswith("-dirty")
    if not clean and not smoke:
        problems["git_sha"] = sha
    return {"passed": not problems, "problems": problems, "git_sha": sha,
            "dirty_excused": bool(smoke and not clean), "smoke": smoke,
            "exposure": joint.get("task_samples")}


def gpm_checks(run_dir: Path) -> dict:
    """Identity at stage 0, T1 pairing, memory artifacts at both stages matching their SHA-256, the
    worst residual over bound <= 1 at stage 1."""
    identity = run_dir / "identity_stage0.json"
    ident = json.loads(identity.read_text()).get("passed") if identity.is_file() else None
    result = json.loads((run_dir / "result.json").read_text())
    t1 = bool((result.get("t1_pairing") or {}).get("passed"))
    mismatched, named = [], []
    for stage in (0, 1):
        ck = torch.load(run_dir / "checkpoints" / f"stage{stage}.pt", map_location="cpu", weights_only=False)
        for art in (ck.get("extra") or {}).get("method_artifacts") or []:
            named.append(art["path"])
            p = run_dir / art["path"]
            if not p.is_file() or file_sha256(p) != art["sha256"]:
                mismatched.append(art["path"])
    logs = run_dir / "method" / "gpm_logs_task1.json"
    worst = None
    if logs.is_file():
        worst = max([r["max_residual_over_bound"]
                     for r in json.loads(logs.read_text()).get("residuals", {}).values()] or [0.0])
    has = {f"method/memory_task{k}.pt" in named for k in (0, 1)}
    passed = bool(ident is True and t1 and has == {True} and not mismatched and worst is not None and worst <= 1.0)
    return {"passed": passed, "identity_stage0": ident, "t1_pairing": t1, "mismatched": mismatched,
            "memory_named": sorted(n for n in named if "memory" in n), "worst_residual_over_bound": worst}


def swap_artifact(path: Path, ckpt: Path, cfg: dict, stage: int, n_episodes: int, smoke: bool) -> dict:
    if not path.is_file():
        return {"status": INCOMPLETE, "problems": [f"{path}: missing"]}
    rec = json.loads(path.read_text())
    problems = []
    if not ckpt.is_file() or rec.get("checkpoint_sha256") != file_sha256(ckpt):
        problems.append("checkpoint_sha256")
    if rec.get("weights_unchanged") is not True:
        problems.append("weights_unchanged")
    if rec.get("model_signatures_equal") is not True:
        problems.append("model_signatures_equal")
    pairs = swap_pairs(cfg, stage)
    keys = sorted({k for p in pairs for k in p})
    want = expected_precision(cfg["curriculum"], keys, bool(rec.get("amp")), str(rec.get("device_type")))
    got = {k: (rec.get("text_cache") or {}).get(k, {}).get("precision") for k in keys}
    if got != want:
        problems.append(f"text_cache precision {got} != {want}")
    if not smoke and rec.get("device_type") != "cuda":
        problems.append(f"device {rec.get('device_type')}")
    for task, requested in pairs:
        s = (rec.get("swaps") or {}).get(f"{task}->{requested}")
        if s is None:
            problems.append(f"{task}->{requested}: missing")
            continue
        if len(s.get("swapped_successes", [])) != n_episodes or len(s.get("requested_reached", [])) != n_episodes:
            problems.append(f"{task}->{requested}: not {n_episodes} scored rollouts")
        if s.get("eval_seeds_match") is not True:
            problems.append(f"{task}->{requested}: seeds")
        if s.get("own_instruction_matches_env") is not True:
            problems.append(f"{task}->{requested}: own instruction")
    return {"status": INVALID if problems else VALID, "problems": problems}


def probe_artifact(path: Path, ckpt: Path, cfg: dict, curriculum: str, smoke: bool) -> dict:
    if not path.is_file():
        return {"status": INCOMPLETE, "problems": [f"{path}: missing"]}
    rec = json.loads(path.read_text())
    problems = []
    if not ckpt.is_file() or rec.get("checkpoint_sha256") != file_sha256(ckpt):
        problems.append("checkpoint_sha256")
    if rec.get("weights_unchanged") is not True:
        problems.append("weights_unchanged")
    keys = list(tasks(cfg).values())
    want = expected_precision(curriculum, keys, bool(rec.get("amp")), str(rec.get("device_type")))
    got = {k: (rec.get("text_cache") or {}).get(k, {}).get("precision") for k in keys}
    if got != want:
        problems.append(f"text_cache precision {got} != {want}")
    if not smoke and rec.get("device_type") != "cuda":
        problems.append(f"device {rec.get('device_type')}")
    return {"status": INVALID if problems else VALID, "problems": problems}


def _rel(a, b) -> float | None:
    if a is None or b is None:
        return None
    scale = max(abs(a), abs(b))
    return 0.0 if scale == 0 else abs(a - b) / scale


def reproduction_check(rec: dict, reference: dict, cfg: dict) -> dict:
    """Probe (a) against similar_cl's stage-0 file: probe loss on A, S, S2 and both instruction
    statistics for S and S2, each within ``reproduction_rtol``."""
    t = tasks(cfg)
    diffs = {}
    for k in (t["A"], t["S2"], t["S"]):
        diffs[f"probe_loss {k}"] = _rel(rec["probe_loss"].get(k), reference["probe_loss"].get(k))
    for b in (t["S2"], t["S"]):
        for d in ("A_data_B_instruction", "B_data_A_instruction"):
            mine, ref = rec["instruction"][b][d], reference["instruction"][b][d]
            diffs[f"velocity {b} {d}"] = _rel(mine["velocity"]["sensitivity"], ref["velocity"]["sensitivity"])
            diffs[f"chunks {b} {d}"] = _rel(mine["chunks"]["relative"], ref["chunks"]["relative"])
    tol = float(cfg["reproduction_rtol"])
    worst = max((v for v in diffs.values() if v is not None), default=None)
    return {"passed": worst is not None and worst <= tol and None not in diffs.values(), "worst_rel": worst,
            "rtol": tol, "diffs": diffs}


# ---- the summary --------------------------------------------------------------------------------------------


def _swap_view(rec: dict, pairs: list[tuple[str, str]]) -> dict:
    return {f"{a}->{b}": {k: rec["swaps"][f"{a}->{b}"][k] for k in
                         ("correct_estimate", "swapped_estimate", "drop_estimate", "requested_estimate",
                          "requested_at_start", "determinism", "correct_reused")}
            for a, b in pairs}


def _mechanism(rec: dict) -> dict:
    out = {"probe_loss": rec["probe_loss"], "token_contrast": rec["token_contrast"],
           "token_projection_gradient": rec["token_projection_gradient"],
           "memory_ranks": (rec.get("memory") or {}).get("ranks", {}).get("halves"),
           "memory_coverage": ({k: v["halves"] for k, v in rec["memory_coverage"].items()}
                               if rec.get("memory_coverage") else None),
           "instruction": {}}
    for b, by in rec["instruction"].items():
        out["instruction"][b] = {d: {"velocity_sensitivity": v["velocity"]["sensitivity"],
                                     "chunk_relative": v["chunks"]["relative"],
                                     "trunk_size_median": (v["size_halves"].get("trunk") or {}).get("median"),
                                     "trunk_outside_share_median":
                                         ((v.get("outside_share_halves") or {}).get("trunk") or {}).get("median")}
                                 for d, v in by.items()}
    return out


def run_summary(cfg: dict, *, results_root=None, a_only_root=None, out: Path | None = None, smoke: bool = False,
                rule_sha256: str | None = None) -> dict:
    """Judge the study and write ``summary.json`` atomically (see the module docstring)."""
    from flowcl.experiments.reverse_order import seed_pairing, stats_check
    from flowcl.experiments.sequence_report import load_run

    root = _root(results_root)
    a_root = Path(a_only_root) if a_only_root else root
    out = Path(out) if out else out_dir(cfg, root)
    bootstrap = _bootstrap()
    n_episodes = int((cfg["smoke"] if smoke else cfg["production"])["n_episodes"])
    t = tasks(cfg)
    keys = [t["A"], t["S2"], t["S"]]
    dirs = {arm: root / run_name(cfg, arm) for arm in ARMS}
    cells, problems = {}, []
    for arm in ARMS:
        cells[arm], p = _cells(dirs[arm], keys, n_episodes)
        problems += p

    checks = {}
    status = VALID
    if problems:
        status = INCOMPLETE
    else:
        try:
            for arm in ARMS:
                checks[f"production_{arm}"] = production_check(dirs[arm], cfg, arm, smoke)
            checks["statistics"] = stats_check(dirs, t["A"])
            checks["episode_pairing"] = seed_pairing(load_run(dirs["seq_ft"]), load_run(dirs["gpm"]))
            checks["gpm"] = gpm_checks(dirs["gpm"])
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            problems.append(f"malformed input: {type(exc).__name__}: {exc}")
            status = INCOMPLETE
        if status == VALID and not all(c["passed"] for c in checks.values()):
            status = INVALID

    # artifacts
    swaps = {}
    for stage, arm in ((0, "seq_ft"), (1, "seq_ft"), (1, "gpm")):
        swaps[f"stage{stage}_{arm}"] = swap_artifact(swap_path(cfg, root, stage, arm),
                                                     dirs[arm] / "checkpoints" / f"stage{stage}.pt", cfg, stage,
                                                     n_episodes, smoke)
    probes = {}
    for label in PROBE_LABELS:
        try:
            target = probe_target(cfg, root, label, a_root)
            probes[label] = probe_artifact(probe_path(cfg, root, label), target["checkpoint"], cfg,
                                           target["curriculum"], smoke)
        except (OSError, KeyError) as exc:
            probes[label] = {"status": INCOMPLETE, "problems": [f"{type(exc).__name__}: {exc}"]}
    reproduction = None
    if probes["a_only_stage0"]["status"] == VALID:
        ref_path = a_root / cfg["a_only"]["stage0_probe"]
        if smoke:
            reproduction = {"passed": None, "excused": "smoke (3 probe batches)"}
        elif not ref_path.is_file():
            reproduction = {"passed": False, "missing": str(ref_path)}
        else:
            reproduction = reproduction_check(json.loads(probe_path(cfg, root, "a_only_stage0").read_text()),
                                              json.loads(ref_path.read_text()), cfg)
        if reproduction["passed"] is False:
            probes["a_only_stage0"] = {"status": INVALID, "problems": ["reproduction of similar_cl's stage 0"]}
    artifacts = [*swaps.values(), *probes.values()]
    if status != INCOMPLETE and any(a["status"] == INVALID for a in artifacts):
        status = INVALID
    elif status == VALID and any(a["status"] == INCOMPLETE for a in artifacts):
        status = INCOMPLETE

    summary = {"git_sha": git_sha(), "rule_sha256": rule_sha256, "smoke": smoke, "namespace": namespace(cfg),
               "n_episodes": n_episodes, "runs": {arm: str(d) for arm, d in dirs.items()}, "checks": checks,
               "problems": problems, "swap_artifacts": swaps, "probe_artifacts": probes,
               "reproduction": reproduction, "config": cfg}
    if cells["seq_ft"] is not None and cells["gpm"] is not None:
        c = cells
        summary["cells"] = {arm: {f"R{stage}": {k.split("/")[1]: {"value": e.estimate.value, "low": e.estimate.low,
                                                                   "high": e.estimate.high}
                                                for k, e in c[arm][stage].items()} for stage in (0, 1)}
                            for arm in ARMS}
        swaps0 = None
        if swaps["stage0_seq_ft"]["status"] == VALID:
            rec0 = json.loads(swap_path(cfg, root, 0, "seq_ft").read_text())
            swaps0 = {k: rec0["swaps"][k] for k in (f"{a}->{b}" for a, b in swap_pairs(cfg, 0))}
            summary["swaps_stage0"] = _swap_view(rec0, swap_pairs(cfg, 0))
        r0 = {t["A"]: c["seq_ft"][0][t["A"]].successes, t["S2"]: c["seq_ft"][0][t["S2"]].successes}
        l1a = (judge_l1a(r0, swaps0, cfg) if swaps0 is not None
               else {"verdict": "not_judged", "reason": "stage-0 swaps missing or invalid"})
        l1b = judge_l1b(swaps0, cfg, l1a) if swaps0 is not None else {"verdict": "not_judged"}
        s1 = {arm: c[arm][1][t["S"]].successes for arm in ARMS}
        l3 = {name: judge_l3(t[name], c["gpm"][0][t[name]].successes,
                             {arm: c[arm][1][t[name]].successes for arm in ARMS},
                             c["seq_ft"][0][t[name]].successes, s1, cfg, bootstrap,
                             l1a if "verdict" in l1a else {"verdict": "not_judged"})
              for name in ("A", "S2")}
        stage1_swaps = {}
        for arm in ARMS:
            if swaps[f"stage1_{arm}"]["status"] == VALID:
                stage1_swaps[arm] = _swap_view(json.loads(swap_path(cfg, root, 1, arm).read_text()),
                                               swap_pairs(cfg, 1))
        summary["rules"] = {
            "L1a": l1a, "L1b": l1b, "L3": l3,
            "L4": {"S_zero_shot": summary["cells"]["seq_ft"]["R0"][t["S"].split("/")[1]],
                   "seq_ft_forgetting": {n: l3[n]["quantities"]["forgetting"]["seq_ft"] for n in ("A", "S2")},
                   "stage1_swaps": stage1_swaps,
                   "exposure": (checks.get("production_seq_ft") or {}).get("exposure"),
                   "a_only_lineage_gpm_A_after_S": float(cfg["a_only"]["gpm_a_after_s"])},
        }
    mechanism = {}
    for label in PROBE_LABELS:
        if probes[label]["status"] == VALID:
            mechanism[label] = _mechanism(json.loads(probe_path(cfg, root, label).read_text()))
    summary["mechanism"] = mechanism
    summary["status"] = status
    summary["exit_code"] = EXIT_CODES[status]
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out / "summary.json", json.dumps(summary, indent=2, default=str) + "\n")
    rules = summary.get("rules", {})
    print(f"[flowcl] language_cl: {status}; L1a {rules.get('L1a', {}).get('verdict')}; L1b "
          f"{rules.get('L1b', {}).get('verdict')}; L3 A {rules.get('L3', {}).get('A', {}).get('outcome')}, S2 "
          f"{rules.get('L3', {}).get('S2', {}).get('outcome')}", flush=True)
    return summary
