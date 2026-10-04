"""Similar-vs-diverse two-task continual study (``configs/analysis/similar_cl.yaml``).

Three curricula A -> B (B similar, diverse, similar) under three arms (seq_ft, allowlist_ft, plain
GPM) share one A state: S's runs train stage 0 (allowlist_ft and gpm bit-identical to seq_ft), D's
and S2's resume at stage 1 from S's stage0.pt of the same arm. This module adds what the runner does
not record, and judges the rule:

1. :func:`run_stage0` (on S seq_ft's stage0.pt): B's zero-shot cells (S's namespace and eval config),
   the fixed probe loss on A and every B, the instruction probes A <-> each B;
2. :func:`run_probe` (a run's stage1.pt): the probe loss on A and B and the instruction probe A <-> B;
3. :func:`run_summary`: the validity checks, the cells and paired differences, P1-P5, and
   ``summary.json`` (written atomically in every case).

The instruction probe has two statistics, both from identical inputs at every checkpoint: the
diagnostic's velocity statistic (:func:`flowcl.experiments.similar_tasks.paired_swap`) and the
sampled-chunk sensitivity (:func:`chunk_sensitivity`, built on
:func:`flowcl.analysis.language_check.instruction_sensitivity`). Before any probe the text cache is
rebuilt as in the runner (:func:`warm_cache`) and its precision recorded.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from flowcl.experiments import similar_tasks as st
from flowcl.utils.libero_paths import repo_root
from flowcl.utils.run import atomic_write_text, file_sha256, git_sha

VALID, INVALID, INCOMPLETE = "valid", "invalid", "incomplete"
EXIT_CODES = {VALID: 0, INVALID: 3, INCOMPLETE: 4}
ARMS = ("seq_ft", "allowlist_ft", "gpm")
FROZEN_ARMS = ("allowlist_ft", "gpm")


def config_path() -> Path:
    return repo_root() / "configs" / "analysis" / "similar_cl.yaml"


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


def run_name(cfg: dict, curriculum: str, arm: str) -> str:
    from flowcl.train.continual import continual_run_id

    return continual_run_id(method_spec(cfg, arm)[2], curriculum, int(cfg["seed"]))


def namespace(cfg: dict) -> str:
    from flowcl.train.continual import seed_namespace_run_id

    return seed_namespace_run_id(cfg["source_curriculum"], int(cfg["seed"]))


def b_key(curriculum: str) -> str:
    conf = OmegaConf.load(repo_root() / "configs" / "curriculum" / f"{curriculum}.yaml")
    return str(conf.tasks[1].task_key)


def _conf(run_dir: Path) -> dict:
    return OmegaConf.to_container(OmegaConf.load(run_dir / "config.yaml"), resolve=True)


def _bootstrap() -> dict:
    return OmegaConf.to_container(
        OmegaConf.load(repo_root() / "configs" / "eval" / "libero_eval.yaml"), resolve=True)["bootstrap"]


# ---- probes ---------------------------------------------------------------------------------------------


def warm_cache(policy, texts: list[str], device, amp: bool) -> list[str]:
    """Encode ``texts`` into a fresh text cache as the runner does: ``texts[0]`` (task A) under AMP
    when the run used AMP on CUDA, every other one in fp32. Returns each text's precision."""
    device_type = torch.device(device).type
    cached = [t for t in texts if t in policy.text_encoder._cache]
    if cached:
        raise RuntimeError(f"the text cache is not fresh: {cached}")
    out = []
    for i, text in enumerate(texts):
        task_amp = bool(amp and i == 0 and device_type == "cuda")
        with torch.no_grad(), torch.autocast(device_type=device_type, enabled=task_amp):
            policy.text_encoder([text])
        out.append("amp" if task_amp else "fp32")
    return out


def expected_precision(n_texts: int, amp: bool, device_type: str) -> list[str]:
    return ["amp" if (i == 0 and amp and device_type == "cuda") else "fp32" for i in range(n_texts)]


def chunk_sensitivity(policy, dataset, swap_text: str, chunks: dict, device, label: str, n_steps: int,
                      n_batches: int | None = None) -> dict:
    """Sampled chunks under the native and ``swap_text`` instruction from identical observations and
    initial noise, over the first fixed batches; pooled ``sum |diff| / sum |chunk|``."""
    from flowcl.analysis.language_check import instruction_sensitivity
    from flowcl.train.trainer import build_dataloader, move_batch
    from flowcl.utils.seeding import derive_seed

    device = torch.device(device)
    loader = build_dataloader(dataset, batch_size=int(chunks["batch_size"]), num_workers=0, shuffle=False)
    limit = int(n_batches if n_batches is not None else chunks["n_batches"])
    diff = scale = 0.0
    n, native = 0, None
    with torch.autocast(device_type=device.type, enabled=False):
        for i, batch in enumerate(loader):
            if i >= limit:
                break
            batch = move_batch(batch, device)
            native = batch["language"][0]
            r = instruction_sensitivity(policy, batch, native, swap_text,
                                        seed=derive_seed(chunks["seed_tag"], label, i), n_steps=n_steps)
            diff += r.absolute_divergence * r.n_samples
            scale += r.chunk_scale * r.n_samples
            n += r.n_samples
    if n == 0:
        raise ValueError("chunk sensitivity saw no samples")
    return {"reference": native, "swap": swap_text, "absolute": diff / n, "scale": scale / n,
            "relative": diff / scale if scale > 0 else None, "n_samples": n}


def instruction_probe(policy, ds_a, ds_b, text_a: str, text_b: str, cfg: dict, sub, device, label: str,
                      n_steps: int, smoke_batches: int | None) -> dict:
    """Both directions of the A <-> B instruction swap, both statistics. ``label`` names the task pair
    only, so every checkpoint sees identical batches, s, noise and positions."""
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
        out[direction] = {
            "reference": ref, "swap": swap,
            "velocity": velocity,
            "size_halves": st.half_summary(size, dims),
            "chunks": chunk_sensitivity(policy, ds, swap, cfg["chunks"], device, tag, n_steps, smoke_batches),
        }
        del cap
    return out


def _probe_cfg(cfg: dict, smoke_batches: int | None) -> dict:
    probe = dict(cfg["probe"])
    if smoke_batches:
        probe["n_batches"] = int(smoke_batches)
    return probe


def _measure(cfg: dict, loaded, keys: list[str], pairs: list[str], conf: dict, *, device, dataset_dir,
             build_datasets, smoke_batches) -> dict:
    """The probe loss on every task in ``keys`` (``keys[0]`` is A) and the instruction probe A <-> each
    of ``pairs``, after rebuilding the text cache; weights verified unchanged."""
    from flowcl.analysis.probes import probe_loss

    policy = loaded.policy
    sub = st._capture_config({"capture_config": cfg["swap"]["capture_config"]}, smoke_batches is not None)
    size = max(int(cfg["probe"]["batch_size"]), int(cfg["chunks"]["batch_size"]), int(sub.batch_size))
    amp = bool(conf["train"].get("amp"))
    n_steps = int(conf["eval"].get("euler_steps", 10))
    probe = _probe_cfg(cfg, smoke_batches)
    a = keys[0]
    if policy.text_encoder._cache:
        raise RuntimeError("the text cache is not fresh")
    weights = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    ds_a = st._dataset(a, loaded, dataset_dir, build_datasets, smoke_batches, size)
    cache = {a: {"text": st.instruction(ds_a)}}
    cache[a]["precision"] = warm_cache(policy, [cache[a]["text"]], device, amp)[0]   # A first, as the runner
    probes = {a: probe_loss(policy, ds_a, probe, device)}
    instr = {}
    for b in keys[1:]:   # one B dataset in memory at a time; each B's instruction is first encoded in fp32
        ds_b = st._dataset(b, loaded, dataset_dir, build_datasets, smoke_batches, size)
        cache[b] = {"text": st.instruction(ds_b)}
        cache[b]["precision"] = warm_cache(policy, [cache[b]["text"]], device, amp=False)[0]
        probes[b] = probe_loss(policy, ds_b, probe, device)
        if b in pairs:
            instr[b] = instruction_probe(policy, ds_a, ds_b, cache[a]["text"], cache[b]["text"], cfg, sub, device,
                                         f"{a}->{b}", n_steps, smoke_batches)
        del ds_b
    unchanged = all(torch.equal(v, weights[k]) for k, v in policy.state_dict().items())
    return {"device_type": torch.device(device).type, "amp": amp, "text_cache": cache,
            "probe_loss": probes, "instruction": instr, "weights_unchanged": unchanged}


# ---- step: stage 0 -----------------------------------------------------------------------------------------


def stage0_path(cfg: dict, root: Path) -> Path:
    return out_dir(cfg, root) / "stage0" / "stage0.json"


def run_stage0(cfg: dict, *, results_root=None, device="cuda", allow_dirty=False, dataset_dir=None,
               build_datasets=None, smoke_batches: int | None = None, evaluator=None) -> dict:
    """B's zero-shot cells, probe losses and instruction probes at the shared stage 0."""
    from flowcl.data.tasks import TaskRef
    from flowcl.envs.evaluation import eval_config_from_dict, evaluate_tasks
    from flowcl.experiments.low_update import refuse_dirty
    from flowcl.train.checkpoint import load_checkpoint

    sha = refuse_dirty(allow_dirty)
    root = _root(results_root)
    canonical = stage0_path(cfg, root).parent
    if (canonical / "stage0.json").is_file():
        print("[flowcl] stage 0: published, reused", flush=True)
        return json.loads((canonical / "stage0.json").read_text())
    src = root / run_name(cfg, cfg["source_curriculum"], "seq_ft")
    ckpt = src / "checkpoints" / "stage0.pt"
    ckpt_sha = file_sha256(ckpt)
    conf = _conf(src)
    loaded = load_checkpoint(ckpt, device=device)
    if loaded.task_key != cfg["task_a"] or loaded.stage != 0:
        raise ValueError(f"{ckpt} is {loaded.task_key!r} stage {loaded.stage}, expected {cfg['task_a']!r} stage 0")
    started = time.perf_counter()
    bs = [b_key(c["name"]) for c in cfg["curricula"]]
    keys = [cfg["task_a"], *bs]
    measured = _measure(cfg, loaded, keys, bs, conf, device=device, dataset_dir=dataset_dir,
                        build_datasets=build_datasets, smoke_batches=smoke_batches)
    zero_keys = [b for b in bs if b != b_key(cfg["source_curriculum"])]   # S's B is in S's own stage-0 eval
    policy = loaded.policy
    weights = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    policy.eval()
    report = (evaluator or evaluate_tasks)(policy, [TaskRef.from_key(k) for k in zero_keys], loaded.spec,
                                           loaded.stats, namespace(cfg), eval_config_from_dict(conf["eval"]),
                                           _bootstrap(), stage=0)
    report.method_run_id = src.name
    report.seed_namespace_run_id = namespace(cfg)
    measured["weights_unchanged"] = bool(measured["weights_unchanged"] and all(
        torch.equal(v, weights[k]) for k, v in policy.state_dict().items()))
    attempt = st._attempt(canonical.parent, "stage0")
    report.save(attempt / "zero_shot.json")
    record = {"git_sha": sha, "smoke": smoke_batches is not None, "source_run": src.name,
              "checkpoint": str(ckpt), "checkpoint_sha256": ckpt_sha, "namespace": namespace(cfg),
              "zero_shot": {"path": "zero_shot.json", "sha256": file_sha256(attempt / "zero_shot.json"),
                            "tasks": zero_keys},
              **measured, "wall_clock_s": time.perf_counter() - started}
    atomic_write_text(attempt / "stage0.json", json.dumps(record, indent=2, default=str) + "\n")
    st._publish(attempt, canonical)
    print(f"[flowcl] stage 0: zero-shot {[(t.task_key.split('/')[1], t.estimate.value) for t in report.tasks]}; "
          f"weights unchanged {record['weights_unchanged']}", flush=True)
    return record


# ---- step: stage-1 probes ------------------------------------------------------------------------------------


def probe_path(cfg: dict, root: Path, curriculum: str, arm: str) -> Path:
    return out_dir(cfg, root) / "probes" / run_name(cfg, curriculum, arm) / "probe.json"


def run_probe(cfg: dict, curriculum: str, arm: str, *, results_root=None, device="cuda", allow_dirty=False,
              dataset_dir=None, build_datasets=None, smoke_batches: int | None = None) -> dict:
    from flowcl.experiments.low_update import refuse_dirty
    from flowcl.train.checkpoint import load_checkpoint

    sha = refuse_dirty(allow_dirty)
    root = _root(results_root)
    canonical = probe_path(cfg, root, curriculum, arm).parent
    if (canonical / "probe.json").is_file():
        print(f"[flowcl] probe {canonical.name}: published, reused", flush=True)
        return json.loads((canonical / "probe.json").read_text())
    run = root / run_name(cfg, curriculum, arm)
    ckpt = run / "checkpoints" / "stage1.pt"
    ckpt_sha = file_sha256(ckpt)
    b = b_key(curriculum)
    loaded = load_checkpoint(ckpt, device=device)
    if loaded.task_key != b or loaded.stage != 1:
        raise ValueError(f"{ckpt} is {loaded.task_key!r} stage {loaded.stage}, expected {b!r} stage 1")
    started = time.perf_counter()
    measured = _measure(cfg, loaded, [cfg["task_a"], b], [b], _conf(run), device=device, dataset_dir=dataset_dir,
                        build_datasets=build_datasets, smoke_batches=smoke_batches)
    record = {"git_sha": sha, "smoke": smoke_batches is not None, "run": run.name, "curriculum": curriculum,
              "arm": arm, "checkpoint": str(ckpt), "checkpoint_sha256": ckpt_sha, **measured,
              "wall_clock_s": time.perf_counter() - started}
    attempt = st._attempt(canonical.parent, canonical.name)
    atomic_write_text(attempt / "probe.json", json.dumps(record, indent=2, default=str) + "\n")
    st._publish(attempt, canonical)
    print(f"[flowcl] probe {run.name}: probe loss {record['probe_loss']}; weights unchanged "
          f"{record['weights_unchanged']}", flush=True)
    return record


def run_probes(cfg: dict, curricula: list[str], **kw) -> list[str]:
    """Probe every arm of ``curricula`` whose stage1.pt exists; returns the problems (missing or failed)."""
    root = _root(kw.get("results_root"))
    problems = []
    for curriculum in curricula:
        for arm in ARMS:
            ckpt = root / run_name(cfg, curriculum, arm) / "checkpoints" / "stage1.pt"
            if not ckpt.is_file():
                problems.append(f"{ckpt}: missing")
                print(f"[flowcl] probe: {ckpt} missing, skipped", flush=True)
                continue
            try:
                run_probe(cfg, curriculum, arm, **kw)
            except Exception as exc:  # one run's failure must not block the others' probes
                problems.append(f"{curriculum} {arm}: {type(exc).__name__}: {exc}")
                print(f"[flowcl] probe {curriculum} {arm} FAILED: {exc}", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return problems


# ---- checks ---------------------------------------------------------------------------------------------------


def _load_view(run_dir: Path, n_episodes: int):
    """The run's view, or ``(None, problems)`` if a cell up to the diagonal is missing or short."""
    from flowcl.experiments.sequence_report import load_run

    if not (run_dir / "result.json").is_file():
        return None, [f"{run_dir.name}: no result.json"]
    try:
        view = load_run(run_dir)
    except (OSError, ValueError, KeyError) as exc:
        return None, [f"{run_dir.name}: {exc}"]
    problems = []
    if view.n_tasks != 2:
        problems.append(f"{run_dir.name}: {view.n_tasks} tasks")
    else:
        for i, j in ((0, 0), (1, 0), (1, 1)):
            entry = view.evals[i].by_task().get(view.task_keys[j])
            if entry is None or entry.n_rollouts != n_episodes or len(entry.seeds) != n_episodes:
                problems.append(f"{run_dir.name}: R[{i}][{j}] has {None if entry is None else entry.n_rollouts} "
                                f"rollouts, expected {n_episodes}")
    return (None if problems else view), problems


def production_check(run_dir: Path, cfg: dict, curriculum: str, arm: str, smoke: bool) -> dict:
    """Curriculum, seed, S's namespace, recipe, evaluation episodes, method config, clean tree."""
    conf = _conf(run_dir)
    name, kwargs, _ = method_spec(cfg, arm)
    want = cfg["smoke"] if smoke else cfg["production"]
    problems = {}
    if conf["curriculum"]["name"] != curriculum:
        problems["curriculum"] = conf["curriculum"]["name"]
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
    sha = (run_dir / "git_sha").read_text().strip() if (run_dir / "git_sha").is_file() else None
    clean = sha is not None and not sha.endswith("-dirty")
    if not clean and not smoke:
        problems["git_sha"] = sha
    return {"passed": not problems, "problems": problems, "git_sha": sha,
            "dirty_excused": bool(smoke and not clean), "smoke": smoke}


def identity_check(run_dir: Path) -> dict:
    path = run_dir / "identity_stage0.json"
    passed = json.loads(path.read_text()).get("passed") if path.is_file() else None
    return {"passed": passed is True, "stage0": passed}


def resume_check(run_dir: Path, source_dir: Path) -> dict:
    """``resume.json`` names the source's stage0.pt with its current SHA-256, and stage 0 is imported
    only: no own stage0.pt, and ``eval/stage0.json`` identical to the source's."""
    path = run_dir / "resume.json"
    if not path.is_file():
        return {"passed": False, "problems": {"resume.json": "missing"}}
    rec = json.loads(path.read_text())
    src_ckpt = source_dir / "checkpoints" / "stage0.pt"
    sha = file_sha256(src_ckpt) if src_ckpt.is_file() else None
    problems = {}
    if Path(str(rec.get("run"))).name != source_dir.name:
        problems["run"] = rec.get("run")
    if rec.get("start_stage") != 1:
        problems["start_stage"] = rec.get("start_stage")
    if sha is None or rec.get("sha256") != sha:
        problems["sha256"] = (rec.get("sha256"), sha)
    if (run_dir / "checkpoints" / "stage0.pt").exists():
        problems["own_stage0_checkpoint"] = True
    mine, theirs = run_dir / "eval" / "stage0.json", source_dir / "eval" / "stage0.json"
    if not (mine.is_file() and theirs.is_file() and file_sha256(mine) == file_sha256(theirs)):
        problems["stage0_eval"] = "not the source's"
    return {"passed": not problems, "problems": problems, "source_sha256": sha}


def frozen_check(run_dir: Path, source_dir: Path, allowed: set[str], arm: str) -> dict:
    """Every state-dict tensor outside the allowlist equal between the arm's source stage0.pt and this
    run's stage1.pt; allowlist_ft's recorded trainable set equal to the allowlist. Allowlisted tensors
    that did not move are reported, not failed (GPM may immobilize a layer its memory spans)."""
    s0 = torch.load(source_dir / "checkpoints" / "stage0.pt", map_location="cpu", weights_only=False)
    s1 = torch.load(run_dir / "checkpoints" / "stage1.pt", map_location="cpu", weights_only=False)
    st0, st1 = s0["state_dict"], s1["state_dict"]
    outside = sorted(set(st0) - allowed)
    changed = [n for n in outside if n not in st1 or not torch.equal(st0[n], st1[n])]
    unmoved = sorted(n for n in allowed if n in st0 and n in st1 and torch.equal(st0[n], st1[n]))
    if arm == "allowlist_ft":
        report = ((s1.get("extra") or {}).get("method_state") or {}).get("freeze_reports", {}).get("1")
        trainable = {"passed": bool(report) and report.get("trainable_tensors") == len(allowed), "report": report}
    else:
        trainable = {"passed": True, "enforced_by": "GPM.on_task_start (freeze_to_allowlist; the run stops otherwise)"}
    return {"passed": not changed and trainable["passed"], "changed": changed, "trainable_set": trainable,
            "n_outside_allowlist": len(outside), "n_allowlisted": len(allowed),
            "n_unmoved_allowlisted": len(unmoved), "unmoved_allowlisted": unmoved}


def memory_check(run_dir: Path) -> dict:
    """GPM's memory artifacts match the SHA-256 the stage-1 checkpoint names; worst residual <= bound."""
    s1 = torch.load(run_dir / "checkpoints" / "stage1.pt", map_location="cpu", weights_only=False)
    arts = (s1.get("extra") or {}).get("method_artifacts") or []
    mismatched = [a["path"] for a in arts
                  if not (run_dir / a["path"]).is_file() or file_sha256(run_dir / a["path"]) != a["sha256"]]
    has_memory = any(a["path"].endswith("memory_task1.pt") for a in arts)
    logs = run_dir / "method" / "gpm_logs_task1.json"
    worst = None
    if logs.is_file():
        worst = max([r["max_residual_over_bound"]
                     for r in json.loads(logs.read_text()).get("residuals", {}).values()] or [0.0])
    return {"passed": bool(has_memory and not mismatched and worst is not None and worst <= 1.0),
            "mismatched": mismatched, "memory_named": has_memory, "worst_residual_over_bound": worst}


def probe_artifact(path: Path, ckpt: Path, keys: list[str], smoke: bool) -> dict:
    """``missing`` / ``invalid`` / ``valid``: the checkpoint SHA, the weight check and the cache
    precision (A amp when the run used AMP on CUDA, every B fp32; production probes run on CUDA)."""
    if not path.is_file():
        return {"status": INCOMPLETE, "problems": [f"{path}: missing"]}
    rec = json.loads(path.read_text())
    problems = []
    if not ckpt.is_file() or rec.get("checkpoint_sha256") != file_sha256(ckpt):
        problems.append("checkpoint_sha256")
    if rec.get("weights_unchanged") is not True:
        problems.append("weights_unchanged")
    cache = rec.get("text_cache") or {}
    want = expected_precision(len(keys), bool(rec.get("amp")), str(rec.get("device_type")))
    got = [(cache.get(k) or {}).get("precision") for k in keys]
    if got != want:
        problems.append(f"text_cache precision {got} != {want}")
    if not smoke and rec.get("device_type") != "cuda":
        problems.append(f"device {rec.get('device_type')}")
    return {"status": INVALID if problems else VALID, "problems": problems}


# ---- the judgment ------------------------------------------------------------------------------------------


def _est(entry) -> dict:
    e = entry.estimate
    return {"value": e.value, "low": e.low, "high": e.high, "n": entry.n_rollouts}


def _paired(a, b, bootstrap: dict) -> dict:
    from flowcl.experiments.reverse_order import _paired as paired

    return paired(list(a), list(b), bootstrap)


def judge_p1(cfg: dict, blocks: dict) -> dict:
    """The similar-minus-diverse contrast of dP_proj, labelled by its basis (see the config)."""
    by = {c["name"]: c for c in cfg["curricula"]}
    diverse = [c["name"] for c in cfg["curricula"] if c["category"] == "diverse"][0]
    sims = [c["name"] for c in cfg["curricula"] if c["category"] == "similar"]
    used = [c for c in sims if blocks[c]["status"] != INCOMPLETE]
    missing = {by[c]["short"]: blocks[c].get("problems", []) for c in sims if c not in used}
    out = {"diverse": by[diverse]["short"], "similar_used": [by[c]["short"] for c in used],
           "similar_missing": missing, "threshold": float(cfg["p1_supported_below"])}
    if blocks[diverse]["status"] == INCOMPLETE or not used:
        return {**out, "judgment": "not_judged", "reason": "the diverse or every similar curriculum is incomplete"}
    shorts = [by[c]["short"] for c in used]
    out["basis"] = (f"both similar curricula ({', '.join(shorts)}) versus {by[diverse]['short']}"
                    if len(used) == len(sims) and len(used) > 1
                    else f"{' and '.join(shorts)} versus {by[diverse]['short']}, "
                         f"{'one similar task' if len(used) == 1 else f'{len(used)} similar tasks'}")
    d = blocks[diverse]["differences"]["dP"]["proj"]["diff"]
    values = {by[c]["short"]: blocks[c]["differences"]["dP"]["proj"]["diff"] for c in used}
    out["contrasts"] = {f"{s}-{by[diverse]['short']}": v - d for s, v in values.items()}
    out["dP_proj"] = {**values, by[diverse]["short"]: d}
    out["mean_contrast"] = statistics.fmean(values.values()) - d
    if any(blocks[c]["status"] == INVALID for c in [*used, diverse]):
        return {**out, "judgment": "inconclusive", "reason": "a used curriculum is invalid"}
    if any(blocks[c]["assess"]["dP_proj"] != "ok" for c in [*used, diverse]):
        return {**out, "judgment": "inconclusive", "reason": "a comparator did not acquire B"}
    m = out["mean_contrast"]
    judgment = ("supported" if m < out["threshold"] else "direction_only" if m < 0 else "not_supported")
    return {**out, "judgment": judgment}


def judge_p2(cfg: dict, blocks: dict) -> dict:
    per = {}
    for c in cfg["curricula"]:
        b = blocks[c["name"]]
        if b["status"] == INCOMPLETE:
            per[c["short"]] = "not_judged"
        elif b["status"] == INVALID:
            per[c["short"]] = "invalid"
        elif b["assess"]["retention"] != "ok":
            per[c["short"]] = "not_assessable"
        else:
            per[c["short"]] = "holds" if b["differences"]["dR"]["tot"]["low"] > 0 else "fails"
    values = set(per.values())
    overall = ("holds_in_every_curriculum" if values == {"holds"} else
               "fails_somewhere" if "fails" in values else "partial")
    return {"per_curriculum": per, "overall": overall}


def _block(cfg: dict, curriculum: dict, root: Path, n_episodes: int, smoke: bool, bootstrap: dict,
           zero_shot, allowed_fn) -> dict:
    """Everything one curriculum contributes: status, checks, cells, differences, flags."""
    from flowcl.experiments.reverse_order import seed_pairing, stats_check

    name = curriculum["name"]
    source = cfg["source_curriculum"]
    b = b_key(name)
    dirs = {arm: root / run_name(cfg, name, arm) for arm in ARMS}
    src_dirs = {arm: root / run_name(cfg, source, arm) for arm in ARMS}
    views, problems = {}, []
    for arm in ARMS:
        views[arm], p = _load_view(dirs[arm], n_episodes)
        problems += p
    block = {"short": curriculum["short"], "category": curriculum["category"], "task_b": b,
             "runs": {arm: str(d) for arm, d in dirs.items()}}
    if problems:
        return {**block, "status": INCOMPLETE, "problems": problems}
    if name != source and zero_shot is None:
        return {**block, "status": INCOMPLETE, "problems": ["the stage-0 zero-shot file is missing"]}
    checks = {}
    try:
        for arm in ARMS:
            checks[f"production_{arm}"] = production_check(dirs[arm], cfg, name, arm, smoke)
        checks["statistics"] = stats_check(dirs, cfg["task_a"])
        checks["episode_pairing"] = {"passed": True, "pairs": {}}
        for x, y in (("seq_ft", "allowlist_ft"), ("seq_ft", "gpm")):
            pair = seed_pairing(views[x], views[y])
            checks["episode_pairing"]["pairs"][f"{x}-{y}"] = pair
            checks["episode_pairing"]["passed"] &= pair["passed"]
        z = (views["seq_ft"].evals[0].by_task().get(b) if name == source
             else (zero_shot["report"].by_task().get(b) if zero_shot else None))
        z_seeds_ok = z is not None and all(list(views[arm].cell(1, 1).seeds) == list(z.seeds) for arm in ARMS)
        checks["zero_shot"] = {"passed": bool(z_seeds_ok and (name == source or zero_shot["sha_ok"])),
                               "seeds_match": z_seeds_ok,
                               "source": "S's stage-0 eval" if name == source else "stage0/zero_shot.json"}
        if name == source:
            for arm in FROZEN_ARMS:
                checks[f"identity_{arm}"] = identity_check(dirs[arm])
            checks["t1_pairing"] = {"passed": bool((views["gpm"].result.get("t1_pairing") or {}).get("passed"))}
        else:
            for arm in ARMS:
                checks[f"resume_{arm}"] = resume_check(dirs[arm], src_dirs[arm])
        allowed = allowed_fn()
        for arm in FROZEN_ARMS:
            checks[f"frozen_{arm}"] = frozen_check(dirs[arm], src_dirs[arm], allowed, arm)
        checks["gpm_memory"] = memory_check(dirs["gpm"])
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        return {**block, "status": INCOMPLETE, "problems": [f"malformed input: {type(exc).__name__}: {exc}"],
                "checks": checks}
    status = VALID if all(c["passed"] for c in checks.values()) else INVALID

    seq, allow, gpm = views["seq_ft"], views["allowlist_ft"], views["gpm"]
    cells = {"R00": {arm: _est(views[arm].cell(0, 0)) for arm in ARMS},
             "Z": _est(z) if z is not None else None,
             "R11": {arm: _est(views[arm].cell(1, 1)) for arm in ARMS},
             "R10": {arm: _est(views[arm].cell(1, 0)) for arm in ARMS}}
    s = {arm: {"R11": views[arm].cell(1, 1).successes, "R10": views[arm].cell(1, 0).successes,
               "R00": views[arm].cell(0, 0).successes} for arm in ARMS}
    diffs = {
        "dP": {"proj": _paired(s["gpm"]["R11"], s["allowlist_ft"]["R11"], bootstrap),
               "freeze": _paired(s["allowlist_ft"]["R11"], s["seq_ft"]["R11"], bootstrap),
               "tot": _paired(s["gpm"]["R11"], s["seq_ft"]["R11"], bootstrap)},
        "dR": {"proj": _paired(s["gpm"]["R10"], s["allowlist_ft"]["R10"], bootstrap),
               "freeze": _paired(s["allowlist_ft"]["R10"], s["seq_ft"]["R10"], bootstrap),
               "tot": _paired(s["gpm"]["R10"], s["seq_ft"]["R10"], bootstrap)},
        "forgetting": {arm: _paired(s[arm]["R00"], s[arm]["R10"], bootstrap) for arm in ARMS},
        "acquisition_over_zero_shot": ({arm: _paired(s[arm]["R11"], z.successes, bootstrap) for arm in ARMS}
                                       if z_seeds_ok else None),
    }
    b_min, a_min = float(cfg["b_acquired_min"]), float(cfg["a_weak_below"])
    acquired = {arm: views[arm].cell(1, 1).estimate.value >= b_min for arm in ARMS}
    a_weak = seq.cell(0, 0).estimate.value < a_min
    assess = {"dP_proj": "ok" if acquired["allowlist_ft"] else "inconclusive",
              "dP_freeze": "ok" if acquired["seq_ft"] else "inconclusive",
              "dP_tot": "ok" if acquired["seq_ft"] else "inconclusive",
              "retention": "not_assessable" if a_weak else "ok"}
    del allow, gpm
    return {**block, "status": status, "problems": [n for n, c in checks.items() if not c["passed"]],
            "checks": checks, "cells": cells, "differences": diffs, "assess": assess,
            "flags": {"A_weak": a_weak, "B_acquired": acquired}}


def _probe_values(rec: dict, b: str) -> dict:
    instr = rec["instruction"][b]
    return {d: {"velocity_sensitivity": v["velocity"]["sensitivity"], "chunk_relative": v["chunks"]["relative"],
                "trunk_size_median": (v["size_halves"].get("trunk") or {}).get("median")}
            for d, v in instr.items()}


def run_summary(cfg: dict, *, results_root=None, out: Path | None = None, smoke: bool = False,
                rule_sha256: str | None = None, allowlist_names: set[str] | None = None) -> dict:
    """Judge the study and write ``summary.json`` atomically (see the module docstring)."""
    from flowcl.envs.evaluation import EvaluationReport

    root = _root(results_root)
    out = Path(out) if out else out_dir(cfg, root)
    bootstrap = _bootstrap()
    n_episodes = int((cfg["smoke"] if smoke else cfg["production"])["n_episodes"])
    source_seqft = root / run_name(cfg, cfg["source_curriculum"], "seq_ft")
    stage0_ckpt = source_seqft / "checkpoints" / "stage0.pt"
    keys0 = [cfg["task_a"], *[b_key(c["name"]) for c in cfg["curricula"]]]

    def allowed_fn():
        nonlocal allowlist_names
        if allowlist_names is None:
            from flowcl.methods.gpm import allowlist
            from flowcl.train.checkpoint import load_checkpoint

            allowlist_names = set(allowlist(load_checkpoint(stage0_ckpt, device="cpu").policy))
        return allowlist_names

    # the stage-0 extras
    s0_path = stage0_path(cfg, root)
    stage0 = {"artifact": probe_artifact(s0_path, stage0_ckpt, keys0, smoke)}
    zero_shot = None
    if s0_path.is_file():
        rec0 = json.loads(s0_path.read_text())
        zpath = s0_path.parent / rec0["zero_shot"]["path"]
        if zpath.is_file():
            report = EvaluationReport.load(zpath)
            sha_ok = (file_sha256(zpath) == rec0["zero_shot"]["sha256"] and stage0_ckpt.is_file()
                      and rec0.get("checkpoint_sha256") == file_sha256(stage0_ckpt)
                      and all(t.n_rollouts == n_episodes for t in report.tasks))
            zero_shot = {"report": report, "sha_ok": sha_ok}
        stage0["zero_shot_sha_ok"] = zero_shot["sha_ok"] if zero_shot else None

    blocks = {c["name"]: _block(cfg, c, root, n_episodes, smoke, bootstrap, zero_shot, allowed_fn)
              for c in cfg["curricula"]}

    probes = {}
    for c in cfg["curricula"]:
        for arm in ARMS:
            run = root / run_name(cfg, c["name"], arm)
            probes[run.name] = probe_artifact(probe_path(cfg, root, c["name"], arm),
                                              run / "checkpoints" / "stage1.pt", [cfg["task_a"], b_key(c["name"])],
                                              smoke)
    artifacts = [stage0["artifact"], *probes.values()]
    if any(b["status"] == INVALID for b in blocks.values()) or any(a["status"] == INVALID for a in artifacts):
        status = INVALID
    elif any(b["status"] == INCOMPLETE for b in blocks.values()) or any(a["status"] == INCOMPLETE for a in artifacts):
        status = INCOMPLETE
    else:
        status = VALID

    # P4 and probe losses (valid artifacts only)
    p4, losses = {}, {}
    if stage0["artifact"]["status"] == VALID:
        rec0 = json.loads(s0_path.read_text())
        losses["stage0"] = rec0["probe_loss"]
        for c in cfg["curricula"]:
            b = b_key(c["name"])
            p4.setdefault(c["short"], {})["stage0"] = _probe_values(rec0, b)
    for c in cfg["curricula"]:
        b = b_key(c["name"])
        for arm in ARMS:
            name = run_name(cfg, c["name"], arm)
            if probes[name]["status"] == VALID:
                rec = json.loads(probe_path(cfg, root, c["name"], arm).read_text())
                p4.setdefault(c["short"], {})[arm] = _probe_values(rec, b)
                losses.setdefault(c["short"], {})[arm] = rec["probe_loss"]

    complete = {n: b for n, b in blocks.items() if b["status"] != INCOMPLETE}
    predictions = {
        "P1": judge_p1(cfg, blocks),
        "P2": judge_p2(cfg, blocks),
        "P2b": {b["short"]: b["differences"]["dR"]["proj"] for b in complete.values()},
        "P3": {b["short"]: {"category": b["category"], "seq_ft_forgetting": b["differences"]["forgetting"]["seq_ft"]}
               for b in complete.values()},
        "P4": p4,
        "P5": {b["short"]: {"dP_tot": b["differences"]["dP"]["tot"], "dP_freeze": b["differences"]["dP"]["freeze"],
                            "Z": b["cells"]["Z"], "assess": b["assess"]} for b in complete.values()},
    }
    summary = {"git_sha": git_sha(), "rule_sha256": rule_sha256, "smoke": smoke, "status": status,
               "exit_code": EXIT_CODES[status], "namespace": namespace(cfg), "n_episodes": n_episodes,
               "stage0": stage0, "probes": probes, "curricula": blocks, "probe_loss": losses,
               "predictions": predictions, "config": cfg}
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out / "summary.json", json.dumps(summary, indent=2, default=str) + "\n")
    print(f"[flowcl] similar_cl: {status}; " + "; ".join(f"{b['short']} {b['status']}" for b in blocks.values())
          + f"; P1 {predictions['P1']['judgment']} ({predictions['P1'].get('basis')}); "
            f"P2 {predictions['P2']['overall']}", flush=True)
    return summary
