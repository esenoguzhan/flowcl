"""Similar-task diagnostic (``configs/analysis/similar_tasks.yaml``). No continual training.

At one single-task policy A (``libero_goal/put_the_bowl_on_the_plate``, trained by Gate 0's recipe):

1. :func:`run_memory` builds ``M_A`` (A's input basis from A's own data), checks it on the Gram it was
   built from, records an independent stochastic recapture, runs Gate 3 on A's own data (keeping A's
   full and split-half gradients), and pins A's checkpoint and ``M_A``'s artifact by SHA-256;
2. :func:`run_measure`, per task B: the input overlap with ``M_A``, Gate 3's blocked-gradient share,
   the local gradient alignment with A (split-half stability), and the paired instruction-swap capture
   (:class:`PairedSwapCapture`: the same observations and s/noise under the native and the swapped
   instruction);
3. :func:`run_summary`: per half, per task, per category and the category differences.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import statistics
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from flowcl.utils.libero_paths import repo_root
from flowcl.utils.run import atomic_write_text, file_sha256, git_sha

SCOPES = ("all", "trunk", "decoder")


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else repo_root() / "configs" / "analysis" / "similar_tasks.yaml"
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def _root(results_root) -> Path:
    return Path(results_root) if results_root else repo_root() / "results"


def a_run_dir(cfg: dict, root: Path) -> Path:
    from flowcl.experiments.gate0 import single_task_run_id

    return root / single_task_run_id(cfg["task_a"], int(cfg["seed"]))


def out_dir(cfg: dict, root: Path) -> Path:
    return root / cfg["out"]


def slug(task_key: str) -> str:
    return task_key.replace("/", "__")


def _half(name: str) -> str:
    return "trunk" if name.startswith("trunk.") else "decoder"


def _in_scope(name: str, scope: str) -> bool:
    return scope == "all" or _half(name) == scope


# ---- pure pieces --------------------------------------------------------------------------------------


def surviving_norm(c: float) -> float:
    """``||G_perp|| / ||G|| = sqrt(1 - c^2)`` for ``c = ||G_par|| / ||G||`` (orthogonal parts)."""
    return math.sqrt(max(0.0, 1.0 - c * c))


def _cos(dot: float, na2: float, nb2: float) -> float | None:
    return None if na2 <= 0.0 or nb2 <= 0.0 else dot / math.sqrt(na2 * nb2)


def gradient_cosines(ga: dict[str, torch.Tensor], gb: dict[str, torch.Tensor]) -> dict:
    """Per layer and pooled scope: ``cos(ga, gb)`` and both squared norms (``None`` at zero norm)."""
    per_layer, pooled = {}, {s: [0.0, 0.0, 0.0] for s in SCOPES}
    for name in ga:
        a, b = ga[name].to(torch.float64), gb[name].to(torch.float64)
        dot, na2, nb2 = float((a * b).sum()), float((a * a).sum()), float((b * b).sum())
        per_layer[name] = {"cos": _cos(dot, na2, nb2), "norm_a": math.sqrt(na2), "norm_b": math.sqrt(nb2)}
        for s in SCOPES:
            if _in_scope(name, s):
                pooled[s][0] += dot
                pooled[s][1] += na2
                pooled[s][2] += nb2
    return {"layers": per_layer,
            "scopes": {s: {"cos": _cos(*v), "norm_a": math.sqrt(v[1]), "norm_b": math.sqrt(v[2])}
                       for s, v in pooled.items()}}


def alignment(full_a, split_a, full_b, split_b, threshold: float) -> dict:
    """A-B cosine with each task's split-half self-consistency; ``unstable`` per layer and scope where
    either self-consistency is below ``threshold`` or undefined."""
    ab = gradient_cosines(full_a, full_b)
    sa = gradient_cosines(split_a["even"], split_a["odd"])
    sb = gradient_cosines(split_b["even"], split_b["odd"])

    def unstable(x, y):
        return x is None or y is None or x < threshold or y < threshold

    layers = {n: {**ab["layers"][n], "self_a": sa["layers"][n]["cos"], "self_b": sb["layers"][n]["cos"],
                  "unstable": unstable(sa["layers"][n]["cos"], sb["layers"][n]["cos"])}
              for n in ab["layers"]}
    scopes = {s: {**ab["scopes"][s], "self_a": sa["scopes"][s]["cos"], "self_b": sb["scopes"][s]["cos"],
                  "unstable": unstable(sa["scopes"][s]["cos"], sb["scopes"][s]["cos"])}
              for s in SCOPES}
    return {"layers": layers, "scopes": scopes}


def half_summary(values: dict[str, float | None], dims: dict[str, tuple[int, int]]) -> dict:
    """Per half: the median and the parameter-weighted mean of the defined values, and the counts."""
    out = {}
    for half in ("trunk", "decoder"):
        names = [n for n in dims if _half(n) == half]
        defined = [n for n in names if values.get(n) is not None]
        if not defined:
            out[half] = {"median": None, "weighted": None, "n_layers": len(names), "n_defined": 0}
            continue
        w = [dims[n][0] * dims[n][1] for n in defined]
        out[half] = {"median": statistics.median(values[n] for n in defined),
                     "weighted": sum(values[n] * wi for n, wi in zip(defined, w)) / sum(w),
                     "n_layers": len(names), "n_defined": len(defined)}
    return out


def category_summary(per_task: dict[str, float | None], categories: dict[str, str],
                     baseline: str = "diverse") -> dict:
    """Mean per category over the defined values, and each category minus ``baseline``."""
    means = {}
    for cat in dict.fromkeys(categories.values()):
        vals = [per_task[k] for k, c in categories.items() if c == cat and per_task.get(k) is not None]
        means[cat] = sum(vals) / len(vals) if vals else None
    diffs = {f"{cat}-{baseline}": (None if means[cat] is None or means.get(baseline) is None
                                   else means[cat] - means[baseline])
             for cat in means if cat != baseline}
    return {"means": means, "differences": diffs}


def outside_share(M: torch.Tensor, D: torch.Tensor) -> float | None:
    """``1 - tr(M^T D M) / tr D``; ``None`` where ``tr D = 0`` (no instruction signal)."""
    tr = float(torch.trace(D))
    if tr <= 0.0:
        return None
    M = M.to(torch.float64)
    return 1.0 - float(torch.trace(M.T @ D @ M)) / tr


# ---- the paired instruction-swap capture ----------------------------------------------------------------


class PairedSwapCapture:
    """Accumulate, per hooked layer, ``D = sum (x_swap - x_ref)(x_swap - x_ref)^T`` and
    ``K_ref = sum x_ref x_ref^T`` at the SAME subsampled positions of a reference and a swapped pass.

    Run the reference forward under ``mode("ref")`` and the swapped forward under ``mode("swap")``,
    with the same observations and s/noise. Positions are drawn in the reference pass (up to
    ``tokens_per_sample`` per sample; action-position layers only at valid positions) and reused.
    """

    def __init__(self, layers, tokens_per_sample: int, seed: int) -> None:
        from flowcl.analysis.hooks import KIND_ACTION, KIND_STATE

        self._action, self._state = KIND_ACTION, KIND_STATE
        self.layers = list(layers)                  # (name, module, kind)
        self.m = int(tokens_per_sample)
        self.gen = torch.Generator(device="cpu").manual_seed(int(seed))
        self.D = {n: torch.zeros(mod.in_features, mod.in_features, dtype=torch.float64) for n, mod, _ in self.layers}
        self.K = {n: torch.zeros(mod.in_features, mod.in_features, dtype=torch.float64) for n, mod, _ in self.layers}
        self.n = {n: 0 for n, _, _ in self.layers}
        self._mode, self._mask, self._ref, self._handles = None, None, {}, []

    @classmethod
    def for_policy(cls, policy, tokens_per_sample: int, seed: int) -> "PairedSwapCapture":
        from flowcl.analysis.hooks import layer_kind

        return cls([(e.name, e.module, layer_kind(e.name)) for e in policy.projectable_layers()],
                   tokens_per_sample, seed)

    def __enter__(self):
        for name, module, kind in self.layers:
            self._handles.append(module.register_forward_pre_hook(self._hook(name, kind)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def mode(self, mode: str | None, action_mask: torch.Tensor | None = None):
        if mode not in (None, "ref", "swap"):
            raise ValueError(mode)
        if mode == "swap" and not self._ref:
            raise RuntimeError("a swap pass needs a reference pass first")
        self._mode = mode
        if action_mask is not None:
            self._mask = action_mask.detach().to("cpu") > 0
        if mode == "ref":
            self._ref = {}

    def _hook(self, name, kind):
        def hook(module, args):
            if self._mode is None:
                return
            x = args[0].detach()
            if x.shape[-1] != module.in_features:
                raise RuntimeError(f"{name}: input width {x.shape[-1]} != in_features {module.in_features}")
            if kind == self._state:
                x = x.unsqueeze(1)
            batch, seq, _ = x.shape
            if self._mode == "ref":
                eligible = torch.ones((batch, seq), dtype=torch.bool)
                if kind == self._action:
                    if self._mask is None or tuple(self._mask.shape) != (batch, seq):
                        raise RuntimeError(f"{name}: action layer without a matching mask")
                    eligible = self._mask
                m = min(self.m, seq)
                scores = torch.rand((batch, seq), generator=self.gen).masked_fill(~eligible, 2.0)
                order = scores.argsort(dim=1)[:, :m]
                keep = eligible.gather(1, order)
                rows = torch.arange(batch).unsqueeze(1).expand(batch, m)
                b, t = rows[keep], order[keep]
                self._ref[name] = (b, t, x[b.to(x.device), t.to(x.device)].to("cpu", torch.float64))
            else:
                b, t, ref_rows = self._ref[name]
                rows = x[b.to(x.device), t.to(x.device)].to("cpu", torch.float64)
                if rows.shape != ref_rows.shape:
                    raise RuntimeError(f"{name}: swapped pass has a different shape")
                delta = rows - ref_rows
                self.D[name].addmm_(delta.T, delta)
                self.K[name].addmm_(ref_rows.T, ref_rows)
                self.n[name] += rows.shape[0]
        return hook


def paired_swap(policy, dataset, swap_text: str, sub_cfg, device, seed: int, n_batches: int | None = None):
    """Each batch twice (native and ``swap_text`` instruction) through the same s/noise. Returns the
    capture and the pooled output sensitivity ``(sum ||v_swap - v_ref||^2, sum ||v_ref||^2)`` over
    valid action positions."""
    from flowcl.models.flow_head import draw_with_generator
    from flowcl.train.trainer import build_dataloader, move_batch

    device = torch.device(device)
    policy.eval()
    cap = PairedSwapCapture.for_policy(policy, sub_cfg.tokens_per_sample, seed)
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    loader = build_dataloader(dataset, batch_size=sub_cfg.batch_size, num_workers=sub_cfg.num_workers,
                              shuffle=False)
    num = den = 0.0
    with cap, torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
        for i, batch in enumerate(loader):
            if n_batches is not None and i >= n_batches:
                break
            batch = move_batch(batch, device)
            size = batch["actions"].shape[0]
            s = policy.s_sampler.sample(size, device, generator=gen)
            noise = draw_with_generator(tuple(batch["actions"].shape), device=device, generator=gen,
                                        dtype=torch.float32, normal=True)
            cap.mode("ref", batch["action_mask"])
            v_ref = policy(batch, s=s, noise=noise)["velocity"]
            swapped = dict(batch)
            swapped["language"] = [swap_text] * size
            cap.mode("swap")
            v_swap = policy(swapped, s=s, noise=noise)["velocity"]
            cap.mode(None)
            m = batch["action_mask"].unsqueeze(-1).to(v_ref.dtype)
            num += float((((v_swap - v_ref) ** 2) * m).sum())
            den += float(((v_ref ** 2) * m).sum())
    return cap, {"numerator": num, "denominator": den, "sensitivity": (num / den) if den > 0 else None}


def swap_report(cap: PairedSwapCapture, M: dict[str, torch.Tensor], dims, out_sens: dict) -> dict:
    size = {n: (float(torch.trace(cap.D[n])) / float(torch.trace(cap.K[n]))
                if float(torch.trace(cap.K[n])) > 0 else None) for n in cap.D}
    share = {n: outside_share(M[n], cap.D[n]) for n in cap.D}
    return {"output_sensitivity": out_sens,
            "size": {"layers": size, "halves": half_summary(size, dims)},
            "outside_share": {"layers": share, "halves": half_summary(share, dims)},
            "n_rows": dict(cap.n)}


# ---- shared setup ---------------------------------------------------------------------------------------


def _capture_config(cfg: dict, smoke: bool):
    from flowcl.experiments.gate2 import load_subspace_config

    path = Path(str(cfg["capture_config"]))
    if path.suffix != ".yaml":
        path = repo_root() / "configs" / "analysis" / f"{cfg['capture_config']}.yaml"
    sub = load_subspace_config(path)
    return dataclasses.replace(sub, min_samples_per_dim=0.0) if smoke else sub


class TailSubset:
    """The LAST ``n`` samples of a dataset (smoke runs only): an episode's tail holds padded chunk
    samples, which the reachability probe needs (a long first episode's head has none)."""

    def __init__(self, base, n: int) -> None:
        self.base = base
        self.n = min(int(n), len(base))
        self.offset = len(base) - self.n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i):
        if not 0 <= i < self.n:
            raise IndexError(i)
        return self.base[self.offset + i]

    def sample_index(self, i):
        return self.base.sample_index(self.offset + i)

    def __getattr__(self, name):
        if name == "base":
            raise AttributeError(name)
        return getattr(self.base, name)


def _dataset(key: str, loaded, dataset_dir, build_datasets, smoke_batches, batch_size):
    from flowcl.data.tasks import TaskRef
    from flowcl.train.pipeline import build_dataset

    ds = (build_datasets(key) if build_datasets
          else build_dataset([TaskRef.from_key(key)], loaded.spec, loaded.stats, n_demos=None,
                             dataset_dir=dataset_dir))
    return TailSubset(ds, smoke_batches * batch_size) if smoke_batches else ds


def instruction(ds) -> str:
    texts = sorted({ep.language for ep in ds.episodes})
    if len(texts) != 1:
        raise ValueError(f"expected one instruction per task dataset, got {texts}")
    return texts[0]


def warm_fp32(policy, texts) -> list[str]:
    """Encode every instruction once in fp32 (deterministic cache state; see the config)."""
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
        for text in sorted(set(texts)):
            policy.text_encoder([text])
    return sorted(set(texts))


def _dims(policy) -> tuple[dict, dict]:
    reg = policy.projectable_layers()
    return ({e.name: (e.module.out_features, e.module.in_features) for e in reg},
            {e.name: e.group for e in reg})


def _publish(attempt: Path, canonical: Path) -> None:
    if canonical.exists():
        raise FileExistsError(f"{canonical} already exists; published results are never overwritten")
    canonical.parent.mkdir(parents=True, exist_ok=True)
    os.rename(attempt, canonical)


def _attempt(parent: Path, name: str) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    d = parent / f".attempt_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{name}"
    d.mkdir()
    return d


def _interference_cfg(smoke_batches):
    from flowcl.experiments.gate3 import load_interference_config

    icfg = load_interference_config()
    return dataclasses.replace(icfg, n_batches=int(smoke_batches)) if smoke_batches else icfg


def _gate3_summary(gi, eps: float) -> dict:
    layers = {n: {"c_energy": l.energy_c(eps), "c_full": l.full_c(eps), "c_mean": l.mean_c(eps),
                  "surviving_norm_energy": surviving_norm(l.energy_c(eps))} for n, l in gi.layers.items()}
    gc = gi.global_c(eps)
    return {"global_c": gc,
            "surviving_norm": {s: {k: surviving_norm(v) for k, v in d.items() if isinstance(v, float)}
                               for s, d in gc.items()},
            "layers": layers, "n_batches": gi.n_batches, "n_samples": gi.n_samples}


def _verify_memory(cfg: dict, root: Path) -> dict:
    mem = json.loads((out_dir(cfg, root) / "memory" / "memory.json").read_text())
    ckpt = a_run_dir(cfg, root) / "checkpoints" / "final.pt"
    problems = []
    if file_sha256(ckpt) != mem["checkpoint_sha256"]:
        problems.append("A's checkpoint changed since the memory step")
    for key in ("basis", "gradients"):
        if file_sha256(out_dir(cfg, root) / "memory" / mem["files"][key]["path"]) != mem["files"][key]["sha256"]:
            problems.append(f"the memory step's {key} artifact changed")
    if problems:
        raise RuntimeError("; ".join(problems))
    return mem


# ---- step 1: A's memory -----------------------------------------------------------------------------------


def run_memory(cfg: dict, *, results_root=None, device="cuda", allow_dirty=False, dataset_dir=None,
               build_datasets=None, smoke_batches: int | None = None) -> dict:
    from flowcl.analysis.subspace import basis_from_gram, captured_energy_fraction, save_bases
    from flowcl.experiments.gate2 import capture_task_grams
    from flowcl.experiments.gate3 import measure_gradient_interference
    from flowcl.experiments.low_update import refuse_dirty
    from flowcl.train.checkpoint import load_checkpoint
    from flowcl.utils.seeding import derive_seed

    sha = refuse_dirty(allow_dirty)
    root = _root(results_root)
    canonical = out_dir(cfg, root) / "memory"
    if (canonical / "memory.json").is_file():
        print("[flowcl] memory: published, reused", flush=True)
        return _verify_memory(cfg, root)
    ckpt = a_run_dir(cfg, root) / "checkpoints" / "final.pt"
    ckpt_sha = file_sha256(ckpt)
    loaded = load_checkpoint(ckpt, device=device)
    if loaded.task_key != cfg["task_a"] or loaded.stage != 0:
        raise ValueError(f"{ckpt} is {loaded.task_key!r} stage {loaded.stage}, expected {cfg['task_a']!r}")
    policy, smoke = loaded.policy, smoke_batches is not None
    sub = _capture_config(cfg, smoke)
    ds_a = _dataset(cfg["task_a"], loaded, dataset_dir, build_datasets, smoke_batches, sub.batch_size)
    texts = warm_fp32(policy, [instruction(ds_a)])
    run_id, key = loaded.run_id, cfg["task_a"]
    tags = cfg["seed_tags"]
    weights = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    started = time.perf_counter()

    cap = capture_task_grams(policy, ds_a, sub, device, probe_seed=derive_seed(f"{tags['probe']}::{run_id}", key, 0),
                             capture_seed=derive_seed(f"{tags['capture']}::{run_id}", key, 0))
    dims, groups = _dims(policy)
    eps_list = [float(e) for e in cfg["eps_reported"]]
    bases, self_control, K_a = {}, {}, {}
    for name, acc in cap.accumulators.items():
        view = cap.primary_view(name)
        K_a[name] = acc.gram[view]
        bases[name] = basis_from_gram(acc.gram[view], layer=name, n_samples=acc.n[view], thresholds=eps_list,
                                      neg_tol=sub.neg_tol, rank_tol=sub.rank_tol)
        self_control[name] = captured_energy_fraction(bases[name].basis(float(cfg["eps"])), K_a[name])
    worst = min(self_control.values())
    recap = capture_task_grams(policy, ds_a, sub, device,
                               probe_seed=derive_seed(f"{tags['probe']}::{run_id}", key, 0),
                               capture_seed=derive_seed(f"{tags['recapture']}::{run_id}", key, 0))
    recapture = {n: captured_energy_fraction(bases[n].basis(float(cfg["eps"])),
                                             recap.accumulators[n].gram[recap.primary_view(n)])
                 for n in bases}
    del recap, cap
    meta = {"kind": "similar_tasks_M_A", "run_id": run_id, "task_idx": 0, "task_key": key,
            "stats_fingerprint": loaded.stats.fingerprint(), "checkpoint_sha256": ckpt_sha,
            "seed_tags": tags, "smoke": smoke}
    attempt = _attempt(out_dir(cfg, root), "memory")
    save_bases(attempt / "M_A.pt", bases, meta)
    gi = measure_gradient_interference(loaded, bases, meta, ds_a, _interference_cfg(smoke_batches), device=device,
                                       label="A_on_A", checkpoint_path=ckpt, keep_full_gradient=True,
                                       keep_split_gradients=True)
    torch.save({"full": {n: g.float().cpu() for n, g in gi.full_gradients.items()},
                "even": {n: g.float().cpu() for n, g in gi.split_gradients["even"].items()},
                "odd": {n: g.float().cpu() for n, g in gi.split_gradients["odd"].items()}},
               attempt / "grad_A.pt")
    unchanged = all(torch.equal(v, weights[k]) for k, v in policy.state_dict().items())
    gate0 = root / cfg["a_gate0_out"] / "gate0.json"
    a_success = (json.loads(gate0.read_text())["evidence"]["per_task"].get(key, {}).get("success_rate")
                 if gate0.is_file() else None)
    tol = float(cfg["self_control_tol"])
    record = {
        "git_sha": sha, "smoke": smoke, "task_a": key, "run_id": run_id,
        "checkpoint": str(ckpt), "checkpoint_sha256": ckpt_sha,
        "a_gate0_success": a_success,
        "a_weak": a_success is None or a_success < float(cfg["a_gate0_threshold"]),
        "instructions": texts,
        "dims": dims, "groups": groups,
        "self_control": {"passed": worst >= float(cfg["eps"]) - tol, "worst": worst, "layers": self_control,
                         "halves": half_summary(self_control, dims)},
        "recapture": {"layers": recapture, "halves": half_summary(recapture, dims)},
        "ranks": {n: {f"{e:g}": b.ranks[e] for e in eps_list} for n, b in bases.items()},
        "gate3_self": _gate3_summary(gi, float(cfg["eps"])),
        "weights_unchanged": unchanged,
        "files": {"basis": {"path": "M_A.pt", "sha256": file_sha256(attempt / "M_A.pt")},
                  "gradients": {"path": "grad_A.pt", "sha256": file_sha256(attempt / "grad_A.pt")}},
        "wall_clock_s": time.perf_counter() - started,
    }
    atomic_write_text(attempt / "memory.json", json.dumps(record, indent=2) + "\n")
    _publish(attempt, canonical)
    print(f"[flowcl] memory: self-control worst {worst:.4f} (>= {cfg['eps']} - tol: "
          f"{record['self_control']['passed']}); A Gate 0 {a_success}; weights unchanged {unchanged}", flush=True)
    return record


# ---- step 2: one task B ------------------------------------------------------------------------------------


def run_measure(cfg: dict, b_key: str, *, results_root=None, device="cuda", allow_dirty=False, dataset_dir=None,
                build_datasets=None, smoke_batches: int | None = None) -> dict:
    from flowcl.analysis.subspace import captured_energy_fraction, load_bases
    from flowcl.experiments.gate2 import capture_task_grams
    from flowcl.experiments.gate3 import measure_gradient_interference
    from flowcl.experiments.low_update import refuse_dirty
    from flowcl.train.checkpoint import load_checkpoint
    from flowcl.utils.seeding import derive_seed

    entry = next((t for t in cfg["tasks_b"] if t["key"] == b_key), None)
    if entry is None:
        raise ValueError(f"{b_key} is not a registered task B")
    sha = refuse_dirty(allow_dirty)
    root = _root(results_root)
    canonical = out_dir(cfg, root) / "tasks" / slug(b_key)
    if (canonical / "result.json").is_file():
        print(f"[flowcl] {b_key}: published, reused", flush=True)
        return json.loads((canonical / "result.json").read_text())
    mem = _verify_memory(cfg, root)
    smoke = smoke_batches is not None
    ckpt = a_run_dir(cfg, root) / "checkpoints" / "final.pt"
    loaded = load_checkpoint(ckpt, device=device)
    policy = loaded.policy
    sub = _capture_config(cfg, smoke)
    bases, meta = load_bases(out_dir(cfg, root) / "memory" / mem["files"]["basis"]["path"])
    dims = {n: tuple(v) for n, v in mem["dims"].items()}
    ds_a = _dataset(cfg["task_a"], loaded, dataset_dir, build_datasets, smoke_batches, sub.batch_size)
    ds_b = _dataset(b_key, loaded, dataset_dir, build_datasets, smoke_batches, sub.batch_size)
    text_a, text_b = instruction(ds_a), instruction(ds_b)
    warm_fp32(policy, [text_a, text_b])
    tags, run_id = cfg["seed_tags"], loaded.run_id
    weights = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    started = time.perf_counter()
    eps = float(cfg["eps"])
    eps_list = [float(e) for e in cfg["eps_reported"]]

    # 1. input overlap
    cap = capture_task_grams(policy, ds_b, sub, device, probe_seed=derive_seed(f"{tags['probe']}::{run_id}", b_key, 0),
                             capture_seed=derive_seed(f"{tags['capture']}::{run_id}", b_key, 0))
    overlap = {f"{e:g}": {n: captured_energy_fraction(bases[n].basis(e), cap.accumulators[n].gram[cap.primary_view(n)])
                          for n in bases} for e in eps_list}
    del cap
    # 2-3. blocked gradient and local alignment
    gi = measure_gradient_interference(loaded, bases, meta, ds_b, _interference_cfg(smoke_batches), device=device,
                                       label=f"{b_key}_on_A", checkpoint_path=ckpt, keep_full_gradient=True,
                                       keep_split_gradients=True)
    grad_a = torch.load(out_dir(cfg, root) / "memory" / mem["files"]["gradients"]["path"], weights_only=False)
    full_b = {n: g.float().cpu() for n, g in gi.full_gradients.items()}
    split_b = {h: {n: g.float().cpu() for n, g in d.items()} for h, d in gi.split_gradients.items()}
    align = alignment(grad_a["full"], {"even": grad_a["even"], "odd": grad_a["odd"]}, full_b, split_b,
                      float(cfg["stability_threshold"]))
    gate3 = _gate3_summary(gi, eps)
    del gi, grad_a, full_b, split_b
    # 4. instruction signal
    M = {n: bases[n].basis(eps) for n in bases}
    swaps = {}
    cap_a, out_a = paired_swap(policy, ds_a, text_b, sub, device,
                               derive_seed(f"{tags['swap']}::{run_id}", f"{cfg['task_a']}->{b_key}", 0),
                               n_batches=smoke_batches)
    swaps["on_A_data"] = {"reference": text_a, "swap": text_b, **swap_report(cap_a, M, dims, out_a)}
    del cap_a
    if entry["category"] == "similar":
        cap_b, out_b = paired_swap(policy, ds_b, text_a, sub, device,
                                   derive_seed(f"{tags['swap']}::{run_id}", f"{b_key}->{cfg['task_a']}", 0),
                                   n_batches=smoke_batches)
        swaps["on_B_data"] = {"reference": text_b, "swap": text_a, **swap_report(cap_b, M, dims, out_b)}
        del cap_b
    unchanged = all(torch.equal(v, weights[k]) for k, v in policy.state_dict().items())

    result = {
        "git_sha": sha, "smoke": smoke, "task_b": b_key, "category": entry["category"],
        "instructions": {"a": text_a, "b": text_b},
        "memory_checkpoint_sha256": mem["checkpoint_sha256"], "memory_basis_sha256": mem["files"]["basis"]["sha256"],
        "overlap": {e: {"layers": v, "halves": half_summary(v, dims)} for e, v in overlap.items()},
        "gate3": gate3,
        "alignment": align,
        "swaps": swaps,
        "weights_unchanged": unchanged,
        "wall_clock_s": time.perf_counter() - started,
    }
    attempt = _attempt(canonical.parent, slug(b_key))
    atomic_write_text(attempt / "result.json", json.dumps(result, indent=2, default=str) + "\n")
    _publish(attempt, canonical)
    t = result["overlap"][f"{eps:g}"]["halves"]["trunk"]["median"]
    print(f"[flowcl] {b_key}: trunk overlap {t:.4f}, global_c trunk energy "
          f"{gate3['global_c'].get('trunk', {}).get('energy')}, weights unchanged {unchanged}", flush=True)
    return result


# ---- step 3: the summary --------------------------------------------------------------------------------------


def run_summary(cfg: dict, *, results_root=None) -> dict:
    root = _root(results_root)
    out = out_dir(cfg, root)
    mem_path = out / "memory" / "memory.json"
    summary: dict = {"git_sha": git_sha(), "config": cfg}
    if not mem_path.is_file():
        summary["status"] = "incomplete"
        summary["missing"] = [str(mem_path)]
        out.mkdir(parents=True, exist_ok=True)
        atomic_write_text(out / "summary.json", json.dumps(summary, indent=2) + "\n")
        return summary
    mem = _verify_memory(cfg, root)
    eps = f"{float(cfg['eps']):g}"
    categories = {t["key"]: t["category"] for t in cfg["tasks_b"]}
    per_task, missing = {}, []
    for t in cfg["tasks_b"]:
        p = out / "tasks" / slug(t["key"]) / "result.json"
        if not p.is_file():
            missing.append(t["key"])
            continue
        r = json.loads(p.read_text())
        if r["memory_basis_sha256"] != mem["files"]["basis"]["sha256"]:
            raise RuntimeError(f"{t['key']} was measured against another M_A")
        sc = r["alignment"]["scopes"]["trunk"]
        per_task[t["key"]] = {
            "category": t["category"],
            "overlap_trunk_median": r["overlap"][eps]["halves"]["trunk"]["median"],
            "overlap_decoder_median": r["overlap"][eps]["halves"]["decoder"]["median"],
            "global_c_trunk_energy": r["gate3"]["global_c"].get("trunk", {}).get("energy"),
            "global_c_trunk_full": r["gate3"]["global_c"].get("trunk", {}).get("full"),
            "global_c_decoder_energy": r["gate3"]["global_c"].get("decoder", {}).get("energy"),
            "alignment_trunk": None if sc["unstable"] else sc["cos"],
            "alignment_trunk_raw": sc["cos"], "alignment_trunk_unstable": sc["unstable"],
            "swap_on_A_outside_trunk_median": r["swaps"]["on_A_data"]["outside_share"]["halves"]["trunk"]["median"],
            "swap_on_A_size_trunk_median": r["swaps"]["on_A_data"]["size"]["halves"]["trunk"]["median"],
            "swap_on_A_output_sensitivity": r["swaps"]["on_A_data"]["output_sensitivity"]["sensitivity"],
            "weights_unchanged": r["weights_unchanged"], "smoke": r["smoke"],
        }
    cats = {}
    for field in ("overlap_trunk_median", "global_c_trunk_energy", "alignment_trunk",
                  "swap_on_A_outside_trunk_median", "swap_on_A_output_sensitivity"):
        cats[field] = category_summary({k: v[field] for k, v in per_task.items()},
                                       {k: categories[k] for k in per_task})
    sim = cats["global_c_trunk_energy"]["means"].get("similar")
    summary.update({
        "status": "complete" if not missing else "incomplete",
        "missing": missing,
        "a": {"gate0_success": mem["a_gate0_success"], "a_weak": mem["a_weak"],
              "self_control": {k: mem["self_control"][k] for k in ("passed", "worst")},
              "recapture_trunk_median": mem["recapture"]["halves"]["trunk"]["median"],
              "gate3_self_global_c": mem["gate3_self"]["global_c"]},
        "per_task": per_task, "categories": cats,
        "predictions": {
            "P1_similar_overlap_ge_diverse": _ge(cats["overlap_trunk_median"]["differences"].get("similar-diverse")),
            "P2_similar_blocked_gt_diverse": _gt(cats["global_c_trunk_energy"]["differences"].get("similar-diverse")),
            "P3_similar_alignment_positive": (None if cats["alignment_trunk"]["means"].get("similar") is None
                                              else cats["alignment_trunk"]["means"]["similar"] > 0),
            "P4": "measured (see swap fields); no direction predicted",
        },
        "followup_candidate": None if sim is None else sim >= float(cfg["followup_c"]),
        "smoke": mem["smoke"] or any(v["smoke"] for v in per_task.values()),
    })
    atomic_write_text(out / "summary.json", json.dumps(summary, indent=2, default=str) + "\n")
    print(f"[flowcl] similar tasks: {summary['status']}; P1 {summary['predictions']['P1_similar_overlap_ge_diverse']}"
          f", P2 {summary['predictions']['P2_similar_blocked_gt_diverse']}, P3 "
          f"{summary['predictions']['P3_similar_alignment_positive']}; follow-up {summary['followup_candidate']}",
          flush=True)
    return summary


def _ge(x):
    return None if x is None else x >= 0


def _gt(x):
    return None if x is None else x > 0
