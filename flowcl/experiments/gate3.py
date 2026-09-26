"""Gate 3 (§10.3): does the new task need protected directions?

Spec: "``c_l`` after T2; ``c_l ≈ 1`` ⇒ hard projection incompatible with plasticity".
Read here as **Task-2 gradients at the start of T2**: the gradients GPM would project
from T2's first step, measured at the T1 checkpoint against the Task-1 basis Gate 2
persisted. No Task-2 training happens in this gate; no optimiser steps, no rollouts.

Per checkpoint, over one pass of a task's dataset in training-sized batches:

* forward in fp32 (no autocast) with ``s`` and ``A_0`` from their own generators, one
  ``backward()``, and the registry weight gradients ``G_l`` read off;
* per layer, ``||G_l||²`` and ``||G_l M_l M_l^T||²`` at every swept ``eps`` from one
  matmul (:func:`flowcl.analysis.interference.projected_energies`);
* three aggregates: the equal-step mean of per-batch ``c_l`` (decides the verdict —
  each batch is one optimiser step), the energy-weighted ``c_l``, and ``c_l`` of the
  loss-weighted full-dataset gradient ``Σ_b n_b G_b / Σ_b n_b`` (``n_b`` = the batch's
  valid-element normaliser; an unweighted sum is *not* the full-data gradient).

Evidence that never decides: the T1 self-gradient control, the realised seq_ft update
``W_stage1 − W_stage0``, a paired replicate on the Gate 0 single-Spatial checkpoint,
and the isotropic RMS baseline ``sqrt(rho_l)`` (``E[c_l²] = rho_l`` for an isotropic
gradient direction).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from omegaconf import OmegaConf

from flowcl.analysis.gates import GateResult, gate3
from flowcl.analysis.interference import (
    assert_orthonormal,
    c_from_energies,
    energy_weighted_ratio,
    mean_ratio,
    projected_energies,
)
from flowcl.analysis.metrics import bootstrap_ci
from flowcl.analysis.subspace import SubspaceBasis, load_bases, validate_thresholds
from flowcl.data.curriculum import Curriculum
from flowcl.data.dataset import ChunkedActionDataset
from flowcl.models.flow_head import draw_with_generator
from flowcl.models.losses import valid_element_count
from flowcl.train.checkpoint import CHECKPOINT_FORMAT_VERSION, LoadedCheckpoint, load_checkpoint
from flowcl.train.trainer import build_dataloader, move_batch
from flowcl.utils.libero_paths import repo_root
from flowcl.utils.run import git_sha
from flowcl.utils.seeding import derive_seed

GENERATOR_ROLES = ("shuffle", "flow_time", "noise")


@dataclass
class InterferenceConfig:
    """``configs/analysis/interference.yaml``; see the comments there."""

    energy_thresholds: tuple[float, ...]
    default_eps: float
    batch_size: int
    n_batches: int | None
    num_workers: int
    n_demos: int | None
    seed_tags: dict
    bootstrap: dict

    def __post_init__(self) -> None:
        self.energy_thresholds = validate_thresholds(self.energy_thresholds)
        if self.default_eps not in self.energy_thresholds:
            raise ValueError(
                f"default_eps {self.default_eps} not among {self.energy_thresholds}"
            )
        if set(self.seed_tags) != set(GENERATOR_ROLES):
            raise ValueError(
                f"seed_tags must have exactly {GENERATOR_ROLES}, got {sorted(self.seed_tags)}"
            )
        if len(set(self.seed_tags.values())) != len(GENERATOR_ROLES):
            raise ValueError(f"seed_tags must be distinct, got {self.seed_tags}")
        if set(self.bootstrap) != {"n", "confidence", "seed"}:
            raise ValueError(
                f"bootstrap must have keys n, confidence, seed; got {sorted(self.bootstrap)}"
            )
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")

    @classmethod
    def from_dict(cls, payload: dict) -> "InterferenceConfig":
        known = set(cls.__dataclass_fields__)
        unknown, missing = sorted(set(payload) - known), sorted(known - set(payload))
        if unknown or missing:
            raise ValueError(
                f"interference config: unknown keys {unknown}, missing keys {missing}"
            )
        return cls(**payload)

    def as_dict(self) -> dict:
        out = asdict(self)
        out["energy_thresholds"] = list(self.energy_thresholds)
        return out


def load_interference_config(path: str | Path | None = None) -> InterferenceConfig:
    path = (
        Path(path)
        if path
        else repo_root() / "configs" / "analysis" / "interference.yaml"
    )
    if not path.is_file():
        raise FileNotFoundError(f"interference config not found: {path}")
    return InterferenceConfig.from_dict(
        OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    )


def batch_generators(cfg: InterferenceConfig, task_key: str) -> dict[str, torch.Generator]:
    """One independent CPU generator per role, seeded from ``(tag, data task)``.

    Independent so that DataLoader behaviour (workers, shuffling) cannot shift the
    flow-time or noise streams. No ``run_id`` in the key: every checkpoint measured on
    the same task sees the same batches, ``s`` and ``A_0``, which makes cross-checkpoint
    comparisons paired.
    """
    return {
        role: torch.Generator(device="cpu").manual_seed(
            derive_seed(cfg.seed_tags[role], task_key, 0)
        )
        for role in GENERATOR_ROLES
    }


# ---- results -------------------------------------------------------------------


@dataclass
class LayerInterference:
    """Per-batch energies for one layer, and the aggregates derived from them."""

    name: str
    group: str
    d_in: int
    ranks: dict[float, int]
    rhos: dict[float, float]
    total: list[float] = field(default_factory=list)
    parallel: dict[float, list[float]] = field(default_factory=dict)
    full_total: float = 0.0
    full_parallel: dict[float, float] = field(default_factory=dict)
    # Per-batch parallel energies against further basis sets, measured in the same pass
    # (label -> eps -> per batch). Empty unless ``extra_bases`` was given.
    extra_parallel: dict[str, dict[float, list[float]]] = field(default_factory=dict)

    def per_batch_c(self, eps: float, basis_set: str | None = None) -> list[float]:
        """Per-batch ``c_l`` against the main bases, or the extra set ``basis_set``."""
        parallel = self.parallel[eps] if basis_set is None else self.extra_parallel[basis_set][eps]
        return [c_from_energies(p, t, self.name) for p, t in zip(parallel, self.total)]

    def mean_c(self, eps: float) -> float:
        return mean_ratio(self.parallel[eps], self.total, self.name)

    def energy_c(self, eps: float) -> float:
        return energy_weighted_ratio(self.parallel[eps], self.total, self.name)

    def full_c(self, eps: float) -> float:
        return c_from_energies(self.full_parallel[eps], self.full_total, self.name)

    def rms_baseline(self, eps: float) -> float:
        """``sqrt(rho_l)``: RMS ``c_l`` of an isotropic gradient direction."""
        return math.sqrt(self.rhos[eps])

    def mean_grad_norm(self) -> float:
        return sum(math.sqrt(t) for t in self.total) / len(self.total)

    def row(self, cfg: InterferenceConfig, include_per_batch: bool = False) -> dict:
        eps0 = cfg.default_eps
        estimate = bootstrap_ci(
            self.per_batch_c(eps0),
            n_bootstrap=cfg.bootstrap["n"],
            confidence=cfg.bootstrap["confidence"],
            seed=cfg.bootstrap["seed"],
        )
        row = {
            "layer": self.name,
            "group": self.group,
            "d_in": self.d_in,
            "mean_grad_norm": self.mean_grad_norm(),
            "full_grad_norm": math.sqrt(self.full_total),
            "ci_low": estimate.low,
            "ci_high": estimate.high,
        }
        for eps in cfg.energy_thresholds:
            row[f"k@{eps}"] = self.ranks[eps]
            row[f"rho@{eps}"] = self.rhos[eps]
            row[f"rms_baseline@{eps}"] = self.rms_baseline(eps)
            row[f"c_mean@{eps}"] = self.mean_c(eps)
            row[f"c_energy@{eps}"] = self.energy_c(eps)
            row[f"c_full@{eps}"] = self.full_c(eps)
        if include_per_batch:
            row[f"c_per_batch@{eps0}"] = self.per_batch_c(eps0)
        return row


def half_of(name: str) -> str:
    if name.startswith("trunk."):
        return "trunk"
    if name.startswith("flow_head."):
        return "decoder"
    raise ValueError(f"cannot assign {name!r} to the trunk or the decoder")


@dataclass
class GradientInterference:
    """One measurement: one checkpoint, one basis, one task's data."""

    label: str
    checkpoint: str
    run_id: str
    basis_task_key: str
    data_task_key: str
    stats_fingerprint: str
    n_batches: int
    n_samples: int
    layers: dict[str, LayerInterference]
    wall_clock_s: float = 0.0
    # In memory only: per-batch s (for determinism tests) and, when requested, the
    # loss-weighted full-dataset gradients.
    s_trace: list[torch.Tensor] = field(default_factory=list, repr=False)
    full_gradients: dict[str, torch.Tensor] | None = field(default=None, repr=False)

    def mean_c(self, eps: float) -> dict[str, float]:
        return {n: layer.mean_c(eps) for n, layer in self.layers.items()}

    def global_c(self, eps: float) -> dict:
        """Share of the *whole* gradient hard projection would remove, overall and per half.

        ``energy`` pools every batch and layer; ``full`` uses the full-dataset gradients.
        """
        out = {}
        for scope in ("all", "trunk", "decoder"):
            members = [
                layer
                for n, layer in self.layers.items()
                if scope == "all" or half_of(n) == scope
            ]
            if not members:
                continue
            par = sum(sum(layer.parallel[eps]) for layer in members)
            tot = sum(sum(layer.total) for layer in members)
            fpar = sum(layer.full_parallel[eps] for layer in members)
            ftot = sum(layer.full_total for layer in members)
            out[scope] = {
                "energy": c_from_energies(par, tot, scope),
                "full": c_from_energies(fpar, ftot, scope),
            }
        return out

    def verdict(self, cfg: InterferenceConfig) -> GateResult:
        eps = cfg.default_eps
        rows = {n: layer.row(cfg) for n, layer in self.layers.items()}
        return gate3(
            self.mean_c(eps),
            groups={n: layer.group for n, layer in self.layers.items()},
            ci={n: (r["ci_low"], r["ci_high"]) for n, r in rows.items()},
            extra_layer_evidence={
                n: {
                    "c_energy": r[f"c_energy@{eps}"],
                    "c_full": r[f"c_full@{eps}"],
                    "rho": r[f"rho@{eps}"],
                    "rms_baseline": r[f"rms_baseline@{eps}"],
                    "c_over_rms_baseline": r[f"c_mean@{eps}"] / r[f"rms_baseline@{eps}"],
                    "mean_grad_norm": r["mean_grad_norm"],
                    "c_sweep": {
                        str(e): r[f"c_mean@{e}"] for e in cfg.energy_thresholds
                    },
                }
                for n, r in rows.items()
            },
            aggregate_evidence={"c_global": self.global_c(eps)},
            eps=eps,
            run_id=self.run_id,
        )

    def summary(self, cfg: InterferenceConfig, include_per_batch: bool = False) -> dict:
        return {
            "label": self.label,
            "checkpoint": self.checkpoint,
            "run_id": self.run_id,
            "basis_task_key": self.basis_task_key,
            "data_task_key": self.data_task_key,
            "stats_fingerprint": self.stats_fingerprint,
            "n_batches": self.n_batches,
            "n_samples": self.n_samples,
            "wall_clock_s": self.wall_clock_s,
            "c_global": {
                str(eps): self.global_c(eps) for eps in cfg.energy_thresholds
            },
            "layers": [
                layer.row(cfg, include_per_batch=include_per_batch)
                for layer in self.layers.values()
            ],
        }


# ---- measurement ---------------------------------------------------------------


def check_provenance(
    loaded: LoadedCheckpoint,
    bases: dict[str, SubspaceBasis],
    meta: dict,
    thresholds,
) -> None:
    """Refuse to pair a checkpoint with bases that were not measured on it."""
    problems = []
    if meta.get("run_id") != loaded.run_id:
        problems.append(f"run_id: bases {meta.get('run_id')!r} vs checkpoint {loaded.run_id!r}")
    if meta.get("task_idx") != loaded.stage:
        problems.append(f"task_idx: bases {meta.get('task_idx')} vs checkpoint stage {loaded.stage}")
    fingerprint = loaded.stats.fingerprint()
    if meta.get("stats_fingerprint") != fingerprint:
        problems.append(
            f"stats_fingerprint: bases {meta.get('stats_fingerprint')} vs checkpoint {fingerprint}"
        )
    registry = {e.name: e for e in loaded.policy.projectable_layers()}
    if list(bases) != list(registry):
        problems.append(
            f"layers differ: missing {sorted(set(registry) - set(bases))}, "
            f"extra {sorted(set(bases) - set(registry))}"
        )
    else:
        for name, basis in bases.items():
            if basis.d_in != registry[name].d_in:
                problems.append(f"{name}: basis d_in {basis.d_in} vs layer {registry[name].d_in}")
            absent = [t for t in thresholds if t not in basis.ranks]
            if absent:
                problems.append(f"{name}: basis lacks eps {absent}")
    if problems:
        raise ValueError(
            "Task-1 bases do not belong to this checkpoint:\n  " + "\n  ".join(problems)
        )


def measure_gradient_interference(
    loaded: LoadedCheckpoint,
    bases: dict[str, SubspaceBasis],
    meta: dict,
    dataset: ChunkedActionDataset,
    cfg: InterferenceConfig,
    device: str | torch.device = "cuda",
    label: str = "",
    checkpoint_path: Path | None = None,
    keep_full_gradient: bool = False,
    extra_bases: dict[str, dict[str, SubspaceBasis]] | None = None,
) -> GradientInterference:
    """Decompose one task's training gradients at ``loaded`` against ``bases``.

    No optimiser step. Every parameter's ``.grad`` is restored afterwards and the
    trainable weights are verified bit-identical.

    ``extra_bases``: further basis sets (label -> layer -> basis) to project the *same*
    per-batch gradients against, in the same pass; their per-batch parallel energies land
    in ``LayerInterference.extra_parallel[label]``. ``None`` leaves everything else as it
    was.
    """
    thresholds = cfg.energy_thresholds
    check_provenance(loaded, bases, meta, thresholds)
    for extra in (extra_bases or {}).values():
        check_provenance(loaded, extra, meta, thresholds)
    if len(dataset.task_ids) != 1:
        raise ValueError(f"expected a single-task dataset, got {dataset.task_ids}")
    data_task = dataset.task_ids[0]

    device = torch.device(device)
    policy = loaded.policy.to(device)
    policy.eval()
    entries = policy.projectable_layers()
    started = time.perf_counter()

    vectors, ranks, layers = {}, {}, {}
    for entry in entries:
        basis = bases[entry.name]
        layer_ranks = [basis.ranks[eps] for eps in thresholds]
        V = basis.vectors[:, : max(layer_ranks)].to(device=device, dtype=torch.float64)
        assert_orthonormal(V, entry.name)
        vectors[entry.name], ranks[entry.name] = V, layer_ranks
        layers[entry.name] = LayerInterference(
            name=entry.name,
            group=entry.group,
            d_in=entry.d_in,
            ranks={eps: basis.ranks[eps] for eps in thresholds},
            rhos={eps: basis.rhos[eps] for eps in thresholds},
            parallel={eps: [] for eps in thresholds},
            extra_parallel={
                set_label: {eps: [] for eps in thresholds} for set_label in (extra_bases or {})
            },
        )

    extra_vectors: dict[str, dict[str, torch.Tensor]] = {}
    extra_ranks: dict[str, dict[str, list[int]]] = {}
    for set_label, extra in (extra_bases or {}).items():
        extra_vectors[set_label], extra_ranks[set_label] = {}, {}
        for entry in entries:
            basis = extra[entry.name]
            layer_ranks = [basis.ranks[eps] for eps in thresholds]
            V = basis.vectors[:, : max(layer_ranks)].to(device=device, dtype=torch.float64)
            assert_orthonormal(V, entry.name)
            extra_vectors[set_label][entry.name] = V
            extra_ranks[set_label][entry.name] = layer_ranks

    params = dict(policy.named_parameters())
    saved_grads = {
        n: (p.grad.detach().clone() if p.grad is not None else None)
        for n, p in params.items()
    }
    weights_before = {n: p.detach().clone() for n, p in params.items() if p.requires_grad}

    generators = batch_generators(cfg, data_task)
    loader = build_dataloader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=True,
        generator=generators["shuffle"],
    )
    full = {
        e.name: torch.zeros(
            e.module.weight.shape, dtype=torch.float64, device=device
        )
        for e in entries
    }
    n_total, n_samples, n_batches = 0.0, 0, 0
    s_trace: list[torch.Tensor] = []

    try:
        for batch_idx, batch in enumerate(loader):
            if cfg.n_batches is not None and batch_idx >= cfg.n_batches:
                break
            batch = move_batch(batch, device)
            size = batch["actions"].shape[0]
            s = policy.s_sampler.sample(size, device, generator=generators["flow_time"])
            noise = draw_with_generator(
                tuple(batch["actions"].shape),
                device=device,
                generator=generators["noise"],
                dtype=torch.float32,
                normal=True,
            )
            s_trace.append(s.detach().cpu())

            policy.zero_grad(set_to_none=True)
            with torch.enable_grad(), torch.autocast(
                device_type=device.type, enabled=False
            ):
                loss = policy(batch, s=s, noise=noise)["loss"]
                loss.backward()
            n_b = float(valid_element_count(batch["action_mask"], policy.d_action))

            for entry in entries:
                G = entry.module.weight.grad
                if G is None:
                    raise RuntimeError(f"{entry.name}: weight received no gradient")
                total, parallel = projected_energies(
                    G, vectors[entry.name], ranks[entry.name], entry.name
                )
                layer = layers[entry.name]
                layer.total.append(total)
                for eps, p in zip(thresholds, parallel):
                    layer.parallel[eps].append(p)
                for set_label in extra_vectors:
                    _, extra_parallel = projected_energies(
                        G, extra_vectors[set_label][entry.name],
                        extra_ranks[set_label][entry.name], entry.name,
                    )
                    for eps, p in zip(thresholds, extra_parallel):
                        layer.extra_parallel[set_label][eps].append(p)
                # Loss-weighted: each batch's loss is a mean over its own n_b valid
                # elements, so n_b * G_b is that batch's share of the summed loss.
                full[entry.name].add_(G.to(torch.float64), alpha=n_b)
            n_total += n_b
            n_samples += size
            n_batches += 1
    finally:
        for n, p in params.items():
            p.grad = saved_grads[n]

    changed = [n for n, w in weights_before.items() if not torch.equal(w, params[n])]
    if changed:
        raise RuntimeError(f"weights changed during a measurement-only pass: {changed[:5]}")
    if n_batches == 0:
        raise ValueError("no batches were measured")

    for entry in entries:
        G_full = full[entry.name] / n_total
        total, parallel = projected_energies(
            G_full, vectors[entry.name], ranks[entry.name], entry.name
        )
        layers[entry.name].full_total = total
        layers[entry.name].full_parallel = dict(zip(thresholds, parallel))
        full[entry.name] = G_full

    return GradientInterference(
        label=label,
        checkpoint=str(checkpoint_path) if checkpoint_path else "<memory>",
        run_id=loaded.run_id,
        basis_task_key=meta.get("task_key", loaded.task_key),
        data_task_key=data_task,
        stats_fingerprint=loaded.stats.fingerprint(),
        n_batches=n_batches,
        n_samples=n_samples,
        layers=layers,
        wall_clock_s=time.perf_counter() - started,
        s_trace=s_trace,
        full_gradients=full if keep_full_gradient else None,
    )


# ---- realised update -----------------------------------------------------------


def update_interference(
    state_before: dict,
    state_after: dict,
    bases: dict[str, SubspaceBasis],
    thresholds,
) -> dict:
    """Decompose ``ΔW = W_after − W_before`` of every registry layer against ``bases``.

    This is what seq_ft actually did, including Adam's per-coordinate scaling (which
    does not preserve subspaces) and weight decay — so it can differ from the
    raw-gradient ``c_l``.
    """
    per_layer, par_sum, tot_sum = {}, {eps: 0.0 for eps in thresholds}, 0.0
    half_par = {h: {eps: 0.0 for eps in thresholds} for h in ("trunk", "decoder")}
    half_tot = {"trunk": 0.0, "decoder": 0.0}
    for name, basis in bases.items():
        key = f"{name}.weight"
        if key not in state_before or key not in state_after:
            raise KeyError(f"{key} missing from a checkpoint state_dict")
        delta = state_after[key].to(torch.float64) - state_before[key].to(torch.float64)
        layer_ranks = [basis.ranks[eps] for eps in thresholds]
        V = basis.vectors[:, : max(layer_ranks)].to(torch.float64)
        total, parallel = projected_energies(delta, V, layer_ranks, name)
        per_layer[name] = {
            "delta_norm": math.sqrt(total),
            # A layer that did not move (e.g. frozen by a full-rank projector) has no
            # interference ratio: recorded as None, not as a number.
            "c": {
                str(eps): (c_from_energies(p, total, name) if total > 0 else None)
                for eps, p in zip(thresholds, parallel)
            },
        }
        tot_sum += total
        half_tot[half_of(name)] += total
        for eps, p in zip(thresholds, parallel):
            par_sum[eps] += p
            half_par[half_of(name)][eps] += p
    present = sorted({half_of(name) for name in bases})

    def pooled(par: float, tot: float, scope: str) -> float | None:
        # Same rule as per layer: nothing moved means no ratio, not a number.
        return c_from_energies(par, tot, scope) if tot > 0 else None

    return {
        "per_layer": per_layer,
        "c_global": {
            str(eps): {
                "all": pooled(par_sum[eps], tot_sum, "all"),
                **{h: pooled(half_par[h][eps], half_tot[h], h) for h in present},
            }
            for eps in thresholds
        },
    }


def update_interference_from_checkpoints(
    before: Path, after: Path, bases: dict[str, SubspaceBasis], meta: dict, thresholds
) -> dict:
    """:func:`update_interference` between two stages of the same sequential run."""
    payloads = []
    for path in (before, after):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("version") != CHECKPOINT_FORMAT_VERSION:
            raise ValueError(f"{path}: unexpected checkpoint version {payload.get('version')}")
        payloads.append(payload)
    a, b = payloads
    problems = []
    if a["run_id"] != meta["run_id"] or b["run_id"] != meta["run_id"]:
        problems.append(f"run_id {a['run_id']} / {b['run_id']} vs bases {meta['run_id']}")
    if a["stage"] != meta["task_idx"] or b["stage"] != a["stage"] + 1:
        problems.append(f"stages {a['stage']} -> {b['stage']}, bases task_idx {meta['task_idx']}")
    if a["stats_fingerprint"] != b["stats_fingerprint"]:
        problems.append("stats fingerprints differ between stages (§3.3 violated)")
    if problems:
        raise ValueError("update decomposition inputs are inconsistent: " + "; ".join(problems))
    result = update_interference(a["state_dict"], b["state_dict"], bases, thresholds)
    result.update(
        {"before": str(before), "after": str(after), "stages": [a["stage"], b["stage"]]}
    )
    return result


# ---- the gate ------------------------------------------------------------------


@dataclass(frozen=True)
class Gate3Inputs:
    t1_checkpoint: Path
    t1_bases: Path
    stage1_checkpoint: Path
    t2_task_key: str
    replicate_checkpoint: Path
    replicate_bases: Path


def default_inputs(
    curriculum: Curriculum, seed: int = 0, results_root: Path | None = None
) -> Gate3Inputs:
    """The Gate 1 seq_ft stages 0/1, their T1 bases, and the single-Spatial replicate."""
    from flowcl.experiments.gate0 import single_task_run_id
    from flowcl.experiments.gate2 import bases_path
    from flowcl.train.continual import continual_run_id

    if len(curriculum.task_keys) < 2:
        raise ValueError("Gate 3 needs a curriculum with at least two tasks")
    root = Path(results_root) if results_root else repo_root() / "results"
    run_id = continual_run_id("seq_ft", curriculum.name, seed)
    replicate_id = single_task_run_id(curriculum.task_keys[0], seed)
    return Gate3Inputs(
        t1_checkpoint=root / run_id / "checkpoints" / "stage0.pt",
        t1_bases=bases_path(run_id, 0, root),
        stage1_checkpoint=root / run_id / "checkpoints" / "stage1.pt",
        t2_task_key=curriculum.task_keys[1],
        replicate_checkpoint=root / replicate_id / "checkpoints" / "final.pt",
        replicate_bases=bases_path(replicate_id, 0, root),
    )


@dataclass
class Gate3Report:
    result: GateResult
    t2: GradientInterference
    cfg: InterferenceConfig
    control: GradientInterference | None = None
    update: dict | None = None
    replicate: GradientInterference | None = None

    def comparisons(self) -> dict:
        eps = self.cfg.default_eps
        out: dict = {}
        t2_c = self.t2.mean_c(eps)
        if self.control is not None:
            ctrl_c = self.control.mean_c(eps)
            below = [n for n in t2_c if ctrl_c[n] < t2_c[n]]
            out["control_below_t2"] = {
                "note": (
                    "Layers where the T1 self-gradient c_l is below T2's. For inspection, "
                    "not a failure: T1 is already trained, so its residual gradients can "
                    "be small, noisy, or concentrated in low-energy directions."
                ),
                "layers": {
                    n: {
                        "c_t1": ctrl_c[n],
                        "c_t2": t2_c[n],
                        "t1_mean_grad_norm": self.control.layers[n].mean_grad_norm(),
                        "t2_mean_grad_norm": self.t2.layers[n].mean_grad_norm(),
                    }
                    for n in below
                },
            }
        if self.replicate is not None:
            rep = self.replicate.verdict(self.cfg)
            out["replicate_verdict"] = {
                "passed": rep.passed,
                "failing_halves": rep.evidence["failing_halves"],
                "per_half": {
                    h: {k: v[k] for k in ("n_blocked", "n_layers", "median_c")}
                    for h, v in rep.evidence["per_half"].items()
                },
                "same_verdict": rep.passed == self.result.passed,
            }
        return out

    def as_dict(self) -> dict:
        payload = self.result.as_dict()
        payload["measurement"] = {
            "config": self.cfg.as_dict(),
            "analysis_git_sha": git_sha(),
            "reading": (
                "Task-2 gradients at the start of T2: measured at the T1 checkpoint, "
                "no Task-2 training"
            ),
            "gradient": "registry weight gradients of the masked flow-matching loss",
            "mode": "policy.eval(), fp32, no autocast, no optimiser step",
            "aggregates": {
                "verdict": "equal-step mean of per-batch c_l",
                "c_energy": "sqrt(sum_b ||G_par||^2 / sum_b ||G||^2)",
                "c_full": "c_l of sum_b n_b G_b / sum_b n_b (n_b = valid elements)",
                "rms_baseline": "sqrt(rho_l), RMS c_l of an isotropic direction",
            },
            "generators": (
                "independent shuffle / flow_time / noise streams keyed on (tag, data "
                "task); identical across checkpoints, so the replicate is paired"
            ),
        }
        payload["t2"] = self.t2.summary(self.cfg, include_per_batch=True)
        payload["t1_control"] = self.control.summary(self.cfg) if self.control else None
        payload["update"] = self.update
        payload["replicate"] = self.replicate.summary(self.cfg) if self.replicate else None
        payload["comparisons"] = self.comparisons()
        return payload

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2) + "\n")
        return path


def _task_dataset(task_key: str, loaded: LoadedCheckpoint, cfg, dataset_dir):
    from flowcl.data.tasks import TaskRef
    from flowcl.train.pipeline import build_dataset

    # The checkpoint's own frozen stats (§3.3): for the sequential run these are the
    # Task-1 stats that Task 2 was trained under.
    return build_dataset(
        [TaskRef.from_key(task_key)],
        loaded.spec,
        loaded.stats,
        n_demos=cfg.n_demos,
        dataset_dir=dataset_dir,
    )


def run_gate3(
    inputs: Gate3Inputs,
    cfg: InterferenceConfig,
    device: str = "cuda",
    dataset_dir: Path | None = None,
    out_dir: Path | None = None,
    control: bool = True,
    replicate: bool = True,
) -> Gate3Report:
    """Measure Task-2 interference at the T1 checkpoint and record Gate 3."""
    required = [inputs.t1_checkpoint, inputs.t1_bases, inputs.stage1_checkpoint]
    if replicate:
        required += [inputs.replicate_checkpoint, inputs.replicate_bases]
    missing = [str(p) for p in required if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Gate 3 inputs not found (run Gates 1 and 2 first): {missing}"
        )

    def measure(checkpoint, bases_file, task_key, label):
        loaded = load_checkpoint(checkpoint, device=device)
        bases, meta = load_bases(bases_file)
        dataset = _task_dataset(task_key, loaded, cfg, dataset_dir)
        print(
            f"[flowcl] gate3 {label}: {loaded.run_id} stage {loaded.stage}, "
            f"{task_key} data ({len(dataset)} samples) vs {meta['task_key']} basis",
            flush=True,
        )
        result = measure_gradient_interference(
            loaded, bases, meta, dataset, cfg, device=device, label=label,
            checkpoint_path=checkpoint,
        )
        del dataset, loaded
        torch.cuda.empty_cache()
        return result, bases, meta

    t2, bases, meta = measure(inputs.t1_checkpoint, inputs.t1_bases, inputs.t2_task_key, "t2")
    control_result = (
        measure(inputs.t1_checkpoint, inputs.t1_bases, meta["task_key"], "t1_control")[0]
        if control
        else None
    )
    print("[flowcl] gate3 update: decomposing W_stage1 - W_stage0", flush=True)
    update = update_interference_from_checkpoints(
        inputs.t1_checkpoint, inputs.stage1_checkpoint, bases, meta, cfg.energy_thresholds
    )
    replicate_result = (
        measure(
            inputs.replicate_checkpoint, inputs.replicate_bases, inputs.t2_task_key,
            "replicate_t2",
        )[0]
        if replicate
        else None
    )

    verdict = t2.verdict(cfg)
    report = Gate3Report(
        result=verdict,
        t2=t2,
        cfg=cfg,
        control=control_result,
        update=update,
        replicate=replicate_result,
    )
    _print_summary(report)
    out_dir = Path(out_dir) if out_dir else repo_root() / "results" / "gate3"
    print(f"[flowcl] wrote {report.save(out_dir / 'gate3.json')}", flush=True)
    return report


def _print_summary(report: Gate3Report) -> None:
    cfg, eps = report.cfg, report.cfg.default_eps
    verdict = report.result
    ev = verdict.evidence
    print("\n" + verdict.describe(), flush=True)
    for half, stats in ev["per_half"].items():
        print(
            f"  {half:8s} blocked {stats['n_blocked']:2d}/{stats['n_layers']} "
            f"({100 * stats['blocked_fraction']:.1f}%), median c {stats['median_c']:.3f}",
            flush=True,
        )
    rms = {n: report.t2.layers[n].rms_baseline(eps) for n in report.t2.layers}
    print(f"  {'group':20s} {'median c':>9s} {'[min, max]':>16s} {'median sqrt(rho)':>17s}")
    for group, stats in ev["per_group"].items():
        members = [n for n, layer in report.t2.layers.items() if layer.group == group]
        baseline = sorted(rms[n] for n in members)[len(members) // 2]
        print(
            f"  {group:20s} {stats['median_c']:9.3f} "
            f"[{stats['min_c']:.3f}, {stats['max_c']:.3f}] {baseline:17.3f}",
            flush=True,
        )
    g = ev["c_global"]
    print(
        f"  c_global energy: all {g['all']['energy']:.3f}, trunk {g['trunk']['energy']:.3f}, "
        f"decoder {g['decoder']['energy']:.3f}",
        flush=True,
    )
    if report.control is not None:
        cg = report.control.global_c(eps)
        print(
            f"  T1 self-control c_global energy: all {cg['all']['energy']:.3f}, "
            f"trunk {cg['trunk']['energy']:.3f}, decoder {cg['decoder']['energy']:.3f}",
            flush=True,
        )
    if report.update is not None:
        ug = report.update["c_global"][str(eps)]
        print(
            f"  realised update dW c_global: all {ug['all']:.3f}, trunk {ug['trunk']:.3f}, "
            f"decoder {ug['decoder']:.3f}",
            flush=True,
        )
    if report.replicate is not None:
        rv = report.comparisons()["replicate_verdict"]
        print(
            f"  replicate (single Spatial): {'PASS' if rv['passed'] else 'FAIL'}, "
            + ", ".join(
                f"{h} blocked {v['n_blocked']}/{v['n_layers']}"
                for h, v in rv["per_half"].items()
            ),
            flush=True,
        )
