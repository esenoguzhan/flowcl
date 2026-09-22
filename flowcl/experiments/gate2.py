"""Gate 2 (§10.3): is projection geometrically plausible?

Threshold: ``rho_l`` after Task 1 must not already be ~1 on most registry layers. If
Task 1's input subspace already fills a layer, hard projection leaves that layer no
free directions and a GPM/SGP comparison would be measuring frozen layers, not
protected ones.

Post-hoc and training-free. Per checkpoint:

1. one diagnostic forward/backward — the gradient-reachability probe
   (:func:`flowcl.analysis.hooks.probe_policy_reachability`) — decides, per
   action-position layer, whether padded chunk positions carry ``δ ≠ 0`` and so belong
   in the basis. No optimiser step; ``.grad`` is restored;
2. forward-only passes over every sample of the checkpoint's own training task, with
   :class:`~flowcl.analysis.hooks.ActivationCapture` streaming float64 Grams;
3. :func:`flowcl.analysis.subspace.basis_from_gram` per layer and view.

Measurement choices (also written into the gate JSON): ``policy.eval()``, fp32 with no
autocast, ``s`` from the policy's own training sampler and ``A_0 ~ N(0, I)`` from a CPU
generator derived from ``(run_id, task)``, the checkpoint's own frozen stats (§3.3,
never refitted), uncentered activations.

The verdict is read from the ``seq_ft`` stage-0 checkpoint of the Gate 1 run — exactly
the Task-1 state GPM would protect. The Gate 0 single-task checkpoints are measured the
same way and reported as robustness evidence only.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import torch
from omegaconf import OmegaConf

from flowcl.analysis.gates import GateResult, gate2
from flowcl.analysis.hooks import (
    VIEW_ALL,
    VIEW_VALID,
    ActivationCapture,
    expected_reachability,
    probe_policy_reachability,
)
from flowcl.analysis.subspace import (
    SubspaceBasis,
    basis_from_gram,
    save_bases,
    validate_thresholds,
)
from flowcl.data.curriculum import Curriculum
from flowcl.data.dataset import ChunkedActionDataset, collate_chunks
from flowcl.models.flow_head import S_BINS, draw_with_generator
from flowcl.train.checkpoint import LoadedCheckpoint, load_checkpoint
from flowcl.train.trainer import build_dataloader, move_batch
from flowcl.utils.libero_paths import repo_root
from flowcl.utils.run import git_sha
from flowcl.utils.seeding import derive_seed


@dataclass
class SubspaceConfig:
    """``configs/analysis/subspace.yaml``; see the comments there."""

    energy_thresholds: tuple[float, ...]
    default_eps: float
    tokens_per_sample: int
    min_samples_per_dim: float
    subsample_seed: int
    s_bin_edges: tuple[float, ...]
    tag_s_bins: bool
    neg_tol: float
    rank_tol: float | None
    batch_size: int
    num_workers: int
    n_demos: int | None
    probe_batch_size: int

    def __post_init__(self) -> None:
        self.energy_thresholds = validate_thresholds(self.energy_thresholds)
        if self.default_eps not in self.energy_thresholds:
            raise ValueError(
                f"default_eps {self.default_eps} is not among the swept thresholds "
                f"{self.energy_thresholds}"
            )
        self.s_bin_edges = tuple(float(e) for e in self.s_bin_edges)
        expected = tuple([low for low, _ in S_BINS] + [S_BINS[-1][1]])
        if self.s_bin_edges != expected:
            raise ValueError(
                f"s_bin_edges {self.s_bin_edges} disagree with flow_head.S_BINS "
                f"{expected}; §7.5 bins must be defined in one place"
            )
        if self.probe_batch_size < 2:
            raise ValueError("probe_batch_size must be >= 2 (padded + valid samples)")

    @classmethod
    def from_dict(cls, payload: dict) -> "SubspaceConfig":
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(payload) - known)
        missing = sorted(known - set(payload))
        if unknown or missing:
            raise ValueError(
                f"subspace config: unknown keys {unknown}, missing keys {missing}"
            )
        return cls(**payload)

    def as_dict(self) -> dict:
        out = asdict(self)
        out["energy_thresholds"] = list(self.energy_thresholds)
        out["s_bin_edges"] = list(self.s_bin_edges)
        return out


def load_subspace_config(path: str | Path | None = None) -> SubspaceConfig:
    path = Path(path) if path else repo_root() / "configs" / "analysis" / "subspace.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"subspace config not found: {path}")
    return SubspaceConfig.from_dict(
        OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    )


@dataclass
class LayerSubspace:
    """One layer's measured bases.

    ``views`` always has ``all``; action-position layers also have ``valid``. For other
    layers there is no masking, so ``valid`` would be the same Gram and is not stored.
    """

    name: str
    group: str
    kind: str
    d_in: int
    reachable: bool | None
    views: dict[str, SubspaceBasis]
    type_counts: dict[str, dict[str, int]]

    @property
    def primary_view(self) -> str:
        """``valid`` only when the probe showed padded positions carry no gradient."""
        return VIEW_VALID if self.reachable is False else VIEW_ALL

    @property
    def primary(self) -> SubspaceBasis:
        return self.views[self.primary_view]

    def row(self, thresholds) -> dict:
        row = {
            "layer": self.name,
            "group": self.group,
            "kind": self.kind,
            "d_in": self.d_in,
            "reachable": self.reachable,
            "primary_view": self.primary_view,
            "n_primary": self.primary.n_samples,
            "n_primary_per_dim": self.primary.samples_per_dim,
            "numerical_rank": self.primary.numerical_rank,
            "token_counts": self.type_counts,
        }
        for view, basis in self.views.items():
            row[f"n_{view}"] = basis.n_samples
            for eps in thresholds:
                row[f"k_{view}@{eps}"] = basis.ranks[eps]
                row[f"rho_{view}@{eps}"] = basis.rhos[eps]
        for eps in thresholds:
            row[f"k@{eps}"] = self.primary.ranks[eps]
            row[f"rho@{eps}"] = self.primary.rhos[eps]
        return row


@dataclass
class CheckpointSubspace:
    """Everything Gate 2 measured on one checkpoint."""

    checkpoint: Path
    run_id: str
    stage: int
    task_key: str
    stats_fingerprint: str
    n_dataset_samples: int
    layers: dict[str, LayerSubspace]
    reachability: dict[str, bool]
    reachability_expected: dict[str, bool]
    wall_clock_s: float = 0.0
    # Per-s-bin Grams when tag_s_bins; in memory only (Gate 4 consumes them).
    binned: dict[str, dict] = field(default_factory=dict)

    @property
    def reachability_mismatches(self) -> list[str]:
        return [
            name
            for name, value in self.reachability.items()
            if self.reachability_expected.get(name) != value
        ]

    def rhos(self, eps: float, view: str = "primary") -> dict[str, float]:
        out = {}
        for name, layer in self.layers.items():
            if view == "primary":
                out[name] = layer.primary.rhos[eps]
            else:
                # Unmasked layers have one Gram: valid == all by definition.
                out[name] = layer.views.get(view, layer.views[VIEW_ALL]).rhos[eps]
        return out

    def verdict(self, cfg: SubspaceConfig) -> GateResult:
        eps = cfg.default_eps
        return gate2(
            self.rhos(eps),
            groups={n: l.group for n, l in self.layers.items()},
            d_in={n: l.d_in for n, l in self.layers.items()},
            alternative_rhos={
                VIEW_VALID: self.rhos(eps, VIEW_VALID),
                VIEW_ALL: self.rhos(eps, VIEW_ALL),
            },
            eps=eps,
            run_id=self.run_id,
        )

    def summary(self, cfg: SubspaceConfig) -> dict:
        return {
            "checkpoint": str(self.checkpoint),
            "run_id": self.run_id,
            "stage": self.stage,
            "task_key": self.task_key,
            "stats_fingerprint": self.stats_fingerprint,
            "n_dataset_samples": self.n_dataset_samples,
            "wall_clock_s": self.wall_clock_s,
            "reachability": self.reachability,
            "reachability_expected": self.reachability_expected,
            "reachability_mismatches": self.reachability_mismatches,
            "layers": [
                layer.row(cfg.energy_thresholds) for layer in self.layers.values()
            ],
        }


def probe_indices(dataset: ChunkedActionDataset, n: int) -> list[int]:
    """Deterministic probe batch: half padded-chunk samples, half fully valid ones."""
    padded, full = [], []
    for i in range(len(dataset)):
        idx = dataset.sample_index(i)
        length = dataset.episodes[idx.episode_idx].length
        (padded if idx.t + dataset.horizon > length else full).append(i)
    n_padded = min(len(padded), n - n // 2)
    n_full = min(len(full), n - n_padded)
    if n_padded == 0:
        raise ValueError("dataset has no padded samples; the probe needs some")
    return padded[:n_padded] + full[:n_full]


def collect_bases(
    loaded: LoadedCheckpoint,
    dataset: ChunkedActionDataset,
    cfg: SubspaceConfig,
    device: str | torch.device = "cuda",
    checkpoint_path: Path | None = None,
) -> CheckpointSubspace:
    """Probe reachability, capture Grams over ``dataset``, and build every basis."""
    device = torch.device(device)
    policy = loaded.policy.to(device)
    policy.eval()
    run_id = loaded.run_id
    task_key = loaded.task_key
    if run_id is None or task_key is None:
        raise ValueError("checkpoint lacks run_id/task_key; cannot tag its bases")
    started = time.perf_counter()

    # 1. Gradient reachability of padded action positions.
    probe_batch = move_batch(
        collate_chunks(
            [dataset[i] for i in probe_indices(dataset, cfg.probe_batch_size)]
        ),
        device,
    )
    probe_generator = torch.Generator(device="cpu").manual_seed(
        derive_seed(f"gate2_probe::{run_id}", task_key, 0)
    )
    reachability = probe_policy_reachability(policy, probe_batch, probe_generator)
    expected = expected_reachability(
        policy.registry_names(), len(policy.flow_head.blocks)
    )
    del probe_batch

    # 2. Forward-only capture over the whole task dataset, in a fixed order.
    binned_views = (
        {n: (VIEW_ALL if r else VIEW_VALID) for n, r in reachability.items()}
        if cfg.tag_s_bins
        else None
    )
    capture = ActivationCapture.for_policy(
        policy,
        tokens_per_sample=cfg.tokens_per_sample,
        subsample_seed=cfg.subsample_seed,
        s_bin_edges=cfg.s_bin_edges if cfg.tag_s_bins else None,
        binned_views=binned_views,
    )
    generator = torch.Generator(device="cpu").manual_seed(
        derive_seed(f"gate2_capture::{run_id}", task_key, 0)
    )
    loader = build_dataloader(
        dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=False
    )
    with capture:
        for batch in loader:
            batch = move_batch(batch, device)
            size = batch["actions"].shape[0]
            s = policy.s_sampler.sample(size, device, generator=generator)
            noise = draw_with_generator(
                tuple(batch["actions"].shape),
                device=device,
                generator=generator,
                dtype=torch.float32,
                normal=True,
            )
            capture.forward_policy(policy, batch, s, noise)

    # 3. Sample-count check on the view each basis will actually use, then bases.
    groups = {entry.name: entry.group for entry in policy.projectable_layers()}
    too_few = []
    for name, acc in capture.accumulators.items():
        view = VIEW_VALID if reachability.get(name) is False else VIEW_ALL
        ratio = acc.n[view] / acc.d_in
        if ratio < cfg.min_samples_per_dim:
            too_few.append(f"{name} ({view}): N={acc.n[view]}, N/d={ratio:.2f}")
    if too_few:
        raise RuntimeError(
            f"§7.1: fewer than {cfg.min_samples_per_dim} samples per input dimension "
            f"on {len(too_few)} layer(s); bases from too few samples are garbage:\n  "
            + "\n  ".join(too_few)
        )

    layers: dict[str, LayerSubspace] = {}
    binned: dict[str, dict] = {}
    for name, acc in capture.accumulators.items():
        views = {
            view: basis_from_gram(
                acc.gram[view],
                layer=name,
                n_samples=acc.n[view],
                thresholds=cfg.energy_thresholds,
                neg_tol=cfg.neg_tol,
                rank_tol=cfg.rank_tol,
            )
            for view in acc.views
        }
        layers[name] = LayerSubspace(
            name=name,
            group=groups[name],
            kind=acc.kind,
            d_in=acc.d_in,
            reachable=reachability.get(name),
            views=views,
            type_counts={v: acc.type_count_dict(v) for v in acc.views},
        )
        if acc.binned_gram is not None:
            binned[name] = {
                "view": acc.binned_view,
                "grams": acc.binned_gram,
                "n": acc.binned_n,
            }
        acc.gram.clear()  # free the float64 Grams as soon as they are decomposed

    return CheckpointSubspace(
        checkpoint=Path(checkpoint_path) if checkpoint_path else Path("<memory>"),
        run_id=run_id,
        stage=int(loaded.stage or 0),
        task_key=task_key,
        stats_fingerprint=loaded.stats.fingerprint(),
        n_dataset_samples=len(dataset),
        layers=layers,
        reachability=reachability,
        reachability_expected=expected,
        wall_clock_s=time.perf_counter() - started,
        binned=binned,
    )


def bases_path(run_id: str, stage: int, results_root: Path | None = None) -> Path:
    """§7.2: bases persist per ``(run_id, task_idx)``; every layer in one file."""
    root = Path(results_root) if results_root else repo_root() / "results"
    return root / run_id / "bases" / f"task{stage}.pt"


def save_checkpoint_bases(
    result: CheckpointSubspace,
    cfg: SubspaceConfig,
    results_root: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Write the primary basis of every layer, plus provenance and alternative views."""
    path = bases_path(result.run_id, result.stage, results_root)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists. Bases are artifacts; pass overwrite=True "
            "(--overwrite) to replace them deliberately."
        )
    bases = {}
    for name, layer in result.layers.items():
        basis = layer.primary
        basis.meta = {
            "group": layer.group,
            "kind": layer.kind,
            "primary_view": layer.primary_view,
            "reachable": layer.reachable,
            "token_counts": layer.type_counts,
            "views": {v: b.summary() for v, b in layer.views.items()},
        }
        bases[name] = basis
    meta = {
        "run_id": result.run_id,
        "task_idx": result.stage,
        "task_key": result.task_key,
        "checkpoint": str(result.checkpoint),
        "stats_fingerprint": result.stats_fingerprint,
        "analysis_git_sha": git_sha(),
        "created": datetime.now(timezone.utc).isoformat(),
        "config": cfg.as_dict(),
        "reachability": result.reachability,
        "centered": False,
        "dtype": "float64",
    }
    return save_bases(path, bases, meta)


def default_checkpoints(
    curriculum: Curriculum, seed: int = 0, results_root: Path | None = None
) -> tuple[Path, list[Path]]:
    """The Gate 1 ``seq_ft`` stage-0 checkpoint, plus the Gate 0 single-task references."""
    from flowcl.experiments.gate0 import single_task_run_id
    from flowcl.train.continual import continual_run_id

    root = Path(results_root) if results_root else repo_root() / "results"
    verdict = (
        root
        / continual_run_id("seq_ft", curriculum.name, seed)
        / "checkpoints"
        / "stage0.pt"
    )
    references = [
        root / single_task_run_id(key, seed) / "checkpoints" / "final.pt"
        for key in curriculum.task_keys
    ]
    return verdict, references


@dataclass
class Gate2Report:
    result: GateResult
    verdict: CheckpointSubspace
    references: list[CheckpointSubspace]
    cfg: SubspaceConfig
    bases_files: list[Path]

    def as_dict(self) -> dict:
        payload = self.result.as_dict()
        payload["measurement"] = {
            "config": self.cfg.as_dict(),
            "analysis_git_sha": git_sha(),
            "activations": "uncentered registry-layer inputs, fp32 forward, float64 Gram",
            "mode": "policy.eval(), torch.no_grad(), no autocast",
            "s_and_noise": "policy.s_sampler and N(0, I), CPU generator from derive_seed",
            "primary_view": (
                "gradient-reachable tokens: all positions where the probe found padded "
                "rows with nonzero output gradient, valid positions otherwise"
            ),
        }
        payload["verdict_checkpoint"] = self.verdict.summary(self.cfg)
        payload["references"] = {
            ref.run_id: {
                "hypothetical_verdict": ref.verdict(self.cfg).as_dict(),
                **ref.summary(self.cfg),
            }
            for ref in self.references
        }
        payload["bases_files"] = [str(p) for p in self.bases_files]
        return payload

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2) + "\n")
        return path


def measure_checkpoint(
    checkpoint: Path,
    cfg: SubspaceConfig,
    device: str,
    dataset_dir: Path | None = None,
) -> CheckpointSubspace:
    """Load a checkpoint, build its own task's dataset with its own stats, measure."""
    from flowcl.data.tasks import TaskRef
    from flowcl.train.pipeline import build_dataset

    loaded = load_checkpoint(checkpoint, device=device)
    dataset = build_dataset(
        [TaskRef.from_key(loaded.task_key)],
        loaded.spec,
        loaded.stats,
        n_demos=cfg.n_demos,
        dataset_dir=dataset_dir,
    )
    print(
        f"[flowcl] gate2: {loaded.run_id} stage {loaded.stage} on {loaded.task_key} "
        f"({len(dataset)} samples)",
        flush=True,
    )
    result = collect_bases(loaded, dataset, cfg, device=device, checkpoint_path=checkpoint)
    if result.reachability_mismatches:
        print(
            "[flowcl] NOTE: measured reachability differs from the hand-derived rule "
            f"on {result.reachability_mismatches}; the measured rule is used",
            flush=True,
        )
    return result


def run_gate2(
    verdict_checkpoint: Path,
    reference_checkpoints: list[Path],
    cfg: SubspaceConfig,
    device: str = "cuda",
    dataset_dir: Path | None = None,
    results_root: Path | None = None,
    out_dir: Path | None = None,
    overwrite: bool = False,
) -> Gate2Report:
    """Measure ``rho_l`` on the verdict checkpoint (and references) and record Gate 2."""
    missing = [
        str(p) for p in [verdict_checkpoint, *reference_checkpoints] if not Path(p).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Gate 2 checkpoints not found (run Gate 0 / Gate 1 first, or pass "
            f"--no-references): {missing}"
        )

    bases_files = []
    verdict_result = measure_checkpoint(verdict_checkpoint, cfg, device, dataset_dir)
    bases_files.append(
        save_checkpoint_bases(verdict_result, cfg, results_root, overwrite=overwrite)
    )
    references = []
    for path in reference_checkpoints:
        ref = measure_checkpoint(path, cfg, device, dataset_dir)
        bases_files.append(save_checkpoint_bases(ref, cfg, results_root, overwrite=overwrite))
        references.append(ref)
        torch.cuda.empty_cache()

    verdict = verdict_result.verdict(cfg)
    report = Gate2Report(
        result=verdict,
        verdict=verdict_result,
        references=references,
        cfg=cfg,
        bases_files=bases_files,
    )

    print("\n" + verdict.describe(), flush=True)
    ev = verdict.evidence
    print(
        f"  saturated {ev['n_saturated']}/{ev['n_layers']} "
        f"({100 * ev['saturated_fraction']:.1f}%) at eps={ev['eps']}",
        flush=True,
    )
    for group, stats in ev["per_group"].items():
        print(
            f"  {group:20s} n={stats['n_layers']:3d} median rho "
            f"{stats['median_rho']:.3f} [{stats['min_rho']:.3f}, "
            f"{stats['max_rho']:.3f}] saturated {stats['n_saturated']}",
            flush=True,
        )
    for ref in references:
        rv = ref.verdict(cfg).evidence
        print(
            f"  reference {ref.run_id}: saturated {rv['n_saturated']}/{rv['n_layers']}",
            flush=True,
        )

    out_dir = Path(out_dir) if out_dir else repo_root() / "results" / "gate2"
    print(f"[flowcl] wrote {report.save(out_dir / 'gate2.json')}")
    return report
