"""End-to-end "train a policy on some tasks" pipeline.

One function covers three things the spec needs, because they differ only in how many
tasks go into the dataset:

* Gate 0's single-task sanity run (§10.3),
* the §10.4 *independent single-task* reference for each task,
* the §10.4 *joint multi-task* reference over the union of a curriculum.

Sequential continual training is **not** this function — it is Phase 4's
:mod:`flowcl.train.continual`, which calls :func:`flowcl.train.trainer.train_one_task`
per stage while carrying one policy across stages.

The §3.3 stats rule is enforced here rather than left to callers: statistics come from
``refs[0]`` only, whatever else is in the dataset. For a joint reference that is not
an approximation of convenience, it is required — the joint run must normalise its
inputs identically to the sequential runs it is being compared against, or the
comparison measures normalisation rather than continual learning.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch
from omegaconf import OmegaConf

from flowcl.data.dataset import ChunkedActionDataset
from flowcl.data.spec import EmbodimentSpec
from flowcl.data.stats import NormalizationStats, compute_stats
from flowcl.data.tasks import TaskRef, load_task_episodes
from flowcl.models.build import build_policy, load_policy_config
from flowcl.models.policy import FlowPolicy
from flowcl.train.checkpoint import save_checkpoint
from flowcl.train.trainer import TrainConfig, TrainLog, train_one_task
from flowcl.utils.run import RunHandle, create_run
from flowcl.utils.seeding import derive_seed


@dataclass
class TrainedPolicy:
    """What a training run produced."""

    policy: FlowPolicy
    spec: EmbodimentSpec
    stats: NormalizationStats
    log: TrainLog
    run: RunHandle
    checkpoint: Path
    task_keys: tuple[str, ...]
    dataset_size: int


def build_dataset(
    refs: list[TaskRef] | tuple[TaskRef, ...],
    spec: EmbodimentSpec,
    stats: NormalizationStats,
    n_demos: int | None = None,
    dataset_dir: Path | None = None,
    verify: bool = True,
) -> ChunkedActionDataset:
    """Chunked dataset over the union of ``refs``."""
    episodes = []
    for ref in refs:
        ref.assert_consistent(dataset_dir)
        episodes.extend(
            load_task_episodes(
                ref, spec, n_demos=n_demos, dataset_dir=dataset_dir, verify=verify
            )
        )
    return ChunkedActionDataset(episodes, spec, stats)


def fit_stats(
    first: TaskRef,
    spec: EmbodimentSpec,
    n_demos: int | None = None,
    dataset_dir: Path | None = None,
) -> NormalizationStats:
    """Fit the §3.3 statistics on the curriculum's *first* task only."""
    episodes = load_task_episodes(
        first, spec, n_demos=n_demos, dataset_dir=dataset_dir
    )
    return compute_stats(
        episodes,
        embodiment=spec.name,
        task_id=first.task_key,
        # §3.2: LIBERO actions are already in [-1, 1]; record, never renormalise.
        normalize_actions=not spec.action.already_normalized,
    )


def train_on_tasks(
    refs: list[TaskRef] | tuple[TaskRef, ...],
    spec: EmbodimentSpec,
    policy_config: str | Path | dict,
    train_cfg: TrainConfig,
    run_id: str,
    seed: int | None = None,
    n_demos: int | None = None,
    dataset_dir: Path | None = None,
    results_root: Path | None = None,
    pretrained: bool = True,
    extra_config: dict | None = None,
    exist_ok: bool = False,
) -> TrainedPolicy:
    """Train one policy on the union of ``refs`` and checkpoint it.

    Args:
        refs: Tasks to train on. ``refs[0]`` supplies the normalization stats (§3.3).
        spec: Embodiment spec.
        policy_config: Name, path or dict for :func:`build_policy`.
        train_cfg: Optimisation settings.
        run_id: Run identity; the run registry directory is named after it (§2) and
            evaluation seeds derive from it (§8.3).
        seed: Resolved seed. ``None`` derives one from ``run_id`` so that a run is
            still reproducible from its own name.
        n_demos: Demos per task; ``None`` means all 50.
        pretrained: Load pretrained encoder weights. Tests pass False.
        extra_config: Merged into the stored ``config.yaml`` for provenance.

    Returns:
        A :class:`TrainedPolicy` whose checkpoint has already been written.
    """
    if not refs:
        raise ValueError("train_on_tasks received no tasks")

    resolved_seed = (
        int(seed) if seed is not None else derive_seed(run_id, "train", 0)
    )
    raw_policy_cfg = load_policy_config(policy_config)

    cfg_payload = {
        "run_id": run_id,
        "seed": resolved_seed,
        "embodiment": spec.to_dict(),
        "policy": raw_policy_cfg,
        "train": {
            k: v
            for k, v in vars(train_cfg).items()
            if not k.startswith("_")
        },
        "tasks": [ref.task_key for ref in refs],
        "data": {"n_demos": n_demos},
        **(extra_config or {}),
    }
    run = create_run(
        run_id=run_id,
        cfg=OmegaConf.create(cfg_payload),
        seed=resolved_seed,
        results_root=results_root,
        exist_ok=exist_ok,
    )

    torch.manual_seed(resolved_seed)
    stats = fit_stats(refs[0], spec, n_demos=n_demos, dataset_dir=dataset_dir)
    stats.save(run.artifact("stats.json"))

    dataset = build_dataset(
        refs, spec, stats, n_demos=n_demos, dataset_dir=dataset_dir
    )
    print(
        f"[flowcl] {run_id}: {len(dataset)} samples over "
        f"{dataset.n_episodes} demos from {len(refs)} task(s)",
        flush=True,
    )

    policy = build_policy(raw_policy_cfg, spec, pretrained=pretrained)
    report = policy.parameter_report()
    print(
        f"[flowcl] trainable {report['trainable'] / 1e6:.1f}M, "
        f"frozen {report['frozen'] / 1e6:.1f}M, "
        f"{report['registry_layers']} registry layers",
        flush=True,
    )

    generator = torch.Generator(device="cpu").manual_seed(resolved_seed)
    started = time.perf_counter()
    log = train_one_task(policy, dataset, train_cfg, generator=generator)
    log.wall_clock_s = time.perf_counter() - started

    checkpoint = save_checkpoint(
        run.subdir("checkpoints") / "final.pt",
        policy=policy,
        policy_config=raw_policy_cfg,
        spec=spec,
        stats=stats,
        run_id=run_id,
        stage=0,
        task_key=refs[0].task_key,
        extra={
            "task_keys": [ref.task_key for ref in refs],
            "final_loss": log.final_loss,
            "mean_last_50_loss": log.mean_last(50),
            "train_wall_clock_s": log.wall_clock_s,
            "parameter_report": report,
        },
    )

    return TrainedPolicy(
        policy=policy,
        spec=spec,
        stats=stats,
        log=log,
        run=run,
        checkpoint=checkpoint,
        task_keys=tuple(ref.task_key for ref in refs),
        dataset_size=len(dataset),
    )
