"""Per-step loss series and peak GPU memory (status.md §5 items 1-2): what each series measures,
the per-stage loss file, and its copy on resume."""

from __future__ import annotations

import json

import pytest
import torch

from flowcl.data.config import load_embodiment_spec
from flowcl.data.dataset import ChunkedActionDataset, collate_chunks
from flowcl.data.stats import compute_stats
from flowcl.envs.libero_env import EvalConfig
from flowcl.methods.base import BaseMethod
from flowcl.train.continual import run_continual
from flowcl.train.trainer import TrainConfig, stage_task_loss, train_one_task
from flowcl.utils.run import file_sha256
from test_continual import TINY_POLICY, fake_evaluation, two_task_curriculum, tiny_train_cfg  # noqa: F401
from test_low_update import KEYS, episodes, spec, tiny_policy  # noqa: F401


def test_stage_task_loss_uses_only_this_stages_samples_under_the_mask():
    velocity = torch.zeros(4, 2, 3)
    target = torch.ones(4, 2, 3)
    target[2:] = 3.0                                   # the two buffered samples err by 3, not 1
    mask = torch.ones(4, 2)
    mask[0, 1] = 0.0                                   # a padded step contributes nothing
    batch = {"task_id": ["new", "new", "old", "old"], "action_mask": mask}
    outputs = {"velocity": velocity, "target_velocity": target}
    assert stage_task_loss(outputs, batch, {"new"}) == pytest.approx(1.0)
    assert stage_task_loss(outputs, batch, {"old"}) == pytest.approx(9.0)
    assert stage_task_loss(outputs, batch, set()) == pytest.approx((3 + 6 + 6 * 9 + 6 * 9) / 21)  # every sample
    assert stage_task_loss(outputs, batch, {"absent"}) is None
    assert stage_task_loss({"loss": torch.tensor(1.0)}, batch, {"new"}) is None


class PenaltyMethod(BaseMethod):
    """modify_loss adds a constant, as EWC's penalty adds a term."""

    name = "penalty"

    def modify_loss(self, loss, batch, policy, *, outputs=None):
        return loss + 0.5


class MixingMethod(BaseMethod):
    """build_batch mixes in samples of another task, as replay does."""

    name = "mixing"

    def __init__(self, other: ChunkedActionDataset):
        super().__init__()
        self.other = other

    def build_batch(self, dataset, task_idx):
        return collate_chunks([dataset[0], self.other[0], self.other[1], dataset[1]])


def _dataset(spec, key):  # noqa: F811
    """The task's demos under T1's (KEYS[3]'s) frozen stats, as the runner builds every stage."""
    stats = compute_stats(episodes(spec, KEYS[3]), embodiment=spec.name, task_id=KEYS[3])
    return ChunkedActionDataset(episodes(spec, key), spec, stats)


def _cfg(**kw):
    return TrainConfig(steps=3, batch_size=4, warmup_steps=1, device="cpu", log_every=0, num_workers=0, **kw)


def test_seq_ft_series_agree_and_peaks_are_not_measured_on_cpu(spec):  # noqa: F811
    dataset = _dataset(spec, KEYS[3])
    log = train_one_task(tiny_policy(spec), dataset, _cfg(), generator=torch.Generator().manual_seed(0))
    assert len(log.losses) == len(log.batch_losses) == len(log.task_losses) == 3
    assert log.batch_losses == log.losses
    assert log.task_losses == pytest.approx(log.batch_losses, rel=1e-5)
    assert (log.peak_gpu_allocated_bytes, log.peak_gpu_reserved_bytes) == (None, None)


def test_a_loss_term_shows_in_loss_but_not_in_batch_loss(spec):  # noqa: F811
    dataset = _dataset(spec, KEYS[3])
    log = train_one_task(tiny_policy(spec), dataset, _cfg(), method=PenaltyMethod(),
                         generator=torch.Generator().manual_seed(0))
    assert [a - b for a, b in zip(log.losses, log.batch_losses)] == pytest.approx([0.5] * 3)


def test_mixed_batches_give_a_separate_new_task_curve(spec):  # noqa: F811
    dataset, other = _dataset(spec, KEYS[3]), _dataset(spec, KEYS[2])
    seen = []
    log = train_one_task(tiny_policy(spec), dataset, _cfg(), method=MixingMethod(other),
                         generator=torch.Generator().manual_seed(0),
                         on_step=lambda step, outputs: seen.append(outputs))
    assert log.task_losses != pytest.approx(log.batch_losses, rel=1e-3)
    assert all(v is not None and v > 0 for v in log.task_losses)


def test_the_runner_writes_a_loss_file_per_stage_and_resume_copies_it(
        dataset_dir, two_task_curriculum, tiny_train_cfg, tmp_path, monkeypatch):  # noqa: F811
    import flowcl.train.continual as continual_module

    monkeypatch.setattr(continual_module, "evaluate_tasks", fake_evaluation([]))
    spec_ = load_embodiment_spec("libero_franka")
    kwargs = dict(method_name="seq_ft", spec=spec_, policy_config=TINY_POLICY, train_cfg=tiny_train_cfg,
                  eval_cfg=EvalConfig(n_episodes=2), seed=0, dataset_dir=dataset_dir, pretrained=False)
    source = run_continual(two_task_curriculum, results_root=tmp_path / "a", **kwargs)
    payload = json.loads((tmp_path / "a" / source.run_id / "result.json").read_text())
    for stage, record in enumerate(payload["stages"]):
        entry = record["loss_file"]
        path = tmp_path / "a" / source.run_id / entry["path"]
        assert entry["path"] == f"train/stage{stage}_losses.json" and file_sha256(path) == entry["sha256"]
        data = json.loads(path.read_text())
        assert data["steps"] == tiny_train_cfg.steps and len(data["task_loss"]) == tiny_train_cfg.steps
        assert set(record["peak_gpu"]) == {"train_allocated_bytes", "train_reserved_bytes",
                                           "eval_allocated_bytes", "eval_reserved_bytes"}
    assert payload["systems"]["peak_gpu"]["train_allocated_bytes"] is None  # CPU: not measured, not 0
    assert not list((tmp_path / "a" / source.run_id / "eval").glob("*losses*"))

    resumed = run_continual(two_task_curriculum, results_root=tmp_path / "b",
                            resume_run=tmp_path / "a" / source.run_id, start_stage=1, **kwargs)
    copied = tmp_path / "b" / resumed.run_id / "train" / "stage0_losses.json"
    assert copied.is_file() and file_sha256(copied) == payload["stages"][0]["loss_file"]["sha256"]
    assert (tmp_path / "b" / resumed.run_id / "train" / "stage1_losses.json").is_file()
