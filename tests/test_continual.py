"""The sequential runner: §3.3 stage-boundary assertion, §6 hook order, §8.2 matrix."""

from __future__ import annotations

import pytest
import torch

from flowcl.data.config import load_embodiment_spec
from flowcl.data.curriculum import load_curriculum
from flowcl.envs.libero_env import EvalConfig
from flowcl.methods.base import BaseMethod
from flowcl.train.continual import continual_run_id, run_continual
from flowcl.train.trainer import TrainConfig

MILK = "libero_object/pick_up_the_milk_and_place_it_in_the_basket"
SAUCE = "libero_object/pick_up_the_tomato_sauce_and_place_it_in_the_basket"

# Smallest legal policy under §4.2 (d_model 384-512, 6-8 trunk layers), trimmed
# elsewhere so a two-stage run finishes in seconds on CPU.
TINY_POLICY = {
    "d_model": 384,
    "n_trunk_layers": 6,
    "n_heads": 8,
    "n_decoder_layers": 2,
    "n_context_tokens": 4,
    "pretrained": False,
    "euler_steps": 2,
}


@pytest.fixture(scope="module")
def two_task_curriculum():
    return load_curriculum(
        {
            "name": "test_pair",
            "tasks": [
                {"task_key": MILK, "n_demos": 1},
                {"task_key": SAUCE, "n_demos": 1},
            ],
        }
    )


@pytest.fixture
def tiny_train_cfg():
    return TrainConfig(
        steps=2,
        batch_size=2,
        num_workers=0,
        device="cpu",
        log_every=0,
        warmup_steps=1,
    )


class RecordingMethod(BaseMethod):
    """Records the order and arguments of every §6 hook call."""

    name = "recording"

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple] = []

    def on_task_start(self, task_idx, policy, dataset):
        self.calls.append(("on_task_start", task_idx))

    def build_batch(self, dataset, task_idx):
        self.calls.append(("build_batch", task_idx))
        return None

    def modify_loss(self, loss, batch, policy, *, outputs=None):
        self.calls.append(("modify_loss", outputs is not None))
        return loss

    def modify_gradients(self, policy, batch_meta):
        self.calls.append(("modify_gradients", batch_meta["task_idx"]))

    def on_task_end(self, task_idx, policy, dataset):
        self.calls.append(("on_task_end", task_idx))


def test_run_id_encodes_everything_that_changes_the_result():
    assert continual_run_id("gpm", "seq_hetero", 2) == "seq_hetero__gpm__seed2"


# ---- §3.3: stats fitted on task 1 and asserted at every boundary ---------------


def test_stats_are_fitted_on_task_one_and_checked_at_every_stage(
    dataset_dir, two_task_curriculum, tiny_train_cfg, tmp_path, monkeypatch
):
    """§3.3's guarantee, verified at the call site rather than by reading the code."""
    import flowcl.train.continual as continual_module

    seen = []
    real = continual_module.assert_frozen

    def spy(stats, embodiment, first_task_id, expected_fingerprint=None):
        seen.append((first_task_id, stats.fingerprint(), expected_fingerprint))
        return real(stats, embodiment, first_task_id, expected_fingerprint)

    monkeypatch.setattr(continual_module, "assert_frozen", spy)

    result = run_continual(
        two_task_curriculum,
        method_name="seq_ft",
        spec=load_embodiment_spec("libero_franka"),
        policy_config=TINY_POLICY,
        train_cfg=tiny_train_cfg,
        eval_cfg=EvalConfig(n_episodes=1),
        seed=0,
        dataset_dir=dataset_dir,
        results_root=tmp_path,
        pretrained=False,
        evaluate=False,
    )

    # One assertion per stage, not just one at the start.
    assert len(seen) == len(two_task_curriculum)
    # Always against the first task, and always the same fingerprint.
    assert {entry[0] for entry in seen} == {MILK}
    assert len({entry[1] for entry in seen}) == 1
    assert seen[0][1] == seen[0][2], "the stage-1 fingerprint must be pinned"

    stats = result.run.artifact("stats.json")
    assert stats.is_file()


def test_stage_boundary_assertion_actually_fires_on_refitted_stats(
    dataset_dir, two_task_curriculum
):
    """The guard must reject stats fitted on a later task, not just record them."""
    from flowcl.data.stats import assert_frozen
    from flowcl.train.pipeline import fit_stats

    spec = load_embodiment_spec("libero_franka")
    refitted = fit_stats(
        two_task_curriculum.stages[1].ref, spec, n_demos=1, dataset_dir=dataset_dir
    )
    with pytest.raises(ValueError, match="§3.3 violation"):
        assert_frozen(
            refitted,
            embodiment=spec.name,
            first_task_id=two_task_curriculum.first_task_key,
        )


# ---- §6: one code path for every method ---------------------------------------


def test_every_hook_is_called_once_per_stage_in_order(
    dataset_dir, two_task_curriculum, tiny_train_cfg, tmp_path, monkeypatch
):
    from flowcl.methods import base as base_module

    recorder = RecordingMethod()
    monkeypatch.setitem(base_module.METHOD_REGISTRY, "recording", lambda **kw: recorder)

    run_continual(
        two_task_curriculum,
        method_name="recording",
        spec=load_embodiment_spec("libero_franka"),
        policy_config=TINY_POLICY,
        train_cfg=tiny_train_cfg,
        eval_cfg=EvalConfig(n_episodes=1),
        seed=0,
        dataset_dir=dataset_dir,
        results_root=tmp_path,
        pretrained=False,
        evaluate=False,
    )

    names = [call[0] for call in recorder.calls]
    assert names.count("on_task_start") == 2
    assert names.count("on_task_end") == 2
    # Per stage: start, then (build_batch, modify_loss) x steps, then end.
    assert names[0] == "on_task_start"
    assert names[-1] == "on_task_end"

    # modify_gradients runs once per optimiser step, with the stage index attached.
    grad_stages = [call[1] for call in recorder.calls if call[0] == "modify_gradients"]
    assert grad_stages == [0, 0, 1, 1]

    # The documented deviation: modify_loss receives the forward outputs ConSFT needs.
    assert all(call[1] is True for call in recorder.calls if call[0] == "modify_loss")


def test_one_policy_is_carried_across_stages(
    dataset_dir, two_task_curriculum, tiny_train_cfg, tmp_path
):
    """Continual learning requires stage 2 to start from stage 1's weights.

    A runner that rebuilt the policy per stage would report zero forgetting for every
    method, since nothing would be shared to forget.
    """
    from flowcl.train.checkpoint import load_checkpoint

    result = run_continual(
        two_task_curriculum,
        method_name="seq_ft",
        spec=load_embodiment_spec("libero_franka"),
        policy_config=TINY_POLICY,
        train_cfg=tiny_train_cfg,
        eval_cfg=EvalConfig(n_episodes=1),
        seed=0,
        dataset_dir=dataset_dir,
        results_root=tmp_path,
        pretrained=False,
        evaluate=False,
    )

    stage0 = load_checkpoint(result.stages[0].checkpoint).policy.state_dict()
    stage1 = load_checkpoint(result.stages[1].checkpoint).policy.state_dict()

    # The frozen encoder must be bit-identical across stages (§0).
    frozen = [k for k in stage0 if k.startswith("vision_encoder.backbone")]
    assert frozen, "expected frozen vision backbone weights in the checkpoint"
    for key in frozen:
        torch.testing.assert_close(stage0[key], stage1[key])

    # The trunk must have moved: stage 1 trained on top of stage 0.
    trunk_key = "trunk.blocks.0.attn.q_proj.weight"
    assert not torch.equal(stage0[trunk_key], stage1[trunk_key])


def test_checkpoints_record_their_stage_and_task(
    dataset_dir, two_task_curriculum, tiny_train_cfg, tmp_path
):
    """The retention matrix is indexed by these, so a mislabel would transpose it."""
    from flowcl.train.checkpoint import load_checkpoint

    result = run_continual(
        two_task_curriculum,
        method_name="seq_ft",
        spec=load_embodiment_spec("libero_franka"),
        policy_config=TINY_POLICY,
        train_cfg=tiny_train_cfg,
        eval_cfg=EvalConfig(n_episodes=1),
        seed=0,
        dataset_dir=dataset_dir,
        results_root=tmp_path,
        pretrained=False,
        evaluate=False,
    )
    for stage_idx, record in enumerate(result.stages):
        loaded = load_checkpoint(record.checkpoint)
        assert loaded.stage == stage_idx
        assert loaded.task_key == two_task_curriculum.task_keys[stage_idx]


def test_matrix_is_empty_without_rollouts_rather_than_full_of_zeros(
    dataset_dir, two_task_curriculum, tiny_train_cfg, tmp_path
):
    """There is no such thing as a rollout-free success rate."""
    result = run_continual(
        two_task_curriculum,
        method_name="seq_ft",
        spec=load_embodiment_spec("libero_franka"),
        policy_config=TINY_POLICY,
        train_cfg=tiny_train_cfg,
        eval_cfg=EvalConfig(n_episodes=1),
        seed=0,
        dataset_dir=dataset_dir,
        results_root=tmp_path,
        pretrained=False,
        evaluate=False,
    )
    with pytest.raises(KeyError, match="never evaluated"):
        result.matrix.get(0, 0)


def test_systems_numbers_are_recorded_for_the_spec_eight_two_table(
    dataset_dir, two_task_curriculum, tiny_train_cfg, tmp_path
):
    result = run_continual(
        two_task_curriculum,
        method_name="seq_ft",
        spec=load_embodiment_spec("libero_franka"),
        policy_config=TINY_POLICY,
        train_cfg=tiny_train_cfg,
        eval_cfg=EvalConfig(n_episodes=1),
        seed=0,
        dataset_dir=dataset_dir,
        results_root=tmp_path,
        pretrained=False,
        evaluate=False,
    )
    systems = result.systems
    assert systems["trainable_params"] > 0
    assert systems["frozen_params"] > 0
    assert systems["method_stored_mb"] == 0.0
    assert systems["is_exemplar_free"] is True
    assert systems["train_wall_clock_s"] > 0
