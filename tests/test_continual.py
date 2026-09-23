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

    def on_task_start(self, policy, task_idx, *, context):
        self.calls.append(("on_task_start", task_idx))

    def build_batch(self, dataset, task_idx):
        self.calls.append(("build_batch", task_idx))
        return None

    def modify_loss(self, loss, batch, policy, *, outputs=None):
        self.calls.append(("modify_loss", outputs is not None))
        return loss

    def modify_gradients(self, policy, batch_meta):
        self.calls.append(("modify_gradients", batch_meta["task_idx"]))

    def on_task_end(self, policy, task_idx, *, context):
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


# ---- build step 8: seed namespace, artifacts, stage order, provenance guards ------

BBQ = "libero_object/pick_up_the_bbq_sauce_and_place_it_in_the_basket"


class ArtifactMethod(BaseMethod):
    """Writes one artifact per stage and records lifecycle events and contexts."""

    name = "artifact_method"

    def __init__(self, events) -> None:
        super().__init__()
        self.events = events
        self.contexts = []

    def on_task_start(self, policy, task_idx, *, context):
        self.contexts.append(context)

    def on_task_end(self, policy, task_idx, *, context):
        self.events.append(("on_task_end", task_idx))

    def save_artifacts(self, directory, task_idx, *, context):
        from flowcl.utils.run import atomic_write_text

        self.events.append(("save_artifacts", task_idx))
        return [atomic_write_text(directory / f"note_task{task_idx}.txt", f"stage {task_idx}\n")]


def fake_evaluation(events):
    from flowcl.analysis.metrics import success_estimate
    from flowcl.envs.evaluation import EvaluationReport, TaskEvaluation

    def evaluate(policy, refs, spec, stats, run_id, cfg, bootstrap=None, stage=None):
        events.append(("evaluate", stage, run_id))
        succ = [True, False]
        return EvaluationReport(run_id=run_id, stage=stage, tasks=[
            TaskEvaluation(r.task_key, succ, [10, 600], [0, 1], success_estimate(succ)) for r in refs
        ])
    return evaluate


def test_seq_ft_rollout_seeds_are_unchanged_by_the_namespace_rule():
    """Pinned against Gate 1's own artifact: its first Spatial rollout seed."""
    from flowcl.train.continual import seed_namespace_run_id
    from flowcl.utils.seeding import derive_seed

    ns = seed_namespace_run_id("seq_hetero", 0)
    assert ns == "seq_hetero__seq_ft__seed0"
    spatial = "libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate"
    assert derive_seed(ns, spatial, 0) == 827087693


def test_namespace_order_hashes_and_provenance(
    dataset_dir, two_task_curriculum, tiny_train_cfg, tmp_path, monkeypatch
):
    import json

    import flowcl.train.continual as continual_module
    from flowcl.methods import base as base_module
    from flowcl.train.checkpoint import load_checkpoint
    from flowcl.utils.run import file_sha256
    from flowcl.utils.seeding import derive_seed

    events = []
    method = ArtifactMethod(events)
    monkeypatch.setitem(base_module.METHOD_REGISTRY, "artifact_method", lambda **kw: method)
    monkeypatch.setattr(continual_module, "evaluate_tasks", fake_evaluation(events))
    real_save = continual_module.save_checkpoint

    def save_spy(*args, **kwargs):
        events.append(("save_checkpoint", kwargs["stage"]))
        return real_save(*args, **kwargs)

    monkeypatch.setattr(continual_module, "save_checkpoint", save_spy)
    seeds = []
    real_train = continual_module.train_one_task

    def train_spy(*args, **kwargs):
        seeds.append(kwargs["generator"].initial_seed())
        return real_train(*args, **kwargs)

    monkeypatch.setattr(continual_module, "train_one_task", train_spy)

    result = run_continual(
        two_task_curriculum, method_name="artifact_method",
        spec=load_embodiment_spec("libero_franka"), policy_config=TINY_POLICY,
        train_cfg=tiny_train_cfg, eval_cfg=EvalConfig(n_episodes=2), seed=0,
        dataset_dir=dataset_dir, results_root=tmp_path, pretrained=False, evaluate=True,
    )
    ns, run_id = "test_pair__seq_ft__seed0", "test_pair__artifact_method__seed0"
    assert result.run_id == run_id and result.seed_namespace_run_id == ns

    # Order per stage: train (ending in on_task_end) -> artifacts -> checkpoint -> evaluate.
    stage_events = events
    expected = []
    for stage in (0, 1):
        expected += [("on_task_end", stage), ("save_artifacts", stage),
                     ("save_checkpoint", stage), ("evaluate", stage, ns)]
    assert stage_events == expected

    # Every stream derives from the seed namespace, not from this run's id.
    assert seeds == [derive_seed(ns, t, i) for i, t in enumerate(two_task_curriculum.task_keys)]
    assert all(c.seed_namespace_run_id == ns and c.method_run_id == run_id for c in method.contexts)
    assert [c.task_key for c in method.contexts] == list(two_task_curriculum.task_keys)

    run_dir = tmp_path / run_id
    for stage in (0, 1):
        extra = load_checkpoint(run_dir / "checkpoints" / f"stage{stage}.pt").payload["extra"]
        assert (extra["method_run_id"], extra["seed_namespace_run_id"]) == (run_id, ns)
        (artifact,) = extra["method_artifacts"]
        assert artifact["path"] == f"method/note_task{stage}.txt"
        assert artifact["sha256"] == file_sha256(run_dir / artifact["path"])
        report = json.loads((run_dir / "eval" / f"stage{stage}.json").read_text())
        assert (report["run_id"], report["method_run_id"], report["seed_namespace_run_id"]) == (ns, run_id, ns)
    config = (run_dir / "config.yaml").read_text()
    assert f"method_run_id: {run_id}" in config and f"seed_namespace_run_id: {ns}" in config
    saved = json.loads((run_dir / "result.json").read_text())
    assert (saved["method_run_id"], saved["seed_namespace_run_id"]) == (run_id, ns)


def test_gpm_three_stages_end_to_end(dataset_dir, tiny_train_cfg, tmp_path):
    """Variant-specific run id, accumulated memory artifacts, frozen tensors from stage 1."""
    from omegaconf import OmegaConf

    from flowcl.analysis.subspace import load_bases
    from flowcl.methods.gpm import GPM, allowlist
    from flowcl.train.checkpoint import load_checkpoint
    from flowcl.utils.libero_paths import repo_root
    from flowcl.utils.run import file_sha256

    cfg = OmegaConf.load(repo_root() / "configs" / "analysis" / "subspace.yaml")
    cfg.min_samples_per_dim = 0.01
    cfg.num_workers = 0
    capture = tmp_path / "capture.yaml"
    OmegaConf.save(cfg, capture)

    trio = load_curriculum({"name": "test_trio", "tasks": [
        {"task_key": MILK, "n_demos": 1}, {"task_key": SAUCE, "n_demos": 1},
        {"task_key": BBQ, "n_demos": 1}]})
    result = run_continual(
        trio, method_name="gpm",
        method_kwargs={"update_memory": True, "capture_config": str(capture), "log_interval": 1},
        spec=load_embodiment_spec("libero_franka"), policy_config=TINY_POLICY,
        train_cfg=tiny_train_cfg, eval_cfg=EvalConfig(n_episodes=1), seed=0,
        dataset_dir=dataset_dir, results_root=tmp_path, pretrained=False, evaluate=False,
    )
    assert result.run_id == "test_trio__gpm_projected_adam__seed0"
    assert result.method == "gpm_projected_adam" and result.method_registry_name == "gpm"
    run_dir = tmp_path / result.run_id

    states, rhos = [], []
    for stage in range(3):
        payload = load_checkpoint(run_dir / "checkpoints" / f"stage{stage}.pt").payload
        artifacts = {a["path"]: a["sha256"] for a in payload["extra"]["method_artifacts"]}
        memory = f"method/memory_task{stage}.pt"
        assert artifacts[memory] == file_sha256(run_dir / memory)
        assert f"method/gpm_logs_task{stage}.json" in artifacts
        bases, meta = load_bases(run_dir / memory)
        assert meta["task_idx"] == stage and meta["method_run_id"] == result.run_id
        rhos.append({n: b.rhos[0.95] for n, b in bases.items()})
        states.append(payload["state_dict"])
    for name in rhos[0]:
        assert rhos[0][name] <= rhos[1][name] <= rhos[2][name], name

    restored = GPM(update_memory=True)
    restored.restore_memory(run_dir / "method" / "memory_task1.pt",
                            file_sha256(run_dir / "method" / "memory_task1.pt"),
                            method_run_id=result.run_id, task_idx=1)

    from flowcl.models.build import build_policy
    policy = build_policy(TINY_POLICY, load_embodiment_spec("libero_franka"), pretrained=False)
    registry = set(allowlist(policy))
    trainable = {n for n, p in policy.named_parameters() if p.requires_grad}
    for name in trainable - registry:  # frozen from stage 1 on
        assert torch.equal(states[0][name], states[2][name]), name

    # The sequence report's checkpoint-verifying provenance checks, on real artifacts.
    from pathlib import Path

    from flowcl.experiments.sequence_report import RunView, provenance_checks

    method_view = RunView(run_dir, {"task_keys": list(trio.task_keys),
                                    "seed_namespace_run_id": "test_trio__seq_ft__seed0"}, {})
    checks = provenance_checks(method_view, RunView(Path("test_trio__seq_ft__seed0"), {}, {}))
    for name in ("seed_namespace", "occupancy_non_decreasing", "residuals_within_bound",
                 "artifact_hashes_match", "frozen_from_stage1"):
        assert checks[name]["passed"], (name, checks[name])


def test_t1_pairing_check_passes_on_a_match_and_catches_a_mismatch(tmp_path):
    import json

    from flowcl.models.build import build_policy
    from flowcl.train.checkpoint import save_checkpoint
    from flowcl.train.continual import t1_pairing_check
    from flowcl.train.trainer import TrainLog

    spec = load_embodiment_spec("libero_franka")
    torch.manual_seed(0)
    policy = build_policy(TINY_POLICY, spec, pretrained=False)
    ref_dir = tmp_path / "ref"
    (ref_dir / "checkpoints").mkdir(parents=True)
    torch.save({"state_dict": policy.state_dict()}, ref_dir / "checkpoints" / "stage0.pt")
    (ref_dir / "result.json").write_text(json.dumps({"stages": [{"final_loss": 0.01, "mean_last_50_loss": 0.011}]}))
    log = TrainLog(losses=[0.02, 0.01], steps=2)

    mine = tmp_path / "mine.pt"
    torch.save({"state_dict": policy.state_dict()}, mine)
    check = t1_pairing_check(mine, ref_dir, log, policy, max_rel_diff=0.05)
    assert check["passed"] and check["rel_weight_diff"] == 0.0
    assert check["final_loss"] == {"this": 0.01, "reference": 0.01}

    perturbed = {k: (v * 1.2 if v.is_floating_point() else v) for k, v in policy.state_dict().items()}
    torch.save({"state_dict": perturbed}, mine)
    check = t1_pairing_check(mine, ref_dir, log, policy, max_rel_diff=0.05)
    assert not check["passed"] and check["rel_weight_diff"] == pytest.approx(0.2, rel=1e-6)


def test_failed_t1_pairing_stops_the_run_before_evaluation(
    dataset_dir, two_task_curriculum, tiny_train_cfg, tmp_path, monkeypatch
):
    import flowcl.train.continual as continual_module

    events = []
    monkeypatch.setattr(continual_module, "evaluate_tasks", fake_evaluation(events))
    monkeypatch.setattr(continual_module, "t1_pairing_check",
                        lambda *a, **k: {"passed": False, "rel_weight_diff": 0.9})
    with pytest.raises(RuntimeError, match="T1 pairing check failed"):
        run_continual(
            two_task_curriculum, method_name="seq_ft",
            spec=load_embodiment_spec("libero_franka"), policy_config=TINY_POLICY,
            train_cfg=tiny_train_cfg, eval_cfg=EvalConfig(n_episodes=2), seed=0,
            dataset_dir=dataset_dir, results_root=tmp_path, pretrained=False,
            t1_reference_run=tmp_path / "whatever",
        )
    assert events == []  # stopped before any rollout
    assert (tmp_path / "test_pair__seq_ft__seed0" / "t1_pairing.json").is_file()


def test_require_clean_tree_refuses_a_dirty_tree(two_task_curriculum, tiny_train_cfg, tmp_path, monkeypatch):
    import flowcl.train.continual as continual_module

    monkeypatch.setattr(continual_module, "git_sha", lambda *a, **k: "abc123-dirty")
    with pytest.raises(RuntimeError, match="dirty"):
        run_continual(
            two_task_curriculum, method_name="seq_ft",
            spec=load_embodiment_spec("libero_franka"), policy_config=TINY_POLICY,
            train_cfg=tiny_train_cfg, eval_cfg=EvalConfig(n_episodes=1), seed=0,
            results_root=tmp_path, pretrained=False, require_clean_tree=True,
        )
    assert not (tmp_path / "test_pair__seq_ft__seed0").exists()
