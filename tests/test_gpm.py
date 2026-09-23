"""GPM with AdamW-safe update projection, the §7.4 allowlist, and the trainer hook order."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch
import torch.nn as nn

from flowcl.analysis.subspace import basis_from_gram
from flowcl.data.config import load_embodiment_spec
from flowcl.data.dataset import ChunkedActionDataset
from flowcl.data.episode import Episode
from flowcl.data.stats import compute_stats
from flowcl.experiments.gate2 import collect_bases, load_subspace_config
from flowcl.methods.base import BaseMethod, TaskContext
from flowcl.methods.gpm import GPM, allowlist, assert_allowlist, freeze_to_allowlist
from flowcl.methods.seq_ft import SeqFT
from flowcl.models.build import build_policy
from flowcl.models.policy import RegistryEntry
from flowcl.train.checkpoint import LoadedCheckpoint
from flowcl.train.trainer import TrainConfig, train_one_task

NEG_TOL = 1e-8
CTX = TaskContext(task_key=None, dataset=None, device="cpu")


class Toy(nn.Module):
    """A one-layer 'policy' exposing the registry interface GPM relies on."""

    def __init__(self, d_in: int = 6, d_out: int = 4, seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.a = nn.Linear(d_in, d_out, bias=False)

    def projectable_layers(self):
        return (RegistryEntry("a", self.a, "trunk_mlp"),)

    def projectable_parameters(self):
        return ("a.weight",)


def basis_for(d_in: int, k: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(d_in, d_in, generator=g, dtype=torch.float64))
    sigmas = torch.cat([torch.full((k,), 10.0), torch.full((d_in - k,), 1e-4)])
    R = q @ torch.diag(sigmas.to(torch.float64))
    basis = basis_from_gram(R @ R.T, "a", n_samples=d_in, thresholds=[0.95], neg_tol=NEG_TOL)
    assert basis.ranks[0.95] == k
    return basis, q


def gpm_for(model, basis, **kwargs):
    method = GPM(eps=0.95, **kwargs)
    method.set_memory({"a": basis})
    method.on_task_start(model, 1, context=CTX)
    return method


def adamw_step(model, method, x, y, project_update: bool, wd: float = 0.1):
    opt = torch.optim.AdamW(model.parameters(), lr=0.05, weight_decay=wd)
    opt.zero_grad()
    ((model.a(x) - y) ** 2).mean().backward()
    method.modify_gradients(model, {"step": 0, "task_idx": 1})
    opt.step()
    if project_update:
        method.after_step(model, {"step": 0, "task_idx": 1})


def residual(D, M):
    return float((D.double() @ M).norm())


def test_projected_update_is_orthogonal_and_gradient_only_is_not():
    """Regression guard for Adam/weight-decay leakage into protected directions."""
    basis, _ = basis_for(6, 2)
    M = basis.basis(0.95)
    x, y = torch.randn(32, 6), torch.randn(32, 4)

    leaky = Toy()
    w0 = leaky.a.weight.detach().clone()
    adamw_step(leaky, gpm_for(leaky, basis), x, y, project_update=False)
    D = leaky.a.weight.detach() - w0
    assert residual(D, M) > 1e-6 * float(w0.norm()) + 1e-3 * float(D.norm())

    safe = Toy()
    method = gpm_for(safe, basis)
    adamw_step(safe, method, x, y, project_update=True)
    D = safe.a.weight.detach() - w0
    assert float(D.norm()) > 0
    assert residual(D, M) <= 1e-6 * float(w0.norm()) + 1e-3 * float(D.norm())


def test_gradient_projection_acts_on_the_input_dimension_of_a_square_layer():
    """G = u v^T with v in span(M), u orthogonal to it: input-side projection gives 0."""
    basis, q = basis_for(6, 2)
    model = Toy(d_in=6, d_out=6)
    method = gpm_for(model, basis)
    u, v = q[:, 3].float(), q[:, 0].float()
    model.a.weight.grad = torch.outer(u, v)
    method.modify_gradients(model, {"step": 1, "task_idx": 1})
    assert float(model.a.weight.grad.norm()) < 1e-6
    # A transposed projection (P G) would have kept it: u is outside span(M).
    P = torch.eye(6) - basis.basis(0.95).float() @ basis.basis(0.95).float().T
    assert float((P @ torch.outer(u, v)).norm()) > 0.9


def test_full_rank_basis_builds_an_exactly_zero_projector():
    basis, _ = basis_for(4, 4)
    model = Toy(d_in=4)
    method = gpm_for(model, basis)
    assert torch.count_nonzero(method._layers[0].P) == 0
    w0 = model.a.weight.detach().clone()
    adamw_step(model, method, torch.randn(8, 4), torch.randn(8, 4), project_update=True)
    assert torch.equal(model.a.weight.detach(), w0)


def test_residual_check_tolerates_zero_updates_and_catches_violations():
    basis, _ = basis_for(6, 2)
    model = Toy()
    method = gpm_for(model, basis)
    method.after_step(model, {"step": 0, "task_idx": 1})  # skipped AMP step: no change
    method._layers[0].P = torch.eye(6)  # sabotage: nothing gets projected
    with torch.no_grad():
        model.a.weight.add_(0.1 * torch.randn_like(model.a.weight))
    with pytest.raises(RuntimeError, match="not orthogonal"):
        method.after_step(model, {"step": 1, "task_idx": 1})


def test_name_memory_and_lifecycle_rules():
    assert GPM().display_name == "gpm_projected_adam"
    with pytest.raises(ValueError, match="not implemented"):
        GPM(projection="gradient")
    model = Toy()
    with pytest.raises(RuntimeError, match="no memory"):
        GPM().on_task_start(model, 1, context=CTX)
    first = GPM()
    first.on_task_start(model, 0, context=CTX)  # task 0 trains unconstrained
    first.modify_gradients(model, {"step": 0})  # inactive: no-op, no grad needed

    basis, _ = basis_for(6, 2)
    # A memory update needs a seed namespace to seed the capture: refuse, don't invent one.
    with pytest.raises(RuntimeError, match="seed_namespace_run_id"):
        gpm_for(model, basis, update_memory=True).on_task_end(model, 1, context=CTX)
    method = gpm_for(model, basis)
    method.on_task_end(model, 1, context=CTX)
    assert method.memory_extended == {1: False}
    assert method.residuals["a"]["k"] == 2
    assert method.stored_bytes() == 2 * 6 * 4


# ---- the real policy ------------------------------------------------------------


@pytest.fixture(scope="module")
def spec():
    return load_embodiment_spec("libero_franka")


def make_episode(spec, task, length, seed):
    rng = np.random.default_rng(seed)
    h, w = spec.observation.image_size
    return Episode(
        images={c: rng.integers(0, 255, (length, h, w, 3), dtype=np.uint8) for c in spec.cameras},
        state=rng.normal(size=(length, spec.d_state)).astype(np.float32),
        action=rng.uniform(-1, 1, size=(length, spec.d_action)).astype(np.float32),
        language=f"do {task}",
        task_id=task,
        embodiment=spec.name,
    )


@pytest.fixture(scope="module")
def setup(spec):
    t1 = [make_episode(spec, "toy/one", 18, s) for s in range(2)]
    t2 = [make_episode(spec, "toy/two", 18, 10 + s) for s in range(2)]
    stats = compute_stats(t1, embodiment=spec.name, task_id="toy/one")
    torch.manual_seed(0)
    policy = build_policy("flowpolicy_small", spec, pretrained=False)
    for p in policy.parameters():
        if p.requires_grad:
            nn.init.normal_(p, std=0.05)
    loaded = LoadedCheckpoint(policy, spec, stats, {"run_id": "toy", "stage": 0, "task_key": "toy/one"})
    sub = collect_bases(
        loaded, ChunkedActionDataset(t1, spec, stats),
        dataclasses.replace(load_subspace_config(), min_samples_per_dim=0.01, num_workers=0, batch_size=16),
        device="cpu",
    )
    bases = {n: layer.primary for n, layer in sub.layers.items()}
    return policy, bases, ChunkedActionDataset(t2, spec, stats)


def small_cfg(steps=3):
    return TrainConfig(steps=steps, batch_size=4, device="cpu", amp=False, num_workers=0,
                       warmup_steps=1, log_every=0)


def test_allowlist_is_exactly_the_registry_weights(setup):
    policy, _, _ = setup
    report = freeze_to_allowlist(policy)
    trainable = {n for n, p in policy.named_parameters() if p.requires_grad}
    assert trainable == set(policy.projectable_parameters()) == set(allowlist(policy))
    assert report["newly_frozen"]
    policy.trunk.position_embedding.requires_grad_(True)
    with pytest.raises(RuntimeError, match="trainable but not allowlisted"):
        assert_allowlist(policy)
    policy.trunk.position_embedding.requires_grad_(False)


def test_train_one_task_with_gpm_keeps_frozen_tensors_and_orthogonality(setup):
    policy, bases, t2_data = setup
    freeze_to_allowlist(policy)
    frozen = {n: p.detach().clone() for n, p in policy.named_parameters() if not p.requires_grad}
    method = GPM(log_interval=1)
    method.set_memory(bases)
    log = train_one_task(policy, t2_data, small_cfg(), method=method, task_idx=1,
                         generator=torch.Generator().manual_seed(0))
    assert all(np.isfinite(log.losses))
    for n, p in policy.named_parameters():
        if n in frozen:
            assert torch.equal(p, frozen[n]), n
    assert set(method.gradient_c) == {0, 1, 2} and set(method.update_c) == {0, 1, 2}
    worst = max(r["max_residual_over_bound"] for r in method.residuals.values())
    assert worst <= 1.0
    # A layer moves if and only if its basis leaves free directions.
    for name, r in method.residuals.items():
        moved = method.cumulative_displacement_norm[name] > 0
        assert moved != r["full_rank"], (name, r)
    assert method.residuals["flow_head.action_in"]["full_rank"]


class Spy(BaseMethod):
    name = "spy"

    def __init__(self, events):
        super().__init__()
        self.events = events

    def modify_gradients(self, policy, batch_meta):
        self.events.append("modify_gradients")

    def after_step(self, policy, step_meta):
        self.events.append("after_step")


def test_trainer_hook_order_per_step(setup, monkeypatch):
    """backward -> unscale_ -> modify_gradients -> clip -> step -> update -> after_step."""
    policy, _, t2_data = setup
    freeze_to_allowlist(policy)
    events: list[str] = []

    def wrap(owner, attr, label):
        original = getattr(owner, attr)

        def wrapper(*args, **kwargs):
            events.append(label)
            return original(*args, **kwargs)

        monkeypatch.setattr(owner, attr, wrapper)

    wrap(torch.amp.GradScaler, "unscale_", "unscale_")
    wrap(torch.amp.GradScaler, "step", "step")
    wrap(torch.amp.GradScaler, "update", "update")
    wrap(torch.nn.utils, "clip_grad_norm_", "clip")
    handle = policy.flow_head.action_out.weight.register_hook(
        lambda g: events.append("backward")
    )
    try:
        train_one_task(policy, t2_data, small_cfg(steps=2), method=Spy(events), task_idx=1,
                       generator=torch.Generator().manual_seed(0))
    finally:
        handle.remove()
    per_step = ["backward", "unscale_", "modify_gradients", "clip", "step", "update", "after_step"]
    assert events == per_step * 2


def test_seq_ft_after_step_is_a_no_op():
    assert SeqFT().after_step(None, {"step": 0}) is None


# ---- accumulated memory (build step 8) -----------------------------------------


@pytest.fixture()
def capture_yaml(tmp_path):
    """The Gate 2 capture config, relaxed so tiny fake datasets pass the N/d check."""
    from omegaconf import OmegaConf

    from flowcl.utils.libero_paths import repo_root

    cfg = OmegaConf.load(repo_root() / "configs" / "analysis" / "subspace.yaml")
    cfg.min_samples_per_dim = 0.01
    cfg.num_workers = 0
    cfg.batch_size = 16
    path = tmp_path / "capture.yaml"
    OmegaConf.save(cfg, path)
    return str(path)


def fresh_policy(spec):
    torch.manual_seed(0)
    policy = build_policy("flowpolicy_small", spec, pretrained=False)
    for p in policy.parameters():
        if p.requires_grad:
            nn.init.normal_(p, std=0.05)
    return policy


def task_data(spec, key, seed):
    episodes = [make_episode(spec, key, 18, seed + s) for s in range(2)]
    return episodes


def run_three_tasks(spec, capture_yaml, monkeypatch=None):
    from flowcl.train import trainer as trainer_module

    keys = ["toy/one", "toy/two", "toy/three"]
    episodes = {k: task_data(spec, k, 10 * i) for i, k in enumerate(keys)}
    stats = compute_stats(episodes[keys[0]], embodiment=spec.name, task_id=keys[0])
    policy = fresh_policy(spec)
    method = GPM(update_memory=True, capture_config=capture_yaml, log_interval=1)
    optimizer_params: dict[int, set] = {}
    if monkeypatch is not None:
        real = trainer_module.build_optimizer

        def spy(policy_, cfg_):
            opt = real(policy_, cfg_)
            ids = {id(p) for g in opt.param_groups for p in g["params"]}
            optimizer_params[len(optimizer_params)] = {
                n for n, p in policy_.named_parameters() if id(p) in ids
            }
            return opt

        monkeypatch.setattr(trainer_module, "build_optimizer", spy)
    snapshots = []
    for idx, key in enumerate(keys):
        data = ChunkedActionDataset(episodes[key], spec, stats)
        ctx = TaskContext(task_key=key, dataset=data, device="cpu",
                          seed_namespace_run_id="toy_ns", method_run_id="toy_run")
        train_one_task(policy, data, small_cfg(), method=method, task_idx=idx,
                       generator=torch.Generator().manual_seed(idx), context=ctx)
        snapshots.append({n: p.detach().clone() for n, p in policy.named_parameters()})
    return policy, method, optimizer_params, snapshots, keys


def test_memory_accumulates_across_tasks_and_freezing_precedes_the_optimizer(
    spec, capture_yaml, monkeypatch
):
    policy, method, optimizer_params, snapshots, keys = run_three_tasks(spec, capture_yaml, monkeypatch)
    assert sorted(method.memory_history) == [0, 1, 2]
    for name in policy.registry_names():
        rhos = [method.memory_history[t][name]["rho_after"] for t in range(3)]
        assert rhos == sorted(rhos), (name, rhos)  # occupancy never decreases
        for t in range(3):
            assert method.memory_history[t][name]["captured_energy_fraction"] >= 0.95 - 1e-6
    # Task 0 trained everything; from task 1 the optimiser only ever saw registry weights.
    assert optimizer_params[0] > set(allowlist(policy))
    assert optimizer_params[1] == optimizer_params[2] == set(allowlist(policy))
    frozen = [n for n in snapshots[0] if n not in set(allowlist(policy))]
    for n in frozen:
        assert torch.equal(snapshots[0][n], snapshots[2][n]), n
    assert sorted(method.task_logs) == [0, 1, 2]
    assert method.task_logs[0]["projected"] is False and method.task_logs[2]["projected"] is True


def test_artifacts_round_trip_and_restore_verifies_the_hash(spec, capture_yaml, tmp_path):
    from flowcl.analysis.subspace import load_bases
    from flowcl.utils.run import file_sha256

    policy, method, _, _, keys = run_three_tasks(spec, capture_yaml)
    ctx = TaskContext(task_key=keys[1], dataset=None, device="cpu",
                      seed_namespace_run_id="toy_ns", method_run_id="toy_run")
    paths = method.save_artifacts(tmp_path, 1, context=ctx)
    memory_path, logs_path = paths
    assert memory_path.name == "memory_task1.pt" and logs_path.name == "gpm_logs_task1.json"
    assert not list(tmp_path.glob("*.tmp"))
    bases, meta = load_bases(memory_path)
    assert meta["kind"] == "accumulated_memory"
    assert (meta["method_run_id"], meta["seed_namespace_run_id"]) == ("toy_run", "toy_ns")
    for name, basis in bases.items():
        torch.testing.assert_close(basis.vectors, method._memory[name])

    restored = GPM(update_memory=True, capture_config=capture_yaml)
    restored.restore_memory(memory_path, file_sha256(memory_path), method_run_id="toy_run", task_idx=1)
    ctx2 = TaskContext(task_key="toy/next", dataset=None, device="cpu")
    method.on_task_start(policy, 3, context=ctx2)
    restored.on_task_start(policy, 3, context=ctx2)
    for a, b in zip(method._layers, restored._layers):
        assert torch.equal(a.P, b.P), a.name

    with open(memory_path, "ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="SHA-256"):
        GPM().restore_memory(memory_path, file_sha256(logs_path))
    with pytest.raises(ValueError, match="does not match"):
        GPM().restore_memory(memory_path, file_sha256(memory_path), task_idx=2)


def test_memory_capture_seeds_depend_on_namespace_task_and_index(spec, monkeypatch):
    import flowcl.experiments.gate2 as gate2
    from flowcl.utils.seeding import derive_seed

    seen = []

    class Stop(Exception):
        pass

    def fake_capture(policy, dataset, cfg, device, probe_seed, capture_seed):
        seen.append((probe_seed, capture_seed))
        raise Stop

    monkeypatch.setattr(gate2, "capture_task_grams", fake_capture)
    method = GPM(update_memory=True)
    for ns, key, idx in (("ns_a", "toy/one", 0), ("ns_b", "toy/one", 0), ("ns_a", "toy/one", 1)):
        ctx = TaskContext(task_key=key, dataset=object(), device="cpu", seed_namespace_run_id=ns)
        with pytest.raises(Stop):
            method.on_task_end(None, idx, context=ctx)
        assert seen[-1][1] == derive_seed(ns, f"gpm_memory::{key}", idx)
    assert len({s[1] for s in seen}) == 3 and len({s[0] for s in seen}) == 3


def test_interrupted_artifact_write_leaves_no_partial_file(tmp_path, monkeypatch):
    import os

    from flowcl.utils.run import atomic_write_text

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        atomic_write_text(tmp_path / "gpm_logs_task0.json", "{}")
    assert not (tmp_path / "gpm_logs_task0.json").exists()
