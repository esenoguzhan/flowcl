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
from flowcl.methods.base import BaseMethod
from flowcl.methods.gpm import GPM, allowlist, assert_allowlist, freeze_to_allowlist
from flowcl.methods.seq_ft import SeqFT
from flowcl.models.build import build_policy
from flowcl.models.policy import RegistryEntry
from flowcl.train.checkpoint import LoadedCheckpoint
from flowcl.train.trainer import TrainConfig, train_one_task

NEG_TOL = 1e-8


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
    method.on_task_start(1, model, None)
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
        GPM().on_task_start(1, model, None)
    first = GPM()
    first.on_task_start(0, model, None)  # task 0 trains unconstrained
    first.modify_gradients(model, {"step": 0})  # inactive: no-op, no grad needed

    basis, _ = basis_for(6, 2)
    with pytest.raises(NotImplementedError, match="step 8"):
        gpm_for(model, basis, update_memory=True).on_task_end(1, model, None)
    method = gpm_for(model, basis)
    method.on_task_end(1, model, None)
    assert method.memory_extended is False
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
