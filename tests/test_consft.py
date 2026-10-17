"""ConSFT: eq. 11–12 against hand values, the per-sample masked loss and its reduction to masked_mse at
ω ≡ 1, the stop-gradient, the ω_min clamp, the identical tensor at T1 and stage 0 bitwise seq_ft, the
per-task step counter, the state round trip, and argument validation."""

from __future__ import annotations

import math

import pytest
import torch

from flowcl.methods.base import TaskContext, build_method
from flowcl.methods.consft import ConSFT, per_sample_terms, temperature
from flowcl.methods.seq_ft import SeqFT
from flowcl.models.losses import masked_mse
from flowcl.train.trainer import train_one_task
from test_ewc import world  # noqa: F401
from test_gpm import fresh_policy, small_cfg, spec  # noqa: F401
from test_replay import TASKS

TABLE5 = {"tau_start": 0.003, "decay_steps": 2000}


def ctx(world, j):
    return TaskContext(task_key=TASKS[j], dataset=world[TASKS[j]], device="cpu", seed_namespace_run_id="ns",
                       method_run_id="run", batch_size=4)


def hand_tau(t, tau_start=0.003, T=2000, lam=0.8, c=3.5, tau_end=5.0):
    rho = (math.exp(-lam * min(t / T, 1.0) ** c) - math.exp(-lam)) / (1 - math.exp(-lam))
    return min(tau_start / max(rho, 1e-8), tau_end)


def batch(lengths=(4, 2, 3), H=4, D=7, seed=0):
    g = torch.Generator().manual_seed(seed)
    B = len(lengths)
    mask = torch.zeros(B, H)
    for i, n in enumerate(lengths):
        mask[i, :n] = 1
    velocity = torch.randn(B, H, D, generator=g, requires_grad=True)
    target = torch.randn(B, H, D, generator=g)
    s = torch.tensor([0.05, 0.5, 0.95])[:B]
    return velocity, target, mask, s


def test_temperature_matches_eq_11_12():
    m = ConSFT(**TABLE5)
    assert m.temperature(0) == pytest.approx(0.003)          # ρ(0) = 1
    for t in (1, 500, 1000, 1500, 1900, 1990, 1999):
        assert m.temperature(t) == pytest.approx(hand_tau(t), rel=1e-12)
    assert m.temperature(2000) == 5.0                        # ρ = 0: the cap
    assert m.temperature(10_000) == 5.0                      # t > T clipped to T
    taus = [m.temperature(t) for t in range(0, 2001, 50)]
    assert taus == sorted(taus)                              # monotone non-decreasing
    assert temperature(1000, 0.003, 2000, 0.8, 3.5, 5.0) == m.temperature(1000)


def test_per_sample_terms_reduce_to_masked_mse():
    velocity, target, mask, _ = batch()
    summed, count = per_sample_terms(velocity, target, mask)
    assert count.tolist() == [4 * 7, 2 * 7, 3 * 7]
    assert float(summed.sum() / count.sum()) == pytest.approx(float(masked_mse(velocity, target, mask)), rel=1e-6)


def stage1_method(**kw):
    m = ConSFT(**{**TABLE5, **kw})
    m.on_task_start(None, 1, context=TaskContext(task_key="toy/two", dataset=None, device="cpu"))
    return m


def test_weighted_loss_value_gradient_and_clamp():
    velocity, target, mask, s = batch()
    m = stage1_method(tau_start=0.5)
    out = m.modify_loss(masked_mse(velocity, target, mask), {"action_mask": mask}, None,
                        outputs={"velocity": velocity, "target_velocity": target, "s": s})
    summed, count = per_sample_terms(velocity.detach(), target, mask)
    per = summed / count
    omega = torch.exp(-25.0 * per / 0.5).clamp(min=1e-3)
    assert float(out) == pytest.approx(float((omega * summed).sum() / count.sum()), rel=1e-6)
    # stop-gradient: the gradient is Σ ω_i ∇S_i / Σ N_i D with ω held constant
    out.backward()
    expected = 2 * (velocity.detach() - target) * mask[..., None] * omega[:, None, None] / count.sum()
    assert torch.allclose(velocity.grad, expected, rtol=1e-5, atol=1e-8)
    # these unit-variance errors put every sample at the floor at τ = 0.5
    assert torch.all(omega == 1e-3)
    log = m.logs["1"]["steps"]["0"]
    assert log["omega_floor_fraction"] == 1.0 and log["tau"] == pytest.approx(0.5)
    assert sum(b["n"] for b in log["by_s_bin"].values()) == 3


def test_weights_are_one_minus_small_for_tiny_losses_and_never_exceed_one():
    m = stage1_method()
    w = m.weights(torch.tensor([0.0, 1e-6, 1e-4, 1.0]), 0)
    assert w[0] == 1.0 and 0.99 < float(w[1]) < 1.0
    assert float(w[2]) == pytest.approx(math.exp(-25 * 1e-4 / 0.003))
    assert float(w[3]) == pytest.approx(1e-3)
    assert torch.all(w <= 1.0)


def test_t1_returns_the_same_tensor_and_the_counter_resets_per_task():
    m = ConSFT(**TABLE5)
    m.on_task_start(None, 0, context=TaskContext(task_key="toy/one", dataset=None, device="cpu"))
    loss = torch.tensor(0.25)
    assert m.modify_loss(loss, None, None) is loss
    for _ in range(7):
        m.after_step(None, {})
    m.on_task_start(None, 1, context=TaskContext(task_key="toy/two", dataset=None, device="cpu"))
    assert m._step == 0
    velocity, target, mask, s = batch()
    outputs = {"velocity": velocity, "target_velocity": target, "s": s}
    m.modify_loss(loss, {"action_mask": mask}, None, outputs=outputs)
    m.after_step(None, {})
    assert m._step == 1
    with pytest.raises(ValueError, match="forward outputs"):
        m.modify_loss(loss, {"action_mask": mask}, None)


def test_autocast_inputs_give_an_fp32_loss():
    velocity, target, mask, s = batch()
    m = stage1_method(tau_start=50.0, tau_end=100.0)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = m.modify_loss(masked_mse(velocity, target, mask), {"action_mask": mask}, None,
                            outputs={"velocity": velocity.to(torch.bfloat16),
                                     "target_velocity": target.to(torch.bfloat16), "s": s})
    assert out.dtype == torch.float32


def test_stage0_is_bitwise_seq_ft(spec, world):
    data = world[TASKS[0]]
    weights = []
    for method in (SeqFT(), ConSFT(**TABLE5)):
        policy = fresh_policy(spec)
        train_one_task(policy, data, small_cfg(), method=method, task_idx=0,
                       generator=torch.Generator().manual_seed(0), context=ctx(world, 0))
        weights.append({n: p.detach().clone() for n, p in policy.named_parameters()})
    assert all(torch.equal(weights[0][n], weights[1][n]) for n in weights[0])


def test_stage1_trains_differently_and_logs(spec, world):
    results = []
    for method in (SeqFT(), ConSFT(tau_start=0.05, decay_steps=2, log_interval=1)):
        policy = fresh_policy(spec)
        for j in (0, 1):
            train_one_task(policy, world[TASKS[j]], small_cfg(), method=method, task_idx=j,
                           generator=torch.Generator().manual_seed(j), context=ctx(world, j))
        results.append({n: p.detach().clone() for n, p in policy.named_parameters()})
        if isinstance(method, ConSFT):
            steps = method.logs["1"]["steps"]
            assert sorted(steps, key=int) == ["0", "1", "2"]
            assert steps["2"]["tau"] == 5.0 and steps["0"]["tau"] == pytest.approx(0.05)
            assert method.logs["1"]["task_key"] == TASKS[1]
    assert any(not torch.equal(results[0][n], results[1][n]) for n in results[0])


def test_state_round_trip_and_refusals():
    m = ConSFT(**TABLE5)
    m.logs = {"1": {"task_key": "a", "steps": {}}, "2": {"task_key": "b", "steps": {}}}
    extra = {"method_state": m.state_dict()}
    fresh = ConSFT(**TABLE5)
    assert fresh.restore_state(extra, None, 2) == {"log_tasks": [1, 2]}
    assert fresh.logs == m.logs
    with pytest.raises(ValueError, match="not this ConSFT"):
        ConSFT(tau_start=0.004, decay_steps=2000).restore_state(extra, None, 2)
    with pytest.raises(ValueError, match="cover tasks"):
        ConSFT(**TABLE5).restore_state(extra, None, 3)
    assert ConSFT(**TABLE5).restore_state({"method_state": ConSFT(**TABLE5).state_dict()}, None, 0) == \
        {"log_tasks": []}
    assert m.stored_bytes() == 0 and m.is_exemplar_free


def test_validation_and_registry():
    assert isinstance(build_method("consft", **TABLE5), ConSFT)
    with pytest.raises(ValueError, match="needs `tau_start`"):
        ConSFT()
    with pytest.raises(TypeError, match="unexpected config keys"):
        ConSFT(**TABLE5, scale=3.0)
    for bad in ({"tau_start": 0.0}, {"decay_steps": 0}, {"decay_steps": 2.5}, {"omega_min": 0.0},
                {"omega_min": 1.5}, {"curve": 0.5}, {"tau_end": -1.0}, {"kappa": float("inf")},
                {"log_interval": 0}):
        with pytest.raises(ValueError):
            ConSFT(**{**TABLE5, **bad})
