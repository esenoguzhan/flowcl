"""EWC: the empirical Fisher against its closed form on a toy, the fp32 penalty under autocast, the
identical loss tensor at T1, the online sum and anchor, train() mode, the artifact round-trip with
SHA checks, and stage 0 bitwise seq_ft on the tiny policy."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from flowcl.data.dataset import ChunkedActionDataset
from flowcl.data.stats import compute_stats
from flowcl.methods.base import TaskContext
from flowcl.methods.ewc import EWC, empirical_fisher, tensor_fingerprint
from flowcl.methods.seq_ft import SeqFT
from flowcl.train.trainer import train_one_task
from flowcl.utils.run import file_sha256
from test_gpm import fresh_policy, small_cfg, spec  # noqa: F401
from test_replay import TASKS, demos


class Toy(nn.Module):
    """loss = mean((state @ W - actions[:, 0]) ** 2): a closed-form per-sample gradient."""

    def __init__(self, d_state, d_action):
        super().__init__()
        self.W = nn.Parameter(torch.linspace(-0.5, 0.5, d_state * d_action).reshape(d_state, d_action))
        self.frozen = nn.Parameter(torch.ones(3), requires_grad=False)
        self.modes = []

    def forward(self, batch, generator=None):
        self.modes.append(self.training)
        pred = batch["state"] @ self.W
        return {"loss": ((pred - batch["actions"][:, 0]) ** 2).mean()}


@pytest.fixture(scope="module")
def world(spec):
    eps = {t: demos(spec, t, offset=10 * j) for j, t in enumerate(TASKS)}
    stats = compute_stats(eps[TASKS[0]], embodiment=spec.name, task_id=TASKS[0])
    return {t: ChunkedActionDataset(e, spec, stats) for t, e in eps.items()}


def ctx(world, j):
    return TaskContext(task_key=TASKS[j], dataset=world[TASKS[j]], device="cpu", seed_namespace_run_id="ns",
                       method_run_id="run", batch_size=4)


def test_fisher_matches_the_closed_form_and_restores_the_mode(world, spec):
    toy = Toy(spec.d_state, spec.d_action)
    toy.eval()
    data = world[TASKS[0]]
    fisher, prov = empirical_fisher(toy, data, n_samples=7, seed=11, device="cpu")
    assert set(fisher) == {"W"} and not toy.training and all(toy.modes)  # train() inside, restored after
    idx = torch.randint(len(data), (7,), generator=torch.Generator().manual_seed(11)).tolist()
    expected = torch.zeros_like(toy.W)
    D = spec.d_action
    for i in idx:
        s = data[i]
        x, a = s["state"].unsqueeze(0), s["actions"][0].unsqueeze(0)
        g = (2.0 / D) * x.T @ (x @ toy.W.detach() - a)  # d/dW mean((xW - a)^2)
        expected += g.pow(2)
    torch.testing.assert_close(fisher["W"], expected / 7)
    assert prov["n_samples"] == 7 and prov["fingerprint"] == tensor_fingerprint(fisher)
    assert "masked-mean" in prov["normalization"]


def test_penalty_is_fp32_closed_form_and_t1_returns_the_same_tensor(world, spec):
    toy = Toy(spec.d_state, spec.d_action)
    m = EWC(lam=4.0, fisher_samples=3)
    m.on_task_start(toy, 0, context=ctx(world, 0))
    loss = torch.tensor(0.25)
    assert m.modify_loss(loss, None, toy) is loss  # T1: untouched
    m.on_task_end(toy, 0, context=ctx(world, 0))
    with torch.no_grad():
        toy.W.add_(0.1)
    m.on_task_start(toy, 1, context=ctx(world, 1))
    expected = 0.25 + 0.5 * 4.0 * float((m.fisher["W"] * (toy.W - m.anchor["W"]) ** 2).sum())
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = m.modify_loss(loss.to(torch.bfloat16), None, toy)
    assert out.dtype == torch.float32
    assert float(out) == pytest.approx(expected, rel=1e-3)  # bf16 loss input; penalty exact fp32
    assert float(m.penalty()) == pytest.approx(float((m.fisher["W"] * (toy.W - m.anchor["W"]) ** 2).sum()))


def test_online_sum_and_anchor(world, spec):
    toy = Toy(spec.d_state, spec.d_action)
    m = EWC(lam=1.0, fisher_samples=5)
    m.on_task_end(toy, 0, context=ctx(world, 0))
    f0 = m.fisher["W"].clone()
    with torch.no_grad():
        toy.W.mul_(1.5)
    m.on_task_end(toy, 1, context=ctx(world, 1))
    from flowcl.utils.seeding import derive_seed
    f1, _ = empirical_fisher(toy, world[TASKS[1]], 5, derive_seed("ns", f"ewc_fisher::{TASKS[1]}", 1), "cpu")
    torch.testing.assert_close(m.fisher["W"], f0 + f1["W"])
    assert torch.equal(m.anchor["W"], toy.W.detach())
    assert m.stored_bytes() == 2 * toy.W.numel() * 4
    assert sorted(m.fisher_tasks) == [0, 1] and m.fisher_tasks[1]["task_key"] == TASKS[1]


def test_artifact_round_trip_and_refusals(world, spec, tmp_path):
    toy = Toy(spec.d_state, spec.d_action)
    m = EWC(lam=2.0, fisher_samples=4)
    m.on_task_end(toy, 0, context=ctx(world, 0))
    run = tmp_path / "run"
    (path,) = m.save_artifacts(run / "method", 0, context=ctx(world, 0))
    extra = {"method_artifacts": [{"path": "method/ewc_task0.pt", "sha256": file_sha256(path)}]}
    r = EWC(lam=2.0, fisher_samples=4)
    out = r.restore_state(extra, run, 0)
    assert out["fisher_tasks"] == [0] and torch.equal(r.fisher["W"], m.fisher["W"].cpu())
    assert torch.equal(r.anchor["W"], m.anchor["W"].cpu())
    with pytest.raises(ValueError, match="config"):
        EWC(lam=3.0, fisher_samples=4).restore_state(extra, run, 0)
    with pytest.raises(ValueError, match="names no"):
        EWC(lam=2.0, fisher_samples=4).restore_state({"method_artifacts": []}, run, 0)
    with open(path, "ab") as f:
        f.write(b"x")
    with pytest.raises(ValueError, match="SHA-256"):
        EWC(lam=2.0, fisher_samples=4).restore_state(extra, run, 0)


def test_validation():
    with pytest.raises(ValueError, match="unset"):
        EWC(lam=None)
    for bad in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(ValueError):
            EWC(lam=bad)
    with pytest.raises(ValueError):
        EWC(lam=1.0, fisher_samples=0)


def test_stage0_is_bitwise_seq_ft_and_the_fisher_leaves_weights_alone(spec, world):
    data = world[TASKS[0]]
    weights = []
    for method in (SeqFT(), EWC(lam=10.0, fisher_samples=3)):
        policy = fresh_policy(spec)
        train_one_task(policy, data, small_cfg(), method=method, task_idx=0,
                       generator=torch.Generator().manual_seed(0), context=ctx(world, 0))
        weights.append({n: p.detach().clone() for n, p in policy.named_parameters()})
    assert all(torch.equal(weights[0][n], weights[1][n]) for n in weights[0])


def test_the_penalty_pulls_toward_the_anchor(spec, world):
    """With a large λ, stage-1 training stays much closer to θ* than without the penalty."""
    drift = {}
    for lam in (1e-12, 1e6):
        policy = fresh_policy(spec)
        m = EWC(lam=lam, fisher_samples=4)
        train_one_task(policy, world[TASKS[0]], small_cfg(), method=m, task_idx=0,
                       generator=torch.Generator().manual_seed(0), context=ctx(world, 0))
        anchor = {n: a.clone() for n, a in m.anchor.items()}
        train_one_task(policy, world[TASKS[1]], small_cfg(steps=4), method=m, task_idx=1,
                       generator=torch.Generator().manual_seed(1), context=ctx(world, 1))
        params = dict(policy.named_parameters())
        drift[lam] = sum(float((m.fisher[n] * (params[n].detach() - anchor[n]) ** 2).sum()) for n in anchor)
    assert drift[1e6] < drift[1e-12]
    assert np.isfinite(list(drift.values())).all()


# ---- v2: the sampled (Gaussian velocity) Fisher, normalization, migration --------------------------


class ToyFlow(nn.Module):
    """velocity[h, :] = state @ W for every chunk step h; it draws s and A_0 from the generator like the
    policy does, and records them, so the two estimators' shared draws can be compared."""

    def __init__(self, d_state, d_action):
        super().__init__()
        self.W = nn.Parameter(torch.linspace(-0.5, 0.5, d_state * d_action).reshape(d_state, d_action))
        self.draws = []

    def forward(self, batch, generator=None):
        actions, mask = batch["actions"], batch["action_mask"]
        s = torch.rand((actions.shape[0],), generator=generator)
        noise = torch.randn(tuple(actions.shape), generator=generator)
        self.draws.append((s.clone(), noise.clone()))
        v = (batch["state"] @ self.W)[:, None, :].expand_as(actions)
        target = actions - noise
        loss = (((v - target) ** 2) * mask[..., None]).sum() / (mask.sum() * actions.shape[-1])
        return {"loss": loss, "velocity": v}


@pytest.fixture(scope="module")
def short(spec):
    """One 3-step episode: with H = 16 every sample is padded, with 3, 2 and 1 valid steps."""
    from test_replay import demos as make
    eps = make(spec, TASKS[0], n=1, length=3)
    stats = compute_stats(eps, embodiment=spec.name, task_id=TASKS[0])
    return ChunkedActionDataset(eps, spec, stats)


def test_sampled_fisher_exact_per_sample_gradient_with_unequal_valid_lengths(spec, short):
    from flowcl.methods.ewc import sampled_fisher

    H, D = short.horizon, spec.d_action
    for seed, eps_seed in ((1, 2), (5, 9), (7, 3)):
        toy = ToyFlow(spec.d_state, D)
        fisher, prov = sampled_fisher(toy, short, n_samples=1, seed=seed, eps_seed=eps_seed, device="cpu")
        i = torch.randint(len(short), (1,), generator=torch.Generator().manual_seed(seed)).item()
        eps = torch.randn((1, H, D), generator=torch.Generator().manual_seed(eps_seed))
        sample = short[i]
        x, m = sample["state"], sample["action_mask"]
        grad = -torch.outer(x, (eps[0] * m[:, None]).sum(0))   # d/dW 0.5 Σ_valid (v - y)^2 = -x ⊗ Σ_valid ε_h
        torch.testing.assert_close(fisher["W"], grad.pow(2))
        assert prov["estimator"] == "sampled" and prov["eps_seed"] == eps_seed


def test_sampled_fisher_expectation_has_no_per_sample_weighting(spec, short):
    """E[g_i^2] = diag(J_i^T J_i) = n_valid_i * x_i^2 per entry: no 1/N_i^2 factor."""
    from flowcl.methods.ewc import sampled_fisher

    toy = ToyFlow(spec.d_state, spec.d_action)
    n = 6000
    fisher, _ = sampled_fisher(toy, short, n_samples=n, seed=3, eps_seed=4, device="cpu")
    idx = torch.randint(len(short), (n,), generator=torch.Generator().manual_seed(3)).tolist()
    expected = torch.zeros_like(toy.W)
    for i in idx:
        s = short[i]
        expected += torch.outer(s["state"] ** 2, torch.ones(spec.d_action)) * s["action_mask"].sum()
    expected /= n
    rel = ((fisher["W"] - expected).abs() / expected).max()
    assert rel < 0.08, float(rel)
    valid = sorted({int(short[i]["action_mask"].sum()) for i in range(len(short))})
    assert valid == [1, 2, 3]  # the samples really have different valid lengths


def test_both_estimators_share_indices_s_and_noise(spec, short):
    from flowcl.methods.ewc import empirical_fisher, sampled_fisher

    a, b = ToyFlow(spec.d_state, spec.d_action), ToyFlow(spec.d_state, spec.d_action)
    empirical_fisher(a, short, n_samples=5, seed=11, device="cpu")
    sampled_fisher(b, short, n_samples=5, seed=11, eps_seed=99, device="cpu")
    assert len(a.draws) == len(b.draws) == 5
    for (sa, na), (sb, nb) in zip(a.draws, b.draws):
        assert torch.equal(sa, sb) and torch.equal(na, nb)
    c = ToyFlow(spec.d_state, spec.d_action)
    f1, p1 = sampled_fisher(c, short, n_samples=5, seed=11, eps_seed=99, device="cpu")
    f2, p2 = sampled_fisher(c, short, n_samples=5, seed=11, eps_seed=99, device="cpu")
    assert p1["fingerprint"] == p2["fingerprint"] and torch.equal(f1["W"], f2["W"])


def test_mean_normalization_and_the_online_sum(world, spec):
    toy = ToyFlow(spec.d_state, spec.d_action)
    m = EWC(lam=1.0, fisher_samples=5, fisher_estimator="sampled", fisher_normalization="mean")
    m.on_task_end(toy, 0, context=ctx(world, 0))
    assert float(m.fisher["W"].mean()) == pytest.approx(1.0)
    assert m.fisher_tasks[0]["mean_normalized_by"] > 0 and "normalized_fingerprint" in m.fisher_tasks[0]
    m.on_task_end(toy, 1, context=ctx(world, 1))
    assert float(m.fisher["W"].mean()) == pytest.approx(2.0)  # two tasks, each of mean 1


def test_estimator_options_validation_config_and_legacy_restore(world, spec, tmp_path):
    for kw in ({"fisher_estimator": "true"}, {"fisher_normalization": "max"}):
        with pytest.raises(ValueError):
            EWC(lam=1.0, **kw)
    assert EWC(lam=1.0).config()["fisher_estimator"] == "empirical"
    toy = Toy(spec.d_state, spec.d_action)
    m = EWC(lam=2.0, fisher_samples=4)
    m.on_task_end(toy, 0, context=ctx(world, 0))
    run = tmp_path / "run"
    (path,) = m.save_artifacts(run / "method", 0, context=ctx(world, 0))
    payload = torch.load(path, weights_only=False)
    for key in ("fisher_estimator", "fisher_normalization"):
        del payload["config"][key]          # an artifact written before the two options existed
    torch.save(payload, path)
    extra = {"method_artifacts": [{"path": "method/ewc_task0.pt", "sha256": file_sha256(path)}]}
    assert EWC(lam=2.0, fisher_samples=4).restore_state(extra, run, 0)["fisher_tasks"] == [0]
    with pytest.raises(ValueError, match="config"):
        EWC(lam=2.0, fisher_samples=4, fisher_estimator="sampled").restore_state(extra, run, 0)
    with pytest.raises(ValueError, match="config"):
        EWC(lam=2.0, fisher_samples=4, fisher_normalization="mean").restore_state(extra, run, 0)


def test_sampled_fisher_restores_train_mode(spec, short):
    from flowcl.methods.ewc import sampled_fisher

    toy = ToyFlow(spec.d_state, spec.d_action)
    toy.eval()
    seen = []
    orig = toy.forward
    toy.forward = lambda batch, generator=None: (seen.append(toy.training), orig(batch, generator))[1]
    sampled_fisher(toy, short, n_samples=2, seed=1, eps_seed=2, device="cpu")
    assert seen == [True, True] and not toy.training
