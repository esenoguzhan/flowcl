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
