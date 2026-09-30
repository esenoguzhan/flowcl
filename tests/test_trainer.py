"""Trainer options: vanilla SGD, ``stop_after`` (a bitwise prefix of the full schedule), and
``schedule_factor`` (exactly the scheduler's multiplier). Defaults are unchanged."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from flowcl.data.dataset import ChunkedActionDataset
from flowcl.data.stats import compute_stats
from flowcl.train.trainer import (
    TrainConfig,
    build_optimizer,
    build_scheduler,
    schedule_factor,
    train_one_task,
)
from test_low_update import KEYS, episodes, spec, tiny_policy  # noqa: F401


class Small(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.a = nn.Linear(4, 3)
        self.b = nn.Linear(3, 2, bias=False)
        self.b.weight.requires_grad_(False)


def test_default_optimizer_is_the_unchanged_adamw():
    model = Small()
    opt = build_optimizer(model, TrainConfig())
    assert type(opt) is torch.optim.AdamW
    decay, no_decay = opt.param_groups
    assert decay["weight_decay"] == 1e-4 and no_decay["weight_decay"] == 0.0
    assert [p is model.a.weight for p in decay["params"]] == [True]
    assert [p is model.a.bias for p in no_decay["params"]] == [True]


def test_sgd_is_vanilla_and_refuses_weight_decay():
    model = Small()
    opt = build_optimizer(model, TrainConfig(optimizer="sgd", weight_decay=0.0, lr=0.1))
    assert type(opt) is torch.optim.SGD
    (group,) = opt.param_groups
    assert group["momentum"] == 0.0 and group["weight_decay"] == 0.0 and group["dampening"] == 0.0
    assert not group["nesterov"] and len(group["params"]) == 2  # the frozen weight is excluded
    x = torch.randn(8, 4)
    model.b(model.a(x)).pow(2).mean().backward()
    before = model.a.weight.detach().clone()
    grad = model.a.weight.grad.detach().clone()
    opt.step()
    assert torch.equal(model.a.weight.detach(), before - 0.1 * grad)
    with pytest.raises(ValueError, match="weight_decay must be 0"):
        build_optimizer(model, TrainConfig(optimizer="sgd"))
    with pytest.raises(ValueError, match="unknown optimizer"):
        build_optimizer(model, TrainConfig(optimizer="adam"))


@pytest.mark.parametrize("schedule", ["cosine", "constant"])
def test_schedule_factor_is_exactly_the_schedulers_multiplier(schedule):
    cfg = TrainConfig(steps=40, warmup_steps=5, lr=1.0, schedule=schedule)
    opt = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=1.0)
    scheduler = build_scheduler(opt, cfg)
    for step in range(cfg.steps):
        assert opt.param_groups[0]["lr"] == schedule_factor(cfg, step)
        opt.step()
        scheduler.step()


def test_stop_after_trains_a_bitwise_prefix_of_the_full_schedule(spec):
    eps = episodes(spec, KEYS[3])
    stats = compute_stats(eps, embodiment=spec.name, task_id=KEYS[3])
    dataset = ChunkedActionDataset(eps, spec, stats)
    base = dict(steps=6, batch_size=2, warmup_steps=2, device="cpu", log_every=0, num_workers=0)

    full, snapshot = tiny_policy(spec), {}

    def at_three(step, outputs):
        if step == 2:
            snapshot.update({k: v.detach().clone() for k, v in full.state_dict().items()})

    log_full = train_one_task(full, dataset, TrainConfig(**base), generator=torch.Generator().manual_seed(0),
                              on_step=at_three)
    prefix = tiny_policy(spec)
    log = train_one_task(prefix, dataset, TrainConfig(**base, stop_after=3),
                         generator=torch.Generator().manual_seed(0))
    assert log.steps == 3 and log.losses == log_full.losses[:3]
    assert all(torch.equal(v, snapshot[k]) for k, v in prefix.state_dict().items())
    for bad in (0, 7):
        with pytest.raises(ValueError, match="stop_after"):
            train_one_task(tiny_policy(spec), dataset, TrainConfig(**base, stop_after=bad))
