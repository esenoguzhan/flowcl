"""Sequential merged LoRA (flowcl/models/lora.py, flowcl/methods/lora.py): the adapter's arithmetic and
initialisation against Hu et al. and loralib, the frozen base and the merge, Task 1 as seq_ft, the
per-stage learning rate, the per-stage trainable counts, and resume with the method's history."""

from __future__ import annotations

import dataclasses
import json
import math

import pytest
import torch
from torch import nn

import flowcl.train.continual as continual_module
from flowcl.data.config import load_embodiment_spec, load_method_config
from flowcl.data.dataset import ChunkedActionDataset
from flowcl.data.stats import compute_stats
from flowcl.envs.libero_env import EvalConfig
from flowcl.methods.base import TaskContext, build_method
from flowcl.methods.gpm import allowlist
from flowcl.methods.lora import LoRA, init_seed
from flowcl.methods.seq_ft import SeqFT
from flowcl.models import lora as adapters
from flowcl.train.continual import method_stage_config, run_continual
from flowcl.train.trainer import TrainConfig, build_optimizer, train_one_task
from test_continual import TINY_POLICY, fake_evaluation
from test_gpm import fresh_policy, small_cfg
from test_replay import TASKS, demos
from test_resume import trio

NS = "seq_hetero__seq_ft__seed0"


@pytest.fixture(scope="module")
def spec():
    return load_embodiment_spec("libero_franka")


def data_for(spec, n=2):
    eps = {t: demos(spec, t, offset=10 * j) for j, t in enumerate(TASKS[:n])}
    stats = compute_stats(eps[TASKS[0]], embodiment=spec.name, task_id=TASKS[0])
    return {j: ChunkedActionDataset(eps[t], spec, stats) for j, t in enumerate(TASKS[:n])}


def ctx(data, j, ns=NS):
    return TaskContext(task_key=TASKS[j], dataset=data[j], device="cpu", seed_namespace_run_id=ns, batch_size=4)


# ---- the adapter ----------------------------------------------------------------------------------


def test_adapter_arithmetic_merge_and_detach():
    torch.manual_seed(0)
    layer = nn.Linear(12, 7)
    x = torch.randn(5, 12)
    base = layer(x).detach()
    keys = list(layer.state_dict())
    w0 = layer.weight.detach().clone()
    n = adapters.attach(layer, 3, 16.0, torch.Generator().manual_seed(1))
    assert n == 3 * (12 + 7) and adapters.is_adapted(layer)
    assert torch.equal(layer.lora_B, torch.zeros(7, 3)) and torch.equal(layer(x), base)  # B = 0: the base
    with torch.no_grad():
        layer.lora_B.normal_()
    hooked = layer(x).detach()
    scale = 16.0 / 3
    want = x @ (w0 + scale * layer.lora_B.detach() @ layer.lora_A.detach()).T + layer.bias.detach()
    assert torch.allclose(hooked, want, atol=1e-5)
    assert torch.equal(layer.weight, w0)  # the base is untouched until the merge
    update = adapters.delta(layer)
    info = adapters.merge_and_detach(layer)
    assert torch.equal(layer.weight, w0 + update)  # W <- W + (alpha/r) B A, in fp32
    assert list(layer.state_dict()) == keys and not adapters.is_adapted(layer)
    assert torch.allclose(layer(x), hooked, atol=1e-5)  # merged output = adapted output
    assert info["delta_fro"] == pytest.approx(float(update.norm()))
    assert info["relative"] == pytest.approx(float(update.norm() / w0.norm()))
    with pytest.raises(RuntimeError, match="no adapter"):
        adapters.merge_and_detach(layer)
    with pytest.raises(TypeError):
        adapters.attach(nn.Conv1d(2, 2, 1), 2, 16.0, torch.Generator())


def test_initialisation_is_the_released_codes_and_leaves_the_global_rng():
    layer = nn.Linear(512, 64)
    before = torch.get_rng_state()
    adapters.attach(layer, 64, 16.0, torch.Generator().manual_seed(7))
    assert torch.equal(torch.get_rng_state(), before)
    want = torch.empty(64, 512)
    nn.init.kaiming_uniform_(want, a=math.sqrt(5), generator=torch.Generator().manual_seed(7))
    assert torch.equal(layer.lora_A.detach(), want)
    a = layer.lora_A.detach()
    assert a.abs().max() <= 1 / math.sqrt(512)
    assert float(a.var()) == pytest.approx(1 / (3 * 512), rel=0.03)  # U(+-1/sqrt(d_in))
    assert "kaiming_uniform_(a=sqrt(5))" in adapters.INIT_SPEC


# ---- the method on a policy -----------------------------------------------------------------------


def test_frozen_base_adapters_only_and_the_merge(spec):
    data = data_for(spec)
    policy = fresh_policy(spec)
    keys = list(policy.state_dict())
    method = LoRA(rank=2, lr=1e-3)
    method.on_task_start(policy, 1, context=ctx(data, 1))
    trainable = {n for n, p in policy.named_parameters() if p.requires_grad}
    layers = policy.projectable_layers()
    assert trainable == {f"{e.name}.lora_{x}" for e in layers for x in "AB"}
    assert {f"{e.name}.weight" for e in layers} == set(allowlist(policy))  # GPM's T2 support
    count = 2 * sum(e.module.in_features + e.module.out_features for e in layers)
    hist = method.history["1"]
    assert hist["trainable_params"] == count == sum(p.numel() for p in policy.parameters() if p.requires_grad)
    assert hist["training_state_bytes_estimate"] == 16 * count and hist["adapted_layers"] == len(layers)
    assert hist["init_seed"] == init_seed(NS, TASKS[1], 1)
    optimizer = build_optimizer(policy, small_cfg())
    in_opt = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert in_opt == {id(p) for p in policy.parameters() if p.requires_grad}
    for e in layers:
        method_bias = e.module.bias
        assert method_bias is None or not method_bias.requires_grad
    with pytest.raises(RuntimeError, match="still attached"):
        method.on_task_start(policy, 1, context=ctx(data, 1))
    method.on_task_end(policy, 1, context=ctx(data, 1))
    assert list(policy.state_dict()) == keys
    with pytest.raises(ValueError, match="seed namespace"):
        LoRA(rank=2, lr=1e-3).on_task_start(policy, 1, context=ctx(data, 1, ns=None))


def test_training_moves_only_adapters_and_the_merge_adds_exactly_their_product(spec):
    data = data_for(spec)
    policy = fresh_policy(spec)
    base = {n: p.detach().clone() for n, p in policy.named_parameters()}
    method = LoRA(rank=2, lr=1e-3)
    gen = lambda: torch.Generator().manual_seed(3)  # noqa: E731
    first = policy(next(iter(torch.utils.data.DataLoader(data[1], batch_size=4, collate_fn=_collate()))),
                   generator=gen())["loss"].detach()
    deltas, unchanged = {}, []

    def on_step(step, outputs):
        unchanged.append(all(torch.equal(p, base[n]) for n, p in policy.named_parameters() if "lora_" not in n))
        if step == 2:
            deltas.update({e.name: adapters.delta(e.module) for e in policy.projectable_layers()})

    cfg = dataclasses.replace(small_cfg(steps=3), lr=1e-2)
    train_one_task(policy, data[1], cfg, method=method, task_idx=1, generator=torch.Generator().manual_seed(0),
                   context=ctx(data, 1), on_step=on_step)
    assert unchanged == [True, True, True]
    params = dict(policy.named_parameters())
    registry = {f"{e.name}.weight" for e in policy.projectable_layers()}
    for n, p in params.items():
        if n in registry:
            assert torch.equal(p, base[n] + deltas[n[: -len(".weight")]]), n
        else:
            assert torch.equal(p, base[n]), n
    assert any(float(d.norm()) > 0 for d in deltas.values())
    merge = method.history["1"]["merge"]
    assert set(merge["per_layer"]) == {e.name for e in policy.projectable_layers()}
    assert merge["relative_max"] >= merge["relative_median"] > 0
    # B = 0 at attachment: the first adapted forward is the base's
    fresh = fresh_policy(spec)
    LoRA(rank=2, lr=1e-3).on_task_start(fresh, 1, context=ctx(data, 1))
    again = fresh(next(iter(torch.utils.data.DataLoader(data[1], batch_size=4, collate_fn=_collate()))),
                  generator=gen())["loss"].detach()
    assert torch.equal(first, again)


def _collate():
    from flowcl.data.dataset import collate_chunks
    return collate_chunks


def test_task1_is_seq_ft_bitwise_and_the_stage_lr(spec):
    data = data_for(spec, 1)
    a, b = fresh_policy(spec), fresh_policy(spec)
    for policy, method in ((a, LoRA(rank=4, lr=1e-3)), (b, SeqFT())):
        train_one_task(policy, data[0], small_cfg(), method=method, task_idx=0,
                       generator=torch.Generator().manual_seed(0), context=ctx(data, 0))
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))
    cfg = TrainConfig(lr=1e-4)
    method = LoRA(rank=4, lr=3e-4)
    assert method.stage_lr(0) is None and method_stage_config(cfg, method, 0) is cfg
    assert method_stage_config(cfg, method, 2).lr == 3e-4 and method_stage_config(cfg, SeqFT(), 2) is cfg


def test_configuration_refusals_and_the_production_config():
    for kw in ({"lr": 1e-3}, {"rank": 4}, {"rank": 0, "lr": 1e-3}, {"rank": 2.5, "lr": 1e-3},
               {"rank": 4, "lr": -1.0}):
        with pytest.raises(ValueError):
            LoRA(**kw)
    with pytest.raises(TypeError):
        LoRA(rank=4, lr=1e-3, dropout=0.1)
    name, kw = load_method_config("lora")  # filled from the bracket's bundle (rank 4, lr 3e-5)
    assert name == "lora" and kw == {"rank": 4, "lr": 3e-5, "alpha": 16}
    assert build_method(name, **kw).stage_lr(1) == 3e-5
    with pytest.raises(ValueError, match="bundle"):
        build_method(name, **{**kw, "rank": None, "lr": None})
    method = LoRA(rank=4, lr=1e-3)
    state = {"method_state": {**LoRA(rank=8, lr=1e-3).state_dict()}}
    with pytest.raises(ValueError, match="not this LoRA"):
        method.restore_state(state, None, 1)
    state = {"method_state": {**method.state_dict(), "history": {}}}
    with pytest.raises(ValueError, match="history covers"):
        method.restore_state(state, None, 1)
    assert method.restore_state({"method_state": method.state_dict()}, None, 0) == {"history_tasks": []}


# ---- through the runner ---------------------------------------------------------------------------


def tiny_cfg():
    return TrainConfig(steps=2, batch_size=2, num_workers=0, device="cpu", log_every=0, warmup_steps=1)


def test_runner_identity_stage_lr_counts_and_resume_with_history(dataset_dir, tmp_path, monkeypatch):
    from flowcl.experiments.resume_check import compare_resume

    monkeypatch.setattr(continual_module, "evaluate_tasks", fake_evaluation([]))
    common = dict(spec=load_embodiment_spec("libero_franka"), policy_config=TINY_POLICY, train_cfg=tiny_cfg(),
                  eval_cfg=EvalConfig(n_episodes=2), seed=0, dataset_dir=dataset_dir, pretrained=False)
    ref = run_continual(trio(), method_name="seq_ft", results_root=tmp_path / "a", **common)
    ref_run = tmp_path / "a" / ref.run_id
    kw = {"rank": 2, "lr": 5e-4, "alpha": 16}
    src = run_continual(trio(), method_name="lora", method_kwargs=kw, results_root=tmp_path / "b",
                        identity_reference_run=ref_run, identity_stages=(0,), **common)
    run = tmp_path / "b" / src.run_id
    assert src.identity_checks["0"]["passed"]  # Task 1 is seq_ft's, bit for bit
    saved = json.loads((run / "result.json").read_text())
    lrs = [s["train_config"]["lr"] for s in saved["stages"]]
    assert lrs == [tiny_cfg().lr, 5e-4, 5e-4]
    by_stage = saved["systems"]["trainable_params_by_stage"]
    assert by_stage["0"] == saved["systems"]["trainable_params"] > by_stage["1"] == by_stage["2"] > 0
    assert [s["trainable_params"] for s in saved["stages"]] == [by_stage[k] for k in ("0", "1", "2")]
    state = torch.load(run / "checkpoints" / "stage2.pt", weights_only=False)["extra"]["method_state"]
    assert sorted(state["history"]) == ["1", "2"] and state["rank"] == 2
    resumed = run_continual(trio(), method_name="lora", method_kwargs=kw, results_root=tmp_path / "c",
                            resume_run=run, start_stage=2, **common)
    out = tmp_path / "c" / resumed.run_id
    report = compare_resume(run, out, 2, all_artifacts=True, method_state=True)
    assert report["passed"], report
    assert report["method_artifacts"]["method_state"]["differences"] == []
    assert json.loads((out / "resume.json").read_text())["method_state"] == {"history_tasks": [1]}
    resumed_saved = json.loads((out / "result.json").read_text())
    assert resumed_saved["systems"]["trainable_params_by_stage"] == by_stage
