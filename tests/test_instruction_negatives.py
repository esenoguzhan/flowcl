"""Instruction negatives (flowcl/methods/instruction_negatives.py): the bank, the hold target, the
inactive no-op, the weight-0 identity against plain GPM and allowlist_ft at both stages, and the
equality of the two variants' stage 0."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from flowcl.data.dataset import ChunkedActionDataset
from flowcl.data.stats import compute_stats
from flowcl.methods.allowlist_ft import AllowlistFT
from flowcl.methods.base import TaskContext, build_method
from flowcl.methods.gpm import GPM
from flowcl.methods.instruction_negatives import (
    AllowlistFTNeg,
    GPMNeg,
    gripper_index,
    hold_rows,
    load_bank,
)
from flowcl.methods.seq_ft import SeqFT
from flowcl.train.trainer import train_one_task
from test_gpm import capture_yaml, fresh_policy, make_episode, small_cfg, spec  # noqa: F401

BANK = "instruction_negatives_bank"
KEYS = ["toy/one", "toy/two"]


@pytest.fixture(scope="module")
def world(spec):
    eps = {k: [make_episode(spec, k, 14, 10 * i + s) for s in range(2)] for i, k in enumerate(KEYS)}
    stats = compute_stats(eps[KEYS[0]], embodiment=spec.name, task_id=KEYS[0])
    return {k: ChunkedActionDataset(e, spec, stats) for k, e in eps.items()}


def ctx(world, j):
    return TaskContext(task_key=KEYS[j], dataset=world[KEYS[j]], device="cpu", seed_namespace_run_id="toy_ns",
                       method_run_id="toy_run", batch_size=4)


def run(method, world, spec, stages=2):
    policy = fresh_policy(spec)
    snaps = []
    for j in range(stages):
        train_one_task(policy, world[KEYS[j]], small_cfg(), method=method, task_idx=j,
                       generator=torch.Generator().manual_seed(j), context=ctx(world, j))
        snaps.append({n: p.detach().clone() for n, p in policy.state_dict().items()})
    return snaps


def equal(a, b):
    return sorted(a) == sorted(b) and all(torch.equal(a[n], b[n]) for n in a)


def test_the_registered_bank_is_valid_and_avoids_the_held_out_nouns():
    bank = load_bank(BANK)
    assert len(bank["strings"]) == 15 and bank["n_noun_swaps"] == 8
    for s in bank["strings"]:
        assert "stove" not in s and "cabinet" not in s
        assert not ("bowl" in s and "plate" in s)
    assert "put the bowl on the plate" not in bank["strings"]


@pytest.mark.parametrize("body,match", [
    ("forbidden_words: [stove]\nnoun_swaps: [put the bowl on the stove]\n", "forbidden"),
    ("noun_swaps: [a b, a b]\n", "duplicates"),
    ("noun_swaps: [Put the bowl]\n", "lower case"),
    ("noun_swaps: []\n", "no strings"),
])
def test_an_invalid_bank_is_refused(tmp_path, body, match):
    path = tmp_path / "bank.yaml"
    path.write_text(body)
    with pytest.raises(ValueError, match=match):
        load_bank(str(path))


def test_hold_rows_zero_pose_previous_gripper_open_at_start():
    action = np.array([[0.3, 0, 0, 0, 0, 0, -1], [0.1, 0, 0, 0, 0, 0, 1], [0.2, 0, 0, 0, 0, 0, 1]], dtype=np.float32)
    rows = hold_rows(action, 6)
    assert np.all(rows[:, :6] == 0)
    assert rows[:, 6].tolist() == [-1.0, -1.0, 1.0]


def test_the_hold_target_is_refused_for_other_control_modes(spec):
    assert gripper_index(spec) == 6
    piper = dataclasses.replace(spec, action=dataclasses.replace(spec.action, control_mode="joint_position"))
    with pytest.raises(ValueError, match="osc_pose_delta"):
        gripper_index(piper)


def test_registry_names_display_names_and_unknown_keys():
    gpm = build_method("gpm_neg", negative_bank=BANK)
    allow = build_method("allowlist_ft_neg", negative_bank=BANK)
    assert isinstance(gpm, GPMNeg) and isinstance(allow, AllowlistFTNeg)
    assert gpm.display_name == "gpm_projected_adam_neg"
    assert gpm.config()["negative_bank_sha256"] == load_bank(BANK)["sha256"]
    with pytest.raises(TypeError):
        build_method("allowlist_ft_neg", negative_bank=BANK, bogus=1)


def test_an_inactive_stage_returns_the_identical_loss_tensor(world, spec):
    method = AllowlistFTNeg(negative_bank=BANK, negative_stages=(0,))
    method.on_task_start(fresh_policy(spec), 1, context=ctx(world, 1))
    loss = torch.tensor(1.0)
    assert method.modify_loss(loss, {}, None) is loss
    assert method.negative_logs["1"] == {"active": False}


def test_the_negative_batch_keeps_observations_and_draws_bank_strings(world, spec):
    method = AllowlistFTNeg(negative_bank=BANK)
    policy = fresh_policy(spec)
    method.on_task_start(policy, 0, context=ctx(world, 0))
    from flowcl.data.dataset import collate_chunks

    data = world[KEYS[0]]
    batch = collate_chunks([data[i] for i in (0, 5, 13)])
    neg = method.negative_batch(batch)
    assert neg["images"] is batch["images"] and neg["state"] is batch["state"]
    assert set(neg["language"]) <= set(load_bank(BANK)["strings"])
    assert torch.all(neg["action_mask"] == 1) and torch.all(neg["actions"][..., :6] == 0)
    for row, (e, t) in enumerate(zip(batch["episode_idx"].tolist(), batch["t"].tolist())):
        expected = -1.0 if t == 0 else float(data.episodes[e].action[t - 1, 6])
        assert torch.all(neg["actions"][row, :, 6] == expected)
    assert all(method.negative_logs["0"]["text_cache"][s] == "fp32" for s in method._neg_strings)


def test_the_stages_own_instruction_is_excluded(spec):
    ep = make_episode(spec, "toy/x", 10, 0)
    ep = dataclasses.replace(ep, language="put the bowl on the rack")
    data = ChunkedActionDataset([ep], spec, compute_stats([ep], embodiment=spec.name, task_id="toy/x"))
    method = AllowlistFTNeg(negative_bank=BANK)
    method.on_task_start(None, 0, context=TaskContext(task_key="toy/x", dataset=data, device="cpu",
                                                     seed_namespace_run_id="ns"))
    assert method.negative_logs["0"]["excluded_own"] == ["put the bowl on the rack"]
    assert "put the bowl on the rack" not in method._neg_strings


def test_weight_zero_equals_plain_gpm_and_allowlist_at_both_stages(world, spec, capture_yaml):
    gpm_kwargs = dict(update_memory=True, capture_config=capture_yaml, log_interval=1)
    plain = run(GPM(**gpm_kwargs), world, spec)
    neg0 = run(GPMNeg(negative_bank=BANK, negative_weight=0.0, **gpm_kwargs), world, spec)
    assert equal(plain[0], neg0[0]) and equal(plain[1], neg0[1])
    allow = run(AllowlistFT(), world, spec)
    allow0 = run(AllowlistFTNeg(negative_bank=BANK, negative_weight=0.0), world, spec)
    assert equal(allow[0], allow0[0]) and equal(allow[1], allow0[1])
    seq = run(SeqFT(), world, spec, stages=1)
    assert equal(seq[0], plain[0])


def test_the_two_variants_share_stage0_and_it_differs_from_seq_ft(world, spec, capture_yaml):
    gpm = GPMNeg(negative_bank=BANK, update_memory=True, capture_config=capture_yaml, log_interval=1,
                 negative_log_interval=1)
    a = run(gpm, world, spec)
    b = run(AllowlistFTNeg(negative_bank=BANK), world, spec, stages=1)
    seq = run(SeqFT(), world, spec, stages=1)
    assert equal(a[0], b[0])
    assert not equal(a[0], seq[0])
    logs = gpm.state_dict()["instruction_negatives"]["logs"]
    assert logs["0"]["active"] is True and logs["0"]["n_steps"] == 3 and len(logs["0"]["steps"]) == 3
    assert logs["1"] == {"active": False}
