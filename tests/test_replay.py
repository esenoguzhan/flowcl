"""Replay: the T1 no-op (stage 0 is seq_ft's path), the batch composition and the equal split across
old tasks, generator independence, the buffer's demos, and the artifact round-trip with SHA checks."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from flowcl.data.dataset import ChunkedActionDataset
from flowcl.data.episode import Episode
from flowcl.data.stats import compute_stats
from flowcl.methods.base import TaskContext
from flowcl.methods.replay import Replay, equal_split, episodes_fingerprint
from flowcl.methods.seq_ft import SeqFT
from flowcl.train.trainer import train_one_task
from flowcl.utils.run import file_sha256
from test_gpm import fresh_policy, small_cfg, spec  # noqa: F401

TASKS = ["toy/one", "toy/two", "toy/three", "toy/four"]


def demos(spec, task, n=3, offset=0, length=12):
    out = []
    for i in range(n):
        rng = np.random.default_rng(offset + i)
        h, w = spec.observation.image_size
        out.append(Episode(
            images={c: rng.integers(0, 255, (length, h, w, 3), dtype=np.uint8) for c in spec.cameras},
            state=rng.normal(size=(length, spec.d_state)).astype(np.float32),
            action=rng.uniform(-1, 1, size=(length, spec.d_action)).astype(np.float32),
            language=f"do {task}", task_id=task, embodiment=spec.name, meta={"demo_idx": i}))
    return out


@pytest.fixture(scope="module")
def world(spec):
    eps = {t: demos(spec, t, offset=10 * j) for j, t in enumerate(TASKS)}
    stats = compute_stats(eps[TASKS[0]], embodiment=spec.name, task_id=TASKS[0])
    return {t: ChunkedActionDataset(e, spec, stats) for t, e in eps.items()}


def ctx(world, j, batch_size=8):
    key = TASKS[j]
    return TaskContext(task_key=key, dataset=world[key], device="cpu", seed_namespace_run_id="ns",
                       method_run_id="run", batch_size=batch_size)


def through(method, world, upto, batch_size=8):
    """Start and end tasks 0..upto-1, then start task ``upto``."""
    for j in range(upto):
        method.on_task_start(None, j, context=ctx(world, j, batch_size))
        method.on_task_end(None, j, context=ctx(world, j, batch_size))
    method.on_task_start(None, upto, context=ctx(world, upto, batch_size))
    return method


def test_equal_split():
    assert equal_split(32, 1) == [32] and equal_split(32, 2) == [16, 16] and equal_split(32, 3) == [11, 11, 10]
    with pytest.raises(ValueError):
        equal_split(32, 0)


def test_t1_is_a_no_op_and_the_batch_mixes_half_and_half(world):
    r = Replay(buffer_demos=2)
    r.on_task_start(None, 0, context=ctx(world, 0))
    assert r.build_batch(world[TASKS[0]], 0) is None
    r.on_task_end(None, 0, context=ctx(world, 0))
    assert [ep.meta["demo_idx"] for ep in r._buffer[TASKS[0]]["episodes"]] == [0, 1]
    r.on_task_start(None, 1, context=ctx(world, 1))
    batch = r.build_batch(world[TASKS[1]], 1)
    assert batch["actions"].shape[0] == 8
    assert batch["task_id"] == [TASKS[1]] * 4 + [TASKS[0]] * 4
    buffered = {(TASKS[0], int(e)) for t, e in zip(batch["task_id"], batch["episode_idx"]) if t == TASKS[0]}
    assert all(e in (0, 1) for _, e in buffered)  # only the buffered demos


def test_equal_split_across_three_old_tasks(world):
    r = through(Replay(buffer_demos=2), world, 3, batch_size=64)
    batch = r.build_batch(world[TASKS[3]], 3)
    counts = [batch["task_id"].count(t) for t in TASKS]
    assert counts == [11, 11, 10, 32]


def test_batches_are_deterministic_and_leave_global_rng_alone(world):
    torch.manual_seed(123)
    before = torch.get_rng_state()
    a = through(Replay(buffer_demos=2), world, 2).build_batch(world[TASKS[2]], 2)
    assert torch.equal(torch.get_rng_state(), before)
    b = through(Replay(buffer_demos=2), world, 2).build_batch(world[TASKS[2]], 2)
    assert a["task_id"] == b["task_id"] and torch.equal(a["actions"], b["actions"])


def test_buffer_requires_the_first_demos_and_the_batch_size(world, spec):
    stats = world[TASKS[0]].stats
    shuffled = demos(spec, TASKS[0])
    for i, ep in enumerate(shuffled):
        ep.meta["demo_idx"] = i + 1  # demos 1..3: demo 0 is missing
    bad = TaskContext(task_key=TASKS[0], dataset=ChunkedActionDataset(shuffled, spec, stats), device="cpu",
                      seed_namespace_run_id="ns", batch_size=8)
    with pytest.raises(ValueError, match="expected demos 0..1"):
        Replay(buffer_demos=2).on_task_end(None, 0, context=bad)
    r = Replay(buffer_demos=2)
    r.on_task_end(None, 0, context=ctx(world, 0))
    no_bs = TaskContext(task_key=TASKS[1], dataset=world[TASKS[1]], device="cpu", seed_namespace_run_id="ns")
    with pytest.raises(RuntimeError, match="batch size"):
        r.on_task_start(None, 1, context=no_bs)
    for kw in ({"buffer_demos": 0}, {"replay_fraction": 1.0}, {"replay_fraction": 0.0}):
        with pytest.raises(ValueError):
            Replay(**kw)


def checkpoint_extra(method, paths, run):
    return {"method_artifacts": [{"path": str(p.relative_to(run)), "sha256": file_sha256(p)} for p in paths],
            "method_state": method.state_dict()}


def test_artifacts_round_trip_restore_and_rewrite(world, tmp_path):
    r = through(Replay(buffer_demos=2), world, 2)
    r.on_task_end(None, 2, context=ctx(world, 2))
    run = tmp_path / "run"
    paths = r.save_artifacts(run / "method", 2, context=ctx(world, 2))
    assert [p.name for p in paths] == ["replay_task0.pt", "replay_task1.pt", "replay_task2.pt"]
    extra = checkpoint_extra(r, paths, run)
    restored = Replay(buffer_demos=2)
    out = restored.restore_state(extra, run, 2)
    assert out["tasks"] == TASKS[:3]
    for key in TASKS[:3]:
        assert restored._buffer[key]["fingerprint"] == r._buffer[key]["fingerprint"] == \
            episodes_fingerprint(r._buffer[key]["episodes"])
    assert restored.stored_bytes() == r.stored_bytes() > 0 and not restored.is_exemplar_free
    # A resumed run writes every earlier buffer file into its own directory, identical in content.
    new_paths = restored.save_artifacts(tmp_path / "resumed" / "method", 2, context=ctx(world, 2))
    a = torch.load(paths[0], weights_only=False)
    b = torch.load(new_paths[0], weights_only=False)
    assert a["fingerprint"] == b["fingerprint"] and torch.equal(a["episodes"][0]["state"], b["episodes"][0]["state"])
    # Same restored state, same batches.
    restored.on_task_start(None, 3, context=ctx(world, 3))
    r.on_task_start(None, 3, context=ctx(world, 3))
    assert torch.equal(restored.build_batch(world[TASKS[3]], 3)["actions"], r.build_batch(world[TASKS[3]], 3)["actions"])
    # Tampering and missing files are refused.
    with open(paths[1], "ab") as f:
        f.write(b"x")
    with pytest.raises(ValueError, match="SHA-256"):
        Replay(buffer_demos=2).restore_state(extra, run, 2)
    extra["method_artifacts"] = extra["method_artifacts"][:1]
    with pytest.raises(ValueError, match="names no"):
        Replay(buffer_demos=2).restore_state(extra, run, 2)
    with pytest.raises(ValueError, match="no replay buffer"):
        Replay(buffer_demos=2).restore_state({"method_artifacts": []}, run, 2)


def test_stage0_is_bitwise_seq_ft(spec, world):
    data = world[TASKS[0]]
    weights = []
    for method in (SeqFT(), Replay(buffer_demos=2)):
        policy = fresh_policy(spec)
        train_one_task(policy, data, small_cfg(), method=method, task_idx=0,
                       generator=torch.Generator().manual_seed(0), context=ctx(world, 0, batch_size=4))
        weights.append({n: p.detach().clone() for n, p in policy.named_parameters()})
    assert all(torch.equal(weights[0][n], weights[1][n]) for n in weights[0])
