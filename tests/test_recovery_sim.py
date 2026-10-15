"""Simulator checks for the recovery study (run in its smoke; marked ``sim``): ``run_episode`` from an
evaluation initial state reproduces ``rollout()`` exactly, and the determinism gate passes on real
nominal rollouts (the interior-query suffix from the saved generator state; order invariance)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from flowcl.experiments import recovery_study as rs
from flowcl.utils.seeding import derive_seed

pytestmark = pytest.mark.sim


@pytest.fixture(scope="module")
def world():
    rule = rs.load_rule()
    loaded, env, demos, sigma, cfg = rs._setup(rule, rule["tasks"][0], rs._root(), "cuda" if torch.cuda.is_available()
                                                 else "cpu")
    yield rule, loaded, env, demos, cfg
    env.close()


def test_run_episode_equals_rollout(world):
    rule, loaded, env, demos, cfg = world
    run_id = "recovery_sim_test"
    seed = derive_seed(run_id, env.task_key, 0)
    ref = env.rollout(loaded.policy, 0, loaded.stats, run_id, dataclasses.replace(cfg, record_video=True))
    mine = env.run_episode(loaded.policy, loaded.stats, env.init_states[0], seed,
                           torch.Generator(device="cpu").manual_seed(seed), cfg)
    assert (mine.success, mine.n_steps) == (ref.success, ref.n_steps)
    assert np.array_equal(mine.actions.astype(np.float32), ref.trace.actions)


def test_determinism_gate_on_real_rollouts(world):
    rule, loaded, env, demos, cfg = world
    ns, key = rule["namespace"], rule["tasks"][0]["key"]
    nominals = {}
    i = 0
    while sum(r.success for _, r in nominals.values()) < 1 and i < 6:
        nominals[i] = rs._nominal(env, loaded, demos, i, ns, key, cfg)
        i += 1
    out = rs.gate(env, loaded, demos, nominals, ns, key, cfg, {**rule, "gate_rollouts": 1})
    assert out["checks"], "no successful nominal rollout in 6 tries"
    assert out["passed"], out
    assert out["hard_reset"] in (True, None)
