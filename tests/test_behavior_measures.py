"""Track 2's offline measures (flowcl/analysis/behavior.py and the pure parts of
behavior_predictors.run_measures): M0 equals the diagnostics' probe loss on the same bank, the M1 and M2
slices partition the elements, the per-demonstration units sum to the batch units, M3 is deterministic
with best-of-K <= mean, the global RNG is untouched, and the instrument checks, the suite gate and the
atomic publication refuse what they must."""

from __future__ import annotations

import copy
import json

import pytest
import torch

from flowcl.analysis import behavior as bh
from flowcl.analysis.probes import probe_batch_losses, probe_loss
from flowcl.experiments import behavior_predictors as bp
from test_gpm import fresh_policy, spec  # noqa: F401
from test_projected_lora import data_for

KEY = "toy/one"


def measures_cfg():
    m = copy.deepcopy(bp.load_config()["measures"])
    m["probe"] = {**m["probe"], "batch_size": 8, "n_batches": 4}
    m["m3"] = {**m["m3"], "n_batches": 2, "k": 3, "euler_steps": 2}
    return m


@pytest.fixture
def cell(spec):
    torch.manual_seed(0)
    policy = fresh_policy(spec)
    dataset = data_for(spec, [KEY])[KEY]
    return policy, dataset


def test_m0_is_the_probe_loss_and_the_slices_partition(cell):
    policy, dataset = cell
    m = measures_cfg()
    rng = torch.get_rng_state()
    out = bh.cell_measures(policy, dataset, KEY, m, "cpu")
    assert torch.equal(rng, torch.get_rng_state())                   # the global RNG is untouched
    el = out["elements"]
    assert el["values"]["m0"] == pytest.approx(probe_loss(policy, dataset, m["probe"], "cpu"), rel=1e-6)
    per_batch = probe_batch_losses(policy, dataset, m["probe"], "cpu")
    assert len(el["batches"]) == len(per_batch) == out["n_batches"]
    for b, (loss, n) in zip(el["batches"], per_batch):
        assert b["m0"][1] == n and b["m0"][0] / b["m0"][1] == pytest.approx(loss, rel=1e-5)

    def total(keys, i):
        return sum(sum(b[k][i] for b in el["batches"]) for k in keys)

    m1 = [k for k in el["values"] if k.startswith("m1_")]
    comps = [f"m2_{c}" for c in m["m2"]["components"]]
    for keys in (m1, comps, ["m2_executed", "m2_rest"]):
        assert total(keys, 1) == total(["m0"], 1)
        assert total(keys, 0) == pytest.approx(total(["m0"], 0), rel=1e-6)
    for k in el["values"]:                                            # per-demo units sum to the batch units
        for i in (0, 1):
            assert sum(d[k][i] for d in el["demos"].values()) == pytest.approx(total([k], i), rel=1e-9)


def test_m3_is_deterministic_with_best_at_most_mean(cell):
    policy, dataset = cell
    m = measures_cfg()
    a = bh.cell_measures(policy, dataset, KEY, m, "cpu")["m3"]
    b = bh.cell_measures(policy, dataset, KEY, m, "cpu")["m3"]
    assert a == b and len(a["batches"]) == 2
    for batch in a["batches"]:
        assert batch["m3_best"][0] <= batch["m3_mean"][0] + 1e-9
        assert batch["m3_best_executed"][0] <= batch["m3_mean_executed"][0] + 1e-9
    other = copy.deepcopy(m)
    other["m3"]["noise_tag"] = "another::noise"
    assert bh.cell_measures(policy, dataset, KEY, other, "cpu")["m3"]["values"] != a["values"]


def test_bank_is_the_probe_order_and_a_wrong_task_is_refused(cell):
    policy, dataset = cell
    m = measures_cfg()
    idx = bh.bank_indices(len(dataset), m["probe"], KEY)
    assert len(idx) == m["probe"]["n_batches"] and all(len(b) == m["probe"]["batch_size"] for b in idx)
    assert len({i for b in idx for i in b}) == sum(len(b) for b in idx)   # no repeats within an epoch
    assert idx == bh.bank_indices(len(dataset), m["probe"], KEY)
    with pytest.raises(ValueError, match="single-task"):
        bh.cell_measures(policy, dataset, "toy/two", m, "cpu")


def test_instrument_checks_suite_gate_and_publication(tmp_path, monkeypatch):
    expected = {"a": [{"expected": 1.0, "source": "x"}], "b": [{"expected": 2.0, "source": "y"}],
                "unmeasured": [{"expected": 3.0, "source": "z"}]}
    ok = bp.instrument_checks({"a": 1.0005, "b": 2.0}, expected, 1e-3)
    assert ok["passed"] and ok["checked"] == 2
    bad = bp.instrument_checks({"a": 1.01, "b": 2.0}, expected, 1e-3)
    assert not bad["passed"] and bad["failures"][0]["cell"] == "a"
    monkeypatch.setattr(bp, "git_sha", lambda: "abcdef1234567890")
    suite = tmp_path / "suite.txt"
    assert not bp.suite_green(suite)[0]
    suite.write_text("1 passed\nexit 0\nMon\nabcdef1\n")
    assert bp.suite_green(suite)[0]
    suite.write_text("1 failed\nexit 1\nMon\nabcdef1\n")
    assert not bp.suite_green(suite)[0]
    suite.write_text("1 passed\nexit 0\nMon\n0000000\n")
    assert not bp.suite_green(suite)[0]
    monkeypatch.setattr(bp, "git_sha", lambda: "abcdef1234567890-dirty")
    suite.write_text("1 passed\nexit 0\nMon\nabcdef1\n")
    assert not bp.suite_green(suite)[0]
    out = tmp_path / "track2" / "measures"
    first = bp.publish_bundle(out, {"cells": {"a": 1}}, {"cells": 1})
    again = bp.publish_bundle(out, {"cells": {"a": 2}}, {"cells": 2})   # never rewritten
    assert first == again and json.loads((out / "measures.json").read_text()) == {"cells": {"a": 1}}
    (out / "summary.json").write_text("{}\n")
    with pytest.raises(RuntimeError, match="done-marker"):
        bp.load_bundle(out)
