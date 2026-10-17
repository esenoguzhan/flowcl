"""The ConSFT bracket's pure parts: the planned grid and the cap, the ω summaries, the rule's order down to the
arm id, the Table 5 arm by type, the edge extension (owed, capped, re-applied), the fallback, and Track 2's
exclusion keys (every arm, both tasks, fingerprints recomputed from checkpoint tensors)."""

from __future__ import annotations

import copy
import math

import pytest
import torch

from flowcl.experiments import consft_selection as cs
from flowcl.methods.consft import ConSFT

RULE = cs.load_config()


def cfg(**over):
    c = copy.deepcopy(RULE)
    c["consft"].update(over)
    return c


def test_the_committed_rule():
    c = RULE["consft"]
    assert c["constants"] == {"kappa": 25.0, "decay_rate": 0.8, "curve": 3.5, "omega_min": 0.001, "tau_end": 5.0}
    assert c["zs"] == [1, 3, 10, 30] and c["decay_steps"] == [2000, 30000]
    assert c["table5"] == {"tau_start": 0.003, "decay_steps": 2000}
    assert c["recipe"] == {"steps": 30000, "lr": 1.0e-4}
    assert set(cs.constants(RULE)) == {"kappa", "decay_rate", "curve", "omega_min", "tau_end"}
    ConSFT(tau_start=0.1, decay_steps=2000, **cs.constants(RULE))   # the constants are the method's kwargs


def test_plan_grid_table5_and_extensions():
    plan = cs.plan_arms(RULE, 0.1)
    keys = [a["key"] for a in plan["grid"]]
    assert keys == ["z1__T2000", "z1__T30000", "z3__T2000", "z3__T30000", "z10__T2000", "z10__T30000",
                    "z30__T2000", "z30__T30000"]
    z3 = next(a for a in plan["grid"] if a["key"] == "z3__T2000")
    assert z3["tau_start"] == pytest.approx(25 * 0.1 / 3) and not z3["capped"]
    assert plan["table5"]["key"] == "table5" and plan["table5"]["z"] == pytest.approx(25 * 0.1 / 0.003)
    ext = {a["key"]: a for a in plan["extensions"]}
    assert set(ext) == {"z0.3__T2000", "z0.3__T30000", "z100__T2000", "z100__T30000"}
    # q50 = 0.10: the grid passes (q50 < 0.20) but the lower extension starts at 8.33 >= 5
    assert ext["z0.3__T2000"]["capped"] and not ext["z100__T2000"]["capped"]
    cs.check_caps(plan)
    assert set(cs.runnable_extensions(plan)) == {"z100__T2000", "z100__T30000"}
    assert len(cs.base_arms(plan)) == 9


def test_cap_refusal_and_bad_q50():
    with pytest.raises(ValueError, match="tau_start >= tau_end"):
        cs.check_caps(cs.plan_arms(RULE, 0.25))   # z = 1 needs q50 < 0.20
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive and finite"):
            cs.plan_arms(RULE, bad)


def test_omega_summary_matches_the_method():
    consts = cs.constants(RULE)
    losses = torch.tensor([1e-5, 1e-3, 0.05, 0.5])
    s = torch.tensor([0.05, 0.35, 0.65, 0.95])
    m = ConSFT(tau_start=0.02, decay_steps=10, **consts)
    assert torch.equal(cs.omega(losses, 0.02, consts), m.weights(losses, 0))
    summ = cs.summarize_omega(losses, s, 0.02, consts)
    assert summ["floor_fraction"] == pytest.approx(0.5)   # 0.05 and 0.5 are at the floor at τ = 0.02
    assert summ["by_s_bin"]["0"]["n"] == 1 and summ["by_s_bin"]["9"]["floor_fraction"] == 1.0
    ls = cs.summarize_losses(losses, s)
    assert ls["n"] == 4 and ls["quantiles"]["0.5"] == pytest.approx(float(torch.quantile(losses.double(), 0.5)))


def arm(key, arm_type, z, T, spatial, obj, sp=0.1, op=0.1):
    return {"key": key, "arm_type": arm_type, "z": float(z), "decay_steps": int(T), "tau_start": 0.0,
            "run_id": f"consftb__{key}__seed0", "spatial": spatial, "object": obj,
            "spatial_probe": sp, "object_probe": op}


def base(values: dict, table5=(0.0, 0.8), q50=0.1):
    """Base arm summaries from ``{key: (spatial, object)}`` (others 0.0 / 0.8)."""
    out = []
    for z in (1, 3, 10, 30):
        for T in (2000, 30000):
            key = f"z{z}__T{T}"
            sp, ob = values.get(key, (0.0, 0.8))
            out.append(arm(key, "grid", z, T, sp, ob))
    out.append(arm("table5", "table5", 25 * q50 / 0.003, 2000, *table5))
    return out


EXPECTED = {f"z{z}__T{T}" for z in (1, 3, 10, 30) for T in (2000, 30000)} | {"table5"}
EXT = cs.plan_arms(RULE, 0.1)["extensions"]   # the lower extension is capped at q50 = 0.1
EXT_OPEN = cs.plan_arms(RULE, 0.01)["extensions"]   # none capped


def select(arms, extensions=EXT, floor=0.68):
    return cs.select_consft(arms, floor, EXPECTED, RULE["consft"]["zs"], extensions)


def test_order_spatial_then_probe_then_z_then_T_then_id():
    out = select(base({"z3__T30000": (0.4, 0.7), "z10__T2000": (0.4, 0.7)}))
    assert out["status"] == "selected" and out["selected"]["key"] == "z3__T30000"   # equal Spatial and probe: smaller z
    arms = base({"z3__T30000": (0.4, 0.7), "z3__T2000": (0.4, 0.7)})
    assert select(arms)["selected"]["key"] == "z3__T2000"                            # then the shorter T
    arms = base({"z3__T30000": (0.4, 0.7), "z10__T2000": (0.4, 0.7)})
    next(a for a in arms if a["key"] == "z10__T2000")["spatial_probe"] = 0.05
    assert select(arms)["selected"]["key"] == "z10__T2000"                           # probe before z
    out = select(base({"z3__T30000": (0.4, 0.5)}))                                   # below the floor
    assert out["selected"]["key"] == "z1__T2000"
    assert out["flags"] == ["all_eligible_spatial_zero", "lower_extension_capped"]
    assert "z3__T30000" not in out["eligible"]


def test_arm_id_breaks_an_exact_tie_and_type_decides_the_extension():
    # q50 = 0.00012 makes the Table 5 arm (z 1, T 2000): identical parameters, identical results
    arms = base({"z1__T2000": (0.5, 0.8)}, table5=(0.5, 0.8), q50=0.00012)
    out = select(arms, extensions=EXT_OPEN)
    assert out["grid_selection"] == "table5"                       # "table5" < "z1__T2000" lexicographically
    assert out["status"] == "selected" and out["owed_extension"] is None   # a table5 arm is never extended
    arms = base({"z1__T2000": (0.5, 0.8)}, table5=(0.4, 0.8), q50=0.00012)
    out = select(arms, extensions=EXT_OPEN)
    assert out["status"] == "needs_extension" and out["missing"] == "z0.3__T2000"   # the grid arm owes it


def test_lower_extension_capped_is_flagged_and_owes_nothing():
    out = select(base({"z1__T30000": (0.3, 0.8)}))
    assert out["status"] == "selected" and out["selected"]["key"] == "z1__T30000"
    assert out["flags"] == ["lower_extension_capped"] and out["owed_extension"] is None


def test_upper_extension_owed_then_reapplied():
    arms = base({"z30__T30000": (0.6, 0.7)})
    out = select(arms)
    assert out["status"] == "needs_extension" and out["missing"] == "z100__T30000"
    ext = arm("z100__T30000", "extension", 100, 30000, 0.7, 0.69)
    out = select(arms + [ext])
    assert out["selected"]["key"] == "z100__T30000"
    assert out["flags"] == ["edge_extension", "optimum_at_boundary"]
    ext_low_object = arm("z100__T30000", "extension", 100, 30000, 0.9, 0.5)
    out = select(arms + [ext_low_object])
    assert out["selected"]["key"] == "z30__T30000" and out["flags"] == ["edge_extension"]


def test_interior_selection_owes_nothing_and_unowed_extensions_are_refused():
    arms = base({"z10__T2000": (0.5, 0.75)})
    out = select(arms)
    assert out["status"] == "selected" and out["owed_extension"] is None and out["flags"] == []
    with pytest.raises(ValueError, match="not owed"):
        select(arms + [arm("z100__T2000", "extension", 100, 2000, 0.9, 0.9)])
    with pytest.raises(ValueError, match="not the rule's"):
        select([a for a in arms if a["key"] != "table5"])


def test_fallback_orders_by_object_and_owes_nothing():
    arms = base({k: (0.0, 0.5) for k in EXPECTED - {"table5"}}, table5=(0.0, 0.5))
    next(a for a in arms if a["key"] == "z30__T30000")["object"] = 0.6
    out = select(arms)
    assert out["selected"]["key"] == "z30__T30000" and out["flags"] == ["no_qualifying_setting"]
    assert out["owed_extension"] is None and out["eligible"] == []
    for a in arms:
        a["object"] = 0.5
    assert select(arms)["selected"]["key"] == "z1__T2000"   # ties: Object probe, z, T, id


def fake_row(key, fingerprint):
    return {"run_id": cs.arm_run_id(key, 0), "checkpoint": "checkpoints/final.pt",
            "cells": {"0": {"task_key": "libero_spatial/a"}, "1": {"task_key": "libero_object/b"}},
            "consft": {"state_fingerprint": fingerprint}}


def test_exclusion_keys_cover_every_arm_and_both_tasks_from_checkpoint_tensors(tmp_path):
    from flowcl.methods.ewc import tensor_fingerprint

    rows, paths = {}, {}
    for i, key in enumerate(["z1__T2000", "table5", "z100__T30000"]):
        state = {"w": torch.full((2, 3), float(i)), "b": torch.arange(3.0)}
        path = tmp_path / key / "final.pt"
        path.parent.mkdir()
        torch.save({"state_dict": state}, path)
        rows[key], paths[key] = fake_row(key, tensor_fingerprint(state)), path
    keys = cs.exclusion_keys(rows)
    assert len(keys) == 6
    assert {(k["run_id"], k["task_key"]) for k in keys} == \
        {(cs.arm_run_id(r, 0), t) for r in rows for t in ("libero_spatial/a", "libero_object/b")}
    recomputed = cs.exclusion_keys(rows, lambda k: cs.checkpoint_fingerprint(paths[k]))
    assert recomputed == keys
    torch.save({"state_dict": {"w": torch.zeros(2, 3), "b": torch.arange(3.0) + 1}}, paths["table5"])
    assert cs.exclusion_keys(rows, lambda k: cs.checkpoint_fingerprint(paths[k])) != keys
    assert all(math.isfinite(len(k["fingerprint"])) and len(k["fingerprint"]) == 64 for k in keys)
