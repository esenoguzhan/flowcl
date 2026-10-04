"""Similar-task diagnostic: the paired instruction-swap capture, the outside-memory share, gradient
alignment with split-half stability, the summaries, Gate 3's split-gradient extension (exact and
off by default), the whole diagnostic end to end on the tiny policy, the queue and the config."""

from __future__ import annotations

import copy
import json
import math
import os
import subprocess
from pathlib import Path

import pytest
import torch
from torch import nn

from flowcl.analysis.hooks import KIND_STATE, KIND_TRUNK
from flowcl.experiments import similar_tasks as st
from flowcl.utils.libero_paths import repo_root
from test_low_update import episodes, fake_world, spec  # noqa: F401

A = "libero_goal/put_the_bowl_on_the_plate"
SIM = "libero_goal/put_the_bowl_on_the_stove"
DIV = "libero_object/pick_up_the_milk_and_place_it_in_the_basket"


# ---- the paired capture ---------------------------------------------------------------------------------


def test_paired_capture_state_layer_and_zero_signal():
    lin = nn.Linear(3, 2, bias=False)
    cap = st.PairedSwapCapture([("L", lin, KIND_STATE)], tokens_per_sample=4, seed=0)
    x = torch.randn(5, 3)
    delta = torch.tensor([1.0, 0.0, 2.0])
    with cap:
        cap.mode("ref")
        lin(x)
        cap.mode("swap")
        lin(x + delta)
        cap.mode(None)
    assert torch.allclose(cap.D["L"], 5 * torch.outer(delta, delta).double())
    assert torch.allclose(cap.K["L"], (x.T @ x).double(), atol=1e-6)
    cap0 = st.PairedSwapCapture([("L", lin, KIND_STATE)], tokens_per_sample=4, seed=0)
    with cap0:
        cap0.mode("ref"); lin(x); cap0.mode("swap"); lin(x)
    assert float(torch.trace(cap0.D["L"])) == 0.0
    assert st.outside_share(torch.eye(3)[:, :1], cap0.D["L"]) is None
    with pytest.raises(RuntimeError, match="reference pass first"):
        st.PairedSwapCapture([("L", lin, KIND_STATE)], 4, 0).mode("swap")


def test_paired_capture_uses_identical_positions():
    lin = nn.Linear(2, 2, bias=False)
    cap = st.PairedSwapCapture([("T", lin, KIND_TRUNK)], tokens_per_sample=3, seed=1)
    x = torch.randn(4, 10, 2)
    shift = torch.zeros(4, 10, 2)
    shift[..., 0] = torch.arange(10.0)          # a position-dependent difference
    with cap:
        cap.mode("ref"); lin(x)
        b, t, _ = cap._ref["T"]
        cap.mode("swap"); lin(x + shift)
    d = shift[b, t].double()
    assert torch.allclose(cap.D["T"], d.T @ d) and cap.n["T"] == 12
    assert torch.allclose(cap.K["T"], (x[b, t].double().T @ x[b, t].double()), atol=1e-9)


def test_smoke_subset_takes_the_tail_with_padded_samples(spec):
    from flowcl.data.dataset import ChunkedActionDataset
    from flowcl.data.stats import compute_stats
    from flowcl.experiments.gate2 import probe_indices

    eps = episodes(spec, A, n=2)
    ds = ChunkedActionDataset(eps, spec, compute_stats(eps, embodiment=spec.name, task_id=A))
    sub = st.TailSubset(ds, 8)
    assert len(sub) == 8 and sub.offset == len(ds) - 8
    assert sub.sample_index(0) == ds.sample_index(len(ds) - 8)
    picked = probe_indices(sub, 4)                   # raises if the subset has no padded samples
    assert picked and all(sub.sample_index(i).t + ds.horizon > ds.episodes[sub.sample_index(i).episode_idx].length
                          for i in picked[:1])


def test_outside_share_and_surviving_norm():
    D = torch.diag(torch.tensor([1.0, 4.0, 0.0], dtype=torch.float64))
    assert st.outside_share(torch.eye(3, dtype=torch.float64)[:, :1], D) == pytest.approx(0.8)
    assert st.surviving_norm(0.95) == pytest.approx(math.sqrt(1 - 0.95 ** 2)) == pytest.approx(0.3122, abs=1e-4)


# ---- alignment and summaries -------------------------------------------------------------------------------


def test_alignment_cosines_stability_and_zero_norm():
    ga = {"trunk.a": torch.tensor([1.0, 0.0]), "flow_head.b": torch.tensor([0.0, 1.0])}
    gb = {"trunk.a": torch.tensor([1.0, 1.0]), "flow_head.b": torch.tensor([0.0, 0.0])}
    split_good = {"even": {k: v.clone() for k, v in ga.items()}, "odd": {k: v.clone() for k, v in ga.items()}}
    split_b = {"even": {"trunk.a": torch.tensor([1.0, 1.0]), "flow_head.b": torch.tensor([0.0, 0.0])},
               "odd": {"trunk.a": torch.tensor([-1.0, -1.0]), "flow_head.b": torch.tensor([0.0, 0.0])}}
    al = st.alignment(ga, split_good, gb, split_b, 0.5)
    assert al["layers"]["trunk.a"]["cos"] == pytest.approx(1 / math.sqrt(2))
    assert al["layers"]["trunk.a"]["self_a"] == pytest.approx(1.0) and al["layers"]["trunk.a"]["self_b"] == pytest.approx(-1.0)
    assert al["layers"]["trunk.a"]["unstable"]                       # B's halves disagree
    assert al["layers"]["flow_head.b"]["cos"] is None and al["layers"]["flow_head.b"]["unstable"]
    assert al["scopes"]["trunk"]["cos"] == pytest.approx(1 / math.sqrt(2))
    stable = st.alignment(ga, split_good, ga, split_good, 0.5)
    assert not stable["scopes"]["all"]["unstable"] and stable["scopes"]["all"]["cos"] == pytest.approx(1.0)


def test_half_and_category_summaries():
    dims = {"trunk.a": (2, 2), "trunk.b": (1, 2), "flow_head.c": (3, 3)}
    h = st.half_summary({"trunk.a": 0.5, "trunk.b": 1.0, "flow_head.c": None}, dims)
    assert h["trunk"]["median"] == pytest.approx(0.75) and h["trunk"]["weighted"] == pytest.approx((0.5 * 4 + 1.0 * 2) / 6)
    assert h["decoder"] == {"median": None, "weighted": None, "n_layers": 1, "n_defined": 0}
    c = st.category_summary({"a": 0.9, "b": 0.8, "c": 0.5, "d": None}, {"a": "similar", "b": "similar", "c": "diverse",
                                                                      "d": "same_scene"})
    assert c["means"]["similar"] == pytest.approx(0.85) and c["means"]["same_scene"] is None
    assert c["differences"]["similar-diverse"] == pytest.approx(0.35) and c["differences"]["same_scene-diverse"] is None


# ---- end to end on the tiny policy ------------------------------------------------------------------------


def tiny_world(tmp_path, spec, monkeypatch):
    from omegaconf import OmegaConf

    from flowcl.data.dataset import ChunkedActionDataset
    from flowcl.experiments.gate0 import single_task_run_id
    from flowcl.experiments.gate3 import load_interference_config
    from flowcl.train.checkpoint import load_checkpoint, save_checkpoint

    fake_world(tmp_path, spec, seeds=(0,))
    base = load_checkpoint(tmp_path / "seq_hetero__seq_ft__seed0" / "checkpoints" / "stage0.pt")
    run = tmp_path / single_task_run_id(A, 0)
    save_checkpoint(run / "checkpoints" / "final.pt", policy=base.policy, policy_config=base.payload["policy_config"],
                    spec=base.spec, stats=base.stats, run_id=run.name, stage=0, task_key=A)
    keys = [A, SIM, DIV]

    def build(key):
        return ChunkedActionDataset(episodes(spec, key, offset=10 * keys.index(key)), spec, base.stats)

    sub = OmegaConf.load(repo_root() / "configs" / "analysis" / "subspace.yaml")
    sub.min_samples_per_dim, sub.num_workers, sub.batch_size, sub.probe_batch_size = 0.01, 0, 4, 4
    OmegaConf.save(sub, tmp_path / "tiny_subspace.yaml")
    icfg = load_interference_config()
    import dataclasses
    monkeypatch.setattr(st, "_interference_cfg",
                        lambda smoke: dataclasses.replace(icfg, num_workers=0, batch_size=4, n_batches=None))
    cfg = copy.deepcopy(st.load_config())
    cfg["capture_config"] = str(tmp_path / "tiny_subspace.yaml")
    cfg["tasks_b"] = [{"key": SIM, "category": "similar"}, {"key": DIV, "category": "diverse"}]
    return cfg, dict(results_root=tmp_path, device="cpu", allow_dirty=True, build_datasets=build)


def test_diagnostic_end_to_end(spec, tmp_path, monkeypatch):
    from flowcl.analysis.subspace import load_bases
    from flowcl.experiments.gate3 import measure_gradient_interference
    from flowcl.train.checkpoint import load_checkpoint

    cfg, kw = tiny_world(tmp_path, spec, monkeypatch)
    assert st.run_summary(cfg, results_root=tmp_path)["status"] == "incomplete"     # nothing yet
    mem = st.run_memory(cfg, **kw)
    assert mem["self_control"]["passed"] and mem["weights_unchanged"] and mem["a_weak"]   # no Gate 0 report
    assert all(v >= 0.95 - 1e-6 for v in mem["self_control"]["layers"].values())
    assert set(mem["gate3_self"]["global_c"]) == {"all", "trunk", "decoder"}
    for b in (SIM, DIV):
        r = st.run_measure(cfg, b, **kw)
        assert r["weights_unchanged"] and set(r["overlap"]) == {"0.9", "0.95", "0.99"}
        assert r["alignment"]["scopes"]["trunk"]["norm_a"] > 0
        assert "on_A_data" in r["swaps"] and (("on_B_data" in r["swaps"]) == (b == SIM))
        sw = r["swaps"]["on_A_data"]
        assert sw["reference"] == f"do {A}" and sw["swap"] == f"do {b}"
        assert sw["output_sensitivity"]["denominator"] > 0 and sw["output_sensitivity"]["numerator"] > 0
        sizes = sw["size"]["layers"]
        assert sizes["trunk.state_projection"] == 0.0 and sw["outside_share"]["layers"]["trunk.state_projection"] is None
        assert sw["outside_share"]["halves"]["trunk"]["n_defined"] < sw["outside_share"]["halves"]["trunk"]["n_layers"]
    s = st.run_summary(cfg, results_root=tmp_path)
    assert s["status"] == "complete" and set(s["per_task"]) == {SIM, DIV}
    assert set(s["categories"]["global_c_trunk_energy"]["differences"]) == {"similar-diverse"}
    assert s["predictions"]["P4"].startswith("measured")
    # Gate 3's split extension: exact recombination, and nothing changes when it is off
    loaded = load_checkpoint(tmp_path / st.a_run_dir(cfg, tmp_path).name / "checkpoints" / "final.pt", device="cpu")
    bases, meta = load_bases(tmp_path / "similar_tasks" / "memory" / "M_A.pt")
    ds = kw["build_datasets"](DIV)
    icfg = st._interference_cfg(None)
    on = measure_gradient_interference(loaded, bases, meta, ds, icfg, device="cpu", keep_full_gradient=True,
                                       keep_split_gradients=True)
    off = measure_gradient_interference(loaded, bases, meta, ds, icfg, device="cpu", keep_full_gradient=True)
    n = on.split_counts
    for name in on.full_gradients:
        mix = (n["even"] * on.split_gradients["even"][name] + n["odd"] * on.split_gradients["odd"][name]) / (n["even"] + n["odd"])
        assert torch.allclose(mix, on.full_gradients[name], atol=1e-12, rtol=1e-9)
        assert torch.equal(on.full_gradients[name], off.full_gradients[name])
        assert on.layers[name].total == off.layers[name].total
    assert off.split_gradients is None and off.split_counts is None
    # a changed checkpoint is refused
    with open(tmp_path / st.a_run_dir(cfg, tmp_path).name / "checkpoints" / "final.pt", "ab") as f:
        f.write(b"x")
    with pytest.raises(RuntimeError, match="checkpoint changed"):
        st._verify_memory(cfg, tmp_path)


def test_config_is_consistent():
    from flowcl.data.tasks import TaskRef

    cfg = st.load_config()
    assert cfg["task_a"] == A and cfg["eps"] == 0.95 and cfg["followup_c"] == 0.95
    cats = [t["category"] for t in cfg["tasks_b"]]
    assert cats == ["similar", "similar", "same_scene", "same_object_goal", "diverse", "diverse"]
    for key in [cfg["task_a"], *[t["key"] for t in cfg["tasks_b"]]]:
        TaskRef.from_key(key)
    assert cfg["task_a"] not in [t["key"] for t in cfg["tasks_b"]]


# ---- the queue ----------------------------------------------------------------------------------------

QUEUE = repo_root() / "scripts" / "queue_similar_tasks.sh"


def run_queue(tmp_path, args=(), fail=""):
    env = {**os.environ, "QUEUE_DRY_RUN": "1", "QUEUE_FAIL_STEP": fail, "QUEUE_LOG_ROOT": str(tmp_path / "logs"),
           "QUEUE_RESULTS_ROOT": str(tmp_path / "results")}
    done = subprocess.run(["bash", str(QUEUE), *args], env=env, capture_output=True, text=True, timeout=180)
    (logdir,) = list((tmp_path / "logs").glob("queue_*"))
    return done.returncode, (logdir / "queue.log").read_text(), logdir


def test_queue_order_commands_and_failures(tmp_path):
    assert subprocess.run(["bash", "-n", str(QUEUE)]).returncode == 0
    rc, log, logdir = run_queue(tmp_path / "a")
    assert rc == 0 and log.rstrip().endswith("QUEUE DONE ok")
    starts = [l.split("START ")[1].split(":")[0] for l in log.splitlines() if " START " in l and "QUEUE START" not in l]
    assert starts[0] == "0_train_a" and starts[1] == "1_memory" and starts[-1] == "8_summary" and len(starts) == 9
    train = (logdir / "0_train_a.log").read_text()
    assert f"--tasks {A}" in train and "--train-steps 30000" in train and "--amp" in train
    assert "gate0_similar" in train
    assert "--part measure --task libero_goal/put_the_bowl_on_the_stove" in (logdir / f"{starts[2]}.log").read_text()
    rc, log, _ = run_queue(tmp_path / "b", fail="1_memory")
    assert rc == 1 and "START 2_" not in log and "START 8_summary" in log and "FAILED: 1_memory" in log
    rc, log, logdir = run_queue(tmp_path / "c", ["--smoke", str(tmp_path / "scratch")])
    train = (logdir / "0_train_a.log").read_text()
    assert "--train-steps 200 --n-episodes 1" in train and f"--results-root {tmp_path / 'scratch'}" in train
    assert "--smoke-batches 3" in (logdir / "1_memory.log").read_text()
    assert subprocess.run(["bash", str(QUEUE), "--smoke", "rel"], capture_output=True).returncode == 2
