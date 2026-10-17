"""E1-B (the angle-B capacity gate): the integration's identity with ``policy.sample``, the exact
coherent Grams and the Hutchinson comparison on linear maps, multi-call rows and shared-row
eligibility, the recovery weights and variants, the alignment helpers, the whole measurement end to
end on the tiny policy (valid, smoke, invalid identity, tampering, pins), the config and the queue."""

from __future__ import annotations

import copy
import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from flowcl.analysis import recovery as rc
from flowcl.analysis.hooks import KIND_ACTION, KIND_STATE, CaptureLayer
from flowcl.analysis.integrated_sensitivity import IntegratedOutputCapture, combine_grams, integrate
from flowcl.experiments import twosided_angleb as tb
from flowcl.experiments import twosided_gonogo as tg
from flowcl.utils.libero_paths import repo_root
from flowcl.utils.run import file_sha256
from test_low_update import KEYS, spec  # noqa: F401
from test_t5_sweep import world

GROUPS3 = [(0, 3), (3, 6), (6, 7)]


# ---- the integration -----------------------------------------------------------------------------------


def test_integrate_equals_sample_bitwise(spec, tmp_path):
    from flowcl.train.trainer import build_dataloader
    from test_low_update import episodes, tiny_policy
    from flowcl.data.dataset import ChunkedActionDataset
    from flowcl.data.stats import compute_stats

    policy = tiny_policy(spec).eval()
    stats = compute_stats(episodes(spec, KEYS[0]), embodiment=spec.name, task_id=KEYS[0])
    batch = next(iter(build_dataloader(ChunkedActionDataset(episodes(spec, KEYS[0]), spec, stats),
                                       batch_size=4, shuffle=False)))
    noise = torch.randn(4, policy.horizon, 7, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        ref = policy.sample(batch, noise=noise)
    with torch.enable_grad():
        out = integrate(policy, batch, noise, policy.euler_steps)
    assert out.requires_grad and torch.equal(out.detach(), ref)


# ---- exactness on linear maps ---------------------------------------------------------------------------


class Linear2(nn.Module):
    """x -> y1 = W1 x (hooked) -> chunk (B, H, D) = reshape(W2 y1)."""

    def __init__(self, d0=5, d1=6, H=4, D=7, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.H, self.D = H, D
        self.first = nn.Linear(d0, d1, bias=False)
        self.second = nn.Linear(d1, H * D, bias=False)

    def forward(self, x):
        return self.second(self.first(x)).view(-1, self.H, self.D)


def test_coherent_grams_are_exact_and_hutchinson_converges():
    net = Linear2()
    K, D = 2, 7
    scale = [0.5, 1.0, 2.0, 1.5, 0.7, 1.1, 3.0]
    cap = IntegratedOutputCapture(net, [CaptureLayer("first", net.first, KIND_STATE)], groups=GROUPS3, n_bins=2,
                                  n_probes=1, tokens_per_sample=1, subsample_seed=0)
    W2 = net.second.weight.detach().to(torch.float64).view(net.H, D, -1)      # (H, D, d1)
    r = torch.tensor(scale, dtype=torch.float64) ** 2 * torch.tensor([0.3, 0.3, 0.3, 0.1, 0.1, 0.1, 0.6],
                                                                     dtype=torch.float64)
    n_per_bin = [0, 0]
    gen = torch.Generator().manual_seed(1)
    with cap:
        for _ in range(20):
            x = torch.randn(500, 5, generator=gen)
            bins = (torch.arange(500) % 3 == 0).long()
            n_per_bin[0] += int((bins == 0).sum())
            n_per_bin[1] += int((bins == 1).sum())
            u = torch.randn(500, K, D, generator=gen) * r.sqrt().to(torch.float32)
            with cap.recording():
                out = net(x)
            cap.backward(out[:, :K], scale, bins, [u])
    acc = cap.accumulators["first"]
    assert acc.n == 10000 and acc.n_bin == n_per_bin
    for b in range(2):
        for g, (lo, hi) in enumerate(GROUPS3):
            expected = torch.zeros(6, 6, dtype=torch.float64)
            for d in range(lo, hi):
                v = W2[:K, d, :].sum(dim=0) / math.sqrt(K)                         # J_d^T for a linear map
                expected += scale[d] ** 2 * torch.outer(v, v)
            assert torch.allclose(acc.coherent[b, g], n_per_bin[b] * expected, rtol=1e-5, atol=1e-8)
    # any weighting is the same linear combination
    w = torch.tensor([[1.0, 0.5, 0.0], [2.0, 0.0, 1.0]], dtype=torch.float64)
    combined = combine_grams(acc.coherent, w)
    assert torch.allclose(combined, sum(w[b, g] * acc.coherent[b, g] for b in range(2) for g in range(3)))
    # the independent-step Hutchinson Gram converges to W2_K^T diag(r) W2_K; steps >= K never enter
    J = W2[:K].reshape(K * D, -1)
    expected = J.T @ torch.diag(r.repeat(K)) @ J
    est = acc.hutch[0] / acc.n
    assert float((est - expected).norm() / expected.norm()) < 0.05
    assert net.first.weight.grad is None and net.second.weight.grad is None


class Repeated(nn.Module):
    """A hooked layer run ``n_calls`` times over T positions (an integration's decoder)."""

    def __init__(self, T=3, d=4, D=7, H=2, n_calls=3, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.T, self.H, self.D, self.n_calls = T, H, D, n_calls
        self.layer = nn.Linear(d, d, bias=False)
        self.read = nn.Linear(T * d, H * D, bias=False)

    def forward(self, x):                 # x: (B, T, d)
        for _ in range(self.n_calls):
            x = torch.tanh(self.layer(x))
        return self.read(x.flatten(1)).view(-1, self.H, self.D)


def test_multi_call_rows_sum_over_calls_and_respect_the_budget():
    net = Repeated()
    x = torch.randn(5, 3, 4, generator=torch.Generator().manual_seed(0))
    bins = torch.tensor([0, 1, 1, 0, 1])
    scale = [1.0] * 7
    cap = IntegratedOutputCapture(net, [CaptureLayer("layer", net.layer, KIND_ACTION)], groups=GROUPS3, n_bins=2,
                                  n_probes=0, tokens_per_sample=100, subsample_seed=0)
    with cap:
        with cap.recording():
            out = net(x)
        cap.backward(out[:, :2], scale, bins)
    acc = cap.accumulators["layer"]
    assert acc.calls == 3 and acc.n == 5 * 3 * 3                    # every (call, position) of every sample
    # the reference: autograd through the same retained outputs, summed over calls and positions
    outs = []
    hook = net.layer.register_forward_hook(lambda m, a, o: outs.append(o))
    out = net(x)
    hook.remove()
    for g, (lo, hi) in enumerate(GROUPS3):
        expected = torch.zeros(2, 4, 4, dtype=torch.float64)
        for d in range(lo, hi):
            grads = torch.autograd.grad(out[:, :2, d].sum() / math.sqrt(2), outs, retain_graph=True)
            for gr in grads:
                for s in range(5):
                    rows = gr[s].detach().to(torch.float64)
                    expected[int(bins[s])] += rows.T @ rows
        assert torch.allclose(acc.coherent[:, g], expected, rtol=1e-6, atol=1e-10)
    small = IntegratedOutputCapture(net, [CaptureLayer("layer", net.layer, KIND_ACTION)], groups=GROUPS3,
                                    n_bins=2, n_probes=0, tokens_per_sample=2, subsample_seed=0)
    with small:
        with small.recording():
            out = net(x)
        small.backward(out[:, :2], scale, bins)
    assert small.accumulators["layer"].n == 5 * 2


class Disjoint(nn.Module):
    """Action dimension d reads only position d of the hooked output; position D is never read."""

    def __init__(self, D=7, d=3, H=2, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.D, self.H = D, H
        self.layer = nn.Linear(d, d, bias=False)
        self.heads = nn.Parameter(torch.randn(D, H, d))

    def forward(self, x):                 # x: (B, D + 1, d)
        y = self.layer(x)
        return torch.einsum("bdj,dhj->bhd", y[:, : self.D], self.heads)


def test_shared_rows_cover_disjoint_supports():
    net = Disjoint()
    x = torch.randn(6, 8, 3, generator=torch.Generator().manual_seed(0))
    cap = IntegratedOutputCapture(net, [CaptureLayer("layer", net.layer, KIND_ACTION)], groups=GROUPS3, n_bins=1,
                                  n_probes=1, tokens_per_sample=8, subsample_seed=0)
    u = torch.randn(6, 2, 7, generator=torch.Generator().manual_seed(1))
    with cap:
        with cap.recording():
            out = net(x)
        cap.backward(out, [1.0] * 7, torch.zeros(6, dtype=torch.long), [u])
    acc = cap.accumulators["layer"]
    assert acc.n == 6 * 7                 # every dimension's position is eligible; the unread one is not
    heads = net.heads.detach().to(torch.float64)
    for g, (lo, hi) in enumerate(GROUPS3):
        expected = sum(6 * torch.outer(heads[d].sum(0), heads[d].sum(0)) / 2 for d in range(lo, hi))
        assert torch.allclose(acc.coherent[0, g], expected, rtol=1e-6, atol=1e-10)


def test_capture_refuses_a_prefix_without_graph_and_bad_bins():
    net = Linear2()
    cap = IntegratedOutputCapture(net, [CaptureLayer("first", net.first, KIND_STATE)], groups=GROUPS3, n_bins=2,
                                  n_probes=0, tokens_per_sample=1, subsample_seed=0)
    with cap:
        with cap.recording(), torch.no_grad():
            out = net(torch.randn(3, 5))
        with pytest.raises(RuntimeError, match="graph"):
            cap.backward(out[:, :2], [1.0] * 7, torch.zeros(3, dtype=torch.long))
        with cap.recording():
            out = net(torch.randn(3, 5))
        with pytest.raises(ValueError, match="bins"):
            cap.backward(out[:, :2], [1.0] * 7, torch.tensor([0, 1, 2]))


# ---- the recovery inputs --------------------------------------------------------------------------------


def rows_for(spec_cells, n_seeds=2):
    """Synthetic recovery rows: spec_cells[(query, group)] = list of (magnitude, noop, nominal_ok, ok)."""
    rows = []
    for (q, g), trials in spec_cells.items():
        for m, (mag, noop, nom_ok, ok) in enumerate(trials):
            seed = m % n_seeds + 10 * m
            rows.append({"rollout": 0, "query": q, "seed": seed, "kind": 0, "group": -1, "success": nom_ok,
                         "magnitude": 0.0, "noop": False, "sign": 0})
            rows.append({"rollout": 0, "query": q, "seed": seed, "kind": 1, "group": g, "success": ok,
                         "magnitude": mag, "noop": noop, "sign": 1, "bank_index": 0})
    return rows


def test_phase_group_weights():
    cells = {(0, 0): [(3.0, False, True, False), (3.0, False, True, True), (1.0, False, True, False),
                      (3.0, True, True, False), (3.0, False, False, False)],       # 1/2 counted
             (1, 0): [(3.0, False, True, True)],                                    # 0/1
             (0, 1): [(3.0, False, True, True)] * 3,                                # 0/3
             (0, 2): [(3.0, False, True, False)] * 2, (1, 2): [(3.0, False, True, True)] * 2,
             (0, 3): [(3.0, False, True, False)]}                                   # isotropic: ignored
    w = rc.phase_group_weights(rows_for(cells), 2, 3.0)
    assert w["trials"] == [[2, 3, 2], [1, 0, 2]] and w["injuries"] == [[1, 0, 2], [0, 0, 0]]
    assert w["pooled"] == pytest.approx([1 / 3, 0.0, 0.5])
    assert np.allclose(w["w"], [[0.5, 0.0, 1.0], [0.0, 0.0, 0.0]])               # (1, rotation): pooled 0
    assert w["fallback_cells"] == [[1, "rotation"]]
    with pytest.raises(ValueError, match="no effective"):
        rc.phase_group_weights(rows_for({(0, 0): [(3.0, False, True, False)]}), 2, 3.0)
    with pytest.raises(ValueError, match="phase bins"):
        rc.phase_group_weights(rows_for({(5, 0): [(3.0, False, True, False)]}), 2, 3.0)


def test_variant_weights_and_phase_bins():
    cfg = tb.load_config()
    weights = {"w": [[0.1 * (b + 1) * (g + 1) for g in range(3)] for b in range(6)], "pooled": [0.2, 0.3, 0.4]}
    v = tb.variant_weights(cfg, weights, "ns", KEYS[0], 0)
    assert set(v) == {"R", "I", "S", "P", "R0"}
    assert np.allclose(v["R"], weights["w"]) and np.all(np.asarray(v["I"]) == 1.0)
    assert sorted(np.ravel(v["S"])) == pytest.approx(sorted(np.ravel(v["R"]))) and v["S"] != v["R"]
    assert v["S"] == tb.variant_weights(cfg, weights, "ns", KEYS[0], 0)["S"]                # deterministic
    assert v["S"] != tb.variant_weights(cfg, weights, "ns", KEYS[1], 1)["S"]                # per task
    assert all(row == pytest.approx([0.2, 0.3, 0.4]) for row in v["P"])
    assert all(row[2] == 0.0 and row[:2] == pytest.approx(r[:2]) for row, r in zip(v["R0"], v["R"]))
    t = torch.tensor([0, 2, 3, 5, 17, 99])
    L = torch.tensor([18, 18, 18, 18, 18, 100])
    assert tb.phase_bins(t, L, 6).tolist() == [0, 0, 1, 1, 5, 5]


def test_gripper_alignment_helpers():
    rows = []
    for q, closed in ((0, False), (1, True), (2, None)):
        for idx, net_dir in ((0, 1.0), (1, -1.0)):
            effective = (closed and net_dir < 0) or (closed is False and net_dir > 0) or closed is None
            rows.append({"rollout": 0, "query": q, "seed": 0, "kind": 1, "group": 2, "magnitude": 3.0,
                         "noop": not effective, "bank_index": idx, "success": True})
    out = rc.gripper_closed_by_bin(rows, lambda r, q, i: 1.0 if i == 0 else -1.0, 3, 3.0)
    assert out["closed_fraction"] == [0.0, 1.0, None] and out["undetermined"] == 1
    demo = rc.demo_gripper_closed_by_bin([np.array([-1, -1, 1, 1, 1, -1.0])], 3)
    assert demo["closed_fraction"] == [0.0, 1.0, 0.5] and demo["frames"] == [2, 2, 2]


# ---- end to end on the tiny policy ----------------------------------------------------------------------

KEEP = ("trunk.state_projection", "trunk.blocks.0.attn.q_proj", "flow_head.blocks.0.self_attn.q_proj",
        "flow_head.blocks.0.cross_attn.k_proj", "flow_head.action_out")


def _fake_recovery(root: Path, cfg: dict) -> dict:
    """Production-shaped recovery inputs for the four keys, bound to the pinned rule."""
    rule_sha = cfg["recovery"]["rule_sha256"]
    bind = {"rule_sha256": rule_sha, "checkpoint_sha256": "x", "git_sha": "x", "smoke": None}
    rng = np.random.default_rng(0)
    for i, key in enumerate(KEYS):
        work = root / cfg["recovery"]["work"] / key.replace("/", "__")
        work.mkdir(parents=True)
        (work / "nominal.json").write_text(json.dumps({
            "binding": bind, "fallback": False, "sigma": [0.3, 0.3, 0.4, 0.05, 0.07, 0.1, 0.98],
            "sigma_fallback_dims": [], "rollouts": {"0": {"seed": 1, "success": True, "n_steps": 20}}}))
        rows = []
        for q in range(6):
            for m in range(2):
                rows.append({"rollout": 0, "query": q, "seed": m, "kind": 0, "group": -1, "success": True,
                             "magnitude": 0.0, "noop": False, "sign": 0})
                for j, (g, sign) in enumerate((g, s) for g in range(4) for s in (1, -1)):
                    rows.append({"rollout": 0, "query": q, "seed": m, "kind": 1, "group": g,
                                 "success": bool(rng.random() > 0.3 * (g != 1)), "magnitude": 3.0,
                                 "noop": g == 2 and sign < 0, "sign": sign, "bank_index": 2 * j})
        (work / "rollout_00.json").write_text(json.dumps({"binding": bind, "phases": [], "queries": [],
                                                          "dropped_phases": 0, "rows": rows}))
    analysis = root / cfg["recovery"]["analysis"]
    analysis.parent.mkdir(parents=True)
    analysis.write_text('{"outcome": "GO"}\n')
    return {**cfg["recovery"], "analysis_sha256": file_sha256(analysis)}


def angleb_world(tmp_path, spec, monkeypatch):
    t5cfg, luc, build = world(tmp_path, spec)
    cfg = copy.deepcopy(tb.load_config())
    cfg["source"]["capture_config"] = t5cfg["source"]["capture_config"]
    paths = tg.source_paths(cfg, tmp_path)
    cfg["source"]["sha256"] = {k: file_sha256(paths[k]) for k in cfg["source"]["sha256"]}
    cfg["recovery"] = _fake_recovery(tmp_path, cfg)
    real = IntegratedOutputCapture.for_policy.__func__

    def only_some(cls, policy, **kw):     # four layers keep the 18 Grams per layer small in a test
        cap = real(cls, policy, **kw)
        cap.layers = tuple(layer for layer in cap.layers if layer.name in KEEP)
        cap.accumulators = {n: a for n, a in cap.accumulators.items() if n in KEEP}
        return cap

    monkeypatch.setattr(IntegratedOutputCapture, "for_policy", classmethod(only_some))
    kw = dict(source_root=tmp_path, device="cpu", allow_dirty=True, build_datasets=build, low_update_cfg=luc)
    return cfg, kw


def test_capture_and_summary_end_to_end(spec, tmp_path, monkeypatch):
    cfg, kw = angleb_world(tmp_path, spec, monkeypatch)
    res = tmp_path / "res"
    for i in range(3):
        rec = tb.run_capture(cfg, i, results_root=res, **kw)
        c = rec["checks"]
        assert rec["valid"] and not rec["smoke"]
        assert c["identity"]["passed"] and c["identity"]["mismatched_batches"] == 0
        assert c["weights_unchanged"] and c["grads_unchanged"] and c["outputs_finite"]
        assert [r["name"] for r in rec["layers"]] == list(KEEP)
        calls = {r["name"]: r["calls"] for r in rec["layers"]}
        assert calls["trunk.state_projection"] == 1 and calls["flow_head.action_out"] == 2    # euler_steps 2
        assert set(rec["variants"]) == {"R", "I", "S", "P", "R0"} and len(rec["scale"]) == 7
        assert sum(rec["layers"][0]["n_bin"]) == rec["layers"][0]["n_rows"] == rec["n_dataset_samples"]
    s = tb.run_summary(cfg, results_root=res, source_root=tmp_path)
    assert s["verdict"]["verdict"] == tb.INCOMPLETE and s["available_tasks"] == [0, 1, 2]
    tb.run_capture(cfg, 3, results_root=res, **kw)
    s = tb.run_summary(cfg, results_root=res, source_root=tmp_path)
    v = s["verdict"]
    assert v["verdict"] in (tb.PASS, tb.FAIL) and v["exit_code"] == 0
    assert set(v["labels"]) <= {"uniform_integrated_suffices", "concentration_suffices", "phase_mapping_sensitive",
                                "near_full_rank"}
    assert set(s["tables"]) == {"R", "I", "S", "P", "R0", "Ri"}
    dec = s["tables"]["R"]["adaptive"]["0.95"]["2"]["fractions"]["trunk"]
    assert v["decisive"] == dec["two_sided"] and v["by_variant"]["I"]["two_sided"] == \
        s["tables"]["I"]["adaptive"]["0.95"]["2"]["fractions"]["trunk"]["two_sided"]
    assert ("uniform_integrated_suffices" in v["labels"]) == (v["by_variant"]["I"]["two_sided"] <= 0.5)
    assert set(s["stability"]["fractions"]) == {"A", "B"} and "estimator_sensitive" in s["stability"]
    assert set(s["alignment"]) == {"0", "1", "2", "3"}
    assert s["alignment"]["0"]["label"].startswith("command-level")
    for var in ("R", "I", "S", "P"):
        for i in range(4):
            name = tb.bases_name(cfg, var, i)
            assert s["bases"][name] == file_sha256(res / "twosided_angleb" / "bases" / name)
    assert not any(n.startswith(("R0", "Ri")) for n in s["bases"])
    # the I variant is the sum of the stored Grams; R and I differ when the weights do
    coh = torch.load(res / "twosided_angleb" / "capture" / "task0" / "grams_coherent.pt", weights_only=False)
    G = coh["grams"]["trunk.state_projection"]
    assert G.shape[:2] == (6, 3)
    assert torch.allclose(combine_grams(G, np.ones((6, 3))), G.sum(dim=(0, 1)))
    assert tb.run_summary(cfg, results_root=res, source_root=tmp_path)["bases"] == s["bases"]      # rerun


def test_smoke_is_incomplete_and_a_broken_identity_is_invalid(spec, tmp_path, monkeypatch):
    cfg, kw = angleb_world(tmp_path, spec, monkeypatch)
    rec = tb.run_capture(cfg, 0, results_root=tmp_path / "smoke", smoke_batches=2, **kw)
    assert rec["smoke"] and rec["valid"] and rec["n_captured_samples"] == 8 < rec["n_dataset_samples"]
    assert rec["checks"]["samples"]["status"] == "unverified"
    s = tb.run_summary(cfg, results_root=tmp_path / "smoke", source_root=tmp_path)
    assert s["verdict"]["verdict"] == tb.INCOMPLETE and s["verdict"]["exit_code"] == 4

    import flowcl.analysis.integrated_sensitivity as isens
    real = isens.integrate
    monkeypatch.setattr(isens, "integrate", lambda *a, **k: real(*a, **k) * (1 + 1e-6))
    rec = tb.run_capture(cfg, 0, results_root=tmp_path / "bad", **kw)
    assert not rec["valid"] and rec["checks"]["identity"]["mismatched_batches"] == rec["n_batches"]
    monkeypatch.setattr(isens, "integrate", real)
    for i in (1, 2, 3):
        tb.run_capture(cfg, i, results_root=tmp_path / "bad", **kw)
    s = tb.run_summary(cfg, results_root=tmp_path / "bad", source_root=tmp_path)
    assert s["verdict"]["verdict"] == tb.INVALID and s["verdict"]["exit_code"] == 3


def test_tampering_and_pins_are_refused(spec, tmp_path, monkeypatch):
    cfg, kw = angleb_world(tmp_path, spec, monkeypatch)
    tb.run_capture(cfg, 0, results_root=tmp_path / "r", **kw)
    path = tmp_path / "r" / "twosided_angleb" / "capture" / "task0" / "grams_hutch_A.pt"
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(RuntimeError, match="changed after publication"):
        tb.run_summary(cfg, results_root=tmp_path / "r", source_root=tmp_path)
    bad = copy.deepcopy(cfg)
    bad["recovery"]["analysis_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="pinned recovery analysis"):
        tb.run_capture(bad, 1, results_root=tmp_path / "x", **kw)
    work = tmp_path / cfg["recovery"]["work"] / KEYS[1].replace("/", "__") / "rollout_00.json"
    payload = json.loads(work.read_text())
    payload["binding"]["rule_sha256"] = "f" * 64
    work.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="not a production work file"):
        tb.run_capture(cfg, 1, results_root=tmp_path / "x", **kw)


def test_config_is_consistent():
    cfg = tb.load_config()
    e1 = tg.load_config()
    assert cfg["source"] == e1["source"] and cfg["u_rule"] == e1["u_rule"]
    assert cfg["rule"] == e1["rule"] and cfg["forecast"] == e1["forecast"]
    assert cfg["decisive_variant"] == "R" and cfg["variants"][0] == "R"
    assert cfg["recovery"]["rule_sha256"] == file_sha256(repo_root() / cfg["recovery"]["rule"])
    groups = cfg["capture"]["groups"]
    assert list(groups) == list(rc.GROUPS) and [tuple(v) for v in groups.values()] == [rc.DIMS[g] for g in rc.GROUPS]
    rr = tb.OmegaConf.to_container(tb.OmegaConf.load(repo_root() / cfg["recovery"]["rule"]))
    assert [t["key"] for t in rr["tasks"]] == KEYS and len(rr["phases"]) == cfg["recovery"]["n_bins"]
    assert cfg["capture"]["execute_k"] == rr["eval"]["execute_k"]
    assert float(cfg["recovery"]["magnitude"]) in [float(m) for m in rr["perturbations"]["magnitudes"]]


# ---- the queue ------------------------------------------------------------------------------------------


def run_queue(tmp_path, args=(), env_extra=None):
    logs = tmp_path / f"logs{len(list(tmp_path.glob('logs*')))}"      # one log root per run
    env = {**os.environ, "QUEUE_DRY_RUN": "1", "QUEUE_LOG_ROOT": str(logs), **(env_extra or {})}
    out = subprocess.run(["bash", str(repo_root() / "scripts" / "queue_twosided_angleb.sh"), *args],
                         capture_output=True, text=True, env=env)
    found = sorted(logs.glob("queue_*"))
    return out, found[-1] if found else None


def _log(d):
    return (d / "queue.log").read_text() if d else ""


def test_queue_order_status_and_smoke_guard(tmp_path):
    out, d = run_queue(tmp_path)
    log = _log(d)
    assert out.returncode == 0 and log.strip().endswith("QUEUE DONE ok")
    starts = [line.split("START ")[1].split(":")[0] for line in log.splitlines()
              if "START " in line and "QUEUE START" not in line]
    assert starts == ["c0", "c1", "c2", "c3", "summary"]
    out, d = run_queue(tmp_path, env_extra={"QUEUE_FAIL_STEP": "c1"})
    log = _log(d)
    assert out.returncode == 1 and "QUEUE DONE FAILED: c1" in log
    starts = [line.split("START ")[1].split(":")[0] for line in log.splitlines()
              if "START " in line and "QUEUE START" not in line]
    assert starts == ["c0", "c1", "c2", "c3", "summary"]              # captures are independent
    out, d = run_queue(tmp_path, ["--smoke", str(tmp_path / "s")])
    assert out.returncode == 0 and "s1_summary rc=4" in _log(d)            # incomplete is the expected status
    assert d.name.endswith("_smoke") and "--smoke-batches 3" in (d / "s0.log").read_text()
    out, _ = run_queue(tmp_path, ["--smoke", str(repo_root() / "results" / "x")])
    assert out.returncode == 2
