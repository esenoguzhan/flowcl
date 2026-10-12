"""Projected LoRA and the InfLoRA adaptation (flowcl/models/lora.py projected adapters,
flowcl/methods/projected_lora.py, flowcl/analysis/projection_diagnostics.py): exact protection of the
memory's span under training, the merge, skipped full-rank layers, InfLoRA's design inside an explicit
complement, the matched ablation's identical start, Task 1 as seq_ft with GPM's memory, resume, and the
diagnostics (the general output-change form, f_outside)."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from flowcl.analysis import projection_diagnostics as pd
from flowcl.data.dataset import ChunkedActionDataset
from flowcl.data.stats import compute_stats
from flowcl.methods.base import TaskContext
from flowcl.methods.gpm import GPM
from flowcl.methods.projected_lora import ProjectedLoRA
from flowcl.models import lora as adapters
from flowcl.train.continual import method_stage_config
from flowcl.train.trainer import TrainConfig, train_one_task
from test_gpm import capture_yaml, fresh_policy, small_cfg, spec, task_data  # noqa: F401


def orthonormal(d, k, seed):
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(d, k, generator=g, dtype=torch.float64))
    return q


# ---- the adapter ----------------------------------------------------------------------------------


def test_projected_adapter_protects_the_memory_under_adam_and_merges():
    torch.manual_seed(0)
    layer = nn.Linear(16, 8)
    keys = list(layer.state_dict())
    M = orthonormal(16, 6, 1)
    base = layer.weight.detach().clone()
    adapters.attach_projected(layer, torch.randn(4, 16), M.float(), 2.0, train_a=True)
    with torch.no_grad():
        layer.lora_B.normal_()
    opt = torch.optim.Adam([layer.lora_A, layer.lora_B], lr=0.1)
    for _ in range(5):   # Adam moves A off the complement; the forward projection still protects span(M)
        opt.zero_grad()
        layer(torch.randn(32, 16)).pow(2).sum().backward()
        opt.step()
    assert float((layer.lora_A.detach().double() @ M).norm()) > 1e-3
    x_old = (M @ torch.randn(6, 10, dtype=torch.float64)).T.float()
    with torch.no_grad():
        change = layer(x_old) - (x_old @ base.T + layer.bias)
    assert float(change.abs().max()) < 1e-5
    intended, realized = adapters.merge_projected(layer)
    assert list(layer.state_dict()) == keys and not adapters.is_projected(layer)
    assert float((realized.double() @ M).norm()) < 1e-5 * float(realized.norm()) + 1e-6
    assert torch.allclose(intended, realized, atol=1e-6)
    with pytest.raises(ValueError):
        adapters.attach_projected(nn.Linear(4, 4), torch.zeros(2, 5), None, 1.0, True)


def test_diagnostics_general_form_f_outside_and_inflora_design():
    torch.manual_seed(0)
    d_in, d_out = 12, 5
    M = orthonormal(d_in, 4, 2)
    P = torch.eye(d_in, dtype=torch.float64) - M @ M.T
    X = torch.randn(d_in, 50, dtype=torch.float64)
    K = X @ X.T
    W0 = torch.randn(d_out, d_in, dtype=torch.float64)
    D = torch.randn(d_out, d_in, dtype=torch.float64)           # unprojected: the forms differ
    out = pd.output_change(D, W0, K, P)
    assert out["energy"] == pytest.approx(float(((D @ X) ** 2).sum()))
    assert abs(out["sandwiched"] - out["energy"]) > 1e-3 * out["energy"]
    Dp = D @ P                                                   # projected: they agree
    out = pd.output_change(Dp, W0, K, P)
    assert out["sandwiched"] == pytest.approx(out["energy"], rel=1e-9)
    Q0 = orthonormal(d_in, 3, 3)
    assert pd.subspace_movement(torch.zeros(d_out, d_in), Q0)["f_outside"] is None
    inside = pd.subspace_movement(torch.randn(d_out, 3, dtype=torch.float64) @ Q0.T, Q0)
    assert inside["f_outside"] == pytest.approx(0.0, abs=1e-12) and max(inside["principal_angles_deg"]) < 1e-4
    C = pd.complement_basis(M, d_in)
    assert C.shape == (d_in, 8) and float((C.T @ M).abs().max()) < 1e-10
    rows, info = pd.inflora_down(K, M, 3)
    assert rows.shape == (3, d_in) and float((rows @ M).abs().max()) < 1e-10
    assert torch.allclose(rows @ rows.T, torch.eye(3, dtype=torch.float64), atol=1e-10)
    Kc = C.T @ K @ C
    top = torch.linalg.eigvalsh(Kc).flip(0)[:3]
    assert torch.allclose(torch.diag(rows @ K @ rows.T), top, rtol=1e-8)
    lowrank = (M[:, :1] + C[:, :1]) @ (M[:, :1] + C[:, :1]).T     # one direction outside the memory
    rows, info = pd.inflora_down(lowrank, M, 3)
    assert rows.shape[0] == 1 and info["effective_rank"] == 1    # zero eigenvalues dropped
    rows, info = pd.inflora_down(K, orthonormal(d_in, d_in, 4), 3)
    assert rows.shape[0] == 0 and info["free_dims"] == 0


# ---- the method -----------------------------------------------------------------------------------


def data_for(spec, keys):
    eps = {k: task_data(spec, k, 10 * i) for i, k in enumerate(keys)}
    stats = compute_stats(eps[keys[0]], embodiment=spec.name, task_id=keys[0])
    return {k: ChunkedActionDataset(eps[k], spec, stats) for k in keys}


def ctx(data, key):
    return TaskContext(task_key=key, dataset=data[key], device="cpu", seed_namespace_run_id="toy_ns",
                       method_run_id="toy_run")


def fake_memory(policy, full=()):
    mem = {}
    for i, e in enumerate(policy.projectable_layers()):
        d = e.module.in_features
        mem[e.name] = orthonormal(d, d if e.name in full else d // 2, 10 + i)
    return mem


def fake_grams(policy, task_idx, context):
    out = {}
    for i, e in enumerate(policy.projectable_layers()):
        X = torch.randn(e.module.in_features, 40, generator=torch.Generator().manual_seed(100 + i),
                        dtype=torch.float64)
        out[e.name] = X @ X.T
    return out


@pytest.mark.parametrize("variant", ["trained", "inflora", "pca_trained"])
def test_variants_train_only_adapters_protect_the_memory_and_skip_full_rank_layers(spec, variant, monkeypatch):
    keys = ["toy/one", "toy/two"]
    data = data_for(spec, keys)
    policy = fresh_policy(spec)
    names = [e.name for e in policy.projectable_layers()]
    method = ProjectedLoRA(variant=variant, rank=2, lr=1e-2, update_memory=False)
    method._memory = fake_memory(policy, full=(names[0],))
    monkeypatch.setattr(method, "_design_grams", fake_grams)
    keys_before = list(policy.state_dict())
    base = {n: p.detach().clone() for n, p in policy.named_parameters()}
    cfg = method_stage_config(small_cfg(steps=3), method, 1)
    assert cfg.lr == 1e-2
    train_one_task(policy, data[keys[1]], cfg, method=method, task_idx=1, generator=torch.Generator().manual_seed(1),
                   context=ctx(data, keys[1]))
    hist = method.history["1"]
    assert hist["skipped_layers"] == [names[0]] and hist["adapted_layers"] == len(names) - 1
    assert hist["merge"]["constraint_ok"]
    assert list(policy.state_dict()) == keys_before
    params = dict(policy.named_parameters())
    moved = 0
    for e in policy.projectable_layers():
        D = (params[f"{e.name}.weight"] - base[f"{e.name}.weight"]).double()
        M = method._memory[e.name]
        if e.name == names[0]:
            assert float(D.norm()) == 0.0
            continue
        moved += float(D.norm()) > 0
        assert float((D @ M).norm()) <= 1e-5 * float(base[f"{e.name}.weight"].norm()) + 1e-4 * float(D.norm())
    assert moved > 0
    for n, p in params.items():   # nothing but registry weights moved
        if not any(n == f"{e.name}.weight" for e in policy.projectable_layers()):
            assert torch.equal(p, base[n]), n
    expected = (sum(e.module.out_features * 2 for e in policy.projectable_layers()[1:])
                + (0 if variant == "inflora" else sum(e.module.in_features * 2 for e in policy.projectable_layers()[1:])))
    assert hist["trainable_params"] == expected
    assert (hist["merge"]["median_f_outside"] is not None) or variant == "inflora"


def test_pca_trained_starts_from_inflora_s_basis_and_all_skipped_raises(spec, monkeypatch):
    keys = ["toy/one", "toy/two"]
    data = data_for(spec, keys)
    starts = {}
    for variant in ("inflora", "pca_trained"):
        policy = fresh_policy(spec)
        method = ProjectedLoRA(variant=variant, rank=2, lr=1e-3, update_memory=False)
        method._memory = fake_memory(policy)
        monkeypatch.setattr(method, "_design_grams", fake_grams)
        method.on_task_start(policy, 1, context=ctx(data, keys[1]))
        e = policy.projectable_layers()[3]
        starts[variant] = (e.module.lora_A.detach().clone(), e.module.lora_A.requires_grad)
    assert torch.equal(starts["inflora"][0], starts["pca_trained"][0])
    assert starts["inflora"][1] is False and starts["pca_trained"][1] is True
    policy = fresh_policy(spec)
    method = ProjectedLoRA(variant="trained", rank=2, lr=1e-3, update_memory=False)
    method._memory = fake_memory(policy, full=tuple(e.name for e in policy.projectable_layers()))
    with pytest.raises(RuntimeError, match="nothing can be adapted"):
        method.on_task_start(policy, 1, context=ctx(data, keys[1]))
    with pytest.raises(ValueError):
        ProjectedLoRA(variant="other", rank=2, lr=1e-3)
    with pytest.raises(ValueError, match="bundle"):
        ProjectedLoRA(variant="trained")


def test_task1_is_seq_ft_and_its_memory_is_gpm_s(spec, capture_yaml):
    keys = ["toy/one", "toy/two"]
    data = data_for(spec, keys)
    results = {}
    for name, method in (("gpm", GPM(update_memory=True, capture_config=capture_yaml)),
                         ("plora", ProjectedLoRA(variant="trained", rank=2, lr=1e-3, capture_config=capture_yaml))):
        policy = fresh_policy(spec)
        train_one_task(policy, data[keys[0]], small_cfg(), method=method, task_idx=0,
                       generator=torch.Generator().manual_seed(0), context=ctx(data, keys[0]))
        results[name] = ({k: v.clone() for k, v in policy.state_dict().items()}, method._memory)
    (wa, ma), (wb, mb) = results["gpm"], results["plora"]
    assert all(torch.equal(wa[k], wb[k]) for k in wa)
    assert sorted(ma) == sorted(mb) and all(torch.equal(ma[k], mb[k]) for k in ma)


def test_state_dict_and_restore_history(spec, monkeypatch):
    keys = ["toy/one", "toy/two"]
    data = data_for(spec, keys)
    policy = fresh_policy(spec)
    method = ProjectedLoRA(variant="trained", rank=2, lr=1e-2, update_memory=False)
    method._memory = fake_memory(policy)
    train_one_task(policy, data[keys[1]], small_cfg(steps=2), method=method, task_idx=1,
                   generator=torch.Generator().manual_seed(1), context=ctx(data, keys[1]))
    state = method.state_dict()
    assert state["lora"]["variant"] == "trained" and "1" in state["history"]
    other = ProjectedLoRA(variant="trained", rank=4, lr=1e-2, update_memory=False)
    monkeypatch.setattr(GPM, "restore_state", lambda self, *a, **k: {})
    with pytest.raises(ValueError, match="config"):
        other.restore_state({"method_state": state}, None, 1)
    same = ProjectedLoRA(variant="trained", rank=2, lr=1e-2, update_memory=False)
    assert same.restore_state({"method_state": state}, None, 1)["history_tasks"] == [1]
    assert same.stage_lr(0) is None and same.stage_lr(1) == 1e-2 and same.scale == 8.0
    assert ProjectedLoRA(variant="inflora", rank=2, lr=1e-2).scale == 1.0


# ---- the bracket on a fake world (flowcl/experiments/lora_selection.py with a projection rule) ------


def rule_file(variant):
    from flowcl.utils.libero_paths import repo_root

    name = {"trained": "projected_lora", "inflora": "inflora", "pca_trained": "pca_trained"}[variant]
    return repo_root() / "configs" / "analysis" / f"baseline_selection_{name}.yaml"


def projection_world(tmp_path, spec, capture_yaml, variant):
    """The LoRA bracket's fake world plus GPM seed 0's stage 0 (= seq_ft's) and its Task-1 memory."""
    import copy
    import shutil

    from flowcl.experiments import lora_selection as ls
    from flowcl.experiments.low_update import seed_context
    from flowcl.train.checkpoint import load_checkpoint
    from flowcl.utils.run import file_sha256
    from test_low_update import BOOT, fake_world, stub_evaluator

    luc, build = fake_world(tmp_path, spec, seeds=(0,))
    cfg = copy.deepcopy(ls.load_config(rule_file(variant)))
    cfg["lora"]["recipe"] = {"steps": 2}
    cfg["projection"]["capture_config"] = capture_yaml
    cfg["projection"]["kold"]["capture_config"] = capture_yaml
    ref = tmp_path / "seq_hetero__seq_ft__seed0"
    gpm = tmp_path / cfg["projection"]["memory"]["run"]
    (gpm / "checkpoints").mkdir(parents=True, exist_ok=True)
    shutil.copy(ref / "checkpoints" / "stage0.pt", gpm / "checkpoints" / "stage0.pt")
    stage = seed_context(luc, 0, tmp_path).curriculum.stages[0]
    loaded = load_checkpoint(ref / "checkpoints" / "stage0.pt", device="cpu")
    context = TaskContext(task_key=stage.task_key, dataset=build(stage), device="cpu",
                          seed_namespace_run_id=ref.name, method_run_id=gpm.name)
    method = GPM(update_memory=True, capture_config=capture_yaml)
    method.on_task_start(loaded.policy, 0, context=context)
    method.on_task_end(loaded.policy, 0, context=context)
    method.save_artifacts(gpm / "method", 0, context=context)
    cfg["projection"]["memory"]["sha256"] = file_sha256(gpm / "method" / "memory_task0.pt")
    kw = dict(results_root=tmp_path, device="cpu", build_datasets=build, low_update_cfg=luc)
    train = dict(kw, evaluator=stub_evaluator, bootstrap=BOOT, allow_dirty=True, rule=rule_file(variant))
    return cfg, kw, train


def test_projection_arm_records_constraint_energy_cost_and_refuses_mispairing(spec, tmp_path, capture_yaml):
    import copy

    from flowcl.experiments import lora_selection as ls
    from flowcl.utils.run import file_sha256

    cfg, kw, train = projection_world(tmp_path, spec, capture_yaml, "trained")
    with pytest.raises(FileNotFoundError, match="capture K_old"):
        ls.run_arm(cfg, 4, 1e-3, **train)
    out = ls.capture_kold(cfg, results_root=tmp_path, device="cpu", low_update_cfg=kw["low_update_cfg"],
                          allow_dirty=True, build_datasets=kw["build_datasets"])
    assert not out["reused"]
    assert ls.capture_kold(cfg, results_root=tmp_path, allow_dirty=True)["sha256"] == out["sha256"]  # immutable
    row = ls.run_arm(cfg, 4, 1e-3, **train)
    rec = row["lora"]
    assert row["run_id"] == "plorab__r4__lr0.001__seed0" and row["method"]["family"] == "projected_lora"
    assert row["method"]["projection"] == {"variant": "trained", "eps": 0.95,
                                           "memory_sha256": cfg["projection"]["memory"]["sha256"]}
    assert rec["inputs"]["projection"]["memory"]["sha256"] == cfg["projection"]["memory"]["sha256"]
    assert rec["inputs"]["projection"]["kold"]["sha256"] == out["sha256"]
    assert rec["history"]["merge"]["constraint_ok"] and rec["history"]["trainable_params"] > 0
    assert set(rec["timings"]) == {"design_seconds", "train_seconds", "peak_gpu_bytes"}
    change = rec["output_change"]
    assert change["total_relative"] > 0
    for layer in change["per_layer"].values():   # projected: the sandwiched form agrees with the general one
        assert layer["sandwiched"] == pytest.approx(layer["energy"], rel=1e-4, abs=1e-12)
    bad = copy.deepcopy(cfg)
    bad["projection"]["memory"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="not the memory the rule names"):
        ls.run_arm(bad, 4, 1e-3, **train)
    gpm0 = tmp_path / cfg["projection"]["memory"]["run"] / "checkpoints" / "stage0.pt"
    payload = torch.load(gpm0, weights_only=False)
    name = next(k for k, v in payload["state_dict"].items() if v.is_floating_point())
    payload["state_dict"][name] = payload["state_dict"][name] + 1.0
    torch.save(payload, gpm0)
    with pytest.raises(RuntimeError, match="mispaired"):
        ls.run_arm(cfg, 4, 1e-3, **train)
    # the energy diagnostic for any checkpoint pair; memories compared basis for basis
    final = ls.arm_dir(cfg, tmp_path, 4, 1e-3, 0) / "checkpoints" / "final.pt"
    ref0 = tmp_path / "seq_hetero__seq_ft__seed0" / "checkpoints" / "stage0.pt"
    energy = ls.reference_output_change(ref0, final, ls.kold_path(cfg, tmp_path))
    assert energy["total_relative"] == pytest.approx(change["total_relative"], rel=1e-6)
    mem = ls.memory_path(cfg, tmp_path)
    assert ls.compare_memories(mem, mem)["passed"]
    assert file_sha256(mem) == cfg["projection"]["memory"]["sha256"]


def test_projection_pipeline_selects_and_verifies_under_its_family(spec, tmp_path, capture_yaml):
    from flowcl.experiments import lora_selection as ls

    cfg, kw, train = projection_world(tmp_path, spec, capture_yaml, "pca_trained")
    ls.capture_kold(cfg, results_root=tmp_path, device="cpu", low_update_cfg=kw["low_update_cfg"],
                    allow_dirty=True, build_datasets=kw["build_datasets"])
    assert ls.extension_candidates(cfg) == [(16, 1e-5), (16, 3e-3)]   # a single rank: lr extensions only
    rows = [ls.run_arm(cfg, r, lr, **train) for r, lr in ls.grid(cfg)]
    assert [r["run_id"] for r in rows] == [f"pcab__r16__lr{ls.lr_tag(lr)}__seed0" for _, lr in ls.grid(cfg)]
    sel = dict(results_root=tmp_path, low_update_cfg=kw["low_update_cfg"], allow_dirty=True, rule=train["rule"])
    out = ls.run_select(cfg, **sel)
    while out["status"] == "needs_extension":
        assert all(r == 16 for r, _ in out["owed"])
        ls.run_extension(cfg, **train)
        out = ls.run_select(cfg, **sel)
    assert out["status"] == "selected" and out["family"] == "pca_trained"
    assert ls.verify(cfg, **sel)["passed"]
    with pytest.raises(ValueError):
        ls.check_bundle(out, out["rule_sha256"])   # the LoRA family's check refuses it


def test_owed_extensions_without_a_rank_axis():
    from flowcl.experiments import lora_selection as ls

    ext = {"rank": None, "lr": {"low": 1e-5, "high": 3e-3}}
    assert ls.owed_extensions({"rank": 16, "lr": 3e-5}, [16], [3e-5, 1e-3], ext) == [(16, 1e-5)]
    assert ls.owed_extensions({"rank": 16, "lr": 1e-4}, [16], [3e-5, 1e-4, 1e-3], ext) == []


def test_smoke_method_builds_the_variant(tmp_path):
    from flowcl.data.config import load_method_config
    from flowcl.experiments import lora_selection as ls
    from flowcl.methods.base import build_method

    for variant in ("trained", "inflora", "pca_trained"):
        cfg = ls.load_config(rule_file(variant))
        path = ls.write_smoke_method(tmp_path / f"{variant}.yaml", 4, 1e-3, 16.0, projection=cfg["projection"])
        name, kwargs = load_method_config(path)
        method = build_method(name, **kwargs)
        assert isinstance(method, ProjectedLoRA) and method.variant == variant and method.update_memory
        assert method.display_name == cfg["family"] and method.stage_lr(1) == 1e-3


# ---- the queues -----------------------------------------------------------------------------------


def _queue(tmp_path, script, args, fail="", step_rc="", present=()):
    import os
    import subprocess

    from flowcl.utils.libero_paths import repo_root

    results = tmp_path / "results"
    for rel in present:
        p = results / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}")
    env = {**os.environ, "QUEUE_DRY_RUN": "1", "QUEUE_FAIL_STEP": fail, "QUEUE_STEP_RC": step_rc,
           "QUEUE_LOG_ROOT": str(tmp_path / "logs"), "QUEUE_RESULTS_ROOT": str(results)}
    done = subprocess.run(["bash", str(repo_root() / "scripts" / script), *args], env=env, capture_output=True,
                          text=True, timeout=300)
    logs = sorted((tmp_path / "logs").glob("queue_*")) if (tmp_path / "logs").exists() else []
    return done.returncode, logs, done


def _log(logs, suffix):
    (d,) = [d for d in logs if d.name.endswith(suffix)]
    return (d / "queue.log").read_text(), d


def _order(log, steps):
    idx = [log.index(f"START {s}:") for s in steps]
    return idx == sorted(idx)


def test_bracket_queue_reads_the_rule_and_orders_arms_select_verify(tmp_path):
    from flowcl.experiments import lora_selection as ls

    rule = str(rule_file("trained"))
    cfg = ls.load_config(rule)
    arms = [f"{k}_arm_r{r}_lr{ls.lr_tag(lr)}" for k, (r, lr) in enumerate(ls.grid(cfg))]
    assert len(arms) == 12
    rc, logs, _ = _queue(tmp_path / "a", "queue_projected_lora_bracket.sh", ["--config", rule])
    log, d = _log(logs, "projected_lora_bracket")
    assert rc == 0 and log.rstrip().endswith("QUEUE DONE ok") and _order(log, arms + ["s_select", "v_verify"])
    assert (d / f"{arms[1]}.log").read_text().strip().endswith(f"--config {rule} --arm 4 0.0001")
    rc, logs, _ = _queue(tmp_path / "b", "queue_projected_lora_bracket.sh", ["--config", rule],
                         step_rc="s_select=10")
    log, _ = _log(logs, "projected_lora_bracket")
    assert rc == 0 and _order(log, ["s_select", "e_extension_1", "s_select_1", "v_verify"])
    rc, logs, _ = _queue(tmp_path / "c", "queue_projected_lora_bracket.sh", ["--config", rule], fail=arms[3])
    log, _ = _log(logs, "projected_lora_bracket")
    assert rc == 1 and f"START {arms[4]}" in log and "SKIP select and verify" in log
    rows = [f"baseline_selection_projected_lora/lora/seed0/plorab__r4__lr{t}__seed0/row.json" for t in ("3e-05",)]
    rc, logs, _ = _queue(tmp_path / "d", "queue_projected_lora_bracket.sh", ["--config", rule, "--from-step", "1"],
                         present=rows)
    log, _ = _log(logs, "projected_lora_bracket")
    assert rc == 0 and f"PRIOR {arms[0]} ok" in log and f"START {arms[0]}" not in log
    pca = str(rule_file("pca_trained"))
    rc, logs, _ = _queue(tmp_path / "e", "queue_projected_lora_bracket.sh", ["--config", pca])
    log, _ = _log(logs, "pca_trained_bracket")
    assert rc == 0 and log.count("_arm_r16_") == 8   # 4 arms, START and END each
    for bad in ([], ["--config", "nope.yaml"], ["--config", rule, "--from-step", "13"],
                ["--config", rule, "--smoke", "rel"], ["--config", rule, "--smoke-shared", "/tmp/x"],
                ["--config", str(rule_file("trained")).replace("projected_lora", "lora")]):
        rc, _, done = _queue(tmp_path / "bad", "queue_projected_lora_bracket.sh", bad)
        assert rc == 2, (bad, done.stderr)


def test_bracket_queue_smoke_commands_and_shared_reuse(tmp_path):
    rule = str(rule_file("inflora"))
    smoke, shared = tmp_path / "smoke", tmp_path / "shared"
    (shared / "seq_hetero__seq_ft__seed0").mkdir(parents=True)
    (shared / "seq_hetero__seq_ft__seed0" / "result.json").write_text("{}")
    rc, logs, _ = _queue(tmp_path / "q", "queue_projected_lora_bracket.sh",
                         ["--config", rule, "--smoke", str(smoke), "--smoke-shared", str(shared)])
    log, d = _log(logs, "inflora_bracket_smoke")
    assert rc == 0 and "REUSE s1_seqft" in log and "START s1_seqft" not in log
    assert _order(log, ["s0_smoke_method", "s2_gpm", "s3_run", "s4_memory_equal", "s5_resume", "s6_compare",
                        "s7_bracket_identity", "s8_benchmark"])
    assert "--rank 4 --lr 0.001" in (d / "s0_smoke_method.log").read_text()
    assert f"{smoke}/seq_hetero__inflora__seed0/method/memory_task0.pt" in (d / "s4_memory_equal.log").read_text()
    assert "--smoke-arm 4 0.001" in (d / "s7_bracket_identity.log").read_text()
    assert "--bench-ranks 0 4 16 64" in (d / "s8_benchmark.log").read_text()
    assert "--method gpm " in (d / "s2_gpm.log").read_text()


def test_tonight_queue_chains_and_gates(tmp_path):
    suite = tmp_path / "suite.txt"
    args = ["--suite-result", str(suite), "--smoke", str(tmp_path / "smoke")]
    steps = (["t0_kold", "t1_energy_gpm", "t2_energy_lora", "t3_measures_smoke"]
             + [f"t4_smoke_{f}" for f in ("projected_lora", "inflora", "pca_trained")]
             + ["t5_suite", "t6_measures"] + [f"t7_bracket_{f}" for f in ("projected_lora", "inflora", "pca_trained")])
    rc, logs, done = _queue(tmp_path / "a", "queue_tonight.sh", args)
    log, d = _log(logs, "_tonight")
    assert rc == 0 and log.rstrip().endswith("QUEUE DONE ok") and _order(log, steps), done.stdout
    assert "--suite-result" in (d / "t6_measures.log").read_text()
    assert "--capture-kold" in (d / "t0_kold.log").read_text()
    rc, logs, _ = _queue(tmp_path / "b", "queue_tonight.sh", args, fail="t4_smoke_inflora")
    log, _ = _log(logs, "_tonight")
    assert rc == 1 and "SKIP t7_bracket_inflora: its smoke failed" in log
    assert "START t7_bracket_projected_lora" in log and "START t7_bracket_pca_trained" in log
    rc, logs, _ = _queue(tmp_path / "c", "queue_tonight.sh", args, fail="t5_suite")
    log, _ = _log(logs, "_tonight")
    assert rc == 1 and "SKIP t6_measures" in log and "START t7_" not in log
    rc, logs, _ = _queue(tmp_path / "d", "queue_tonight.sh", args, fail="t3_measures_smoke")
    log, _ = _log(logs, "_tonight")
    assert rc == 1 and "SKIP t6_measures" in log and "START t7_bracket_inflora" in log
    rc, logs, _ = _queue(tmp_path / "e", "queue_tonight.sh", args, fail="t0_kold")
    log, _ = _log(logs, "_tonight")
    assert rc == 1 and "SKIP t7_bracket_projected_lora: K_old was not captured" in log
    for bad in ([], ["--suite-result", "rel.txt", "--smoke", "/tmp/x"], ["--suite-result", str(suite)],
                [*args, "--suite-wait-min", "x"]):
        rc, _, _ = _queue(tmp_path / "bad", "queue_tonight.sh", bad)
        assert rc == 2, bad
