"""C1 high-protection report: the verdict and flags (pure), the replication rule and its refusals,
the energy check, the T2 rank prediction against extend_basis itself, the protected fractions, every
gate wired into build_report on fake run directories, and the queue's dry run (order, identity stop,
explicit T1 references, smoke)."""

from __future__ import annotations

import copy
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from flowcl.analysis.subspace import SubspaceBasis, extend_basis, save_bases
from flowcl.data.curriculum import load_curriculum
from flowcl.experiments import high_protection as hp
from flowcl.experiments.reverse_order import method_spec
from flowcl.utils.libero_paths import repo_root
from flowcl.utils.run import file_sha256

KEYS = [s.task_key for s in load_curriculum("seq_hetero").stages]
N_EP = 50
MARGIN = 0.10


def D(diff, low, high):
    return {"diff": diff, "low": low, "high": high}


ALL_OK = {"identity": True, "energy": True, "valid": True, "premise": True, "reference": True,
          "t3": True, "gain": True}


# ---- the verdict (pure) --------------------------------------------------------------------------


@pytest.mark.parametrize("key,verdict", [
    ("identity", "invalid_comparison"), ("energy", "invalid_implementation"), ("valid", "invalid_run"),
    ("premise", "not_applicable"), ("reference", "invalid_reference")])
def test_gates_stop_in_order(key, verdict):
    out = hp.classify({**ALL_OK, key: False}, None, MARGIN)
    assert out["verdict"] == verdict and out["flags"] == []


@pytest.mark.parametrize("d,gain,verdict,flags", [
    (D(0.02, -0.05, 0.09), True, "protection_suffices", []),
    # GPT's example: adaptive significantly better AND e99 non-inferior; both reported.
    (D(-0.06, -0.09, -0.02), True, "protection_suffices", ["adaptive_advantage"]),
    (D(0.12, 0.04, 0.20), True, "protection_suffices", ["e99_advantage"]),
    (D(-0.30, -0.44, -0.16), True, "flat_control_inferior", ["adaptive_advantage"]),
    (D(-0.30, -0.44, -0.16), False, "flat_control_inferior", ["adaptive_advantage", "gain_not_reproduced"]),
    (D(-0.04, -0.18, 0.10), True, "unresolved", ["within_margin_by_point_only"]),
    (D(-0.14, -0.26, -0.02), True, "unresolved", ["adaptive_advantage"]),
    (D(0.02, -0.05, 0.09), False, "unresolved", ["gain_not_reproduced"]),
    (D(-0.10, -0.20, 0.0), True, "unresolved", ["within_margin_by_point_only"]),  # inclusive at -margin
])
def test_practical_verdict_and_independent_flags(d, gain, verdict, flags):
    out = hp.classify({**ALL_OK, "gain": gain}, d, MARGIN)
    assert out["verdict"] == verdict and out["flags"] == flags


def test_plasticity_failure_keeps_the_flags():
    out = hp.classify({**ALL_OK, "t3": False}, D(-0.30, -0.44, -0.16), MARGIN)
    assert out["verdict"] == "plasticity_failure" and "adaptive_advantage" in out["flags"]


@pytest.mark.parametrize("verdicts,outcome", [
    (["protection_suffices"] * 3, "protection_suffices_replicated"),
    (["flat_control_inferior"] * 3, "flat_control_inferior_replicated"),
    (["plasticity_failure"] * 3, "plasticity_failure_replicated"),
    (["protection_suffices", "unresolved", "protection_suffices"], "mixed"),
    (["protection_suffices", "flat_control_inferior", "protection_suffices"], "mixed"),
    (["protection_suffices", "not_applicable", "invalid_run"], "inconclusive_invalid"),
    (["protection_suffices", "not_applicable", "unresolved"], "not_applicable"),
    (["invalid_reference", "protection_suffices", "protection_suffices"], "inconclusive_invalid"),
])
def test_replication_rows(verdicts, outcome):
    assert hp.classify_replication(dict(zip("012", verdicts)))["replication"] == outcome


# ---- energy, rank prediction, fractions ------------------------------------------------------------

ENERGY = {"eps": 0.95, "eps_later": 0.99, "tasks": [0, 1, 2, 3], "tol": 1e-6, "target_match_atol": 1e-12}
LAYERS = ["trunk.a", "flow_head.b"]


def info(target, captured=None, p=0.5, d_in=4, k_before=0, k_added=1):
    return {"proj_energy_fraction": p, "target_fraction": target,
            "captured_energy_fraction": target if captured is None else captured,
            "d_in": d_in, "k_before": k_before, "k_added": k_added, "k_after": k_before + k_added,
            "rho_after": (k_before + k_added) / d_in, "capacity_exhausted": False}


def history(targets=(0.95, 0.99, 0.99, 0.99), layers=LAYERS):
    return {str(t): {n: info(target) for n in layers} for t, target in enumerate(targets)}


def test_energy_check_passes_and_fails_on_missing_or_wrong_evidence():
    ok = hp.energy_check(history(), {"eps_later": 0.99}, LAYERS, ENERGY)
    assert ok["passed"] and ok["n_checked"] == 8
    cases = []
    h = history(); h["1"]["trunk.a"]["target_fraction"] = 0.9895          # adaptive-like target
    cases.append((h, {"eps_later": 0.99}))
    h = history(); h["2"]["flow_head.b"]["captured_energy_fraction"] = 0.98  # under-captured
    cases.append((h, {"eps_later": 0.99}))
    cases.append((history(targets=(0.99, 0.99, 0.99, 0.99)), {"eps_later": 0.99}))  # task 0 not 0.95
    h = history(); del h["3"]                                               # a missing task
    cases.append((h, {"eps_later": 0.99}))
    h = history(); del h["1"]["trunk.a"]                                    # a missing layer
    cases.append((h, {"eps_later": 0.99}))
    h = history(); h["0"]["trunk.extra"] = info(0.95)                       # an extra layer
    cases.append((h, {"eps_later": 0.99}))
    h = history(); h["2"]["trunk.a"]["captured_energy_fraction"] = float("nan")
    cases.append((h, {"eps_later": 0.99}))
    cases.append((history(), {}))                                           # config lacks eps_later
    h = history(); h["1"]["trunk.a"]["p_saturated"] = True; h["1"]["trunk.a"]["proj_energy_fraction"] = 1.0
    h["1"]["trunk.a"]["target_fraction"] = 0.95                             # saturated layers are checked too
    cases.append((h, {"eps_later": 0.99}))
    for h, conf in cases:
        assert not hp.energy_check(h, conf, LAYERS, ENERGY)["passed"]


def _psd(d, seed, decay=0.6):
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
    return Q @ torch.diag(decay ** torch.arange(d, dtype=torch.float64)) @ Q.T


@pytest.mark.parametrize("seed", range(6))
def test_predicted_t2_ranks_equal_extend_basis_at_eps_later(seed):
    d = 24
    M, _ = extend_basis(None, _psd(d, 100 + seed, 0.8), 0.95, "toy", neg_tol=1e-9)
    K = _psd(d, seed, 0.75)
    _, adaptive = extend_basis(M, K, 0.95, "toy", neg_tol=1e-9, new_energy_fraction=0.9)
    spectrum = adaptive.pop("residual_spectrum")
    sigma = torch.zeros(d, dtype=torch.float64)
    sigma[: spectrum.numel()] = spectrum.clamp(min=0).sqrt()
    predicted = hp.predicted_t2_ranks({"toy": SimpleNamespace(singular_values=sigma)}, {"toy": adaptive}, 0.99)
    _, flat = extend_basis(M, K, 0.99, "toy", neg_tol=1e-9)
    assert predicted["toy"] == flat["k_added"]


def test_predicted_t2_ranks_saturated_and_unpredictable_layers():
    sat = info(0.99, p=0.995, k_added=0)
    none_added = info(0.95, captured=0.96, p=0.96, k_added=0)
    out = hp.predicted_t2_ranks({}, {"sat": sat, "none": none_added}, 0.99)
    assert out == {"sat": 0, "none": -1}


def test_protected_fractions_are_parameter_weighted_per_half():
    dims = {"trunk.a": (10, 4), "flow_head.b": (2, 4)}
    h = {"1": {"trunk.a": info(0.99, k_added=2), "flow_head.b": info(0.99, k_added=4)}}
    out = hp.protected_fractions(h, dims)["1"]
    assert out["trunk"] == pytest.approx(0.5) and out["decoder"] == pytest.approx(1.0)
    assert out["all"] == pytest.approx((10 * 2 + 2 * 4) / (10 * 4 + 2 * 4))
    with pytest.raises(ValueError, match="d_in"):
        hp.protected_fractions(h, {"trunk.a": (10, 5), "flow_head.b": (2, 4)})


def test_t2_allocation_matched_and_not_matched():
    hist = {r: {"1": {"trunk.a": info(0.99), "flow_head.b": info(0.99)}} for r in ("plain", "adaptive", "e99")}
    hist["adaptive"]["1"]["trunk.a"]["target_fraction"] = 0.995
    frac = {"plain": {"1": {"all": 0.37, "trunk": 0.5, "decoder": 0.1}},
            "adaptive": {"1": {"all": 0.5595, "trunk": 0.78, "decoder": 0.23}},
            "e99": {"1": {"all": 0.5585, "trunk": 0.79, "decoder": 0.22}}}
    out = hp.t2_allocation(hist, frac, {"trunk.a": 1, "flow_head.b": 2}, 0.99, {"e99": 55.85}, 0.5)
    assert out["protection_matched"] and out["reading"] == "approximately matched total protection"
    assert out["per_half"]["trunk"]["share_adaptive_target_above"] == 1.0
    assert out["ranks_matching_prediction"]["matching"] == 1
    frac["e99"]["1"]["all"] = 0.5695
    out = hp.t2_allocation(hist, frac, {}, 0.99, None, 0.5)
    assert not out["protection_matched"] and out["reading"].startswith("protection not matched (delta = +1.00")


def test_config_is_consistent():
    cfg = hp.load_config()
    names = hp.run_names(cfg, 1)
    assert names["e99"] == "seq_hetero__gpm_projected_adam_e99__seed1"
    assert names["adaptive"] == "seq_hetero__gpm_projected_adam_ne90__seed1"
    assert cfg["energy"]["eps_later"] == method_spec(cfg, "e99")[1]["eps_later"]
    assert cfg["energy"]["eps"] == method_spec(cfg, "e99")[1]["eps"] == method_spec(cfg, "plain")[1]["eps"]
    assert cfg["seeds"] == [0, 1, 2] and set(cfg["predicted_t2_protected_fraction"]) == {0, 1, 2}
    adaptive_rule = OmegaConf.load(repo_root() / "configs" / "analysis" / "adaptive_gpm.yaml")
    assert cfg["min_improvement"] == adaptive_rule.retention.min_improvement
    assert cfg["cells"]["primary"] == list(adaptive_rule.retention.transition_cell)


# ---- build_report on fake runs ------------------------------------------------------------------

SEQFT = [[0.9, 0, 0, 0], [0.0, 0.9, 0, 0], [0, 0.0, 1.0, 0], [0, 0, 0, 0.98]]
PLAIN = [[0.9, 0, 0, 0], [0.9, 0.8, 0, 0], [0.88, 0.02, 0.94, 0], [0.88, 0.0, 0.56, 0.9]]
ADAPT = [[0.9, 0, 0, 0], [0.9, 0.8, 0, 0], [0.98, 0.68, 0.94, 0], [0.98, 0.84, 0.94, 0.92]]
E99 = [[0.9, 0, 0, 0], [0.9, 0.8, 0, 0], [0.96, 0.70, 0.94, 0], [0.96, 0.80, 0.90, 0.94]]
THETA = [0.75, 0.63, 0.85, 0.83]
W = {"trunk.a.weight": torch.arange(16.0).reshape(4, 4), "flow_head.b.weight": torch.ones(2, 4)}


def write_memory(d: Path, role: str, targets):
    hist = history(targets)
    for t in range(4):
        bases = {n: SubspaceBasis(layer=n, d_in=4, n_samples=10, numerical_rank=t + 1,
                                  singular_values=torch.zeros(4, dtype=torch.float64), thresholds=(0.95,),
                                  ranks={0.95: t + 1}, rhos={0.95: (t + 1) / 4},
                                  vectors=torch.eye(4, dtype=torch.float64)[:, : t + 1])
                 for n in LAYERS}
        config = {"eps": 0.95, **({"eps_later": 0.99} if role == "e99" else {})}
        save_bases(d / "method" / f"memory_task{t}.pt", bases,
                   {"kind": "accumulated_memory", "eps": 0.95, "config": config,
                    "memory_history": {k: v for k, v in hist.items() if int(k) <= t}})


def write_run(root: Path, cfg: dict, role: str, matrix, *, seed=0, fp="fp1", seeds_offset=0, steps=None,
              weights=None, targets=None):
    from flowcl.analysis.metrics import success_estimate

    name = hp.run_names(cfg, seed)[role]
    d = root / name
    (d / "eval").mkdir(parents=True)
    (d / "checkpoints").mkdir()
    (d / "method").mkdir()
    method_name, kwargs, label = method_spec(cfg, role)
    ns = f"seq_hetero__seq_ft__seed{seed}"
    train = {**cfg["production"]["train"], **({"steps": steps} if steps else {})}
    OmegaConf.save(OmegaConf.create({
        "run_id": name, "seed": seed, "seed_namespace_run_id": ns,
        "method": {"name": method_name, "display_name": label, **kwargs},
        "curriculum": {"name": "seq_hetero"}, "train": train,
        "eval": {"n_episodes": N_EP, "max_steps": 600}}), d / "config.yaml")
    (d / "git_sha").write_text("abc\n")
    (d / "stats.json").write_text(json.dumps({"fingerprint": fp, "fitted_on_task_id": KEYS[0]}))
    for i in range(4):
        torch.save({"stats_fingerprint": fp, "state_dict": (weights or W)}, d / "checkpoints" / f"stage{i}.pt")
        tasks = []
        for j, key in enumerate(KEYS):
            k = round(matrix[i][j] * N_EP)
            succ = [True] * k + [False] * (N_EP - k)
            e = success_estimate(succ)
            tasks.append({"task_key": key, "successes": succ, "n_steps": [100] * N_EP,
                          "seeds": list(range(seeds_offset + 1000 * j, seeds_offset + 1000 * j + N_EP)),
                          "success_rate": e.value, "ci_low": e.low, "ci_high": e.high, "confidence": 0.95,
                          "n_rollouts": N_EP, "wall_clock_s": 0.0})
        (d / "eval" / f"stage{i}.json").write_text(json.dumps({"run_id": ns, "stage": i, "tasks": tasks}))
    (d / "result.json").write_text(json.dumps({
        "task_keys": KEYS, "seed_namespace_run_id": ns, "t1_pairing": {"passed": True},
        "metrics": {"F_1": 0.5, "NBT": 0.1, "AUC": 0.6}}))
    if role in hp.MEMORY_ROLES:
        write_memory(d, role, targets or {"e99": (0.95, 0.99, 0.99, 0.99), "plain": (0.95,) * 4,
                                          "adaptive": (0.95, 0.991, 0.998, 0.999)}[role])
    return d


def write_seq_report(root: Path, cfg: dict, role: str, seed=0, checks=None):
    names = hp.run_names(cfg, seed)
    default = {name: {"passed": True} for name in cfg["required_checks"]}
    p = root / cfg["reports"][role][seed] / "report.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "method_run_id": names[role],
        "provenance_checks": default if checks is None else checks,
        "threshold_block": {"mode": "registered", "margin": 0.15, "task_keys": KEYS, "thresholds": THETA,
                            "reference_run_id": names["seq_ft"],
                            "reference_result_sha256": file_sha256(root / names["seq_ft"] / "result.json"),
                            "seed_namespace_run_id": names["seq_ft"]}}))


def world(root: Path, *, e99=E99, **e99_kw):
    cfg = copy.deepcopy(hp.load_config())
    write_run(root, cfg, "seq_ft", SEQFT)
    write_run(root, cfg, "plain", PLAIN)
    write_run(root, cfg, "adaptive", ADAPT)
    write_run(root, cfg, "e99", e99, **e99_kw)
    for role in hp.MEMORY_ROLES:
        write_seq_report(root, cfg, role)
    return cfg


def test_valid_world_gives_a_verdict_and_pins_its_inputs(tmp_path):
    cfg = world(tmp_path)
    r = hp.run_report(cfg, 0, results_root=tmp_path)
    assert r["checks"] == {**ALL_OK}
    assert r["verdict"]["verdict"] == "protection_suffices"
    assert r["primary"]["G"]["diff"] == pytest.approx(0.68) and r["primary"]["D"]["diff"] == pytest.approx(0.02)
    assert all(c["passed"] for c in r["validity"].values())
    inputs = r["inputs"]
    assert inputs["rule"]["sha256"] == file_sha256(hp.rule_path())
    e99_files = inputs["runs"]["e99"]
    assert all(e99_files[f"eval/stage{i}.json"] for i in range(4)) and e99_files["method/memory_task3.pt"]
    assert inputs["runs"]["adaptive"]["method/memory_task1.pt"] and inputs["bootstrap"]["n_resamples"] > 0
    assert (tmp_path / "high_protection" / "seed0" / "report.json").is_file()


def test_flat_control_inferior_with_the_adaptive_advantage_flag(tmp_path):
    e99 = copy.deepcopy(E99)
    e99[2][1] = 0.30
    r = hp.build_report(world(tmp_path, e99=e99), 0, results_root=tmp_path)
    assert r["verdict"]["verdict"] == "flat_control_inferior"
    assert "adaptive_advantage" in r["verdict"]["flags"]


@pytest.mark.parametrize("break_it,verdict,check", [
    (lambda root, cfg: write_seq_report(root, cfg, "e99", checks={}), "invalid_run", "sequence_report_e99"),
    (lambda root, cfg: (root / cfg["reports"]["plain"][0] / "report.json").unlink(), "invalid_run",
     "sequence_report_plain"),
    (lambda root, cfg: write_seq_report(root, cfg, "adaptive", checks={
        n: {"passed": True} for n in cfg["required_checks"] if n != "t1_pairing"}), "invalid_run",
     "sequence_report_adaptive"),
    (lambda root, cfg: (root / hp.run_names(cfg, 0)["seq_ft"] / "result.json").write_text(
        json.dumps({"task_keys": KEYS, "metrics": {}, "changed": True})), "invalid_run", "thresholds"),
])
def test_missing_or_stale_evidence_is_invalid(tmp_path, break_it, verdict, check):
    cfg = world(tmp_path)
    break_it(tmp_path, cfg)
    r = hp.build_report(cfg, 0, results_root=tmp_path)
    assert r["verdict"]["verdict"] == verdict and not r["validity"][check]["passed"]


def test_mismatched_episode_seeds_are_invalid(tmp_path):
    r = hp.build_report(world(tmp_path, seeds_offset=7), 0, results_root=tmp_path)
    assert r["verdict"]["verdict"] == "invalid_run" and not r["validity"]["episode_pairing"]["passed"]


def test_a_comparator_failing_production_is_invalid(tmp_path):
    cfg = copy.deepcopy(hp.load_config())
    write_run(tmp_path, cfg, "seq_ft", SEQFT)
    write_run(tmp_path, cfg, "plain", PLAIN, steps=20000)
    write_run(tmp_path, cfg, "adaptive", ADAPT)
    write_run(tmp_path, cfg, "e99", E99)
    for role in hp.MEMORY_ROLES:
        write_seq_report(tmp_path, cfg, role)
    r = hp.build_report(cfg, 0, results_root=tmp_path)
    assert r["verdict"]["verdict"] == "invalid_run" and not r["validity"]["production_plain"]["passed"]


def test_identity_and_energy_gates(tmp_path):
    other = {k: v + 1 for k, v in W.items()}
    r = hp.build_report(world(tmp_path / "a", weights=other), 0, results_root=tmp_path / "a")
    assert r["verdict"]["verdict"] == "invalid_comparison"
    r = hp.build_report(world(tmp_path / "b", targets=(0.95, 0.99, 0.99, 0.98)), 0, results_root=tmp_path / "b")
    assert r["verdict"]["verdict"] == "invalid_implementation"
    cfg = world(tmp_path / "c")
    e99_dir = tmp_path / "c" / hp.run_names(cfg, 0)["e99"]
    (e99_dir / "method" / "memory_task3.pt").unlink()
    assert hp.build_report(cfg, 0, results_root=tmp_path / "c")["verdict"]["verdict"] == "invalid_implementation"


def test_seq_ft_seed0_namespace_excuse_is_narrow(tmp_path):
    cfg = world(tmp_path)
    seqft = tmp_path / hp.run_names(cfg, 0)["seq_ft"]
    conf = OmegaConf.load(seqft / "config.yaml")
    del conf["seed_namespace_run_id"]
    OmegaConf.save(conf, seqft / "config.yaml")
    check = hp.production(seqft, cfg, "seq_ft", 0, False)
    assert check["passed"] and "excused" in check
    plain = tmp_path / hp.run_names(cfg, 0)["plain"]
    conf = OmegaConf.load(plain / "config.yaml")
    del conf["seed_namespace_run_id"]
    OmegaConf.save(conf, plain / "config.yaml")
    assert not hp.production(plain, cfg, "plain", 0, False)["passed"]


def fake_seed_report(root: Path, cfg: dict, seed: int, verdict="protection_suffices", smoke=False, rule_sha=None):
    p = root / cfg["out"].format(seed=seed)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "seed": seed, "smoke": smoke, "runs": hp.run_names(cfg, seed),
        "inputs": {"rule": {"sha256": rule_sha or file_sha256(hp.rule_path())}},
        "verdict": {"verdict": verdict, "flags": ["adaptive_advantage"] if seed else []},
        "primary": {"D": D(0.0, -0.05, 0.05)}, "descriptive": {}}))


def test_replication_reads_every_seed_and_refuses_bad_inputs(tmp_path):
    cfg = hp.load_config()
    for s in (0, 1, 2):
        fake_seed_report(tmp_path, cfg, s)
    out = hp.run_replication(cfg, results_root=tmp_path)
    assert out["replication"] == "protection_suffices_replicated"
    assert out["flags"]["adaptive_advantage"] == [1, 2]
    for bad in ({"smoke": True}, {"rule_sha": "0" * 64}):
        root = tmp_path / str(len(bad)) / next(iter(bad))
        for s in (0, 1, 2):
            fake_seed_report(root, cfg, s, **(bad if s == 1 else {}))
        with pytest.raises(ValueError, match="replication refused"):
            hp.build_replication(cfg, results_root=root)
    root = tmp_path / "missing"
    for s in (0, 2):
        fake_seed_report(root, cfg, s)
    with pytest.raises(ValueError, match="seed 1: no report"):
        hp.build_replication(cfg, results_root=root)


# ---- the queue ----------------------------------------------------------------------------------

QUEUE = repo_root() / "scripts" / "queue_high_protection.sh"
INPUTS = [f"seq_hetero__{m}__seed{s}/result.json" for s in (0, 1, 2)
          for m in ("seq_ft", "gpm_projected_adam", "gpm_projected_adam_ne90")] + \
         [f"{r}/report.json" for r in ("gpm_seq", "gpm_seq_seed1", "gpm_seq_seed2", "gpm_seq_ne90",
                                       "gpm_seq_ne90_seed1", "gpm_seq_ne90_seed2")]


def run_queue(tmp_path, args=(), fail="", present=tuple(INPUTS), files=None):
    results = tmp_path / "results"
    for rel in present:
        p = results / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}")
    for rel, text in (files or {}).items():
        p = results / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    env = {**os.environ, "QUEUE_DRY_RUN": "1", "QUEUE_FAIL_STEP": fail,
           "QUEUE_LOG_ROOT": str(tmp_path / "logs"), "QUEUE_RESULTS_ROOT": str(results)}
    done = subprocess.run(["bash", str(QUEUE), *args], env=env, capture_output=True, text=True, timeout=120)
    (logdir,) = list((tmp_path / "logs").glob("queue_*"))
    return done.returncode, (logdir / "queue.log").read_text(), logdir


def test_queue_commands_order_and_paths(tmp_path):
    assert subprocess.run(["bash", "-n", str(QUEUE)]).returncode == 0
    reports = [f"high_protection/seed{s}/report.json" for s in (0, 1, 2)]
    rc, log, logdir = run_queue(tmp_path, present=INPUTS + reports)
    steps = ["0_e99_s0", "1_seqrep_s0", "2_report_s0", "3_e99_s1", "4_seqrep_s1", "5_report_s1",
             "6_e99_s2", "7_seqrep_s2", "8_report_s2", "9_replication"]
    assert rc == 0 and log.rstrip().endswith("QUEUE DONE ok")
    assert [log.index(f"START {s}:") for s in steps] == sorted(log.index(f"START {s}:") for s in steps)
    res = str(tmp_path / "results")
    for s, k in ((0, 0), (1, 3), (2, 6)):
        cmd = (logdir / f"{k}_e99_s{s}.log").read_text()
        assert "--curriculum seq_hetero --amp" in cmd and f"--seed {s} --method gpm_e99" in cmd
        assert f"--t1-reference-run {res}/seq_hetero__seq_ft__seed{s}" in cmd
        assert (f"--identity-reference-run {res}/seq_hetero__gpm_projected_adam__seed{s} "
                "--identity-stages 0 1") in cmd
        seqrep = (logdir / f"{k + 1}_seqrep_s{s}.log").read_text()
        assert f"--out {res}/gpm_seq_e99_seed{s}/report.json" in seqrep
        assert f"--seed {s}" in (logdir / f"{k + 2}_report_s{s}.log").read_text()
    assert "--replication" in (logdir / "9_replication.log").read_text()


def test_queue_identity_stop_skips_every_later_seed(tmp_path):
    identity = {"seq_hetero__gpm_projected_adam_e99__seed0/identity_stage0.json": '{"passed": true}',
                "seq_hetero__gpm_projected_adam_e99__seed0/identity_stage1.json": '{\n  "passed": false\n}'}
    rc, log, _ = run_queue(tmp_path, fail="0_e99_s0", files=identity)
    assert rc == 1 and "STOP identity failure at seed 0" in log
    assert "START 3_e99_s1" not in log and "START 6_e99_s2" not in log and "START 9_replication" not in log
    assert "seed1(identity_stop)" in log and "seed2(identity_stop)" in log


def test_queue_other_failures_skip_only_that_seed(tmp_path):
    rc, log, _ = run_queue(tmp_path, fail="0_e99_s0")
    assert rc == 1 and "STOP" not in log and "START 1_seqrep_s0" not in log and "START 3_e99_s1" in log
    rc, log, _ = run_queue(tmp_path / "b", fail="4_seqrep_s1")
    assert rc == 1 and "START 5_report_s1" not in log and "START 6_e99_s2" in log
    rc, log, _ = run_queue(tmp_path / "c", present=[p for p in INPUTS if "seed1" not in p or "gpm_seq" in p])
    assert rc == 1 and "seed1(inputs)" in log and "START 3_e99_s1" not in log and "START 6_e99_s2" in log


def test_queue_resume_and_bad_arguments(tmp_path):
    prior = ["seq_hetero__gpm_projected_adam_e99__seed0/result.json", "gpm_seq_e99_seed0/report.json",
             "high_protection/seed0/report.json"]
    rc, log, _ = run_queue(tmp_path, ["--from-step", "3"], present=INPUTS + prior)
    assert "PRIOR 0_e99_s0 ok" in log and "PRIOR 2_report_s0 ok" in log and "START 3_e99_s1" in log
    for bad in (["--from-step", "10"], ["--from-step", "x"], ["--bogus"], ["--smoke", "rel/dir"]):
        assert subprocess.run(["bash", str(QUEUE), *bad], capture_output=True).returncode == 2


def test_queue_smoke_mode(tmp_path):
    smoke = tmp_path / "scratch"
    for rel in [p for p in INPUTS if "seed2" in p]:  # what the dry run's s0-s4 would have written
        (smoke / rel).parent.mkdir(parents=True, exist_ok=True)
        (smoke / rel).write_text("{}")
    rc, log, logdir = run_queue(tmp_path, ["--smoke", str(smoke)], present=())
    assert rc == 0 and log.rstrip().endswith("QUEUE DONE ok")
    names = {p.name[:-4] for p in logdir.glob("*.log")} - {"queue"}
    assert names == {"s0_seqft", "s1_gpm", "s2_ne90", "s3_seqrep_gpm", "s4_seqrep_ne90",
                     "6_e99_s2", "7_seqrep_s2", "8_report_s2"}
    assert logdir.name.endswith("_smoke")
    for s in ("s0_seqft", "s1_gpm", "s2_ne90", "6_e99_s2"):
        cmd = (logdir / f"{s}.log").read_text()
        assert f"--results-root {smoke}" in cmd and "--steps-per-task 200 --n-episodes 1 --allow-dirty" in cmd
        if s != "s0_seqft":
            assert f"--t1-reference-run {smoke}/seq_hetero__seq_ft__seed2" in cmd
    assert f"--identity-reference-run {smoke}/seq_hetero__gpm_projected_adam__seed2" in (logdir / "6_e99_s2.log").read_text()
    report = (logdir / "8_report_s2.log").read_text()
    assert f"--seed 2 --results-root {smoke} --smoke" in report
    bad = subprocess.run(["bash", str(QUEUE), "--smoke", str(repo_root() / "results" / "x")],
                         capture_output=True, env={**os.environ, "QUEUE_LOG_ROOT": str(tmp_path / "l2")})
    assert bad.returncode == 2


@pytest.mark.parametrize("script, flags", [
    ("run_continual.py", ["--curriculum", "--amp", "--seed", "--method", "--t1-reference-run",
                          "--identity-reference-run", "--identity-stages", "--results-root",
                          "--steps-per-task", "--n-episodes", "--allow-dirty"]),
    ("sequence_report.py", ["--method-run", "--reference-run", "--out"]),
    ("high_protection.py", ["--config", "--seed", "--replication", "--results-root", "--out", "--smoke"]),
])
def test_every_queue_step_accepts_its_flags(script, flags):
    import sys

    done = subprocess.run([sys.executable, str(repo_root() / "scripts" / script), "--help"],
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    for flag in flags:
        assert flag in done.stdout, (script, flag)
