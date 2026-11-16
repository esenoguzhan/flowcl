"""The sequence report on resumed, seen-only and extended-curriculum runs (pre-E3 plan, A2): it follows
a resumed run back through its sources, inherits their T1 pairing and identity checks, verifies each
resume link, compares namespaces by what was used, lists legitimately unevaluated cells, and takes
pinned single-task thresholds and a config path."""

from __future__ import annotations

import json
import shutil

import pytest
from omegaconf import OmegaConf

import flowcl.train.continual as continual_module
from flowcl.data.config import load_embodiment_spec
from flowcl.data.curriculum import load_curriculum
from flowcl.envs.libero_env import EvalConfig
from flowcl.experiments import sequence_report as sr
from flowcl.train.trainer import TrainConfig
from flowcl.utils.libero_paths import repo_root
from flowcl.utils.run import file_sha256
from test_continual import BBQ, MILK, SAUCE, TINY_POLICY, fake_evaluation

KETCHUP = "libero_object/pick_up_the_ketchup_and_place_it_in_the_basket"
REF3, REF4 = "test_trio__seq_ft__seed0", "test_quad__seq_ft__seed0"


def curriculum(name, keys):
    return load_curriculum({"name": name, "tasks": [{"task_key": k, "n_demos": 1} for k in keys]})


def trio():
    return curriculum("test_trio", [MILK, SAUCE, BBQ])


def quad():
    return curriculum("test_quad", [MILK, SAUCE, BBQ, KETCHUP])


def config(*refs):
    cfg = sr.load_report_config()
    cfg["criteria"]["expected_thresholds_by_reference"] = {r: {"mode": "derived", "margin": 0.15} for r in refs}
    cfg["criteria"]["fallback_tasks"] = [2]
    return cfg


@pytest.fixture(scope="module")
def world(tmp_path_factory, dataset_dir):
    """3-task seq_ft, gpm and ewc runs; each resumed into a 4-task extension at stage 3, seen-only;
    and a gpm resume at stage 2 interrupted during stage 3, then resumed from its own folder."""
    root = tmp_path_factory.mktemp("resumed")
    capture_cfg = OmegaConf.load(repo_root() / "configs" / "analysis" / "subspace.yaml")
    capture_cfg.min_samples_per_dim = 0.01
    capture_cfg.num_workers = 0
    capture = root / "capture.yaml"
    OmegaConf.save(capture_cfg, capture)
    train = TrainConfig(steps=2, batch_size=2, num_workers=0, device="cpu", log_every=0, warmup_steps=1)
    common = dict(spec=load_embodiment_spec("libero_franka"), policy_config=TINY_POLICY, train_cfg=train,
                  eval_cfg=EvalConfig(n_episodes=2), seed=0, dataset_dir=dataset_dir, pretrained=False)
    methods = {"seq_ft": {}, "gpm": {"update_memory": True, "capture_config": str(capture), "log_interval": 1},
               "ewc": {"lam": 1e-3, "fisher_samples": 4, "fisher_estimator": "sampled",
                       "fisher_normalization": "mean"}}
    three, four = {}, {}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(continual_module, "evaluate_tasks", fake_evaluation([]))
        run = continual_module.run_continual
        ref = run(trio(), method_name="seq_ft", results_root=root / "three", **common)
        three["seq_ft"] = root / "three" / ref.run_id
        three["gpm"] = root / "three" / run(trio(), method_name="gpm", method_kwargs=methods["gpm"],
                                            t1_reference_run=three["seq_ft"], results_root=root / "three",
                                            **common).run_id
        three["ewc"] = root / "three" / run(trio(), method_name="ewc", method_kwargs=methods["ewc"],
                                            t1_reference_run=three["seq_ft"], identity_reference_run=three["seq_ft"],
                                            identity_stages=(0,), results_root=root / "three", **common).run_id
        for name, src in three.items():
            res = run(quad(), method_name=name, method_kwargs=methods[name], resume_run=src, start_stage=3,
                      eval_unseen=False, results_root=root / "four", **common)
            four[name] = root / "four" / res.run_id
        cut = run(quad(), method_name="gpm", method_kwargs=methods["gpm"], resume_run=three["gpm"], start_stage=2,
                  eval_unseen=False, results_root=root / "cut", **common)
        cut_dir = root / "cut" / cut.run_id
        (cut_dir / "result.json").unlink()   # what a reboot during stage 3 leaves
        for path in [cut_dir / "checkpoints" / "stage3.pt", cut_dir / "eval" / "stage3.json",
                     *cut_dir.glob("train/stage3_*"), *cut_dir.glob("method/*task3*")]:
            path.unlink()
        again = run(quad(), method_name="gpm", method_kwargs=methods["gpm"], resume_run=cut_dir, start_stage=3,
                    eval_unseen=False, results_root=root / "again", **common)
    return {"three": three, "four": four, "cut": cut_dir, "again": root / "again" / again.run_id}


def test_a_resumed_gpm_run_is_followed_back_and_passes_strict(world):
    report = sr.build_report(world["four"]["gpm"], world["four"]["seq_ft"], config=config(REF4))
    checks = report["provenance_checks"]
    assert sr.strict_failures(report, allow_dirty=True) == []
    assert "resume_source" in report["provenance_required"] and checks["resume_source"]["passed"]
    assert checks["seed_namespace"]["passed"] and checks["seed_namespace"]["seed_namespace_run_id"] == REF3
    assert checks["t1_pairing"]["passed"] and checks["t1_pairing"]["inherited_from"] == str(world["three"]["gpm"])
    assert checks["memory_chained"]["passed"] and set(checks["memory_chained"]["steps"]) == {"0->1", "1->2", "2->3"}
    assert checks["occupancy_non_decreasing"]["passed"] and checks["artifact_hashes_match"]["passed"]
    assert checks["frozen_from_stage1"]["passed"]
    # rows 0-2 come from the 3-task source: task 3 legitimately absent there, and listed
    assert set(report["not_evaluated"]["method"]) == {"0,3", "1,3", "2,3"}
    assert report["paired_cells"]["0,3"]["not_evaluated"] and "diff" in report["paired_cells"]["3,3"]
    assert report["matrices"]["method"][0][3] == "-"
    assert report["resume_chains"]["method"][0]["source"] == str(world["three"]["gpm"])


def test_a_resumed_run_inherits_its_sources_identity_checks(world):
    report = sr.build_report(world["four"]["ewc"], world["four"]["seq_ft"], config=config(REF4))
    identity = report["provenance_checks"]["identity"]
    assert identity["passed"] and identity["expected_stages"] == [0]
    assert identity["inherited_from"] == {"0": str(world["three"]["ewc"])}
    assert "identity" in report["provenance_required"]
    assert report["provenance_checks"]["memory_chained"]["passed"] is None   # not applicable for EWC
    assert sr.strict_failures(report, allow_dirty=True) == []


def test_a_run_resumed_from_an_interrupted_run_follows_both_links(world):
    report = sr.build_report(world["again"], world["four"]["seq_ft"], config=config(REF4))
    links = report["provenance_checks"]["resume_source"]["links"]
    assert [link["source"] for link in links] == [str(world["cut"]), str(world["three"]["gpm"])]
    assert all(link["checkpoint_sha256_ok"] and link["evaluations_equal"] and link["imported_from_ok"]
               for link in links)
    assert report["provenance_checks"]["memory_chained"]["passed"]
    assert sr.strict_failures(report, allow_dirty=True) == []


def _copy_pair(world, tmp_path):
    """Copies of the 3-task gpm source and its 4-task resume, the resume pointing at the copy."""
    src = tmp_path / "src" / world["three"]["gpm"].name
    dst = tmp_path / "dst" / world["four"]["gpm"].name
    shutil.copytree(world["three"]["gpm"], src)
    shutil.copytree(world["four"]["gpm"], dst)
    old = str(world["three"]["gpm"])
    for name in ("resume.json", "result.json", *[f"train/stage{k}_record.json" for k in range(3)]):
        path = dst / name
        path.write_text(path.read_text().replace(old, str(src)))
    return src, dst


def test_a_tampered_source_checkpoint_or_imported_evaluation_fails_the_resume_check(world, tmp_path):
    src, dst = _copy_pair(world, tmp_path)
    ok = sr.build_report(dst, world["four"]["seq_ft"], config=config(REF4))
    assert ok["provenance_checks"]["resume_source"]["passed"]
    with open(src / "checkpoints" / "stage2.pt", "ab") as f:
        f.write(b"x")
    bad = sr.build_report(dst, world["four"]["seq_ft"], config=config(REF4), verify_checkpoints=False)
    assert not bad["provenance_checks"]["resume_source"]["passed"]
    assert "resume_source: failed" in sr.strict_failures(bad, allow_dirty=True)

    src, dst = _copy_pair(world, tmp_path / "second")
    report = json.loads((dst / "eval" / "stage0.json").read_text())
    report["tasks"][0]["n_steps"][0] += 1
    (dst / "eval" / "stage0.json").write_text(json.dumps(report))
    bad = sr.build_report(dst, world["four"]["seq_ft"], config=config(REF4), verify_checkpoints=False)
    assert not bad["provenance_checks"]["resume_source"]["links"][0]["evaluations_equal"]


def test_a_seen_only_run_lists_its_unevaluated_cells_and_a_missing_seen_cell_raises(world, tmp_path, dataset_dir,
                                                                                   monkeypatch):
    monkeypatch.setattr(continual_module, "evaluate_tasks", fake_evaluation([]))
    common = dict(spec=load_embodiment_spec("libero_franka"), policy_config=TINY_POLICY,
                  train_cfg=TrainConfig(steps=2, batch_size=2, num_workers=0, device="cpu", log_every=0,
                                        warmup_steps=1),
                  eval_cfg=EvalConfig(n_episodes=2), seed=0, dataset_dir=dataset_dir, pretrained=False,
                  results_root=tmp_path, eval_unseen=False)
    ref = continual_module.run_continual(trio(), method_name="seq_ft", **common)
    ewc = continual_module.run_continual(trio(), method_name="ewc", t1_reference_run=tmp_path / ref.run_id,
                                         method_kwargs={"lam": 1e-3, "fisher_samples": 4,
                                                        "fisher_estimator": "sampled",
                                                        "fisher_normalization": "mean"}, **common)
    report = sr.build_report(tmp_path / ewc.run_id, tmp_path / ref.run_id, config=config(REF3))
    assert set(report["not_evaluated"]["method"]) == {"0,1", "0,2", "1,2"}
    assert all("seen-only" in r for r in report["not_evaluated"]["method"].values())
    assert report["paired_cells"]["0,1"]["not_evaluated"] and "diff" in report["paired_cells"]["2,1"]
    assert report["outcome"]["plasticity"][2]["ok"] in (True, False)
    # a seen cell that should exist but does not
    path = tmp_path / ewc.run_id / "eval" / "stage2.json"
    payload = json.loads(path.read_text())
    payload["tasks"] = [t for t in payload["tasks"] if t["task_key"] != MILK]
    path.write_text(json.dumps(payload))
    with pytest.raises(KeyError, match=r"cell \(2,0\)"):
        sr.build_report(tmp_path / ewc.run_id, tmp_path / ref.run_id, config=config(REF3))


def test_pinned_reference_thresholds(tmp_path):
    single = tmp_path / "single.json"
    single.write_text(json.dumps({"run_id": "single", "stage": 0, "tasks": [
        {"task_key": BBQ, "success_rate": 0.88, "successes": [], "n_steps": [], "seeds": []}]}))
    gate0 = tmp_path / "gate0.json"
    gate0.write_text(json.dumps({"evidence": {"per_task": {SAUCE: {"success_rate": 0.96}}}}))
    entry = {"mode": "pinned_reference", "margin": 0.15,
             "pinned": {BBQ: {"report": str(single), "sha256": file_sha256(single)},
                        SAUCE: {"report": str(gate0), "sha256": file_sha256(gate0)}}}
    criteria = {"margin_pp": 15, "expected_thresholds_by_reference": {"r": entry}}
    got = sr.criteria_thresholds([0.9, 0.8, 0.7], criteria, "r", task_keys=[MILK, SAUCE, BBQ])
    assert got == pytest.approx([0.75, 0.81, 0.73], abs=1e-12)
    with pytest.raises(ValueError, match="task keys"):
        sr.criteria_thresholds([0.9, 0.8, 0.7], criteria, "r")
    entry["pinned"][BBQ]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="pinned sha256"):
        sr.criteria_thresholds([0.9, 0.8, 0.7], criteria, "r", task_keys=[MILK, SAUCE, BBQ])
    entry["pinned"] = {KETCHUP: {"report": str(single), "sha256": file_sha256(single)}}
    with pytest.raises(ValueError, match="not in the sequence"):
        sr.criteria_thresholds([0.9, 0.8, 0.7], criteria, "r", task_keys=[MILK, SAUCE, BBQ])


def test_the_report_takes_a_config_path(world, tmp_path):
    path = tmp_path / "report.yaml"
    OmegaConf.save(OmegaConf.create(config(REF4)), path)
    report = sr.run_sequence_report(world["four"]["gpm"], world["four"]["seq_ft"], None, tmp_path / "out.json",
                                    strict=True, allow_dirty=True, config=path)
    assert report["reference_run_id"] == REF4 and (tmp_path / "out.json").is_file()
    with pytest.raises(ValueError, match="no thresholds pre-registered"):
        sr.run_sequence_report(world["four"]["gpm"], world["four"]["seq_ft"], None, tmp_path / "x.json")
