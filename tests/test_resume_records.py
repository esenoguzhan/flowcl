"""Resume sources for E3: a source that predates the namespace field, per-stage records that survive an
interrupted run, the chain of immediate sources, and the check-only mode (pre-E3 plan, A1)."""

from __future__ import annotations

import json
import shutil

import pytest
import torch
from omegaconf import OmegaConf

import flowcl.train.continual as continual_module
from flowcl.data.config import load_embodiment_spec
from flowcl.data.curriculum import load_curriculum
from flowcl.envs.libero_env import EvalConfig
from flowcl.train.continual import LEGACY_NAMESPACE, load_resume_source, run_continual
from flowcl.train.trainer import TrainConfig
from flowcl.utils.libero_paths import repo_root
from test_continual import BBQ, MILK, SAUCE, TINY_POLICY, fake_evaluation

KETCHUP = "libero_object/pick_up_the_ketchup_and_place_it_in_the_basket"


def tiny_cfg():
    return TrainConfig(steps=2, batch_size=2, num_workers=0, device="cpu", log_every=0, warmup_steps=1)


def curriculum(name, keys):
    return load_curriculum({"name": name, "tasks": [{"task_key": k, "n_demos": 1} for k in keys]})


def trio():
    return curriculum("test_trio", [MILK, SAUCE, BBQ])


def quad():
    return curriculum("test_quad", [MILK, SAUCE, BBQ, KETCHUP])


def run(dataset_dir, root, cur=None, **kw):
    return run_continual(cur or trio(), method_name="seq_ft", spec=load_embodiment_spec("libero_franka"),
                         policy_config=TINY_POLICY, train_cfg=tiny_cfg(), eval_cfg=EvalConfig(n_episodes=2),
                         seed=0, dataset_dir=dataset_dir, results_root=root, pretrained=False, **kw)


def load(src, K, cur=None):
    return load_resume_source(src, K, cur or trio(), "seq_ft", {}, dict(TINY_POLICY), tiny_cfg(), 0)


def state(run_dir, stage):
    return torch.load(run_dir / "checkpoints" / f"stage{stage}.pt", weights_only=False)["state_dict"]


def bitwise(a, b):
    return sorted(a) == sorted(b) and all(torch.equal(a[k], b[k]) for k in a)


@pytest.fixture(scope="module")
def source(tmp_path_factory, dataset_dir):
    """A complete three-stage seq_ft run (fake evaluations)."""
    root = tmp_path_factory.mktemp("source")
    mp = pytest.MonkeyPatch()
    mp.setattr(continual_module, "evaluate_tasks", fake_evaluation([]))
    try:
        result = run(dataset_dir, root / "a")
    finally:
        mp.undo()
    return root / "a" / result.run_id


@pytest.fixture
def fake_eval(monkeypatch):
    monkeypatch.setattr(continual_module, "evaluate_tasks", fake_evaluation([]))


def interrupt(run_dir, stage):
    """What a reboot during ``stage`` leaves: no result.json and nothing of that stage."""
    (run_dir / "result.json").unlink()
    for path in [run_dir / "checkpoints" / f"stage{stage}.pt", run_dir / "eval" / f"stage{stage}.json",
                 *run_dir.glob(f"train/stage{stage}_*"), *run_dir.glob(f"method/*task{stage}*"),
                 *run_dir.glob(f"identity_stage{stage}.json")]:
        path.unlink(missing_ok=True)


def strip_namespace(run_dir, checkpoint_stage):
    """A source written at 8132abc: no namespace field anywhere, no provenance, no stage records."""
    conf = OmegaConf.to_container(OmegaConf.load(run_dir / "config.yaml"))
    for key in ("seed_namespace_run_id", "method_run_id", "provenance"):
        conf.pop(key, None)
    OmegaConf.save(OmegaConf.create(conf), run_dir / "config.yaml")
    result = json.loads((run_dir / "result.json").read_text())
    result.pop("seed_namespace_run_id", None)
    for stage in result["stages"]:
        for key in ("seed_namespace_run_id", "method_run_id"):
            stage["evaluation"].pop(key, None)
    (run_dir / "result.json").write_text(json.dumps(result))
    for path in run_dir.glob("eval/stage*.json"):
        report = json.loads(path.read_text())
        for key in ("seed_namespace_run_id", "method_run_id"):
            report.pop(key, None)
        path.write_text(json.dumps(report))
    for path in run_dir.glob("train/stage*_record.json"):
        path.unlink()
    ckpt = run_dir / "checkpoints" / f"stage{checkpoint_stage}.pt"
    payload = torch.load(ckpt, weights_only=False)
    payload["extra"].pop("seed_namespace_run_id", None)
    torch.save(payload, ckpt)


def test_every_stage_writes_its_record_and_it_equals_result_json(source):
    saved = json.loads((source / "result.json").read_text())["stages"]
    for k, entry in enumerate(saved):
        assert json.loads((source / "train" / f"stage{k}_record.json").read_text()) == entry
        assert entry["loss_file"] and "peak_gpu" in entry


def test_a_legacy_seq_ft_source_resumes_into_an_extension_under_its_own_namespace(source, tmp_path, fake_eval,
                                                                                    dataset_dir):
    legacy = tmp_path / "legacy" / source.name
    shutil.copytree(source, legacy)
    strip_namespace(legacy, 2)
    check = run(dataset_dir, tmp_path / "c", quad(), resume_run=legacy, start_stage=3, check_resume_only=True)
    assert check["namespace"] == "test_trio__seq_ft__seed0" and check["namespace_from"] == LEGACY_NAMESPACE
    assert check["run_id"] == "test_quad__seq_ft__seed0" and not (tmp_path / "c").exists()
    result = run(dataset_dir, tmp_path / "b", quad(), resume_run=legacy, start_stage=3)
    resumed = tmp_path / "b" / result.run_id
    assert result.seed_namespace_run_id == "test_trio__seq_ft__seed0"
    record = json.loads((resumed / "resume.json").read_text())
    assert record["namespace_from"] == LEGACY_NAMESPACE and record["namespace"] == "test_trio__seq_ft__seed0"
    assert OmegaConf.load(resumed / "config.yaml").seed_namespace_run_id == "test_trio__seq_ft__seed0"


def test_a_namespace_less_source_that_is_not_the_rules_run_is_refused_before_any_directory(source, tmp_path,
                                                                                          dataset_dir):
    pilot = tmp_path / "pilot" / source.name
    shutil.copytree(source, pilot)
    strip_namespace(pilot, 2)
    conf = OmegaConf.load(pilot / "config.yaml")
    conf.run_id = "test_trio__gpm_projected_adam_pilot__seed0"   # its evaluations carry seq_ft's id
    OmegaConf.save(conf, pilot / "config.yaml")
    with pytest.raises(ValueError, match="namespace"):
        run(dataset_dir, tmp_path / "out", quad(), resume_run=pilot, start_stage=3)
    assert not (tmp_path / "out").exists()


def test_a_tampered_eval_run_id_is_refused(source, tmp_path):
    copy = tmp_path / source.name
    shutil.copytree(source, copy)
    report = json.loads((copy / "eval" / "stage0.json").read_text())
    report["run_id"] = "someone_else__seq_ft__seed0"
    (copy / "eval" / "stage0.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="evaluations"):
        load(copy, 2)


def test_an_interrupted_source_resumes_with_its_stage_records(source, tmp_path, fake_eval, dataset_dir):
    cut = tmp_path / "cut" / source.name
    shutil.copytree(source, cut)
    interrupt(cut, 2)
    assert load(cut, 2).stage_records_from == "stage record files"
    result = run(dataset_dir, tmp_path / "b", resume_run=cut, start_stage=2)
    resumed = tmp_path / "b" / result.run_id
    saved = json.loads((resumed / "result.json").read_text())["stages"]
    assert [s.get("imported_from") for s in saved] == [str(cut), str(cut), None]
    assert all(s["loss_file"] and "peak_gpu" in s for s in saved)
    for k in (0, 1):
        assert (resumed / saved[k]["loss_file"]["path"]).is_file()
        assert json.loads((resumed / "train" / f"stage{k}_record.json").read_text()) == saved[k]
    assert bitwise(state(resumed, 2), state(source, 2))


def test_a_source_with_records_missing_one_is_refused(source, tmp_path):
    cut = tmp_path / source.name
    shutil.copytree(source, cut)
    interrupt(cut, 2)
    (cut / "train" / "stage0_record.json").unlink()
    with pytest.raises(ValueError, match="stage_records"):
        load(cut, 2)


def test_an_interrupted_source_from_before_the_records_keeps_minimal_entries(source, tmp_path):
    cut = tmp_path / source.name
    shutil.copytree(source, cut)
    interrupt(cut, 2)
    for path in cut.glob("train/stage*_record.json"):
        path.unlink()
    loaded = load(cut, 2)
    assert loaded.stage_records_from.startswith("minimal entries")
    assert all("loss_file" not in d for d in loaded.stage_dicts)


def test_imported_from_names_the_immediate_source_along_a_chain(source, tmp_path, fake_eval, dataset_dir):
    """a (complete) -> b (resumed at 1, interrupted during 2) -> c (resumed from b at 2)."""
    b = run(dataset_dir, tmp_path / "b", resume_run=source, start_stage=1)
    b_dir = tmp_path / "b" / b.run_id
    interrupt(b_dir, 2)
    assert [json.loads((b_dir / "train" / f"stage{k}_record.json").read_text()).get("imported_from")
            for k in (0, 1)] == [str(source), None]
    c = run(dataset_dir, tmp_path / "c", resume_run=b_dir, start_stage=2)
    c_dir = tmp_path / "c" / c.run_id
    saved = json.loads((c_dir / "result.json").read_text())
    assert [s.get("imported_from") for s in saved["stages"]] == [str(b_dir), str(b_dir), None]
    assert saved["resume"]["stage_records_from"] == "stage record files"
    assert bitwise(state(c_dir, 2), state(source, 2))


def test_cli_check_resume_only_needs_a_source():
    import subprocess

    done = subprocess.run(["uv", "run", "python", str(repo_root() / "scripts" / "run_continual.py"),
                           "--curriculum", "seq_hetero", "--check-resume-only"],
                          capture_output=True, text=True, timeout=300, cwd=repo_root())
    assert done.returncode == 2 and "needs --resume-run" in done.stderr
