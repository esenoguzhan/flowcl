"""Track 2's manifest (flowcl/experiments/behavior_predictors.py) on a fake results tree: every source
layout, the exclusions and their reconciliation, deduplication by tensor content with aliases and
flags kept, both branch definitions on a shared trunk, a trajectory as one root, the reserved test set,
per-cell file hashes, and atomic publication."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from flowcl.experiments import behavior_predictors as bp
from flowcl.utils.run import file_sha256

A = "libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate"
B = "libero_object/pick_up_the_milk_and_place_it_in_the_basket"
C = "libero_goal/open_the_middle_drawer_of_the_cabinet"
CURRICULUM = {"name": "seq_hetero", "tasks": [{"task_key": A, "n_demos": 1}, {"task_key": B, "n_demos": 1},
                                              {"task_key": C, "n_demos": 1}]}


def ckpt(path: Path, value: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {"w": torch.full((3,), float(value))}, "extra": {"any": value}}, path)
    return path


def evaluation(path: Path, tasks: dict, n=50, offset=0) -> Path:
    """``tasks``: task -> number of successes (first k episodes succeed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for j, (task, k) in enumerate(tasks.items()):
        succ = [i < k for i in range(n)]
        out.append({"task_key": task, "successes": succ, "n_steps": [100] * n,
                    "seeds": [1000 * j + i + offset for i in range(n)], "success_rate": k / n,
                    "ci_low": 0.0, "ci_high": 1.0, "confidence": 0.95, "n_rollouts": n, "wall_clock_s": 1.0})
    path.write_text(json.dumps({"run_id": "x", "stage": 0, "tasks": out}))
    return path


def config(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(payload), path)
    return path


def runner(root, name, stages, method="seq_ft", embodiment="libero_franka"):
    """``stages``: list of (checkpoint value or None for 'no checkpoint', {task: successes})."""
    d = root / name
    config(d / "config.yaml", {"curriculum": CURRICULUM, "method": {"name": method},
                               "embodiment": {"name": embodiment}})
    for k, (value, tasks) in enumerate(stages):
        if value is not None:
            ckpt(d / "checkpoints" / f"stage{k}.pt", value)
        evaluation(d / "eval" / f"stage{k}.json", tasks)
    return d


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "results"
    seq = runner(root, "seq_hetero__seq_ft__seed0", [(0.0, {A: 40, B: 0}), (1.0, {A: 0, B: 39, C: 0})])
    runner(root, "seq_hetero__gpm__seed0", [(0.0, {A: 40, B: 0}), (2.0, {A: 45, B: 38}), (3.0, {A: 44, B: 30, C: 20})],
           method="gpm")
    runner(root, "seq_hetero__adaptive__seed0", [(0.0, {A: 40}), (2.0, {A: 45, B: 38}), (4.0, {A: 46, B: 37, C: 21})],
           method="adaptive")
    # a resumed reproduction of gpm's stage 2 (one episode differs: a disagreeing duplicate)
    res = runner(root / "resume_check", "seq_hetero__gpm__seed0",
                 [(None, {A: 40, B: 0}), (None, {A: 45, B: 38}), (3.0, {A: 44, B: 31, C: 20})], method="gpm")
    (res / "resume.json").write_text(json.dumps({"start_stage": 2, "imported_stages": [0, 1],
                                                 "checkpoint": "results/seq_hetero__gpm__seed0/checkpoints/stage1.pt"}))
    # a bracket row and a legacy hparam row from seq_ft's stage 0
    row = root / "baseline_selection_lora" / "lora" / "seed0" / "lorab__r4__seed0"
    ckpt(row / "checkpoints" / "final.pt", 5.0)
    evaluation(row / "eval.json", {A: 24, B: 36})
    (row / "row.json").write_text(json.dumps({"run_id": row.name, "stage": 1, "task_key": B,
                                              "checkpoint": "checkpoints/final.pt", "method": {"family": "lora"},
                                              "parent": {"path": str(seq / "checkpoints" / "stage0.pt")}}))
    leg = root / "hparam_study" / "cl_lr" / "cl__b__seed0"
    ckpt(leg / "checkpoints" / "final.pt", 6.0)
    evaluation(leg / "eval.json", {A: 10, B: 40})
    (leg / "row.json").write_text(json.dumps({"run_id": leg.name}))
    config(leg / "config.yaml", {"reference_run": "seq_hetero__seq_ft__seed0", "start_checkpoint": "stage0.pt",
                                 "task_key": B})
    # a trajectory: one root
    traj = root / "hparam_study" / "trajectory" / "traj__b__seed0"
    config(traj / "config.yaml", {"part": "trajectory", "task_key": B})
    for step, value, k in ((1000, 7.0, 5), (2000, 8.0, 30)):
        ckpt(traj / "checkpoints" / f"step{step}.pt", value)
        evaluation(traj / "eval" / f"step{step}.json", {B: k})
    # a single-task policy
    single = root / f"single__{C.replace('/', '__')}__seed0"
    config(single / "config.yaml", {"tasks": [C], "embodiment": {"name": "libero_franka"}})
    ckpt(single / "checkpoints" / "final.pt", 9.0)
    evaluation(single / "eval.json", {C: 48})
    # exclusions
    runner(root, "seq_hetero__gpm__seed0_crashed_20260923", [(2.5, {A: 1})], method="gpm")
    runner(root, "dobot_puzzle__seq_ft__seed0", [(11.0, {"dobot/puzzle": 3})], embodiment="dobot")
    evaluation(root / "high_protection_followup" / "seed0" / "e99.rep1.eval.json", {A: 40})
    short = runner(root, "seq_hetero__short__seed0", [(12.0, {A: 3})])
    evaluation(short / "eval" / "stage0.json", {A: 3}, n=10)
    return root


def build(root, **kw):
    return bp.build_manifest(bp.load_config(), results_root=root, **kw)


def test_discovery_layouts_exclusions_and_reconciliation(tree):
    cfg = bp.load_config()
    sources, excluded, files = bp.discover(tree, cfg)
    kinds = sorted(s.kind for s in sources)
    assert kinds.count("runner") == 10 and kinds.count("row") == 1 and kinds.count("legacy_row") == 1
    assert kinds.count("trajectory") == 2 and kinds.count("single") == 1
    reasons = sorted(e["reason"] for e in excluded)
    assert reasons == ["embodiment", "invalid_dir", "repeated_states", "resume_import_copy", "resume_import_copy"]
    assert len(sources) + len(excluded) == len(files)
    out = build(tree, dry_run=True)
    s = out["summary"]
    assert s["reconciliation"] == {"evaluation_files": len(files), "used": len(sources), "excluded_files": len(excluded)}
    assert s["excluded_by_reason"]["unseen_task"] == 3          # B at stage 0 (seq_ft, gpm), C at seq_ft stage 1
    assert s["excluded_by_reason"]["n_rollouts"] == 1
    assert not (tree / "track2" / "manifest").exists()          # a dry run publishes nothing
    assert (tree / "track2_cache" / "fingerprints.json").is_file()


def test_dedup_aliases_lineage_and_both_branches(tree):
    out = build(tree)
    m = out["manifest"]

    class Cells:
        def __getitem__(self, key):
            run, stage, task = key
            (cell,) = [c for c in m["cells"] if c["task_key"] == task
                       and any(a["run"] == run and a["stage"] == stage for a in c["aliases"])]
            return cell

        def __contains__(self, key):
            try:
                self[key]
                return True
            except ValueError:
                return False

    cells = Cells()
    # gpm and adaptive share stages 0-1 (same tensors): one cell each, both runs as aliases
    shared = cells[("seq_hetero__gpm__seed0", 1, A)]
    assert shared is cells[("seq_hetero__adaptive__seed0", 1, A)]
    assert sorted(a["run"] for a in shared["aliases"]) == ["seq_hetero__adaptive__seed0", "seq_hetero__gpm__seed0"]
    # stage 0's A cell: seq_ft, gpm and adaptive share the root checkpoint
    root_cell = cells[("seq_hetero__seq_ft__seed0", 0, A)]
    assert len(root_cell["aliases"]) == 3 and root_cell["depth"] == 0
    assert root_cell["branch_depth1"] == root_cell["root"] == root_cell["fingerprint"]
    # the resumed reproduction is an alias of gpm's stage 2, and one episode disagrees
    g2 = cells[("seq_hetero__gpm__seed0", 2, B)]
    assert any(a["run"] == "resume_check/seq_hetero__gpm__seed0" for a in g2["aliases"]) and g2["disagreeing_duplicate"]
    assert m["duplicates"]["disagreeing"] and out["summary"]["disagreeing_duplicates"] == 1
    # branches: depth 1 groups gpm's and adaptive's later stages under the shared stage 1;
    # the literal first divergence separates them and keeps the shared stage 1 its own branch
    a2 = cells[("seq_hetero__adaptive__seed0", 2, C)]
    g1 = shared["fingerprint"]
    assert g2["branch_depth1"] == a2["branch_depth1"] == g1
    assert g2["branch_first_divergence"] == g2["fingerprint"] and a2["branch_first_divergence"] == a2["fingerprint"]
    assert shared["branch_first_divergence"] == g1
    s1 = cells[("seq_hetero__seq_ft__seed0", 1, B)]
    assert s1["branch_depth1"] == s1["branch_first_divergence"] == s1["fingerprint"]
    # rows: parent's trained tasks plus their own; bracket flag kept
    lora = cells[("baseline_selection_lora/lora/seed0/lorab__r4__seed0", 1, A)]
    assert lora["bracket"] and lora["root"] == root_cell["root"] and lora["depth"] == 1
    assert ("hparam_study/cl_lr/cl__b__seed0", None, B) in cells
    # the trajectory is one root and one branch
    t1, t2 = (cells[("hparam_study/trajectory/traj__b__seed0", s, B)] for s in (1000, 2000))
    assert t1["root"] == t2["root"] == "trajectory:hparam_study/trajectory/traj__b__seed0"
    assert t1["root_kind"] == "trajectory" and t1["branch_depth1"] == t2["branch_depth1"] == t1["fingerprint"]
    single = cells[(f"single__{C.replace('/', '__')}__seed0", None, C)]
    assert single["root"] == single["fingerprint"] and single["depth"] == 0
    # binding: every file read, hashed; lineage nodes with paths, SHAs and fingerprints
    alias = next(a for a in g2["aliases"] if a["run"] == "seq_hetero__gpm__seed0")
    for rel in ("seq_hetero__gpm__seed0/config.yaml", "seq_hetero__gpm__seed0/eval/stage2.json",
                "seq_hetero__gpm__seed0/checkpoints/stage2.pt"):
        assert alias["files"][rel] == file_sha256(tree / rel)
    resumed = next(a for a in g2["aliases"] if a["run"].startswith("resume_check"))
    assert "resume_check/seq_hetero__gpm__seed0/resume.json" in resumed["files"]
    assert [n["fingerprint"] for n in alias["lineage"]][-1] == g2["fingerprint"] and len(alias["lineage"]) == 3
    assert [g2["fingerprint"], B] in [list(k) for k in m["development_keys"]]
    assert all(c["n"] == 50 for c in m["cells"])


def test_reserved_test_runs_refuse_and_publication_is_atomic_and_immutable(tree):
    first = build(tree)
    again = build(tree)
    assert again["manifest_sha256"] == first["manifest_sha256"]
    out = tree / "track2" / "manifest"
    assert sorted(p.name for p in out.iterdir()) == ["done.json", "manifest.json", "summary.json"]
    assert not list((tree / "track2").glob(".manifest_attempt_*"))
    payload = json.loads((out / "manifest.json").read_text())
    payload["cells"] = payload["cells"][1:]
    (out / "manifest.json").write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="done-marker"):
        bp.load_manifest(out)
    (tree / "seq_hetero__lora__seed0").mkdir()
    with pytest.raises(RuntimeError, match="reserved test runs"):
        build(tree, dry_run=True)


def test_first_divergence_on_merged_chains():
    runs = {("r", "a", "b1"), ("r", "a", "b2"), ("r", "c")}
    assert bp.first_divergence(("r",), runs) == "r"
    assert bp.first_divergence(("r", "a"), runs) == "a"            # shared by two runs: the cell itself
    assert bp.first_divergence(("r", "a", "b1"), runs) == "b1"
    assert bp.first_divergence(("r", "c"), runs) == "c"
    merged = runs | {("r", "a", "b1")}                              # an identical chain adds nothing
    assert bp.first_divergence(("r", "a", "b1"), merged) == "b1"
    cfg = copy.deepcopy(bp.load_config())
    assert cfg["manifest"]["n_rollouts"] == 50 and cfg["manifest"]["precedence"][0] == "runner"
