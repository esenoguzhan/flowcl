"""Spec §2: a run without config/git_sha/freeze/seed is invalid."""

from __future__ import annotations

import json

import pytest
from omegaconf import OmegaConf

from flowcl.utils.run import (
    REQUIRED_ARTIFACTS,
    create_run,
    dependency_freeze,
    git_sha,
    validate_run,
)


def test_create_run_writes_every_required_artifact(tmp_path):
    cfg = OmegaConf.create({"policy": {"d_model": 512}, "seed": 3})
    handle = create_run("r0", cfg, seed=3, results_root=tmp_path)

    for name in REQUIRED_ARTIFACTS:
        artifact = handle.path / name
        assert artifact.is_file(), f"{name} not written"
        assert artifact.stat().st_size > 0, f"{name} is empty"

    assert OmegaConf.load(handle.path / "config.yaml").policy.d_model == 512
    assert json.loads((handle.path / "seed.json").read_text())["seed"] == 3


def test_validate_run_raises_on_missing_artifact(tmp_path):
    handle = create_run("r1", {"a": 1}, seed=0, results_root=tmp_path)
    (handle.path / "git_sha").unlink()

    with pytest.raises(FileNotFoundError, match="git_sha"):
        validate_run(handle.path)


def test_validate_run_raises_on_empty_artifact(tmp_path):
    handle = create_run("r2", {"a": 1}, seed=0, results_root=tmp_path)
    (handle.path / "requirements.txt").write_text("")

    with pytest.raises(ValueError, match="empty artifacts"):
        validate_run(handle.path)


def test_validate_run_raises_on_missing_directory(tmp_path):
    with pytest.raises(NotADirectoryError):
        validate_run(tmp_path / "nope")


def test_refuses_to_overwrite_existing_run(tmp_path):
    create_run("r3", {"a": 1}, seed=0, results_root=tmp_path)
    with pytest.raises(FileExistsError, match="already exists"):
        create_run("r3", {"a": 2}, seed=1, results_root=tmp_path)


def test_exist_ok_allows_reuse(tmp_path):
    create_run("r4", {"a": 1}, seed=0, results_root=tmp_path)
    handle = create_run("r4", {"a": 2}, seed=1, results_root=tmp_path, exist_ok=True)
    assert OmegaConf.load(handle.path / "config.yaml").a == 2


def test_records_both_flowcl_and_libero_shas(tmp_path):
    """The LIBERO SHA is load-bearing: it defines the evaluation initial states."""
    handle = create_run("r5", {"a": 1}, seed=0, results_root=tmp_path)
    flowcl_sha = (handle.path / "git_sha").read_text().strip()
    libero_sha = (handle.path / "libero_submodule_sha").read_text().strip()

    assert len(flowcl_sha.removesuffix("-dirty")) == 40
    assert len(libero_sha.removesuffix("-dirty")) == 40
    assert flowcl_sha != libero_sha


def test_git_sha_marks_dirty_tree():
    """Whatever the tree state, the marker must be consistent with git status."""
    import subprocess

    from flowcl.utils.libero_paths import repo_root

    porcelain = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root(),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert git_sha().endswith("-dirty") == bool(porcelain)


def test_dependency_freeze_lists_core_packages():
    freeze = dependency_freeze()
    assert "torch==" in freeze
    assert "numpy==" in freeze
