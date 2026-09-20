"""Dataset acquisition bookkeeping (§3.2)."""

from __future__ import annotations

import pytest

from flowcl.data import libero_setup


def test_required_suites_match_spec():
    """§3.2 names exactly these four suites; §5 builds both curricula from them."""
    assert libero_setup.REQUIRED_SUITES == (
        "libero_spatial",
        "libero_object",
        "libero_goal",
        "libero_10",
    )


def test_libero_90_is_known_but_not_required():
    """libero_90 exists upstream but no curriculum uses it, and it is ~64 GB."""
    assert "libero_90" in libero_setup.SUITE_TASK_COUNTS
    assert "libero_90" not in libero_setup.REQUIRED_SUITES


def test_no_libero_100_pseudo_suite():
    """Guards against reintroducing upstream's bug.

    ``libero.libero.utils.download_utils.download_from_huggingface`` passes
    ``allow_patterns="libero_100/*"``, but the HuggingFace repo has no such
    directory — it stores ``libero_10/`` and ``libero_90/`` separately. That call
    matches zero files and downloads nothing while appearing to succeed.
    """
    assert "libero_100" not in libero_setup.SUITE_TASK_COUNTS


def test_task_counts():
    assert libero_setup.SUITE_TASK_COUNTS["libero_spatial"] == 10
    assert libero_setup.SUITE_TASK_COUNTS["libero_90"] == 90
    assert libero_setup.DEMOS_PER_TASK == 50


def test_check_reports_missing_suite_without_raising(tmp_path):
    statuses = libero_setup.check(dataset_dir=tmp_path)
    assert len(statuses) == len(libero_setup.REQUIRED_SUITES)
    assert all(not s.complete for s in statuses)
    assert all(s.n_files == 0 for s in statuses)
    assert "BAD" in statuses[0].describe()


def test_assert_complete_raises_with_full_report(tmp_path):
    with pytest.raises(FileNotFoundError, match="LIBERO datasets incomplete"):
        libero_setup.assert_complete(dataset_dir=tmp_path)


def test_check_rejects_unknown_suite(tmp_path):
    with pytest.raises(ValueError, match="Unknown suite"):
        libero_setup.check(dataset_dir=tmp_path, suites=("libero_nope",))


def test_download_rejects_unknown_suite(tmp_path):
    with pytest.raises(ValueError, match="Unknown suites"):
        libero_setup.download(suites=["libero_100"], dataset_dir=tmp_path)


def test_complete_suite_detected(tmp_path):
    suite_dir = tmp_path / "libero_goal"
    suite_dir.mkdir()
    for i in range(10):
        (suite_dir / f"task{i}_demo.hdf5").write_bytes(b"x")

    status = libero_setup.check(dataset_dir=tmp_path, suites=("libero_goal",))[0]
    assert status.complete
    assert "OK" in status.describe()
