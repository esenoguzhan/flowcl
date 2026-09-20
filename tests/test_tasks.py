"""Task identity: the HDF5 file, the BDDL env and the retention-matrix key agree."""

from __future__ import annotations

import pytest

from flowcl.data.tasks import TaskRef, resolve_tasks, suite_task_names

MILK = "libero_object/pick_up_the_milk_and_place_it_in_the_basket"


def test_suite_task_names_are_ten_per_suite():
    """§3.2: 10 tasks each in the four suites the thesis uses."""
    for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
        assert len(suite_task_names(suite)) == 10


def test_from_key_round_trips():
    ref = TaskRef.from_key(MILK)
    assert ref.task_key == MILK
    assert ref.suite == "libero_object"
    assert TaskRef.from_suite_index(ref.suite, ref.task_idx) == ref


def test_from_key_rejects_unknown_task_and_lists_candidates():
    with pytest.raises(ValueError, match="has no task named"):
        TaskRef.from_key("libero_object/pick_up_the_nonexistent_thing")


def test_from_key_rejects_unknown_suite():
    with pytest.raises(ValueError, match="Unknown LIBERO suite"):
        TaskRef.from_key("libero_imaginary/whatever")


def test_from_key_rejects_malformed_key():
    with pytest.raises(ValueError, match="must be"):
        TaskRef.from_key("no_slash_here")


def test_from_suite_index_rejects_out_of_range():
    with pytest.raises(IndexError, match="out of range"):
        TaskRef.from_suite_index("libero_object", 99)


def test_language_is_the_bddl_instruction():
    ref = TaskRef.from_key(MILK)
    assert "milk" in ref.language.lower()


def test_resolve_tasks_rejects_duplicates():
    """A repeated task would make two retention-matrix rows claim the same key."""
    with pytest.raises(ValueError, match="repeats tasks"):
        resolve_tasks([MILK, MILK])


def test_demo_path_matches_the_task_key(dataset_dir):
    from flowcl.data.libero_adapter import default_task_id

    ref = TaskRef.from_key(MILK)
    path = ref.demo_path(dataset_dir)
    assert path.is_file()
    assert default_task_id(path) == ref.task_key


def test_assert_consistent_passes_for_every_curriculum_task(dataset_dir):
    """The check that matters: demo instruction == BDDL instruction.

    A mismatch would train on one instruction and evaluate on another, which no loss
    curve would reveal. Run over the whole seq_hetero curriculum, since that is the
    set the thesis actually uses.
    """
    from flowcl.data.curriculum import load_curriculum

    load_curriculum("seq_hetero").assert_consistent(dataset_dir)


def test_assert_consistent_detects_a_key_mismatch(dataset_dir, monkeypatch):
    """A TaskRef pointing at the wrong file must not pass silently."""
    import flowcl.data.tasks as tasks_module

    ref = TaskRef.from_key(MILK)
    wrong_path = TaskRef.from_key(
        "libero_object/pick_up_the_tomato_sauce_and_place_it_in_the_basket"
    ).demo_path(dataset_dir)

    monkeypatch.setattr(
        tasks_module.TaskRef,
        "demo_path",
        lambda self, dataset_dir=None: wrong_path,
    )
    with pytest.raises(ValueError, match="task key mismatch"):
        ref.assert_consistent(dataset_dir)


def test_load_task_episodes_respects_n_demos(dataset_dir):
    from flowcl.data.config import load_embodiment_spec
    from flowcl.data.tasks import load_task_episodes

    spec = load_embodiment_spec("libero_franka")
    episodes = load_task_episodes(
        TaskRef.from_key(MILK), spec, n_demos=2, dataset_dir=dataset_dir, verify=False
    )
    assert len(episodes) == 2
    assert {ep.task_id for ep in episodes} == {MILK}
