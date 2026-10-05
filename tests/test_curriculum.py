"""Curricula (§5): ordered task lists, and the reverse-order control."""

from __future__ import annotations

import pytest

from flowcl.data.curriculum import Curriculum, load_curriculum

SPATIAL = "libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate"
MILK = "libero_object/pick_up_the_milk_and_place_it_in_the_basket"
DRAWER = "libero_goal/open_the_middle_drawer_of_the_cabinet"


def test_seq_hetero_is_one_task_per_suite_in_spec_order():
    """§5: SPATIAL -> OBJECT -> GOAL -> LONG."""
    curriculum = load_curriculum("seq_hetero")
    assert [k.split("/")[0] for k in curriculum.task_keys] == [
        "libero_spatial",
        "libero_object",
        "libero_goal",
        "libero_10",
    ]
    assert len(curriculum) == 4
    assert all(stage.n_demos == 50 for stage in curriculum.stages)


def test_first_task_key_is_where_stats_come_from():
    """§3.3 fits normalization stats on the first task only."""
    curriculum = load_curriculum("seq_hetero")
    assert curriculum.first_task_key == curriculum.task_keys[0]


def test_reversed_flips_the_order_and_the_stats_task():
    """§5 runs seq_hetero in reverse; that changes which task §3.3 fits stats on."""
    forward = load_curriculum("seq_hetero")
    backward = forward.reversed()

    assert backward.task_keys == tuple(reversed(forward.task_keys))
    assert backward.first_task_key == forward.task_keys[-1]
    assert backward.first_task_key != forward.first_task_key
    assert backward.name == "seq_hetero_reverse"


def test_reversed_twice_is_the_original_order():
    forward = load_curriculum("seq_hetero")
    assert forward.reversed().reversed().task_keys == forward.task_keys


def test_load_from_dict_accepts_bare_keys_and_mappings():
    curriculum = load_curriculum(
        {"name": "mixed", "tasks": [SPATIAL, {"task_key": MILK, "n_demos": 10}]},
        default_n_demos=50,
    )
    assert curriculum.task_keys == (SPATIAL, MILK)
    assert [s.n_demos for s in curriculum.stages] == [50, 10]


def test_load_rejects_an_empty_task_list():
    with pytest.raises(ValueError, match="declares no tasks"):
        load_curriculum({"name": "empty", "tasks": []})


def test_load_rejects_a_repeated_task():
    with pytest.raises(ValueError, match="repeats tasks"):
        load_curriculum({"name": "dup", "tasks": [MILK, MILK]})


def test_load_rejects_a_missing_config():
    with pytest.raises(FileNotFoundError, match="Curriculum config not found"):
        load_curriculum("no_such_curriculum")


def test_empty_curriculum_is_rejected_at_construction():
    with pytest.raises(ValueError, match="has no stages"):
        Curriculum(name="x", stages=())


def test_reverse_yaml_equals_the_derived_reverse():
    """configs/curriculum/seq_hetero_reverse.yaml says its task list is asserted here."""
    yaml_cur = load_curriculum("seq_hetero_reverse")
    derived = load_curriculum("seq_hetero").reversed()
    assert yaml_cur.name == derived.name == "seq_hetero_reverse"
    assert [(s.task_key, s.n_demos) for s in yaml_cur.stages] == [(s.task_key, s.n_demos) for s in derived.stages]


# ---- joint (co-trained) stages ---------------------------------------------------------------

PLATE = "libero_goal/put_the_bowl_on_the_plate"
CABINET = "libero_goal/put_the_bowl_on_top_of_the_cabinet"
STOVE = "libero_goal/put_the_bowl_on_the_stove"


def test_joint_stage_parses_and_orders_its_tasks():
    cur = load_curriculum("langbase_cabinet_stove")
    assert cur.task_keys == (PLATE, STOVE)                     # retention-matrix columns
    assert [r.task_key for r in cur.stages[0].train_refs] == [PLATE, CABINET]
    assert cur.stages[0].steps_factor == 2 and cur.stages[1].steps_factor == 1
    assert [r.task_key for r in cur.eval_refs] == [PLATE, STOVE, CABINET]
    assert cur.first_task_key == PLATE                          # stats on A only
    assert cur.stages[0].record() == {"task_key": PLATE, "n_demos": 50, "co_train": [CABINET], "steps_factor": 2}
    assert cur.stages[1].record() == {"task_key": STOVE, "n_demos": 50}


def test_a_plain_curriculum_records_only_its_task_and_demos():
    cur = load_curriculum("seq_hetero")
    assert all(set(s.record()) == {"task_key", "n_demos"} for s in cur.stages)
    assert cur.eval_refs == cur.refs
    assert all(not s.is_joint and s.train_refs == (s.ref,) for s in cur.stages)


@pytest.mark.parametrize("tasks, match", [
    ([{"task_key": PLATE, "co_train": [STOVE]}, STOVE], "repeats tasks"),
    ([{"task_key": PLATE, "co_train": [CABINET, CABINET]}], "repeats tasks"),
    ([{"task_key": PLATE, "co_train": [PLATE]}], "repeats tasks"),
    ([{"task_key": PLATE, "steps_factor": 0}], "steps_factor"),
    ([{"task_key": PLATE, "steps_factor": "2"}], "steps_factor"),
    ([{"task_key": PLATE, "steps_factor": True}], "steps_factor"),
    ([{"task_key": PLATE, "steps_factor": 1.5}], "steps_factor"),
    ([{"task_key": PLATE, "cotrain": [CABINET]}], "unknown keys"),
])
def test_joint_stage_rejects_bad_declarations(tasks, match):
    with pytest.raises(ValueError, match=match):
        load_curriculum({"name": "bad", "tasks": tasks})


def test_a_joint_curriculum_has_no_reverse_order():
    with pytest.raises(ValueError, match="reverse order is not defined"):
        load_curriculum("langbase_cabinet_stove").reversed()
