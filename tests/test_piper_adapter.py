"""AgileX Piper adapter: LeRobot v3.0 episode/frame mapping, exclusions, cache checks."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from flowcl.data.config import load_embodiment_spec
from flowcl.data.piper_adapter import (
    CAMERA_KEYS,
    build_cache,
    iter_cache_episodes,
    read_source,
    source_fingerprint,
    verify_cache,
)
from flowcl.data.piper_tasks import PiperSource, PiperTask
from flowcl.data.tasks import TaskRef, load_task_episodes, suite_task_names

# Episodes 0 and 1 share file 0; episode 2 is alone in file 1 (LeRobot v3 layout).
LENGTHS = [4, 3, 5]
FILES = [0, 0, 1]
LEVEL_STEP = 20  # frame of global row i is a uniform gray of i * LEVEL_STEP


@pytest.fixture(scope="module")
def spec():
    return load_embodiment_spec("agilex_piper")


def _write_video(path, levels):
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15, (640, 480))
    for level in levels:
        writer.write(np.full((480, 640, 3), level, np.uint8))
    writer.release()


def write_dataset(root, lengths=LENGTHS, files=FILES, drop_last_frame=False):
    """A tiny LeRobot v3.0 folder: state = row index, action = row index + 0.5."""
    info = {
        "codebase_version": "v3.0", "fps": 15, "total_episodes": len(lengths),
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "observation.state": {"shape": [7]}, "action": {"shape": [7]},
            **{key: {"shape": [480, 640, 3]} for key in CAMERA_KEYS.values()},
        },
    }
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps(info))
    pq.write_table(pa.table({"task_index": [0], "__index_level_0__": ["pick the toy"]}),
                   root / "meta" / "tasks.parquet")
    starts = np.cumsum([0, *lengths])
    rows, metas, offsets = [], [], {}
    for e, (n, file) in enumerate(zip(lengths, files)):
        first = offsets.get(file, 0)
        offsets[file] = first + n
        meta = {"episode_index": e, "length": n, "dataset_from_index": int(starts[e]),
                "dataset_to_index": int(starts[e + 1])}
        for key in CAMERA_KEYS.values():
            meta.update({f"videos/{key}/chunk_index": 0, f"videos/{key}/file_index": file,
                         f"videos/{key}/from_timestamp": first / 15.0,
                         f"videos/{key}/to_timestamp": (first + n) / 15.0})
        metas.append(meta)
        for t in range(n):
            i = int(starts[e]) + t
            rows.append({"observation.state": [float(i)] * 7, "action": [i + 0.5] * 7,
                         "timestamp": t / 15.0, "frame_index": t, "episode_index": e,
                         "index": i, "task_index": 0, "file": file})
    pq.write_table(pa.Table.from_pylist(metas), root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    for file in sorted(set(files)):
        part = [r for r in rows if r["file"] == file]
        path = root / "data" / "chunk-000" / f"file-{file:03d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist([{k: v for k, v in r.items() if k != "file"} for r in part]), path)
        levels = [r["index"] * LEVEL_STEP for r in part]
        if drop_last_frame and file == max(files):
            levels = levels[:-1]
        for key in CAMERA_KEYS.values():
            _write_video(root / "videos" / key / "chunk-000" / f"file-{file:03d}.mp4", levels)
    return root


def toy_task(raw_root, exclude=(), holdout=1, language="put the toy away"):
    folder = raw_root / "toy"
    n = len(LENGTHS) - len(exclude)
    return PiperTask(name="toy", language=language,
                     sources=(PiperSource("toy", source_fingerprint(folder), tuple(exclude)),),
                     n_episodes=n, holdout=holdout)


def level(frame):
    """The gray level of a policy frame, read away from the masked corner."""
    return int(round(float(frame[-20:, -20:].mean()) / LEVEL_STEP))


def test_frames_follow_their_episode_across_shared_files(tmp_path, spec):
    write_dataset(tmp_path / "toy")
    episodes = list(read_source(tmp_path / "toy", spec))
    assert [e["raw_episode_index"] for e in episodes] == [0, 1, 2]
    for ep, n, start in zip(episodes, LENGTHS, np.cumsum([0, *LENGTHS])):
        rows = np.arange(start, start + n)
        np.testing.assert_array_equal(ep["state"][:, 0], rows)
        np.testing.assert_array_equal(ep["action"][:, 0], rows + 0.5)
        for cam in spec.cameras:
            assert ep["images"][cam].shape == (n, 128, 128, 3)
            assert [level(f) for f in ep["images"][cam]] == rows.tolist()
        # the front mask blacks out the top-left corner; the wrist camera has none
        assert ep["images"]["front"][-1, 0, 0].max() == 0
        assert ep["images"]["wrist"][-1, 0, 0].max() > 0


def test_exclusions_skip_episodes(tmp_path, spec):
    write_dataset(tmp_path / "toy")
    kept = [e["raw_episode_index"] for e in read_source(tmp_path / "toy", spec, exclude=(1,))]
    assert kept == [0, 2]
    with pytest.raises(ValueError, match="do not exist"):
        list(read_source(tmp_path / "toy", spec, exclude=(7,)))


def test_video_short_of_rows_fails(tmp_path, spec):
    write_dataset(tmp_path / "toy", drop_last_frame=True)
    with pytest.raises(ValueError, match="video frames"):
        list(read_source(tmp_path / "toy", spec))


def test_cache_round_trip_and_checks(tmp_path, spec):
    write_dataset(tmp_path / "toy")
    task = toy_task(tmp_path, exclude=(0,))
    cache = tmp_path / "cache" / "toy.h5"
    summary = build_cache(task, tmp_path, cache, spec)
    assert summary["n_episodes"] == 2 and summary["frames"] == 8
    assert summary["raw_labels"] == {"toy": ["pick the toy"]}
    verify_cache(cache, task, n_demos=1)
    with pytest.raises(ValueError, match="held out"):
        verify_cache(cache, task, n_demos=2)
    with pytest.raises(ValueError, match="language"):
        verify_cache(cache, dataclasses.replace(task, language="other"))
    episodes = list(iter_cache_episodes(cache, spec))
    assert [e.meta["raw_episode_index"] for e in episodes] == [1, 2]
    assert episodes[0].language == "put the toy away" and episodes[0].task_id == "piper/toy"
    assert [level(f) for f in episodes[1].images["wrist"]] == list(range(7, 12))
    # a spec whose mask differs from the cache's must not read it
    obs = dataclasses.replace(spec.observation, image_masks=(("front", (((0, 0), (10, 0), (0, 10)),)),))
    with pytest.raises(ValueError, match="masks"):
        list(iter_cache_episodes(cache, dataclasses.replace(spec, observation=obs)))


def test_build_refuses_a_changed_folder(tmp_path, spec):
    write_dataset(tmp_path / "toy")
    task = toy_task(tmp_path)
    (tmp_path / "toy" / "meta" / "stats.json").write_text("{}")  # the folder changed
    with pytest.raises(ValueError, match="fingerprint"):
        build_cache(task, tmp_path, tmp_path / "cache" / "toy.h5", spec)


def test_taskref_dispatches_the_piper_suite(tmp_path, spec, monkeypatch):
    assert "sausage_pot_lid" in suite_task_names("piper")
    ref = TaskRef.from_key("piper/sausage_pot_lid")
    assert ref.language == "put the sausage in the pot and cover it with the lid"
    with pytest.raises(FileNotFoundError, match="prepare_piper"):
        ref.demo_path(tmp_path)

    import flowcl.data.piper_tasks as piper_tasks

    write_dataset(tmp_path / "raw" / "toy")
    task = toy_task(tmp_path / "raw", holdout=1)
    monkeypatch.setitem(piper_tasks.TASKS, "toy", task)
    build_cache(task, tmp_path / "raw", piper_tasks.cache_path("toy", tmp_path), spec)
    toy = TaskRef.from_key("piper/toy")
    toy.assert_consistent(tmp_path)
    # n_demos=None means the training demos only: the held-out episode is never loaded
    episodes = load_task_episodes(toy, spec, dataset_dir=tmp_path)
    assert [e.meta["raw_episode_index"] for e in episodes] == [0, 1]
    with pytest.raises(ValueError, match="held out"):
        load_task_episodes(toy, spec, n_demos=3, dataset_dir=tmp_path)


def test_a_task_with_nothing_held_out_trains_on_every_episode(tmp_path, spec, monkeypatch):
    import flowcl.data.piper_tasks as piper_tasks
    from flowcl.deploy.piper_server import training_summary

    root = write_dataset(tmp_path / "raw" / "toy")
    # the operator retyped the label mid-session: two recorded tasks, listed out of order
    pq.write_table(pa.table({"task_index": [1, 0], "__index_level_0__": ["pick up the toy", "pick the toy"]}),
                   root / "meta" / "tasks.parquet")
    task = toy_task(tmp_path / "raw", holdout=0)
    monkeypatch.setitem(piper_tasks.TASKS, "toy", task)
    summary = build_cache(task, tmp_path / "raw", piper_tasks.cache_path("toy", tmp_path), spec)
    assert summary["raw_labels"] == {"toy": ["pick the toy", "pick up the toy"]}
    episodes = load_task_episodes(TaskRef.from_key("piper/toy"), spec, dataset_dir=tmp_path)
    assert [e.meta["raw_episode_index"] for e in episodes] == [0, 1, 2]
    assert all(e.language == "put the toy away" for e in episodes)
    meta = training_summary(task, dataset_dir=tmp_path)
    assert meta["n_demos"] == 3 and meta["held_out"] == []


def test_every_registered_task_names_a_loadable_embodiment():
    import flowcl.data.piper_tasks as piper_tasks

    for task in piper_tasks.TASKS.values():
        assert load_embodiment_spec(task.embodiment).d_action == 7
    # the mask ablation: the same recordings and instruction, only the camera treatment differs
    masked, unmasked = piper_tasks.get_task("milk_plate"), piper_tasks.get_task("milk_plate_unmasked")
    assert dataclasses.replace(unmasked, name=masked.name, notes=masked.notes, embodiment=masked.embodiment) == masked
    assert load_embodiment_spec(masked.embodiment).observation.masks_for("front")
    assert not load_embodiment_spec(unmasked.embodiment).observation.image_masks


def test_an_unmasked_cache_keeps_the_corner_and_loads_only_unmasked(tmp_path, spec):
    unmasked = load_embodiment_spec("agilex_piper_unmasked")
    write_dataset(tmp_path / "toy")
    task = toy_task(tmp_path, holdout=0)
    build_cache(task, tmp_path, tmp_path / "masked.h5", spec)
    build_cache(task, tmp_path, tmp_path / "unmasked.h5", unmasked)
    corner = (slice(0, 10), slice(0, 10))
    masked_frame = next(iter_cache_episodes(tmp_path / "masked.h5", spec)).images["front"][2]
    plain_frame = next(iter_cache_episodes(tmp_path / "unmasked.h5", unmasked)).images["front"][2]
    assert masked_frame[corner].max() == 0 and level(plain_frame[corner]) == 2
    # a run of the wrong embodiment must not read either cache
    with pytest.raises(ValueError, match="masks"):
        list(iter_cache_episodes(tmp_path / "unmasked.h5", spec))
    with pytest.raises(ValueError, match="masks"):
        list(iter_cache_episodes(tmp_path / "masked.h5", unmasked))
