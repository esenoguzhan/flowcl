"""Task registry for the Dobot X-Trainer try-out (suite ``"dobot"``, branch dobot-hw).

:class:`~flowcl.data.tasks.TaskRef` dispatches the ``dobot`` suite here, so curricula,
``train_on_tasks``, ``run_continual`` and GPM run unchanged on the lab's recordings.
Each task is pinned to a Hugging Face dataset revision; ``scripts/prepare_dobot.py``
turns that revision into the HDF5 cache :mod:`flowcl.data.dobot_adapter` reads.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from flowcl.utils.libero_paths import repo_root

SUITE = "dobot"


@dataclass(frozen=True)
class DobotTask:
    name: str
    repo_id: str
    revision: str
    # The recordings' own ``prompt`` column (checked against the cache).
    language: str
    n_episodes: int


TASKS: dict[str, DobotTask] = {
    t.name: t
    for t in (
        DobotTask(
            name="puzzle_red_retrieve",
            repo_id="Physical-41/dobot_dataset.puzzle.red.retrieve",
            revision="82612c17cc7dc2382a3e38f56d49889b608499dd",
            language="Use the red block to open the lid of the box and retrieve the object inside",
            n_episodes=120,
        ),
        DobotTask(
            name="puzzle_green_retrieve",
            repo_id="Physical-41/dobot_dataset.puzzle.green.retrieve",
            revision="31d7c64627ca375e3a8e11192f0623208ac33a16",
            language="Combine the green sticks and use it to push the object out of the tube",
            n_episodes=124,
        ),
        DobotTask(
            name="puzzle_yellow_retrieve",
            repo_id="Physical-41/dobot_dataset.puzzle.yellow.retrieve",
            revision="b73e4012caff3c601698b489cd5148a33f4ce474",
            language="Use the yellow tool to retrieve the object out of the cage",
            n_episodes=142,
        ),
    )
}


def get_task(name: str) -> DobotTask:
    if name not in TASKS:
        raise ValueError(f"{SUITE} has no task named {name!r}. Available: {list(TASKS)}")
    return TASKS[name]


def dobot_root(dataset_dir: Path | None = None) -> Path:
    """``datasets/dobot`` in the checkout, or ``$FLOWCL_DOBOT_DIR``, or ``dataset_dir``."""
    if dataset_dir is not None:
        return Path(dataset_dir)
    env = os.environ.get("FLOWCL_DOBOT_DIR")
    return Path(env) if env else repo_root() / "datasets" / "dobot"


def raw_dir(name: str, dataset_dir: Path | None = None) -> Path:
    return dobot_root(dataset_dir) / "raw" / get_task(name).repo_id.split("/", 1)[1]


def cache_path(name: str, dataset_dir: Path | None = None) -> Path:
    return dobot_root(dataset_dir) / "cache" / f"{name}.h5"
