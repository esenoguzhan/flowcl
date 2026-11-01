"""Task registry for the AgileX Piper recordings (suite ``"piper"``, thesis plan F1).

:class:`~flowcl.data.tasks.TaskRef` dispatches the ``piper`` suite here, so ``gate0``
and ``train_on_tasks`` run unchanged on the robot's recordings. Each task names the
LeRobot folders it is built from, pinned by a content fingerprint (there is no dataset
revision), plus the raw episodes left out and why. ``scripts/prepare_piper.py --build``
turns a task into the HDF5 cache :mod:`flowcl.data.piper_adapter` reads.

Cache episodes are numbered in registry order (sources in order, raw episodes ascending,
exclusions skipped). The last ``holdout`` of them are never trained on: they supply the
offline open-loop check and the rollouts' initial-condition photos.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from flowcl.utils.libero_paths import repo_root

SUITE = "piper"


@dataclass(frozen=True)
class PiperSource:
    # Folder under the raw root.
    folder: str
    # flowcl.data.piper_adapter.source_fingerprint of the folder.
    sha256: str
    # Raw episode indices left out (reasons in the task's notes).
    exclude: tuple[int, ...] = ()


@dataclass(frozen=True)
class PiperTask:
    name: str
    # The instruction the policy is conditioned on. The recordings' own label is kept in
    # the cache (``raw_labels``) but may be wrong.
    language: str
    sources: tuple[PiperSource, ...]
    # Episodes after exclusions, over all sources.
    n_episodes: int
    holdout: int
    notes: str = ""

    @property
    def n_train(self) -> int:
        return self.n_episodes - self.holdout


# The 30 Oct 2026 sessions, labelled 'pick the sausage'. Every demo puts the sausage in the
# pot and then the lid on the pot, from and back to the home pose. pick_sausage episode 0
# was filmed with the front camera in another pose; episode 15 is a 3.7 s arm return with
# no task. The v3 session's operator is faster (22-33 s per episode against 24-47 s).
PICK_SAUSAGE = PiperSource(
    folder="pick_sausage",
    sha256="31459649316c92d4884883d65d030531f56f8e0d60bb2e9f84112dc6ac587202",
    exclude=(0, 15),
)
PICK_SAUSAGE_V3 = PiperSource(
    folder="pick_sausage_v3",
    sha256="4618ba974fb38292b077f70c27700798acdc85a3d6eeb7128999d615d248cf40",
)
SAUSAGE_POT_LID = "put the sausage in the pot and cover it with the lid"

TASKS: dict[str, PiperTask] = {
    t.name: t
    for t in (
        PiperTask(
            name="sausage_pot_lid",
            language=SAUSAGE_POT_LID,
            sources=(PICK_SAUSAGE, PICK_SAUSAGE_V3),
            n_episodes=36,
            holdout=6,
            notes="Both sessions. The hold-out is pick_sausage_v3 episodes 10-15, the latest session.",
        ),
        PiperTask(
            name="sausage_pot_lid_v3",
            language=SAUSAGE_POT_LID,
            sources=(PICK_SAUSAGE_V3,),
            n_episodes=16,
            holdout=6,
            notes=(
                "The v3 session only: trains on v3 0-9 and holds out v3 10-15, the same six "
                "episodes sausage_pot_lid holds out, so the two are compared on identical "
                "episodes (session style against demo count)."
            ),
        ),
    )
}


def get_task(name: str) -> PiperTask:
    if name not in TASKS:
        raise ValueError(f"{SUITE} has no task named {name!r}. Available: {list(TASKS)}")
    return TASKS[name]


def raw_root() -> Path:
    """Where the robot PC's recordings are copied: ``$FLOWCL_PIPER_RAW`` or ``~/piper_datasets``."""
    env = os.environ.get("FLOWCL_PIPER_RAW")
    return Path(env) if env else Path.home() / "piper_datasets"


def piper_root(dataset_dir: Path | None = None) -> Path:
    """``datasets/piper`` in the checkout, or ``$FLOWCL_PIPER_DIR``, or ``dataset_dir``."""
    if dataset_dir is not None:
        return Path(dataset_dir)
    env = os.environ.get("FLOWCL_PIPER_DIR")
    return Path(env) if env else repo_root() / "datasets" / "piper"


def cache_path(name: str, dataset_dir: Path | None = None) -> Path:
    return piper_root(dataset_dir) / "cache" / f"{name}.h5"
