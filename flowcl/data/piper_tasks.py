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
    # The embodiment config (configs/embodiment/) the cache is built and trained with. It
    # fixes the image masks, so two tasks on the same recordings can differ only in it.
    embodiment: str = "agilex_piper"

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
# The 2 Nov 2026 session, the same task (episode 0 labelled 'pick the sausage', the rest
# 'pick up the sausage'). The scene sits lower in the front camera than on 30 Oct and the
# lid starts off the pot. Episode 0 ends with the arm over the pot, not at home.
PICK_SAUSAGE_V4 = PiperSource(
    folder="pick_sausage_v4",
    sha256="9e059e744398f1d953daaa2474090c461518a1cce063b2c5d7f2cd6917b2778d",
)
SAUSAGE_POT_LID = "put the sausage in the pot and cover it with the lid"
# The 2 Nov 2026 afternoon session (16:53-17:22), labelled 'pick up the milk'. Every demo
# picks a small milk carton off the table, puts it on the plate and returns home; the
# carton and plate positions vary. The plate is turquoise-blue on both cameras.
PICK_MILK = PiperSource(
    folder="pick_milk",
    sha256="b198239c8ad825fa38cc45ac427fcb812b7099586a9ca54330f56e21c942daf2",
)
MILK_PLATE = "pick up the milk carton and put it on the blue plate"
# The 3 Nov 2026 midday session (12:14-12:59, daylight), labelled 'pick up the grape and put
# it on blue plate'. Every demo picks a grape off the table and puts it on the blue plate.
PICK_GRAPE = PiperSource(
    folder="pick_grape",
    sha256="a7318705b9430b0fcf6f0e8210c2521ce75ac893a9ac836f08f410c7f73f5c0f",
)
GRAPE_PLATE = "pick up the grape and put it on the blue plate"

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
            holdout=4,
            notes=(
                "The v3 session only: trains on v3 0-11 and holds out v3 12-15. sausage_pot_lid "
                "never trains on those four either (it holds out v3 10-15), so the two are "
                "compared on identical unseen episodes (session style against demo count)."
            ),
        ),
        PiperTask(
            name="sausage_pot_lid_all",
            language=SAUSAGE_POT_LID,
            sources=(PICK_SAUSAGE, PICK_SAUSAGE_V3, PICK_SAUSAGE_V4),
            n_episodes=53,
            holdout=0,
            notes=(
                "Every usable demo of the three sessions (20 + 16 + 17), all trained on, the "
                "user's choice for the robot: nothing is held out, so the offline check has "
                "only training episodes."
            ),
        ),
        PiperTask(
            name="sausage_pot_lid_v1",
            language=SAUSAGE_POT_LID,
            sources=(PICK_SAUSAGE,),
            n_episodes=20,
            holdout=4,
            notes=(
                "The first session (pick_sausage) only: trains on its first 16 kept episodes "
                "(raw 1-14, 16, 17) and holds out raw 18-21. sausage_pot_lid trained on those "
                "four, so the common unseen ground of all Piper policies is v3 12-15."
            ),
        ),
        PiperTask(
            name="milk_plate",
            language=MILK_PLATE,
            sources=(PICK_MILK,),
            n_episodes=40,
            holdout=0,
            notes=(
                "Every demo of the milk session, all trained on (as sausage_pot_lid_all). The "
                "front camera is masked where the operator and the leader arm are."
            ),
        ),
        PiperTask(
            name="milk_plate_unmasked",
            language=MILK_PLATE,
            sources=(PICK_MILK,),
            n_episodes=40,
            holdout=0,
            notes=(
                "milk_plate with the front camera as recorded: the policy also sees the operator "
                "and the leader arm, whose pose is the action. The mask ablation."
            ),
            embodiment="agilex_piper_unmasked",
        ),
        PiperTask(
            name="grape_plate",
            language=GRAPE_PLATE,
            sources=(PICK_GRAPE,),
            n_episodes=40,
            holdout=0,
            notes=(
                "Every demo of the grape session, all trained on (as milk_plate), recorded in the "
                "daylight the robot is tested in. The front camera is masked as for milk_plate."
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
