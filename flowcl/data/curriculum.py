"""Curricula: ordered task sequences, declared in ``configs/curriculum/``.

Spec §5: "A curriculum is an ordered list of ``(task_key, dataset_path, n_demos)``."
The dataset path is derived rather than declared — :class:`~flowcl.data.tasks.TaskRef`
resolves it from LIBERO's benchmark registry, so a curriculum YAML cannot pin a path
that disagrees with the BDDL task the rollout env builds.

Order is the experiment. §5 runs ``seq_hetero`` forwards *and* backwards precisely
because LIBERO showed ordering matters, so :meth:`Curriculum.reversed` exists rather
than leaving a second hand-maintained YAML to drift out of sync.

A stage may also **co-train** further tasks (``co_train: [keys]``): its dataset is then the
union of its task and the co-trained ones, which are evaluated after every stage like any
other task but are not columns of the retention matrix (the stage's own task is). With
``steps_factor`` the stage trains for that multiple of the recipe's steps. Both exist for
multi-task starting points (``configs/analysis/language_cl.yaml``); a curriculum without
them is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf

from flowcl.data.tasks import TaskRef, resolve_tasks
from flowcl.utils.libero_paths import repo_root


@dataclass(frozen=True)
class CurriculumStage:
    """One stage: a task and how many of its demos to train on.

    ``co_train`` tasks join the stage's dataset (with the same ``n_demos``); ``steps_factor``
    multiplies the recipe's steps for this stage.
    """

    ref: TaskRef
    n_demos: int
    co_train: tuple[TaskRef, ...] = ()
    steps_factor: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.steps_factor, int) or self.steps_factor < 1:
            raise ValueError(f"steps_factor must be a positive integer, got {self.steps_factor!r}")

    @property
    def task_key(self) -> str:
        return self.ref.task_key

    @property
    def train_refs(self) -> tuple[TaskRef, ...]:
        """The tasks this stage trains on: its own, then the co-trained ones."""
        return (self.ref, *self.co_train)

    @property
    def is_joint(self) -> bool:
        return bool(self.co_train)

    def record(self) -> dict:
        """The stage as ``config.yaml`` records it (old configs have only the first two keys)."""
        out = {"task_key": self.task_key, "n_demos": self.n_demos}
        if self.co_train or self.steps_factor != 1:
            out["co_train"] = [r.task_key for r in self.co_train]
            out["steps_factor"] = self.steps_factor
        return out


@dataclass(frozen=True)
class Curriculum:
    """An ordered task sequence (§5)."""

    name: str
    stages: tuple[CurriculumStage, ...]
    description: str = ""
    expectation: str = ""

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError(f"curriculum {self.name!r} has no stages")
        keys = [r.task_key for stage in self.stages for r in stage.train_refs]
        duplicates = sorted({k for k in keys if keys.count(k) > 1})
        if duplicates:
            raise ValueError(
                f"curriculum {self.name!r} repeats tasks {duplicates} across its stages and "
                "co-trained tasks; retention-matrix rows would be ambiguous"
            )

    def __len__(self) -> int:
        return len(self.stages)

    @property
    def task_keys(self) -> tuple[str, ...]:
        """Retention-matrix column order (§8.2)."""
        return tuple(stage.task_key for stage in self.stages)

    @property
    def refs(self) -> tuple[TaskRef, ...]:
        return tuple(stage.ref for stage in self.stages)

    @property
    def eval_refs(self) -> tuple[TaskRef, ...]:
        """Every task evaluated after every stage: the stage tasks, then the co-trained ones."""
        return (*self.refs, *(r for stage in self.stages for r in stage.co_train))

    @property
    def first_task_key(self) -> str:
        """The task §3.3 fits normalization statistics on."""
        return self.stages[0].task_key

    def reversed(self, name: str | None = None) -> "Curriculum":
        """The same tasks in reverse order (§5's ordering control).

        Note the *normalization stats change* as a result: §3.3 fits them on the first
        task, which is now a different task. That is correct — the reverse run is a
        genuinely different curriculum, not a relabelling of the forward one — and is
        the reason this returns a new object instead of a view.
        """
        if any(stage.co_train or stage.steps_factor != 1 for stage in self.stages):
            raise ValueError(
                f"curriculum {self.name!r} has co-trained tasks or a steps factor; its reverse "
                "order is not defined"
            )
        return Curriculum(
            name=name or f"{self.name}_reverse",
            stages=tuple(reversed(self.stages)),
            description=f"Reverse order of {self.name}. {self.description}".strip(),
            expectation=self.expectation,
        )

    def assert_consistent(self, dataset_dir: Path | None = None) -> None:
        """Check every stage's demos, key and instruction agree (§3.2)."""
        for stage in self.stages:
            for ref in stage.train_refs:
                ref.assert_consistent(dataset_dir)


def curriculum_config_path(name: str) -> Path:
    return repo_root() / "configs" / "curriculum" / f"{name}.yaml"


def load_curriculum(
    source: str | Path | dict,
    default_n_demos: int = 50,
) -> Curriculum:
    """Load a curriculum from a config name, path, or already-parsed mapping.

    Args:
        source: ``"seq_hetero"``, a path to a YAML, or a dict with ``name`` and
            ``tasks``.
        default_n_demos: Used for entries that do not declare ``n_demos``.

    The ``tasks`` list accepts either a bare task key string or a mapping with
    ``task_key`` and optional ``n_demos``, ``co_train`` (a list of task keys trained jointly
    with this stage's task) and ``steps_factor`` (a positive integer).
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.suffix:
            path = curriculum_config_path(str(source))
        if not path.is_file():
            raise FileNotFoundError(f"Curriculum config not found: {path}")
        payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    else:
        payload = dict(source)

    if not isinstance(payload, dict):
        raise TypeError(f"curriculum config must be a mapping, got {type(payload)}")

    entries = payload.get("tasks")
    if not entries:
        raise ValueError(f"curriculum {payload.get('name')!r} declares no tasks")

    keys, demo_counts, co_train, factors = [], [], [], []
    for entry in entries:
        if isinstance(entry, str):
            keys.append(entry)
            demo_counts.append(default_n_demos)
            co_train.append(())
            factors.append(1)
        elif isinstance(entry, dict):
            unknown = sorted(set(entry) - {"task_key", "n_demos", "co_train", "steps_factor"})
            if unknown:
                raise ValueError(f"curriculum task entry has unknown keys {unknown}: {entry!r}")
            keys.append(entry["task_key"])
            demo_counts.append(int(entry.get("n_demos", default_n_demos)))
            co_train.append(tuple(TaskRef.from_key(k) for k in (entry.get("co_train") or ())))
            factor = entry.get("steps_factor", 1)
            if isinstance(factor, bool) or not isinstance(factor, int):
                raise ValueError(f"steps_factor must be a positive integer, got {factor!r}")
            factors.append(factor)
        else:
            raise TypeError(
                f"curriculum task entry must be a string or a mapping, got {entry!r}"
            )

    refs = resolve_tasks(keys)
    return Curriculum(
        name=payload.get("name") or "unnamed",
        stages=tuple(
            CurriculumStage(ref=ref, n_demos=n, co_train=co, steps_factor=f)
            for ref, n, co, f in zip(refs, demo_counts, co_train, factors)
        ),
        description=payload.get("description", ""),
        expectation=payload.get("expectation", ""),
    )
