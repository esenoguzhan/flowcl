"""Curricula: ordered task sequences, declared in ``configs/curriculum/``.

Spec §5: "A curriculum is an ordered list of ``(task_key, dataset_path, n_demos)``."
The dataset path is derived rather than declared — :class:`~flowcl.data.tasks.TaskRef`
resolves it from LIBERO's benchmark registry, so a curriculum YAML cannot pin a path
that disagrees with the BDDL task the rollout env builds.

Order is the experiment. §5 runs ``seq_hetero`` forwards *and* backwards precisely
because LIBERO showed ordering matters, so :meth:`Curriculum.reversed` exists rather
than leaving a second hand-maintained YAML to drift out of sync.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf

from flowcl.data.tasks import TaskRef, resolve_tasks
from flowcl.utils.libero_paths import repo_root


@dataclass(frozen=True)
class CurriculumStage:
    """One stage: a task and how many of its demos to train on."""

    ref: TaskRef
    n_demos: int

    @property
    def task_key(self) -> str:
        return self.ref.task_key


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
        return Curriculum(
            name=name or f"{self.name}_reverse",
            stages=tuple(reversed(self.stages)),
            description=f"Reverse order of {self.name}. {self.description}".strip(),
            expectation=self.expectation,
        )

    def assert_consistent(self, dataset_dir: Path | None = None) -> None:
        """Check every stage's demos, key and instruction agree (§3.2)."""
        for stage in self.stages:
            stage.ref.assert_consistent(dataset_dir)


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
    ``task_key`` and optional ``n_demos``.
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

    keys, demo_counts = [], []
    for entry in entries:
        if isinstance(entry, str):
            keys.append(entry)
            demo_counts.append(default_n_demos)
        elif isinstance(entry, dict):
            keys.append(entry["task_key"])
            demo_counts.append(int(entry.get("n_demos", default_n_demos)))
        else:
            raise TypeError(
                f"curriculum task entry must be a string or a mapping, got {entry!r}"
            )

    refs = resolve_tasks(keys)
    return Curriculum(
        name=payload.get("name") or "unnamed",
        stages=tuple(
            CurriculumStage(ref=ref, n_demos=n)
            for ref, n in zip(refs, demo_counts)
        ),
        description=payload.get("description", ""),
        expectation=payload.get("expectation", ""),
    )
