"""Task identity: one place that ties a curriculum entry to its demos and its env.

A "task" has three separate identities in this project, and a mismatch between any
two of them is a silent correctness bug:

* the HDF5 file the demos come from (:mod:`flowcl.data.libero_adapter`),
* the BDDL problem + ``.pruned_init`` initial states the rollout env loads
  (:mod:`flowcl.envs.libero_env`),
* the ``task_key`` string used to index the §8.2 retention matrix.

LIBERO's benchmark object is the authority on the first two: ``get_task(i)`` gives the
problem folder and name, and ``get_task_demonstration(i)`` gives the demo path
relative to the dataset root. :class:`TaskRef` resolves everything from that single
source so no caller re-derives a filename by string surgery.

The ``task_key`` convention is ``"<suite>/<task name>"``, which is exactly what
:func:`flowcl.data.libero_adapter.default_task_id` produces for the corresponding
HDF5 file. :func:`TaskRef.assert_consistent` checks that identity against the real
file rather than trusting the convention to hold.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from flowcl.data.spec import EmbodimentSpec
from flowcl.utils.libero_paths import default_dataset_dir, ensure_libero_config


@lru_cache(maxsize=None)
def _benchmark(suite: str):
    """LIBERO benchmark object for a suite. Cached: construction reads init files."""
    ensure_libero_config()
    from libero.libero import benchmark as libero_benchmark

    registry = libero_benchmark.get_benchmark_dict()
    if suite not in registry:
        raise ValueError(
            f"Unknown LIBERO suite {suite!r}; available: {sorted(registry)}"
        )
    return registry[suite]()


def _is_dobot(suite: str) -> bool:
    from flowcl.data.dobot_tasks import SUITE

    return suite == SUITE


def suite_task_names(suite: str) -> tuple[str, ...]:
    """Task names for a suite, in LIBERO's own index order.

    The ``dobot`` suite (Dobot X-Trainer recordings, branch dobot-hw) comes from
    :mod:`flowcl.data.dobot_tasks` instead of LIBERO's registry.
    """
    if _is_dobot(suite):
        from flowcl.data.dobot_tasks import TASKS

        return tuple(TASKS)
    bench = _benchmark(suite)
    return tuple(bench.get_task_names())


@dataclass(frozen=True)
class TaskRef:
    """A curriculum entry, resolved against LIBERO's benchmark registry.

    Construct with :meth:`from_suite_index` or :meth:`from_key` rather than directly,
    so ``name`` and ``task_idx`` can never disagree.
    """

    suite: str
    task_idx: int
    name: str

    @classmethod
    def from_suite_index(cls, suite: str, task_idx: int) -> "TaskRef":
        names = suite_task_names(suite)
        if not 0 <= task_idx < len(names):
            raise IndexError(
                f"task_idx {task_idx} out of range for {suite}, which has "
                f"{len(names)} tasks"
            )
        return cls(suite=suite, task_idx=task_idx, name=names[task_idx])

    @classmethod
    def from_key(cls, task_key: str) -> "TaskRef":
        """Parse ``"<suite>/<task name>"``.

        Raises:
            ValueError: If the suite is unknown or the name is not one of its tasks.
                Listing the candidates matters here, because the most common cause is
                a typo in a curriculum YAML.
        """
        if task_key.count("/") != 1:
            raise ValueError(
                f"task key {task_key!r} must be '<suite>/<task name>'"
            )
        suite, name = task_key.split("/")
        names = suite_task_names(suite)
        if name not in names:
            raise ValueError(
                f"{suite} has no task named {name!r}. Available: {list(names)}"
            )
        return cls(suite=suite, task_idx=names.index(name), name=name)

    # ---- identities -----------------------------------------------------------

    @property
    def task_key(self) -> str:
        """``"<suite>/<task name>"`` — the retention-matrix index (§8.2)."""
        return f"{self.suite}/{self.name}"

    @property
    def language(self) -> str:
        """The natural-language instruction LIBERO ships with the task."""
        if _is_dobot(self.suite):
            from flowcl.data.dobot_tasks import get_task

            return get_task(self.name).language
        return _benchmark(self.suite).get_task(self.task_idx).language

    def demo_path(self, dataset_dir: Path | None = None) -> Path:
        """Absolute path to this task's HDF5 demos (the decoded cache for ``dobot``)."""
        if _is_dobot(self.suite):
            from flowcl.data.dobot_tasks import cache_path

            path = cache_path(self.name, dataset_dir)
            if not path.is_file():
                raise FileNotFoundError(
                    f"Cache for {self.task_key} not found at {path}. Run "
                    f"`uv run python scripts/prepare_dobot.py --tasks {self.name}`."
                )
            return path
        root = Path(dataset_dir) if dataset_dir else default_dataset_dir()
        relative = _benchmark(self.suite).get_task_demonstration(self.task_idx)
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(
                f"Demos for {self.task_key} not found at {path}. Run "
                "`uv run python scripts/prepare_libero.py`."
            )
        return path

    def assert_consistent(self, dataset_dir: Path | None = None) -> None:
        """Check the demo file agrees with this ref's key and instruction.

        Guards the one failure mode the convention cannot rule out: a demo file whose
        recorded ``language_instruction`` differs from the BDDL task the rollout env
        builds. That would train the policy on one instruction and evaluate it on
        another, which no loss curve would reveal.
        """
        if _is_dobot(self.suite):
            from flowcl.data.dobot_adapter import verify_cache
            from flowcl.data.dobot_tasks import get_task

            task = get_task(self.name)
            verify_cache(self.demo_path(dataset_dir), repo_id=task.repo_id,
                         revision=task.revision, language=task.language, n_demos=None)
            return
        from flowcl.data.libero_adapter import default_task_id, read_task_metadata

        path = self.demo_path(dataset_dir)
        derived = default_task_id(path)
        if derived != self.task_key:
            raise ValueError(
                f"task key mismatch: TaskRef says {self.task_key!r} but the demo file "
                f"{path} maps to {derived!r}"
            )
        recorded = read_task_metadata(path).language
        if recorded.strip().lower() != self.language.strip().lower():
            raise ValueError(
                f"{self.task_key}: demos were recorded with instruction "
                f"{recorded!r} but the BDDL task declares {self.language!r}. Training "
                "and evaluation would use different language conditioning."
            )


def resolve_tasks(task_keys: list[str] | tuple[str, ...]) -> tuple[TaskRef, ...]:
    """Resolve a curriculum's task keys, rejecting duplicates.

    A repeated task in a curriculum would make the retention matrix ambiguous (two
    rows claiming the same ``task_key``), so it is an error rather than a warning.
    """
    refs = tuple(TaskRef.from_key(key) for key in task_keys)
    keys = [ref.task_key for ref in refs]
    duplicates = sorted({k for k in keys if keys.count(k) > 1})
    if duplicates:
        raise ValueError(
            f"curriculum repeats tasks {duplicates}; retention-matrix rows would be "
            "ambiguous"
        )
    return refs


def load_task_episodes(
    ref: TaskRef,
    spec: EmbodimentSpec,
    n_demos: int | None = None,
    dataset_dir: Path | None = None,
    verify: bool = True,
) -> list:
    """Load a task's demos as canonical episodes.

    Args:
        ref: The task.
        spec: Embodiment spec; shapes are asserted against it.
        n_demos: Take the first ``n_demos`` demos. ``None`` means all 50.
        dataset_dir: Dataset root; defaults to LIBERO's own location.
        verify: Run the §3.2 demo-count and action-stats check first. Leave on
            outside of tight loops; it is the check that catches a corrupt download.
    """
    if _is_dobot(ref.suite):
        from flowcl.data.dobot_adapter import iter_cache_episodes, verify_cache
        from flowcl.data.dobot_tasks import get_task

        path = ref.demo_path(dataset_dir)
        if verify:
            task = get_task(ref.name)
            verify_cache(path, repo_id=task.repo_id, revision=task.revision,
                         language=task.language, n_demos=n_demos)
        return list(iter_cache_episodes(path, spec, n_demos=n_demos, task_id=ref.task_key))
    from flowcl.data.libero_adapter import iter_episodes, verify_task_file

    path = ref.demo_path(dataset_dir)
    if verify:
        verify_task_file(path)
    return list(
        iter_episodes(path, spec, n_demos=n_demos, task_id=ref.task_key)
    )


def recorded_language(ref: TaskRef, dataset_dir: Path | None = None) -> str:
    """The instruction stored with a task's demos (LIBERO HDF5 metadata or the dobot cache)."""
    if _is_dobot(ref.suite):
        from flowcl.data.dobot_adapter import cache_attrs

        return str(cache_attrs(ref.demo_path(dataset_dir))["language"])
    from flowcl.data.libero_adapter import read_task_metadata

    return read_task_metadata(ref.demo_path(dataset_dir)).language
