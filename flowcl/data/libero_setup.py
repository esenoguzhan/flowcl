"""Acquire and verify the LIBERO demo datasets.

Spec §3.2: use LIBERO's own HDF5 demos, do not regenerate, and *fail loudly* on a
demo-count mismatch. This module does the acquisition and the verification; the
canonical-episode conversion lives in :mod:`flowcl.data.libero_adapter`.

We call :func:`huggingface_hub.snapshot_download` directly rather than going through
``benchmark_scripts/download_libero_datasets.py``, which prompts on stdin (and
defaults to expiring UT Box links).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from flowcl.utils.libero_paths import (
    assert_init_files_present,
    default_dataset_dir,
    ensure_libero_config,
)

HF_REPO_ID = "yifengzhu-hf/LIBERO-datasets"

# Number of tasks (= HDF5 files) per suite.
#
# NOTE: the HuggingFace repo lays out one directory per *suite*: libero_spatial/,
# libero_object/, libero_goal/, libero_10/, libero_90/. There is deliberately no
# "libero_100" directory, even though upstream's own
# download_utils.download_from_huggingface() passes allow_patterns="libero_100/*".
# That upstream call matches zero files and silently downloads nothing, so we address
# suites directly instead.
SUITE_TASK_COUNTS: dict[str, int] = {
    "libero_spatial": 10,
    "libero_object": 10,
    "libero_goal": 10,
    "libero_10": 10,
    "libero_90": 90,
}

# Suites the thesis actually uses (§3.2, §5). libero_90 exists in the repo but is not
# part of any curriculum and is ~64 GB of the repo's ~93 GB, so it is never fetched
# by default.
REQUIRED_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")

# §3.2: 50 demos per task.
DEMOS_PER_TASK = 50


@dataclass(frozen=True)
class SuiteStatus:
    """Integrity report for one task suite."""

    suite: str
    directory: Path
    n_files: int
    n_expected: int

    @property
    def complete(self) -> bool:
        return self.directory.is_dir() and self.n_files == self.n_expected

    def describe(self) -> str:
        mark = "OK " if self.complete else "BAD"
        return (
            f"[{mark}] {self.suite}: {self.n_files}/{self.n_expected} hdf5 "
            f"in {self.directory}"
        )


def download(
    suites: tuple[str, ...] | list[str] | None = None,
    dataset_dir: Path | None = None,
) -> Path:
    """Download the requested LIBERO suites from HuggingFace.

    Args:
        suites: Keys of :data:`SUITE_TASK_COUNTS`. Defaults to
            :data:`REQUIRED_SUITES` (deliberately excluding the unused ~64 GB
            libero_90).
        dataset_dir: Destination. Defaults to LIBERO's own default location.

    Returns:
        The dataset directory.
    """
    from huggingface_hub import snapshot_download

    suites = tuple(suites) if suites else REQUIRED_SUITES
    unknown = sorted(set(suites) - set(SUITE_TASK_COUNTS))
    if unknown:
        raise ValueError(
            f"Unknown suites {unknown}; valid suites are {sorted(SUITE_TASK_COUNTS)}"
        )

    dataset_dir = Path(dataset_dir) if dataset_dir else default_dataset_dir()
    dataset_dir.mkdir(parents=True, exist_ok=True)

    for suite in suites:
        print(f"[flowcl] downloading {suite} from {HF_REPO_ID} -> {dataset_dir}")
        snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            local_dir=str(dataset_dir),
            allow_patterns=f"{suite}/*",
        )
        # Fail immediately rather than after every suite: an allow_patterns typo
        # matches nothing and downloads nothing, which must not look like success.
        got = len(list((dataset_dir / suite).glob("*.hdf5")))
        if got == 0:
            raise FileNotFoundError(
                f"Downloading suite {suite!r} from {HF_REPO_ID} produced no .hdf5 "
                f"files in {dataset_dir / suite}. The repo layout may have changed; "
                f"expected a top-level {suite}/ directory."
            )
    return dataset_dir


def check(
    dataset_dir: Path | None = None,
    suites: tuple[str, ...] = REQUIRED_SUITES,
) -> list[SuiteStatus]:
    """Report per-suite HDF5 counts without raising."""
    dataset_dir = Path(dataset_dir) if dataset_dir else default_dataset_dir()

    statuses = []
    for suite in suites:
        if suite not in SUITE_TASK_COUNTS:
            raise ValueError(
                f"Unknown suite {suite!r}; valid suites are {sorted(SUITE_TASK_COUNTS)}"
            )
        directory = dataset_dir / suite
        n_files = (
            len(list(directory.glob("*.hdf5"))) if directory.is_dir() else 0
        )
        statuses.append(
            SuiteStatus(
                suite=suite,
                directory=directory,
                n_files=n_files,
                n_expected=SUITE_TASK_COUNTS[suite],
            )
        )
    return statuses


def assert_complete(
    dataset_dir: Path | None = None,
    suites: tuple[str, ...] = REQUIRED_SUITES,
) -> None:
    """Raise with the full report unless every requested suite is complete."""
    statuses = check(dataset_dir=dataset_dir, suites=suites)
    bad = [s for s in statuses if not s.complete]
    if bad:
        report = "\n".join(s.describe() for s in statuses)
        raise FileNotFoundError(
            "LIBERO datasets incomplete. Run:\n"
            "    uv run python scripts/prepare_libero.py\n"
            f"Status:\n{report}"
        )


def prepare(
    dataset_dir: Path | None = None,
    suites: tuple[str, ...] | list[str] | None = None,
    skip_download: bool = False,
) -> Path:
    """Full setup: write LIBERO's config, download if needed, then verify.

    Returns:
        The dataset directory, guaranteed complete for :data:`REQUIRED_SUITES`.
    """
    dataset_dir = Path(dataset_dir) if dataset_dir else default_dataset_dir()
    config_file = ensure_libero_config(dataset_dir)
    print(f"[flowcl] LIBERO config at {config_file}")

    init_dir = assert_init_files_present()
    n_init = len(list(init_dir.rglob("*.pruned_init")))
    print(f"[flowcl] init_files OK: {n_init} .pruned_init files in {init_dir}")

    wanted = tuple(suites) if suites else REQUIRED_SUITES
    statuses = check(dataset_dir, suites=wanted)
    for status in statuses:
        print(f"[flowcl] {status.describe()}")

    incomplete = tuple(s.suite for s in statuses if not s.complete)
    if incomplete and not skip_download:
        download(suites=incomplete, dataset_dir=dataset_dir)

    assert_complete(dataset_dir, suites=wanted)
    print(f"[flowcl] all required suites complete in {dataset_dir}")
    return dataset_dir
