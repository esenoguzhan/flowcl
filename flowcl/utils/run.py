"""Run registry.

Spec §2: every run writes ``config.yaml``, ``git_sha``, a dependency freeze and the
resolved seed. *A run without these is invalid* — so :func:`validate_run` raises
rather than warning, and :func:`create_run` writes every artifact up front before a
single gradient step happens.

We additionally record the LIBERO submodule SHA. LIBERO's ``init_files`` define the
evaluation initial states and its ``bddl_files`` define the tasks themselves, so a
run is not reproducible from the flowcl SHA alone.
"""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from flowcl.utils.libero_paths import libero_submodule_root, repo_root

# Artifacts that must exist for a run directory to be considered valid (§2).
REQUIRED_ARTIFACTS = (
    "config.yaml",
    "git_sha",
    "libero_submodule_sha",
    "requirements.txt",
    "seed.json",
)


def _git(args: list[str], cwd: Path) -> str:
    """Run a git command, failing loudly with the offending invocation."""
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed in {cwd} (exit {result.returncode}): "
            f"{result.stderr.strip()}"
        )
    return result.stdout.strip()


def git_sha(cwd: Path | None = None) -> str:
    """HEAD SHA, suffixed with ``-dirty`` when the working tree has changes.

    The dirty marker matters: a SHA alone would claim reproducibility that an
    uncommitted edit silently breaks.
    """
    cwd = cwd or repo_root()
    sha = _git(["rev-parse", "HEAD"], cwd)
    status = _git(["status", "--porcelain"], cwd)
    return f"{sha}-dirty" if status else sha


def dependency_freeze() -> str:
    """``pip freeze``-style listing of the active environment.

    Read from installed distribution metadata rather than shelling out to pip,
    which is not guaranteed to be present inside a uv-managed virtualenv.
    """
    entries = sorted(
        f"{dist.metadata['Name']}=={dist.version}"
        for dist in importlib.metadata.distributions()
        if dist.metadata["Name"]
    )
    header = (
        f"# python {platform.python_version()} on {platform.platform()}\n"
        f"# generated {datetime.now(timezone.utc).isoformat()}\n"
    )
    return header + "\n".join(entries) + "\n"


@dataclass(frozen=True)
class RunHandle:
    """Handle to a validated run directory."""

    run_id: str
    path: Path
    seed: int

    def artifact(self, name: str) -> Path:
        return self.path / name

    def subdir(self, name: str) -> Path:
        """Create and return a subdirectory, e.g. ``bases`` or ``checkpoints``."""
        d = self.path / name
        d.mkdir(parents=True, exist_ok=True)
        return d


def create_run(
    run_id: str,
    cfg: DictConfig | dict,
    seed: int,
    results_root: Path | None = None,
    exist_ok: bool = False,
) -> RunHandle:
    """Create ``results/<run_id>/`` and write every required artifact.

    Args:
        run_id: Unique run identifier.
        cfg: Fully resolved config. Stored verbatim so the run can be replayed.
        seed: The *resolved* seed, i.e. the integer actually used, never ``None``.
        results_root: Defaults to ``<repo>/results``.
        exist_ok: Allow writing into an existing directory. Off by default so a
            typo'd run_id cannot silently overwrite finished results.

    Returns:
        A :class:`RunHandle` whose directory has already passed
        :func:`validate_run`.
    """
    results_root = results_root or (repo_root() / "results")
    run_dir = results_root / run_id

    if run_dir.exists() and not exist_ok:
        raise FileExistsError(
            f"Run directory {run_dir} already exists. Pass exist_ok=True to reuse it, "
            "or choose a different run_id; overwriting finished results silently "
            "would destroy evidence."
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    container = cfg if isinstance(cfg, DictConfig) else OmegaConf.create(cfg)
    (run_dir / "config.yaml").write_text(OmegaConf.to_yaml(container, resolve=True))
    (run_dir / "git_sha").write_text(git_sha() + "\n")
    (run_dir / "libero_submodule_sha").write_text(
        git_sha(libero_submodule_root()) + "\n"
    )
    (run_dir / "requirements.txt").write_text(dependency_freeze())
    (run_dir / "seed.json").write_text(
        json.dumps(
            {"seed": int(seed), "created": datetime.now(timezone.utc).isoformat()},
            indent=2,
        )
        + "\n"
    )

    validate_run(run_dir)
    return RunHandle(run_id=run_id, path=run_dir, seed=int(seed))


def validate_run(run_dir: Path) -> None:
    """Raise unless every artifact required by §2 is present and non-empty."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise NotADirectoryError(f"Run directory {run_dir} does not exist")

    missing = [name for name in REQUIRED_ARTIFACTS if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Run {run_dir} is invalid per spec §2; missing artifacts: {missing}"
        )
    empty = [
        name
        for name in REQUIRED_ARTIFACTS
        if (run_dir / name).stat().st_size == 0
    ]
    if empty:
        raise ValueError(f"Run {run_dir} has empty artifacts: {empty}")
