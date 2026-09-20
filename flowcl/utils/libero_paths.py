"""Non-interactive configuration of LIBERO's path registry.

``benchmark/LIBERO/libero/libero/__init__.py`` builds ``~/.libero/config.yaml`` on
first import and calls :func:`input` to ask whether the user wants a custom dataset
directory. In a batch run that blocks forever on a closed stdin. Every entry point
that touches LIBERO must therefore call :func:`ensure_libero_config` *before*
importing ``libero``.

The config file is a flat mapping of five keys, matching
``libero.libero.get_default_path_dict``::

    benchmark_root: <repo>/benchmark/LIBERO/libero/libero
    bddl_files:     <benchmark_root>/bddl_files
    init_states:    <benchmark_root>/init_files
    datasets:       <configurable, default <benchmark_root>/../datasets>
    assets:         <benchmark_root>/assets
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

# Keys LIBERO's get_libero_path() will accept. Asserting on this set means a future
# LIBERO bump that renames a key fails loudly here instead of much later.
LIBERO_PATH_KEYS = frozenset(
    {"benchmark_root", "bddl_files", "init_states", "datasets", "assets"}
)


def repo_root() -> Path:
    """Absolute path to the flowcl checkout root."""
    return Path(__file__).resolve().parents[2]


def libero_submodule_root() -> Path:
    """Absolute path to the LIBERO submodule checkout."""
    path = repo_root() / "benchmark" / "LIBERO"
    if not path.is_dir():
        raise FileNotFoundError(
            f"LIBERO submodule missing at {path}. Run: git submodule update --init --recursive"
        )
    return path


def libero_benchmark_root() -> Path:
    """The directory LIBERO calls ``benchmark_root`` (its inner package dir)."""
    return libero_submodule_root() / "libero" / "libero"


def default_dataset_dir() -> Path:
    """Where LIBERO demo HDF5s live unless overridden.

    Mirrors LIBERO's own default of ``<benchmark_root>/../datasets`` so that a
    checkout configured by upstream tooling and one configured by us agree.
    """
    return libero_benchmark_root().parent / "datasets"


def ensure_libero_config(dataset_dir: str | os.PathLike[str] | None = None) -> Path:
    """Write ``~/.libero/config.yaml`` so importing ``libero`` never prompts.

    Idempotent, but *not* silently tolerant: if a config already exists and points
    somewhere other than this checkout, it is rewritten and the change is reported,
    because a stale config silently redirects evaluation initial states to another
    LIBERO copy.

    Args:
        dataset_dir: Where demo HDF5s live. Defaults to :func:`default_dataset_dir`.

    Returns:
        Path to the config file that was written.
    """
    benchmark_root = libero_benchmark_root()
    datasets = Path(dataset_dir).expanduser().resolve() if dataset_dir else default_dataset_dir()

    desired = {
        "benchmark_root": str(benchmark_root),
        "bddl_files": str(benchmark_root / "bddl_files"),
        "init_states": str(benchmark_root / "init_files"),
        "datasets": str(datasets),
        "assets": str(benchmark_root / "assets"),
    }
    assert set(desired) == set(LIBERO_PATH_KEYS), (
        f"config keys {sorted(desired)} do not match LIBERO's expected "
        f"{sorted(LIBERO_PATH_KEYS)}"
    )

    config_dir = Path(os.environ.get("LIBERO_CONFIG_PATH", Path.home() / ".libero"))
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "config.yaml"

    if config_file.exists():
        existing = yaml.safe_load(config_file.read_text()) or {}
        if existing != desired:
            differing = sorted(
                k for k in desired if str(existing.get(k, "")) != desired[k]
            )
            print(
                f"[flowcl] rewriting {config_file}; keys changed: {differing}. "
                f"Previous benchmark_root={existing.get('benchmark_root')!r}"
            )
        else:
            return config_file

    config_file.write_text(yaml.safe_dump(desired, sort_keys=True))
    return config_file


def assert_init_files_present() -> Path:
    """Fail loudly unless LIBERO's fixed initial-state files are checked out.

    Spec §8.1 requires the same 50 initial states for every method; those live in
    ``init_files/<suite>/<task>.pruned_init``. A partially-initialised submodule
    would otherwise surface much later as a confusing evaluation error.
    """
    init_dir = libero_benchmark_root() / "init_files"
    if not init_dir.is_dir():
        raise FileNotFoundError(
            f"LIBERO init_files missing at {init_dir}. "
            "Run: git submodule update --init --recursive"
        )
    n = len(list(init_dir.rglob("*.pruned_init")))
    if n == 0:
        raise FileNotFoundError(
            f"No .pruned_init files under {init_dir}; the submodule checkout is incomplete."
        )
    return init_dir
