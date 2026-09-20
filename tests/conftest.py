"""Shared test fixtures.

LIBERO's package ``__init__`` prompts on stdin when ``~/.libero/config.yaml`` is
absent, so the config must exist before *any* test imports ``libero``. Doing it in
``conftest`` means no individual test has to remember.
"""

from __future__ import annotations

import pytest

from flowcl.utils.libero_paths import ensure_libero_config

ensure_libero_config()


@pytest.fixture(scope="session")
def dataset_dir():
    """Dataset directory, skipping the test if the suites are not downloaded."""
    from flowcl.data import libero_setup

    statuses = libero_setup.check()
    incomplete = [s for s in statuses if not s.complete]
    if incomplete:
        pytest.skip(
            "LIBERO datasets not downloaded: "
            + "; ".join(s.describe() for s in incomplete)
        )
    return libero_setup.default_dataset_dir()
