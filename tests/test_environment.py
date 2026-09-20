"""Phase 0 green test: the environment is actually usable.

Split by requirement so the suite is honest about where it runs:

* unmarked  — pure filesystem/import checks, run anywhere (including WSL)
* ``gpu``   — needs a CUDA device
* ``sim``   — needs MuJoCo offscreen rendering plus the LIBERO datasets

The ``sim`` and ``gpu`` tests are the ones that matter on the training box; the
unmarked ones catch a broken checkout during development.
"""

from __future__ import annotations

import os

import pytest

from flowcl.data import libero_setup
from flowcl.utils.libero_paths import (
    assert_init_files_present,
    ensure_libero_config,
    libero_benchmark_root,
    libero_submodule_root,
)


def test_libero_submodule_checked_out():
    root = libero_submodule_root()
    assert (root / "setup.py").is_file(), f"{root} is not a LIBERO checkout"
    assert (libero_benchmark_root() / "bddl_files").is_dir()


def test_init_files_present():
    """Spec §8.1 needs the fixed initial states; there are 130 tasks total."""
    init_dir = assert_init_files_present()
    n = len(list(init_dir.rglob("*.pruned_init")))
    assert n == 130, f"expected 130 .pruned_init files, found {n} in {init_dir}"


def test_libero_config_is_written_without_prompting():
    """Importing libero must never block on stdin."""
    config_file = ensure_libero_config()
    assert config_file.is_file()

    import yaml

    config = yaml.safe_load(config_file.read_text())
    assert set(config) == {
        "benchmark_root",
        "bddl_files",
        "init_states",
        "datasets",
        "assets",
    }
    # init_states must point at the submodule we actually validated above.
    assert config["init_states"] == str(libero_benchmark_root() / "init_files")


def test_imports_robosuite_and_libero():
    ensure_libero_config()
    import robosuite  # noqa: F401
    import libero.libero  # noqa: F401
    from libero.libero import benchmark

    suites = benchmark.get_benchmark_dict()
    for name in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
        assert name in suites, f"{name} missing from LIBERO benchmark registry"


def test_no_robomimic_dependency():
    """We deliberately do not depend on robomimic (§3.1 mandates our own adapter)."""
    import importlib.util

    assert importlib.util.find_spec("robomimic") is None, (
        "robomimic is installed; the pyproject deliberately excludes it because its "
        "pins conflict with PyTorch 2.4"
    )


@pytest.mark.gpu
def test_torch_sees_cuda_device():
    import torch

    assert torch.cuda.is_available(), "no CUDA device visible to torch"
    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    print(f"CUDA device: {name}, capability {capability}, torch {torch.__version__}")
    # RTX 4090 is Ada Lovelace, sm_89. Assert the major arch rather than the exact
    # marketing name so an equivalent Ada card also passes.
    assert capability[0] >= 8, f"expected sm_80+ (Ada/Ampere), got {capability}"


@pytest.mark.sim
def test_datasets_complete(dataset_dir):
    libero_setup.assert_complete(dataset_dir)


@pytest.mark.sim
def test_offscreen_render_one_frame(dataset_dir):
    """MUJOCO_GL=egl renders a 128x128 agentview frame from a real LIBERO task."""
    assert os.environ.get("MUJOCO_GL") == "egl", (
        "set MUJOCO_GL=egl for headless rendering; got "
        f"{os.environ.get('MUJOCO_GL')!r}"
    )

    ensure_libero_config()
    import numpy as np
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()["libero_object"]()
    task = suite.get_task(0)
    bddl = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )

    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=128,
        camera_widths=128,
        camera_names=["agentview", "robot0_eye_in_hand"],
    )
    try:
        env.seed(0)
        env.reset()
        init_states = suite.get_task_init_states(0)
        obs = env.set_init_state(init_states[0])
        for key in ("agentview_image", "robot0_eye_in_hand_image"):
            assert key in obs, f"{key} missing from observation; got {sorted(obs)}"
            assert obs[key].shape == (128, 128, 3), f"{key} shape {obs[key].shape}"
            assert obs[key].dtype == np.uint8
        # A frame that is entirely one colour means the GL context produced nothing.
        assert obs["agentview_image"].std() > 1.0, "rendered frame is constant"
    finally:
        env.close()
