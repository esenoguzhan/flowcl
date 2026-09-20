# Environment notes

Findings that cost real debugging time, recorded so they are not rediscovered.

## Dependency pins that are load-bearing

### `mujoco==2.3.7`

robosuite 1.4.1 only declares `mujoco>=2.3.0`, so an unpinned resolve installs
mujoco 3.x. Every LIBERO scene then fails to load inside
`robosuite/utils/binding_utils.py::joint_name2addr` at

```
assert joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
```

The failure is an opaque bare `AssertionError` with no mention of versions, and it
happens on `env.reset()`, i.e. only once you try to run an actual rollout. 2.3.7 is
the last 2.3 release and the series robosuite 1.4 targets.

### LIBERO's undeclared runtime dependencies

`benchmark/LIBERO/setup.py` declares `install_requires=[]`, so *none* of LIBERO's
real dependencies resolve transitively. The four needed to reach
`from libero.libero.envs import OffScreenRenderEnv` are:

| Package | Needed by |
| --- | --- |
| `future` | `bddl` 1.0.1 does `from future.utils import with_metaclass` |
| `easydict` | `libero/libero/envs/objects/articulated_objects.py` |
| `gym` | `libero/libero/envs/venv.py`, imported by `envs/__init__.py` |
| `cloudpickle` | same |

All four appear in LIBERO's own `requirements.txt`. We deliberately do **not**
install `robomimic`, `thop` or `opencv-python`: those are only used by
`libero/lifelong/`, which we never import because §3.1 requires our own adapter.

### `libero` needs `editable_mode=compat`

LIBERO's `setup.py` uses `find_packages()`, which returns `[]` because
`benchmark/LIBERO/libero/` has no `__init__.py` — it is a PEP 420 namespace package.
A modern PEP 660 editable install therefore maps nothing and `import libero` fails
with `ModuleNotFoundError` despite `uv pip list` showing the package as installed.
`[tool.uv.config-settings-package]` sets setuptools' compat editable mode, which
writes a plain `.pth` pointing at the project root. That is how upstream
`pip install -e .` used to behave. The submodule is never edited.

## LIBERO quirks

- **`~/.libero/config.yaml` must exist before importing `libero`.** Its package
  `__init__` calls `input()` to ask about a custom dataset directory, which blocks
  forever on a closed stdin. `flowcl.utils.libero_paths.ensure_libero_config()`
  writes it; `tests/conftest.py` calls that at import time.
- **There is no `libero_100` directory on HuggingFace.** The dataset repo
  (`yifengzhu-hf/LIBERO-datasets`) stores `libero_10/` and `libero_90/` separately,
  but upstream's `download_from_huggingface` passes
  `allow_patterns="libero_100/*"`, which matches zero files and downloads nothing
  while appearing to succeed. `flowcl/data/libero_setup.py` addresses suites
  directly and raises if a suite download yields no files.
- **`ControlEnv` is not re-exported** by `libero/libero/envs/__init__.py`; import it
  from `libero.libero.envs.env_wrapper`. Only the rendering variants are surfaced.
- **Flattened MuJoCo state width is per-scene**, not constant: 92 for a
  `libero_spatial` tabletop, 110 for the `libero_object` scene. Never hardcode it.
- **HDF5 demo groups sort alphabetically**, so a naive `sorted()` puts `demo_10`
  immediately after `demo_1`. The adapter sorts numerically so demo indices mean the
  same thing here as in LIBERO's tooling.
- **Image convention.** Demos record `macros_image_convention='opengl'` and the live
  env applies `robosuite.macros.IMAGE_CONVENTION`. If those ever disagree, training
  frames are vertically flipped relative to rollout frames — training loss looks
  healthy while evaluation collapses. The adapter asserts they match.

## Rendering

Offscreen rendering needs `MUJOCO_GL=egl` and a real GPU. Tests are split so this is
explicit:

| Marker | Requires | Runs in WSL? |
| --- | --- | --- |
| *(none)* | nothing | yes |
| `physics` | MuJoCo physics + LIBERO datasets, no GL | yes |
| `sim` | offscreen rendering via EGL | no |
| `gpu` | a CUDA device | no |

The §10.1 demo-replay test is `physics`, not `sim`: success is a pure-physics
property, so the most important data-layer check does not depend on a GL context.

Run locally with `-m 'not sim and not gpu'`; run everything on the training box with
`MUJOCO_GL=egl`.

### What "no GPU in this WSL instance" looks like

Diagnosed once, recorded because the two symptoms look unrelated but share a cause.

`torch.cuda.is_available()` is `False` and offscreen rendering dies with:

```
libEGL warning: failed to open /dev/dri/renderD128: Permission denied
ImportError: Cannot initialize a EGL device display. This likely means that your
EGL driver does not support the PLATFORM_DEVICE extension ...
```

Check, in this order:

1. `ls /usr/lib/wsl/lib` — with a working NVIDIA WSL driver this contains
   `libcuda.so.1`, `libnvidia-ml.so.1` and `nvidia-smi`. If it holds only
   `libd3d12.so`, `libd3d12core.so` and `libdxcore.so`, the Windows-side NVIDIA
   driver is missing or predates WSL CUDA support, and *no* amount of Linux-side
   installation fixes it. CUDA is unavailable until it is updated on Windows.
2. `id` — the user must be in `render` and `video`. `/dev/dri/card0` is
   `root:video 0660` and `/dev/dri/renderD128` is `root:render 0660`, so without
   those groups Mesa's EGL cannot open either device and `egl_get_devices()`
   enumerates devices that all fail to initialise. Fix with
   `sudo usermod -aG render,video $USER` followed by `wsl --shutdown`.

Both are environment problems, not code problems: every non-`sim`, non-`gpu` test
passes regardless. Gate 0 onwards cannot produce a verdict without them, since every
gate needs training plus rollouts.
