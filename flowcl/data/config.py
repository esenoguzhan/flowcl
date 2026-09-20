"""Build dataclass specs from Hydra/OmegaConf configs.

Spec §11: configs are the only place experiment parameters live. This module is the
single bridge from YAML to the frozen dataclasses in :mod:`flowcl.data.spec`, so no
other module needs to know config key names.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from flowcl.data.spec import ActionSpec, EmbodimentSpec, ObservationSpec
from flowcl.utils.libero_paths import repo_root


def configs_root() -> Path:
    return repo_root() / "configs"


def load_embodiment_spec(cfg: DictConfig | dict | str | Path) -> EmbodimentSpec:
    """Build an :class:`EmbodimentSpec` from a config node, or from a name/path.

    Args:
        cfg: Either the already-composed ``embodiment`` config node, or an
            embodiment name such as ``"libero_franka"``, or a path to a YAML file.
    """
    if isinstance(cfg, (str, Path)):
        path = Path(cfg)
        if not path.suffix:
            path = configs_root() / "embodiment" / f"{path.name}.yaml"
        if not path.is_file():
            raise FileNotFoundError(f"Embodiment config not found: {path}")
        node = OmegaConf.load(path)
    else:
        node = cfg if isinstance(cfg, DictConfig) else OmegaConf.create(cfg)

    raw = OmegaConf.to_container(node, resolve=True)
    if not isinstance(raw, dict):
        raise TypeError(f"embodiment config must be a mapping, got {type(raw)}")

    for key in ("name", "observation", "action"):
        if key not in raw:
            raise KeyError(
                f"embodiment config missing {key!r}; present keys {sorted(raw)}"
            )

    obs = raw["observation"]
    act = raw["action"]

    observation = ObservationSpec(
        cameras=tuple(obs["cameras"]),
        image_size=tuple(obs["image_size"]),
        d_state=int(obs["d_state"]),
        state_keys=tuple((str(k), int(v)) for k, v in obs.get("state_keys", [])),
    )
    action = ActionSpec(
        d_action=int(act["d_action"]),
        control_mode=str(act["control_mode"]),
        control_rate_hz=float(act["control_rate_hz"]),
        chunk_horizon=int(act["chunk_horizon"]),
        execute_k=int(act["execute_k"]),
        already_normalized=bool(act.get("already_normalized", False)),
        component_keys=tuple((str(k), int(v)) for k, v in act.get("component_keys", [])),
    )
    return EmbodimentSpec(
        name=str(raw["name"]),
        observation=observation,
        action=action,
        notes=str(raw.get("notes", "")),
    )


def load_method_config(cfg: DictConfig | dict | str | Path) -> tuple[str, dict]:
    """Resolve ``configs/method/<name>.yaml`` into ``(name, kwargs)``.

    ``name`` is popped out because it selects the class, while everything else is passed
    to the constructor. :class:`~flowcl.methods.base.BaseMethod` rejects unknown keys, so
    a stale key left in a method YAML fails at construction instead of letting the run
    proceed with a default the config does not mention.
    """
    if isinstance(cfg, (str, Path)):
        path = Path(cfg)
        if not path.suffix:
            path = configs_root() / "method" / f"{path.name}.yaml"
        if not path.is_file():
            raise FileNotFoundError(f"Method config not found: {path}")
        node = OmegaConf.load(path)
    else:
        node = cfg if isinstance(cfg, DictConfig) else OmegaConf.create(cfg)

    raw = OmegaConf.to_container(node, resolve=True)
    if not isinstance(raw, dict):
        raise TypeError(f"method config must be a mapping, got {type(raw)}")
    if "name" not in raw:
        raise KeyError(
            f"method config must declare `name`; present keys {sorted(raw)}"
        )

    kwargs = {k: v for k, v in raw.items() if k != "name"}
    return str(raw["name"]), kwargs
