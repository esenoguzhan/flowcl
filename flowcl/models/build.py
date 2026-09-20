"""Build a :class:`~flowcl.models.policy.FlowPolicy` from config.

Keeps config key names in one place (§11: configs are the only home for experiment
parameters), so no training or analysis code constructs a policy by hand.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from flowcl.data.spec import EmbodimentSpec
from flowcl.models.flow_head import build_s_sampler
from flowcl.models.policy import FlowPolicy
from flowcl.utils.libero_paths import repo_root


def policy_config_path(name: str) -> Path:
    return repo_root() / "configs" / "policy" / f"{name}.yaml"


def load_policy_config(cfg: DictConfig | dict | str | Path) -> dict:
    """Resolve a policy config from a node, a name, or a path."""
    if isinstance(cfg, (str, Path)):
        path = Path(cfg)
        if not path.suffix:
            path = policy_config_path(path.name)
        if not path.is_file():
            raise FileNotFoundError(f"Policy config not found: {path}")
        node = OmegaConf.load(path)
    else:
        node = cfg if isinstance(cfg, DictConfig) else OmegaConf.create(cfg)

    raw = OmegaConf.to_container(node, resolve=True)
    if not isinstance(raw, dict):
        raise TypeError(f"policy config must be a mapping, got {type(raw)}")
    return raw


def build_policy(
    cfg: DictConfig | dict | str | Path,
    spec: EmbodimentSpec,
    pretrained: bool | None = None,
) -> FlowPolicy:
    """Instantiate the policy described by ``cfg`` for embodiment ``spec``.

    Args:
        cfg: Policy config node, name (e.g. ``"flowpolicy_base"``), or path.
        spec: Embodiment spec, which supplies ``d_state``, ``d_action`` and ``H``.
        pretrained: Override the config's ``pretrained`` flag. Tests set False to
            avoid network access.
    """
    raw = load_policy_config(cfg)

    sampler_cfg = dict(raw.get("s_sampler") or {"name": "uniform"})
    sampler_name = sampler_cfg.pop("name")
    s_sampler = build_s_sampler(sampler_name, **sampler_cfg)

    policy = FlowPolicy(
        spec=spec,
        vision_backbone=raw.get("vision_backbone", "dinov2_s"),
        text_backbone=raw.get("text_backbone", "clip_b"),
        d_model=int(raw.get("d_model", 512)),
        n_trunk_layers=int(raw.get("n_trunk_layers", 8)),
        n_heads=int(raw.get("n_heads", 8)),
        n_decoder_layers=int(raw.get("n_decoder_layers", 4)),
        n_context_tokens=int(raw.get("n_context_tokens", 32)),
        mlp_ratio=float(raw.get("mlp_ratio", 4.0)),
        max_tokens=int(raw.get("max_tokens", 1024)),
        s_sampler=s_sampler,
        pretrained=raw.get("pretrained", True) if pretrained is None else pretrained,
        euler_steps=int(raw.get("euler_steps", 10)),
    )
    policy.assert_registry_complete()
    return policy
