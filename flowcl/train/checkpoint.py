"""Checkpoint format: policy weights plus everything needed to reproduce a rollout.

A bare ``state_dict`` is not enough. Evaluating a policy requires the embodiment spec
(which fixes ``d_state``, ``d_action`` and ``H``), the policy config (which fixes the
architecture the weights belong to), and the frozen normalization stats (§3.3) — and
an evaluation run that guessed any of those wrong would produce plausible-looking
failure rather than an error. So all four travel together, and :func:`load_checkpoint`
reconstructs the policy rather than leaving the caller to rebuild it by hand.

The stage index and task key are recorded too, because the §8.2 retention matrix is
indexed by them and a mislabelled checkpoint would silently transpose the matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from flowcl.data.spec import EmbodimentSpec
from flowcl.data.stats import NormalizationStats
from flowcl.models.policy import FlowPolicy

# Bump when the payload's meaning changes, so old checkpoints cannot be silently
# reinterpreted.
CHECKPOINT_FORMAT_VERSION = 1


@dataclass(frozen=True)
class LoadedCheckpoint:
    """A checkpoint, reconstructed into usable objects."""

    policy: FlowPolicy
    spec: EmbodimentSpec
    stats: NormalizationStats
    payload: dict

    @property
    def stage(self) -> int | None:
        return self.payload.get("stage")

    @property
    def task_key(self) -> str | None:
        return self.payload.get("task_key")

    @property
    def run_id(self) -> str | None:
        return self.payload.get("run_id")


def save_checkpoint(
    path: str | Path,
    policy: FlowPolicy,
    policy_config: dict,
    spec: EmbodimentSpec,
    stats: NormalizationStats,
    run_id: str,
    stage: int | None = None,
    task_key: str | None = None,
    extra: dict | None = None,
) -> Path:
    """Write a checkpoint.

    Args:
        policy_config: The resolved dict that :func:`flowcl.models.build.build_policy`
            consumed. Stored verbatim so the architecture is reconstructible.
        stats: Frozen stats used for this run; stored so evaluation cannot normalise
            differently from training.
        stage: Curriculum stage index this checkpoint is the end of.
        task_key: The task just trained, for the retention matrix.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "version": CHECKPOINT_FORMAT_VERSION,
            "state_dict": policy.state_dict(),
            "policy_config": dict(policy_config),
            "embodiment": spec.to_dict(),
            "stats": stats.to_dict(),
            "stats_fingerprint": stats.fingerprint(),
            "run_id": run_id,
            "stage": stage,
            "task_key": task_key,
            "extra": dict(extra or {}),
        },
        path,
    )
    return path


def load_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
    pretrained: bool = False,
) -> LoadedCheckpoint:
    """Rebuild the policy, spec and stats from a checkpoint.

    Args:
        pretrained: Whether the encoders should download pretrained weights while
            being constructed. False by default and correct: the checkpoint's own
            ``state_dict`` overwrites them anyway, so downloading would be wasted
            network traffic, and ``load_state_dict(strict=True)`` still proves every
            encoder weight was present in the checkpoint.
    """
    from flowcl.models.build import build_policy

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    payload = torch.load(path, map_location="cpu", weights_only=False)
    version = payload.get("version")
    if version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"{path}: checkpoint format version {version} != "
            f"{CHECKPOINT_FORMAT_VERSION}; refusing to reinterpret it."
        )

    spec = EmbodimentSpec.from_dict(payload["embodiment"])
    stats = NormalizationStats.from_dict(payload["stats"])
    recorded = payload.get("stats_fingerprint")
    if recorded is not None and recorded != stats.fingerprint():
        raise ValueError(
            f"{path}: stored normalization stats do not match their recorded "
            f"fingerprint ({recorded} vs {stats.fingerprint()}); the checkpoint is "
            "corrupt and evaluating it would use the wrong normalisation."
        )

    policy = build_policy(payload["policy_config"], spec, pretrained=pretrained)
    policy.load_state_dict(payload["state_dict"], strict=True)
    policy.to(torch.device(device))
    policy.eval()

    return LoadedCheckpoint(policy=policy, spec=spec, stats=stats, payload=payload)
