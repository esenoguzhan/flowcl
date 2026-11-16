"""Instruction negatives: train a stage with wrong instructions mapped to a hold target.

The rule and its reasons are ``configs/analysis/instruction_negatives.yaml``. In a single-task
stage the instruction is a constant input, so nothing makes the policy select its behaviour by
it; a later task that shares the scene then rewrites the behaviour (``docs/runs/2026-10-05_similar_cl.md``).
This mixin adds, at the stages it is active, a second flow-matching loss on the *same*
observations paired with a wrong instruction and a hold-still target:

    loss = L_flow(o, l, a) + w * L_flow(o, l_neg, a_hold)

* ``l_neg`` is drawn uniformly from a fixed bank (``configs/analysis/<bank>.yaml``) with the
  method's own generator, so the run's batches, flow times and noise are untouched;
* ``a_hold`` (LIBERO ``osc_pose_delta`` only) is a zero pose delta with the gripper at the
  previous step's command (open, -1, at ``t = 0``), repeated over the chunk, mask all ones,
  normalised by the dataset's stats. A zero gripper *command* would not hold: the Panda controller
  integrates the command's sign, so the previous command keeps a saturated gripper where it is.

The bank is encoded once in fp32 at the stage start (a fixed text-cache state). Instructions of
the stage's own dataset are excluded. Inactive stages return the loss tensor unchanged. GPM's
memory is built from the task's dataset as before, so from positives only.

Registered variants: ``gpm_neg`` (display ``gpm_projected_adam_neg``) and ``allowlist_ft_neg``.
"""

from __future__ import annotations

import numpy as np
import torch

from flowcl.methods.allowlist_ft import AllowlistFT
from flowcl.methods.base import register_method
from flowcl.methods.gpm import GPM
from flowcl.utils.run import file_sha256

HOLD_CONTROL_MODE = "osc_pose_delta"
GRIPPER_OPEN = -1.0


def bank_path(bank: str):
    """``configs/analysis/<bank>.yaml``, or ``bank`` itself if it is a YAML path."""
    from pathlib import Path

    from flowcl.utils.libero_paths import repo_root

    path = Path(bank)
    if path.suffix != ".yaml":
        path = repo_root() / "configs" / "analysis" / f"{bank}.yaml"
    return path


def load_bank(bank: str) -> dict:
    """The bank's strings (``noun_swaps`` then ``unrelated``), its forbidden words and SHA-256.

    Raises on an empty bank, a duplicate, a string that is not lower case and stripped, or one
    containing a forbidden word (the held-out instructions' nouns)."""
    from omegaconf import OmegaConf

    path = bank_path(bank)
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    forbidden = [str(w).lower() for w in raw.get("forbidden_words") or []]
    strings = [str(s) for s in (raw.get("noun_swaps") or [])] + [str(s) for s in (raw.get("unrelated") or [])]
    problems = []
    if not strings:
        problems.append("no strings")
    if len(set(strings)) != len(strings):
        problems.append("duplicates")
    for s in strings:
        if s != s.strip().lower():
            problems.append(f"{s!r} is not lower case and stripped")
        hits = [w for w in forbidden if w in s.split()]
        if hits:
            problems.append(f"{s!r} contains forbidden {hits}")
    if problems:
        raise ValueError(f"{path}: invalid negative bank: {problems}")
    return {"path": str(path), "sha256": file_sha256(path), "strings": strings, "forbidden_words": forbidden,
            "n_noun_swaps": len(raw.get("noun_swaps") or [])}


def gripper_index(spec) -> int:
    """The gripper channel of a LIBERO action (the last component, width 1); raises otherwise."""
    keys = tuple(tuple(k) for k in spec.action.component_keys)
    if spec.action.control_mode != HOLD_CONTROL_MODE:
        raise ValueError(f"the hold target is defined for {HOLD_CONTROL_MODE!r} only, not "
                         f"{spec.action.control_mode!r} (a Piper hold needs its own definition)")
    if not keys or keys[-1] != ("gripper", 1):
        raise ValueError(f"expected the last action component to be ('gripper', 1), got {keys}")
    return spec.action.d_action - 1


def hold_rows(episode_action: np.ndarray, g: int) -> np.ndarray:
    """Per timestep, the raw hold action: zeros, gripper = the previous step's command (open at 0)."""
    action = np.asarray(episode_action, dtype=np.float32)
    rows = np.zeros_like(action)
    rows[0, g] = GRIPPER_OPEN
    rows[1:, g] = action[:-1, g]
    return rows


class InstructionNegatives:
    """Mixin: the negative-instruction loss at ``negative_stages``; see the module docstring."""

    def __init__(self, *, negative_bank: str, negative_weight: float = 1.0, negative_stages=(0,),
                 negative_log_interval: int = 100, **kwargs) -> None:
        super().__init__(**kwargs)
        if not (isinstance(negative_weight, (int, float)) and negative_weight >= 0):
            raise ValueError(f"negative_weight must be >= 0, got {negative_weight!r}")
        if int(negative_log_interval) < 1:
            raise ValueError("negative_log_interval must be >= 1")
        self.negative_bank = str(negative_bank)
        self.negative_weight = float(negative_weight)
        self.negative_stages = tuple(int(s) for s in negative_stages)
        self.negative_log_interval = int(negative_log_interval)
        self._bank = load_bank(self.negative_bank)
        self.negative_logs: dict[str, dict] = {}
        self._neg_active = False
        self._neg_task: int | None = None
        self._neg_step = 0
        self._neg_strings: list[str] = []
        self._neg_rows: list[torch.Tensor] = []
        self._neg_gen: torch.Generator | None = None
        self._neg_sum = None
        self._neg_n = 0

    def negative_config(self) -> dict:
        return {"negative_bank": self.negative_bank, "negative_bank_sha256": self._bank["sha256"],
                "negative_weight": self.negative_weight, "negative_stages": list(self.negative_stages),
                "negative_log_interval": self.negative_log_interval}

    # ---- hooks -----------------------------------------------------------------------------------

    def on_task_start(self, policy, task_idx, *, context) -> None:
        from flowcl.utils.seeding import derive_seed

        self._neg_task, self._neg_step = int(task_idx), 0
        self._neg_active = int(task_idx) in self.negative_stages
        if not self._neg_active:
            self.negative_logs[str(task_idx)] = {"active": False}
            super().on_task_start(policy, task_idx, context=context)
            return
        dataset = context.dataset
        if dataset is None or context.seed_namespace_run_id is None:
            raise RuntimeError("instruction negatives need the stage's dataset and the seed namespace")
        g = gripper_index(dataset.spec)
        own = {ep.language.strip().lower() for ep in dataset.episodes}
        self._neg_strings = [s for s in self._bank["strings"] if s not in own]
        excluded = [s for s in self._bank["strings"] if s in own]
        if not self._neg_strings:
            raise ValueError("every bank string is one of the stage's own instructions")
        self._neg_rows = [torch.from_numpy(np.ascontiguousarray(dataset.stats.normalize_action(
            hold_rows(ep.action, g)))) for ep in dataset.episodes]
        self._neg_gen = torch.Generator(device="cpu").manual_seed(
            derive_seed(context.seed_namespace_run_id, f"instruction_negatives::{context.task_key}", int(task_idx)))
        precision = {}
        if policy is not None:
            device_type = torch.device(context.device).type
            with torch.no_grad(), torch.autocast(device_type=device_type, enabled=False):
                for text in self._neg_strings:
                    precision[text] = "cached_before" if text in policy.text_encoder._cache else "fp32"
                    policy.text_encoder([text])
        self._neg_sum, self._neg_n = None, 0
        self.negative_logs[str(task_idx)] = {
            "active": True, "task_key": context.task_key, "bank_sha256": self._bank["sha256"],
            "n_strings": len(self._neg_strings), "excluded_own": excluded, "text_cache": precision,
            "gripper_index": g,
            "first_gripper_commands": [float(ep.action[0, g]) for ep in dataset.episodes],
            "steps": {},
        }
        super().on_task_start(policy, task_idx, context=context)

    def negative_batch(self, batch: dict) -> dict:
        """The same observations with drawn negative instructions and hold targets."""
        n = len(batch["language"])
        draw = torch.randint(len(self._neg_strings), (n,), generator=self._neg_gen).tolist()
        episodes = batch["episode_idx"].tolist()
        times = batch["t"].tolist()
        rows = torch.stack([self._neg_rows[e][t] for e, t in zip(episodes, times)])
        horizon = batch["actions"].shape[1]
        actions = rows[:, None, :].expand(n, horizon, rows.shape[1]).contiguous().to(batch["actions"].device)
        return {"images": batch["images"], "state": batch["state"], "actions": actions,
                "action_mask": torch.ones_like(batch["action_mask"]),
                "language": [self._neg_strings[i] for i in draw]}

    def modify_loss(self, loss, batch, policy, *, outputs=None):
        loss = super().modify_loss(loss, batch, policy, outputs=outputs)
        if not self._neg_active:
            return loss
        neg = policy(self.negative_batch(batch), generator=self._neg_gen)["loss"]
        value = neg.detach().float()
        self._neg_sum = value if self._neg_sum is None else self._neg_sum + value
        self._neg_n += 1
        return loss + self.negative_weight * neg

    def after_step(self, policy, step_meta) -> None:
        super().after_step(policy, step_meta)
        if not self._neg_active:
            return
        if self._neg_n and (self._neg_step == 0 or (self._neg_step + 1) % self.negative_log_interval == 0):
            self.negative_logs[str(self._neg_task)]["steps"][str(self._neg_step)] = {
                "negative_loss_mean": float(self._neg_sum) / self._neg_n, "n": self._neg_n}
            self._neg_sum, self._neg_n = None, 0
        self._neg_step += 1

    def on_task_end(self, policy, task_idx, *, context) -> None:
        if self._neg_active:
            log = self.negative_logs[str(task_idx)]
            if self._neg_n:
                log["steps"][str(self._neg_step - 1)] = {
                    "negative_loss_mean": float(self._neg_sum) / self._neg_n, "n": self._neg_n}
            log["n_steps"] = self._neg_step
        self._neg_active = False
        self._neg_rows, self._neg_sum, self._neg_n = [], None, 0
        super().on_task_end(policy, task_idx, context=context)

    def state_dict(self) -> dict:
        return {**super().state_dict(), "instruction_negatives": {**self.negative_config(),
                                                                   "logs": self.negative_logs}}


@register_method
class GPMNeg(InstructionNegatives, GPM):
    """GPM (``gpm_projected_adam``) with instruction negatives at ``negative_stages``."""

    name = "gpm_neg"

    @property
    def display_name(self) -> str:
        return GPM.display_name.fget(self) + "_neg"

    def config(self) -> dict:
        return {**GPM.config(self), **self.negative_config()}


@register_method
class AllowlistFTNeg(InstructionNegatives, AllowlistFT):
    """allowlist_ft with instruction negatives at ``negative_stages`` (the no-projection control)."""

    name = "allowlist_ft_neg"
