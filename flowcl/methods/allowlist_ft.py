"""``allowlist_ft``: sequential fine-tuning restricted to GPM's trainable set (a control).

GPM differs from ``seq_ft`` in two ways from Task 2 on: every parameter outside the §7.4
allowlist (the projectable registry weights) is frozen, and the allowlisted weights are
projected. ``allowlist_ft`` applies the first and not the second, so ``gpm`` against
``allowlist_ft`` isolates projection's cost and ``allowlist_ft`` against ``seq_ft``
isolates freezing's. Task 1 trains every parameter, as in GPM (paper Alg. 1), so its
stage-0 model equals ``seq_ft``'s and ``gpm``'s bit for bit. Nothing persists across tasks.
"""

from __future__ import annotations

from flowcl.methods.base import BaseMethod, register_method
from flowcl.methods.gpm import freeze_to_allowlist


@register_method
class AllowlistFT(BaseMethod):
    """Fine-tune every parameter on Task 1, only GPM's allowlist from Task 2 on."""

    name = "allowlist_ft"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.freeze_reports: dict[int, dict] = {}

    def on_task_start(self, policy, task_idx, *, context) -> None:
        if task_idx >= 1:
            self.freeze_reports[task_idx] = freeze_to_allowlist(policy)

    def state_dict(self) -> dict:
        return {"name": self.name,
                "freeze_reports": {str(k): v for k, v in self.freeze_reports.items()}}
