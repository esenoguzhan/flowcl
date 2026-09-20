"""``seq_ft``: plain sequential fine-tuning. Baseline B1.

Spec §6: "plain sequential fine-tuning. Baseline B1 and the reference for every
forgetting number."

It has no body, and that is the point. Every forgetting number in the thesis is
measured *relative* to this, so it must differ from the other methods in nothing but
the absence of their mechanism — same runner, same hooks, same batch construction,
same seeds. Implementing it as ``method=None`` would give it a shorter code path than
its competitors, and any difference in results would then be partly attributable to
that rather than to the mechanism under study.
"""

from __future__ import annotations

from flowcl.methods.base import BaseMethod, register_method


@register_method
class SeqFT(BaseMethod):
    """Train each task in turn with no continual-learning mechanism at all (B1)."""

    name = "seq_ft"
