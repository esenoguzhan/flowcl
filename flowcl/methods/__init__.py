"""Continual-learning methods behind one interface (spec §6).

The sequential runner is method-agnostic: it never branches on which method is
active. Every method is a :class:`~flowcl.methods.base.ContinualMethod`, and new ones
register themselves in :data:`~flowcl.methods.base.METHOD_REGISTRY` rather than adding
a case to the runner.
"""

from flowcl.methods.base import (
    METHOD_REGISTRY,
    BaseMethod,
    ContinualMethod,
    build_method,
    register_method,
)

__all__ = [
    "METHOD_REGISTRY",
    "BaseMethod",
    "ContinualMethod",
    "build_method",
    "register_method",
]
