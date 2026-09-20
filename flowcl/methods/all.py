"""Import every method so :data:`~flowcl.methods.base.METHOD_REGISTRY` is populated.

Separate from ``flowcl/methods/__init__.py`` so that importing the *interface* does not
drag in every implementation — which matters because later methods pull in optimiser
and subspace machinery that the interface itself must not depend on.

Methods appear in the §6 implementation order.
"""

from flowcl.methods.seq_ft import SeqFT

__all__ = ["SeqFT"]
