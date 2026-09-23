"""The §6 ``ContinualMethod`` interface and its registry.

Spec §6 gives the interface verbatim::

    class ContinualMethod(Protocol):
        def on_task_start(self, task_idx, policy, dataset) -> None: ...
        def modify_loss(self, loss, batch, policy) -> Tensor: ...        # EWC, ConSFT
        def modify_gradients(self, policy, batch_meta) -> None: ...      # GPM, SGP, s-binned
        def build_batch(self, dataset, task_idx) -> Batch: ...           # replay mixing
        def on_task_end(self, task_idx, policy, dataset) -> None: ...    # basis/Fisher update
        def state_dict(self) -> dict: ...

Three deliberate additions, all documented here because §11 requires deviations to be
written down rather than absorbed:

1. ``modify_loss`` takes an extra keyword-only ``outputs``. ConSFT scales the loss by a
   confidence derived from the model's *prediction*, which is not recoverable from
   ``(loss, batch, policy)`` alone; recomputing the forward pass to get it would double
   the cost of every step. The positional signature is unchanged.
2. ``build_batch`` may return ``None`` to mean "use the runner's own dataloader". Only
   ``replay`` needs to own sampling, and forcing every other method to reimplement
   batching would duplicate the collate and worker setup five times.
3. ``after_step(policy, step_meta)`` runs once per optimiser step, *after*
   ``optimizer.step()``. §7.3 projects gradients before ``step()``, but with Adam that
   does not make the *applied* update orthogonal: its per-coordinate scaling and AdamW's
   decoupled weight decay move weights back into protected directions (Gate 3 §6.2).
   Projecting the realised update needs a hook after the step. Per-step order in
   :func:`flowcl.train.trainer.train_one_task`::

       backward -> unscale_ -> modify_gradients -> clip -> step -> update -> after_step

:class:`BaseMethod` implements every hook as a no-op, so a method subclasses it and
overrides only what it changes. That is also why ``seq_ft`` is a real class with no
body rather than ``method=None``: the runner then has exactly one code path, and the
B1 baseline is exercised by the same machinery as everything else.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch

from flowcl.data.dataset import ChunkedActionDataset
from flowcl.models.policy import FlowPolicy


@runtime_checkable
class ContinualMethod(Protocol):
    """Spec §6. See the module docstring for the two documented additions."""

    name: str

    def on_task_start(
        self, task_idx: int, policy: FlowPolicy, dataset: ChunkedActionDataset
    ) -> None:
        """Called once before a stage's optimisation begins."""

    def build_batch(
        self, dataset: ChunkedActionDataset, task_idx: int
    ) -> dict | None:
        """Produce a training batch, or ``None`` to use the runner's dataloader."""

    def modify_loss(
        self,
        loss: torch.Tensor,
        batch: dict,
        policy: FlowPolicy,
        *,
        outputs: dict | None = None,
    ) -> torch.Tensor:
        """Add regularisation or reweighting to the flow-matching loss."""

    def modify_gradients(self, policy: FlowPolicy, batch_meta: dict) -> None:
        """Rewrite ``.grad`` in place, after ``backward()`` and before ``step()``."""

    def after_step(self, policy: FlowPolicy, step_meta: dict) -> None:
        """Called once after each ``optimizer.step()`` (documented addition 3)."""

    def on_task_end(
        self, task_idx: int, policy: FlowPolicy, dataset: ChunkedActionDataset
    ) -> None:
        """Called once after a stage, e.g. to update a basis or a Fisher estimate."""

    def state_dict(self) -> dict:
        """Everything needed to resume, and the §8.2 memory accounting."""


class BaseMethod:
    """No-op implementation of every §6 hook.

    Subclass and override only the hooks a method actually uses. The no-op defaults are
    not merely convenient: they mean the runner calls the same five hooks in the same
    order for every method, so a difference between two methods' results cannot come
    from a difference in control flow.
    """

    name = "base"

    def __init__(self, **kwargs) -> None:
        if kwargs:
            # Silently ignoring an unknown config key is how a method ends up running
            # with default hyperparameters while its config claims otherwise.
            raise TypeError(
                f"{type(self).__name__} received unexpected config keys "
                f"{sorted(kwargs)}"
            )

    # ---- §6 hooks --------------------------------------------------------------

    def on_task_start(self, task_idx, policy, dataset) -> None:
        return None

    def build_batch(self, dataset, task_idx) -> dict | None:
        return None

    def modify_loss(self, loss, batch, policy, *, outputs=None):
        return loss

    def modify_gradients(self, policy, batch_meta) -> None:
        return None

    def after_step(self, policy, step_meta) -> None:
        return None

    def on_task_end(self, task_idx, policy, dataset) -> None:
        return None

    def state_dict(self) -> dict:
        return {"name": self.name}

    # ---- §8.2 systems reporting ------------------------------------------------

    def stored_bytes(self) -> int:
        """Extra memory this method persists across tasks, in bytes.

        §8.2 requires reporting "stored memory for bases and buffers in MB". Zero for
        methods that store nothing, which is the honest number for ``seq_ft`` and the
        thing ``replay`` and ``gpm`` must be compared against.
        """
        return 0

    def describe(self) -> str:
        return f"{self.name} (stores {self.stored_bytes() / 1e6:.2f} MB)"

    # ---- exemplar-free accounting ----------------------------------------------

    @property
    def is_exemplar_free(self) -> bool:
        """Whether the method keeps raw data from earlier tasks.

        §6 flags ``replay`` as violating exemplar-free and requires that to appear in
        the results table. Making it a property of the method means the table cannot be
        produced without the flag.
        """
        return True


METHOD_REGISTRY: dict[str, type] = {}


def register_method(cls: type) -> type:
    """Register a method class under its ``name``, rejecting duplicates."""
    name = getattr(cls, "name", None)
    if not name or name == "base":
        raise ValueError(
            f"{cls.__name__} must set a distinct class-level `name` to be registered"
        )
    if name in METHOD_REGISTRY and METHOD_REGISTRY[name] is not cls:
        raise ValueError(
            f"method name {name!r} is already registered to "
            f"{METHOD_REGISTRY[name].__name__}"
        )
    METHOD_REGISTRY[name] = cls
    return cls


def build_method(name: str, **kwargs) -> ContinualMethod:
    """Instantiate a registered method.

    Importing :mod:`flowcl.methods.all` populates the registry; this function does that
    import lazily so a partially-built method module cannot break unrelated code.
    """
    import flowcl.methods.all  # noqa: F401

    if name not in METHOD_REGISTRY:
        raise ValueError(
            f"Unknown method {name!r}; registered methods are "
            f"{sorted(METHOD_REGISTRY)}"
        )
    return METHOD_REGISTRY[name](**kwargs)
