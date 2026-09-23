"""The §6 ContinualMethod interface, its registry, and seq_ft as a real method."""

from __future__ import annotations

import pytest
import torch

from flowcl.methods.base import (
    METHOD_REGISTRY,
    BaseMethod,
    ContinualMethod,
    build_method,
    register_method,
)
from flowcl.methods.seq_ft import SeqFT


def test_seq_ft_is_registered_and_buildable():
    method = build_method("seq_ft")
    assert isinstance(method, SeqFT)
    assert method.name == "seq_ft"


def test_unknown_method_lists_the_registered_ones():
    with pytest.raises(ValueError, match="Unknown method"):
        build_method("definitely_not_a_method")


def test_base_method_satisfies_the_protocol():
    """§6 is a Protocol, so conformance is checkable rather than assumed."""
    assert isinstance(SeqFT(), ContinualMethod)


def test_every_spec_six_hook_exists_with_the_documented_signature():
    """Pins the §6 interface against drift.

    If a hook is renamed or loses a parameter, the runner would stop calling it and the
    method would silently become a no-op -- which for a projection method means it looks
    like it ran and produced no protection.
    """
    import inspect

    # Deviation 4 (documented in flowcl/methods/base.py): the lifecycle hooks take
    # (policy, task_idx, *, context) instead of §6's (task_idx, policy, dataset).
    expected = {
        "on_task_start": ["self", "policy", "task_idx", "context"],
        "build_batch": ["self", "dataset", "task_idx"],
        "modify_loss": ["self", "loss", "batch", "policy", "outputs"],
        "modify_gradients": ["self", "policy", "batch_meta"],
        "after_step": ["self", "policy", "step_meta"],
        "on_task_end": ["self", "policy", "task_idx", "context"],
        "save_artifacts": ["self", "directory", "task_idx", "context"],
        "state_dict": ["self"],
    }
    for hook, params in expected.items():
        signature = inspect.signature(getattr(BaseMethod, hook))
        assert list(signature.parameters) == params, hook


def test_modify_loss_outputs_is_keyword_only():
    """The documented deviation: §6's positional signature stays (loss, batch, policy)."""
    import inspect

    signature = inspect.signature(BaseMethod.modify_loss)
    assert signature.parameters["outputs"].kind is inspect.Parameter.KEYWORD_ONLY


def test_seq_ft_hooks_are_all_no_ops():
    """B1 must change nothing, or it is not a reference."""
    method = SeqFT()
    loss = torch.tensor(1.25, requires_grad=True)

    assert method.build_batch(dataset=None, task_idx=0) is None
    assert method.modify_loss(loss, batch={}, policy=None) is loss
    assert method.on_task_start(None, 0, context=None) is None
    assert method.modify_gradients(None, {}) is None
    assert method.on_task_end(None, 0, context=None) is None
    assert method.save_artifacts(None, 0, context=None) == []


def test_seq_ft_stores_nothing_and_is_exemplar_free():
    """§8.2's memory column and §6's exemplar-free flag, for the reference row."""
    method = SeqFT()
    assert method.stored_bytes() == 0
    assert method.is_exemplar_free is True


def test_method_rejects_unknown_config_keys():
    """A stale key in configs/method/*.yaml must fail loudly, not use a default."""
    with pytest.raises(TypeError, match="unexpected config keys"):
        build_method("seq_ft", lambda_ewc=0.5)


def test_state_dict_identifies_the_method():
    assert SeqFT().state_dict()["name"] == "seq_ft"


def test_registering_a_nameless_class_is_rejected():
    class Nameless(BaseMethod):
        pass

    with pytest.raises(ValueError, match="distinct class-level"):
        register_method(Nameless)


def test_registering_a_duplicate_name_is_rejected():
    class Impostor(BaseMethod):
        name = "seq_ft"

    with pytest.raises(ValueError, match="already registered"):
        register_method(Impostor)


def test_registry_is_populated_by_importing_all():
    import flowcl.methods.all  # noqa: F401

    assert "seq_ft" in METHOD_REGISTRY


def test_method_config_loads_into_name_and_kwargs():
    from flowcl.data.config import load_method_config

    name, kwargs = load_method_config("seq_ft")
    assert name == "seq_ft"
    assert kwargs == {}, "B1 must have no hyperparameters"


def test_method_config_requires_a_name():
    from flowcl.data.config import load_method_config

    with pytest.raises(KeyError, match="must declare `name`"):
        load_method_config({"lambda": 1.0})
