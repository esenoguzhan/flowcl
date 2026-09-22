"""§7.1 activation capture and the gradient-reachability probe."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from flowcl.analysis.hooks import (
    KIND_ACTION,
    KIND_CONTEXT,
    ActivationCapture,
    CaptureLayer,
    expected_reachability,
    gradient_reachability,
    probe_policy_reachability,
)
from flowcl.data.config import load_embodiment_spec
from flowcl.models.build import build_policy
from flowcl.models.trunk import SelfAttention


def toy_capture(module, kind=KIND_CONTEXT, tokens=100, seed=0, **kwargs):
    return ActivationCapture(
        module,
        [CaptureLayer("lin", module, kind)],
        tokens_per_sample=tokens,
        subsample_seed=seed,
        **kwargs,
    )


def run(capture, module, x, s=None, mask=None):
    s = s if s is not None else torch.full((x.shape[0],), 0.5)
    with torch.no_grad(), capture.batch_context(s, mask):
        module(x)


# ---- capture -------------------------------------------------------------------


def test_captured_gram_equals_hand_computed_xtx():
    torch.manual_seed(0)
    lin = nn.Linear(4, 3)
    x = torch.randn(2, 5, 4)
    with toy_capture(lin) as cap:
        run(cap, lin, x)
    flat = x.reshape(-1, 4).double()
    acc = cap.accumulators["lin"]
    torch.testing.assert_close(acc.gram["all"], flat.T @ flat)
    assert acc.n["all"] == 10


def test_orientation_assertion_fires_on_wrong_width():
    lin = nn.Linear(4, 3)
    with toy_capture(lin) as cap:
        with pytest.raises(RuntimeError, match="in_features"):
            run(cap, lin, torch.randn(2, 5, 6))


def test_hook_outside_batch_context_raises():
    lin = nn.Linear(4, 3)
    with toy_capture(lin):
        with pytest.raises(RuntimeError, match="batch_context"):
            lin(torch.randn(1, 2, 4))


def test_hooks_are_removed_on_exit():
    lin = nn.Linear(4, 3)
    with toy_capture(lin) as cap:
        assert len(lin._forward_pre_hooks) == 1
        assert cap.active
    assert len(lin._forward_pre_hooks) == 0
    assert not cap.active
    assert getattr(lin, "_flowcl_active_capture", None) is None


def test_subsampling_is_deterministic_under_a_fixed_seed():
    torch.manual_seed(0)
    lin = nn.Linear(4, 3)
    x = torch.randn(3, 10, 4)

    def gram(seed):
        with toy_capture(lin, tokens=3, seed=seed) as cap:
            run(cap, lin, x)
        acc = cap.accumulators["lin"]
        assert acc.n["all"] == 9
        return acc.gram["all"]

    torch.testing.assert_close(gram(7), gram(7))
    assert not torch.allclose(gram(7), gram(8))


def test_valid_view_drops_masked_rows_and_all_view_keeps_them():
    torch.manual_seed(0)
    lin = nn.Linear(3, 2)
    x = torch.randn(2, 4, 3)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.float32)
    with toy_capture(lin, kind=KIND_ACTION) as cap:
        run(cap, lin, x, mask=mask)
    acc = cap.accumulators["lin"]

    valid = x[mask.bool()].double()
    everything = x.reshape(-1, 3).double()
    torch.testing.assert_close(acc.gram["valid"], valid.T @ valid)
    torch.testing.assert_close(acc.gram["all"], everything.T @ everything)
    assert (acc.n["valid"], acc.n["all"]) == (5, 8)


def test_s_bin_grams_sum_to_pooled_and_skip_s_independent_layers():
    torch.manual_seed(0)
    action = nn.Linear(3, 2)
    context = nn.Linear(3, 2)
    edges = (0.0, 0.25, 0.5, 0.75, 1.0)
    cap = ActivationCapture(
        action,
        [
            CaptureLayer("action", action, KIND_ACTION),
            CaptureLayer("context", context, KIND_CONTEXT),
        ],
        tokens_per_sample=100,
        subsample_seed=0,
        s_bin_edges=edges,
        binned_views={"action": "valid"},
    )
    x = torch.randn(4, 3, 3)
    s = torch.tensor([0.1, 0.25, 0.3, 1.0])  # bins 0, 1, 1, 3
    mask = torch.ones(4, 3)
    mask[1, 2] = 0
    with cap:
        with torch.no_grad(), cap.batch_context(s, mask):
            action(x)
            context(x)

    acc = cap.accumulators["action"]
    torch.testing.assert_close(sum(acc.binned_gram), acc.gram["valid"])
    assert acc.binned_n == [3, 5, 0, 3]
    assert cap.accumulators["context"].binned_gram is None


def test_s_binning_requires_a_view_for_every_s_dependent_layer():
    lin = nn.Linear(3, 2)
    with pytest.raises(ValueError, match="binned_views"):
        toy_capture(lin, kind=KIND_ACTION, s_bin_edges=(0.0, 0.5, 1.0))


# ---- reachability ---------------------------------------------------------------


def masked_loss(y, target, mask):
    return (((y - target) ** 2).sum(-1) * mask).sum() / mask.sum()


def test_positionwise_model_padded_rows_unreachable():
    torch.manual_seed(0)
    a, b = nn.Linear(3, 4), nn.Linear(4, 2)
    model = nn.Sequential(a, nn.GELU(), b)
    x, target = torch.randn(2, 5, 3), torch.randn(2, 5, 2)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.float32)
    result = gradient_reachability(
        model,
        [CaptureLayer("a", a, KIND_ACTION), CaptureLayer("b", b, KIND_ACTION)],
        lambda: masked_loss(model(x), target, mask),
        mask == 0,
    )
    assert result == {"a": False, "b": False}


def test_unmasked_self_attention_makes_padded_rows_reachable():
    torch.manual_seed(0)
    pre, attn, head = nn.Linear(4, 4), SelfAttention(4, 1), nn.Linear(4, 2)
    model = nn.ModuleList([pre, attn, head])
    x, target = torch.randn(2, 5, 4), torch.randn(2, 5, 2)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.float32)
    layers = [
        CaptureLayer("pre", pre, KIND_ACTION),
        CaptureLayer("q", attn.q_proj, KIND_ACTION),
        CaptureLayer("k", attn.k_proj, KIND_ACTION),
        CaptureLayer("v", attn.v_proj, KIND_ACTION),
        CaptureLayer("out", attn.out_proj, KIND_ACTION),
        CaptureLayer("head", head, KIND_ACTION),
    ]
    result = gradient_reachability(
        model, layers, lambda: masked_loss(head(attn(pre(x))), target, mask), mask == 0
    )
    assert result == {
        "pre": True,  # feeds K/V of valid queries
        "q": False,
        "k": True,
        "v": True,
        "out": False,
        "head": False,
    }


def test_zero_initialised_layer_is_classified_by_its_output_gradient():
    """Regression guard: the input gradient W^T δ of a zero layer is zero, δ is not."""
    torch.manual_seed(0)
    first, last = nn.Linear(3, 4), nn.Linear(4, 2)
    nn.init.zeros_(last.weight)
    model = nn.Sequential(first, last)
    x, target = torch.randn(2, 5, 3), torch.randn(2, 5, 2)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.float32)
    loss = lambda: masked_loss(model(x), target, mask)  # noqa: E731

    assert gradient_reachability(
        model, [CaptureLayer("last", last, KIND_ACTION)], loss, mask == 0
    ) == {"last": False}
    # Upstream of the zero layer δ really is zero everywhere: refuse, do not guess.
    with pytest.raises(RuntimeError, match="uninformative"):
        gradient_reachability(
            model, [CaptureLayer("first", first, KIND_ACTION)], loss, mask == 0
        )


def test_probe_restores_parameter_grads():
    torch.manual_seed(0)
    a = nn.Linear(3, 2)
    sentinel = torch.full_like(a.weight, 7.0)
    a.weight.grad = sentinel.clone()
    x, target = torch.randn(1, 4, 3), torch.randn(1, 4, 2)
    mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.float32)
    weight_before = a.weight.detach().clone()
    gradient_reachability(
        a, [CaptureLayer("a", a, KIND_ACTION)],
        lambda: masked_loss(a(x), target, mask), mask == 0,
    )
    torch.testing.assert_close(a.weight.grad, sentinel)
    assert a.bias.grad is None
    torch.testing.assert_close(a.weight, weight_before)
    assert len(a._forward_hooks) == 0


def test_probe_requires_padded_and_valid_positions():
    a = nn.Linear(3, 2)
    with pytest.raises(ValueError, match="both padded and valid"):
        gradient_reachability(
            a, [CaptureLayer("a", a, KIND_ACTION)], lambda: a(torch.ones(1, 2, 3)).sum(),
            torch.zeros(1, 2, dtype=torch.bool),
        )


# ---- the real policy ------------------------------------------------------------


@pytest.fixture(scope="module")
def spec():
    return load_embodiment_spec("libero_franka")


@pytest.fixture(scope="module")
def policy(spec):
    """flowpolicy_small with *randomised* weights.

    At init the AdaLN gates and ``action_out`` are zero, so δ vanishes on every decoder
    layer and the probe would (correctly) refuse to answer.
    """
    torch.manual_seed(0)
    p = build_policy("flowpolicy_small", spec, pretrained=False)
    for param in p.parameters():
        if param.requires_grad:
            nn.init.normal_(param, std=0.05)
    p.eval()
    return p


def make_batch(spec, batch_size=3):
    h, w = spec.observation.image_size
    horizon = spec.action.chunk_horizon
    mask = torch.ones(batch_size, horizon)
    mask[0, 5:] = 0
    mask[1, 12:] = 0
    return {
        "images": {
            c: torch.randint(0, 255, (batch_size, h, w, 3), dtype=torch.uint8)
            for c in spec.cameras
        },
        "state": torch.randn(batch_size, spec.d_state),
        "actions": torch.randn(batch_size, horizon, spec.d_action),
        "action_mask": mask,
        "language": ["pick up the milk"] * batch_size,
    }


def test_policy_probe_matches_the_hand_derived_rule(policy, spec):
    batch = make_batch(spec)
    measured = probe_policy_reachability(
        policy, batch, torch.Generator().manual_seed(0)
    )
    expected = expected_reachability(
        policy.registry_names(), len(policy.flow_head.blocks)
    )
    assert measured == expected
    assert expected["flow_head.action_in"] is True
    assert expected["flow_head.action_out"] is False
    assert all(p.grad is None for p in policy.parameters())


def test_policy_probe_refuses_on_an_untrained_zero_gated_model(spec):
    torch.manual_seed(0)
    fresh = build_policy("flowpolicy_small", spec, pretrained=False)
    with pytest.raises(RuntimeError, match="uninformative"):
        probe_policy_reachability(fresh, make_batch(spec), torch.Generator().manual_seed(0))


def test_token_type_counts_match_the_layout(policy, spec):
    batch = make_batch(spec)
    b = batch["actions"].shape[0]
    cap = ActivationCapture.for_policy(policy, tokens_per_sample=10_000, subsample_seed=0)
    with cap:
        cap.forward_policy(
            policy, batch, torch.full((b,), 0.5), torch.randn_like(batch["actions"])
        )
    acc = cap.accumulators

    n_vision = 2 * 82  # two cameras, DINOv2-S at 128 px: CLS + 9x9 patches
    trunk = acc["trunk.blocks.0.attn.q_proj"].type_count_dict("all")
    assert trunk == {
        "context_query": 32 * b,
        "state": b,
        "language": 32 * b,
        "vision": n_vision * b,
    }
    assert acc["trunk.state_projection"].type_count_dict("all") == {"state": b}
    assert acc["flow_head.blocks.0.cross_attn.k_proj"].type_count_dict("all") == {
        "context": 32 * b
    }
    action = acc["flow_head.action_in"]
    assert action.type_count_dict("all") == {"action": 16 * b}
    assert action.type_count_dict("valid") == {"action": int(batch["action_mask"].sum())}
    assert set(acc) == set(policy.registry_names())


def test_evaluate_tasks_refuses_while_a_capture_is_active(policy):
    from flowcl.envs.evaluation import evaluate_tasks

    with ActivationCapture.for_policy(policy, tokens_per_sample=4, subsample_seed=0):
        with pytest.raises(RuntimeError, match="evaluation rollouts"):
            evaluate_tasks(policy, [object()], None, None, "run", None)
