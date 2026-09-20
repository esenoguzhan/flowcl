"""Flow-matching mechanics (§4.3, §4.4) and the loss masking contract."""

from __future__ import annotations

import math

import pytest
import torch

from flowcl.models.flow_head import (
    S_BINS,
    BinnedSSampler,
    FlowHead,
    LogitNormalSSampler,
    SinusoidalSEmbedding,
    UniformSSampler,
    bin_index,
    build_s_sampler,
)
from flowcl.models.losses import (
    flow_matching_loss,
    interpolate_actions,
    masked_mse,
    valid_element_count,
)

B, H, D, DM = 4, 16, 7, 64


@pytest.fixture
def head():
    torch.manual_seed(0)
    return FlowHead(d_action=D, horizon=H, d_model=DM, n_layers=2, n_heads=4)


# ---- the interpolant ----------------------------------------------------------


def test_interpolant_endpoints():
    """A_s = (1-s)A_0 + s A_1, so s=0 gives A_0 and s=1 gives A_1 exactly."""
    a0 = torch.randn(B, H, D)
    a1 = torch.randn(B, H, D)

    at_zero = interpolate_actions(a0, a1, torch.zeros(B))
    at_one = interpolate_actions(a0, a1, torch.ones(B))
    torch.testing.assert_close(at_zero, a0)
    torch.testing.assert_close(at_one, a1)


def test_interpolant_midpoint_is_the_average():
    a0 = torch.randn(B, H, D)
    a1 = torch.randn(B, H, D)
    mid = interpolate_actions(a0, a1, torch.full((B,), 0.5))
    torch.testing.assert_close(mid, 0.5 * (a0 + a1))


def test_interpolant_is_per_sample_in_s():
    """Each batch element uses its own s, not a shared scalar."""
    a0 = torch.zeros(B, H, D)
    a1 = torch.ones(B, H, D)
    s = torch.tensor([0.0, 0.25, 0.75, 1.0])
    out = interpolate_actions(a0, a1, s)
    for i, expected in enumerate([0.0, 0.25, 0.75, 1.0]):
        assert torch.allclose(out[i], torch.full((H, D), expected))


def test_interpolant_rejects_wrong_s_shape():
    with pytest.raises(ValueError, match="expected .4,. flow times"):
        interpolate_actions(torch.randn(B, H, D), torch.randn(B, H, D), torch.zeros(B, 1))


# ---- the regression target ----------------------------------------------------


def test_target_velocity_is_a1_minus_a0():
    """d/ds[(1-s)A_0 + s A_1] = A_1 - A_0, independent of s."""
    a0 = torch.randn(B, H, D)
    a1 = torch.randn(B, H, D)
    mask = torch.ones(B, H)

    # A prediction that exactly equals A_1 - A_0 must give zero loss.
    loss = flow_matching_loss(a1 - a0, a0, a1, mask)
    assert float(loss) == 0.0


def test_loss_matches_hand_computed_value():
    """Known answer: constant unit error over all valid elements gives loss 1."""
    a0 = torch.zeros(B, H, D)
    a1 = torch.zeros(B, H, D)
    prediction = torch.ones(B, H, D)
    mask = torch.ones(B, H)
    assert float(flow_matching_loss(prediction, a0, a1, mask)) == pytest.approx(1.0)


# ---- masking ------------------------------------------------------------------


def test_masked_steps_contribute_exactly_zero(head):
    """§3.3: masked steps contribute zero loss — bitwise, not approximately."""
    torch.manual_seed(1)
    prediction = torch.randn(B, H, D)
    target = torch.randn(B, H, D)
    mask = torch.ones(B, H)
    mask[:, 8:] = 0.0

    baseline = masked_mse(prediction, target, mask)

    perturbed = prediction.clone()
    perturbed[:, 8:] += 1e6
    assert masked_mse(perturbed, target, mask) == baseline


def test_masked_loss_is_mean_over_valid_elements_only():
    """Half-masked unit error still gives 1.0, because padding is not in the divisor."""
    prediction = torch.ones(2, 4, 3)
    target = torch.zeros(2, 4, 3)
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    # 3 valid timesteps * 3 dims = 9 elements, each with squared error 1.
    assert float(masked_mse(prediction, target, mask)) == pytest.approx(1.0)


def test_shared_normalizer_makes_partial_batches_additive():
    """The property §7.5's equivalence check depends on.

    Splitting a batch and summing the two masked sums divided by the *whole* batch's
    valid-element count reproduces the full-batch mean exactly.
    """
    torch.manual_seed(2)
    prediction = torch.randn(8, H, D)
    target = torch.randn(8, H, D)
    mask = torch.ones(8, H)
    mask[:, 12:] = 0.0

    full = masked_mse(prediction, target, mask)
    total = valid_element_count(mask, D)

    first = masked_mse(prediction[:4], target[:4], mask[:4], normalizer=total)
    second = masked_mse(prediction[4:], target[4:], mask[4:], normalizer=total)
    torch.testing.assert_close(first + second, full, rtol=1e-6, atol=1e-8)


def test_all_masked_batch_raises():
    with pytest.raises(ValueError, match="every timestep in the batch is masked"):
        masked_mse(torch.ones(2, 4, 3), torch.zeros(2, 4, 3), torch.zeros(2, 4))


def test_masked_mse_rejects_mismatched_mask():
    with pytest.raises(ValueError, match="mask shape"):
        masked_mse(torch.ones(2, 4, 3), torch.zeros(2, 4, 3), torch.ones(2, 5))


# ---- p(s) samplers ------------------------------------------------------------


def test_uniform_sampler_range_and_mean():
    s = UniformSSampler().sample(20000, torch.device("cpu"))
    assert s.min() >= 0.0 and s.max() <= 1.0
    assert float(s.mean()) == pytest.approx(0.5, abs=0.02)


def test_logit_normal_sampler_is_in_range_and_centred():
    s = LogitNormalSSampler(mean=0.0, std=1.0).sample(20000, torch.device("cpu"))
    assert s.min() > 0.0 and s.max() < 1.0
    # sigmoid of a zero-mean symmetric variable has median 0.5.
    assert float(s.median()) == pytest.approx(0.5, abs=0.02)


def test_logit_normal_concentrates_more_than_uniform():
    uniform = UniformSSampler().sample(20000, torch.device("cpu"))
    logit = LogitNormalSSampler().sample(20000, torch.device("cpu"))
    assert float(logit.std()) < float(uniform.std())


def test_samplers_are_swappable_by_name():
    assert isinstance(build_s_sampler("uniform"), UniformSSampler)
    assert isinstance(build_s_sampler("logit_normal"), LogitNormalSSampler)
    with pytest.raises(ValueError, match="Unknown s sampler"):
        build_s_sampler("gaussian")


def test_sampler_is_reproducible_under_a_generator():
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    a = UniformSSampler().sample(100, torch.device("cpu"), generator=g1)
    b = UniformSSampler().sample(100, torch.device("cpu"), generator=g2)
    torch.testing.assert_close(a, b, rtol=0, atol=0)


# ---- s bins (§9) --------------------------------------------------------------


def test_s_bins_partition_the_unit_interval():
    assert S_BINS == ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0))
    assert S_BINS[0][0] == 0.0 and S_BINS[-1][1] == 1.0
    for (_, high), (low, _) in zip(S_BINS, S_BINS[1:]):
        assert high == low


def test_bin_index_boundaries():
    """Right-open bins, except s=1.0 which belongs to the last bin."""
    s = torch.tensor([0.0, 0.249, 0.25, 0.5, 0.749, 0.75, 0.999, 1.0])
    assert bin_index(s).tolist() == [0, 0, 1, 2, 2, 3, 3, 3]


def test_binned_sampler_stays_inside_its_bin():
    for idx, (low, high) in enumerate(S_BINS):
        s = BinnedSSampler(idx).sample(2000, torch.device("cpu"))
        assert s.min() >= low
        assert s.max() <= high
        assert bin_index(s).unique().tolist() in ([idx], [idx, idx + 1])


def test_binned_sampler_rejects_bad_index():
    with pytest.raises(ValueError, match="out of range"):
        BinnedSSampler(4)


# ---- s embedding --------------------------------------------------------------


def test_s_embedding_shape_and_sensitivity():
    emb = SinusoidalSEmbedding(dim=32)
    out = emb(torch.tensor([0.0, 0.5, 1.0]))
    assert out.shape == (3, 32)
    # Different flow times must produce different embeddings, or s conditioning is
    # silently a no-op.
    assert not torch.allclose(out[0], out[1])
    assert not torch.allclose(out[1], out[2])


def test_s_embedding_rejects_odd_dim():
    with pytest.raises(ValueError, match="must be even"):
        SinusoidalSEmbedding(dim=31)


def test_s_embedding_rejects_wrong_rank():
    with pytest.raises(ValueError, match=r"expected \(B,\) flow times"):
        SinusoidalSEmbedding(dim=16)(torch.zeros(2, 1))


# ---- the head itself ----------------------------------------------------------


def test_head_output_shape(head):
    context = torch.randn(B, 8, DM)
    out = head(torch.randn(B, H, D), context, torch.rand(B))
    assert out.shape == (B, H, D)


def test_head_rejects_wrong_action_shape(head):
    context = torch.randn(B, 8, DM)
    with pytest.raises(ValueError, match=r"expected \(B, 16, 7\) actions"):
        head(torch.randn(B, 8, D), context, torch.rand(B))


def test_head_output_depends_on_s(head):
    """AdaLN must actually route s into the network.

    The modulation projection is zero-initialised, so a freshly built head is
    s-invariant by construction; perturb it first, then check sensitivity.
    """
    with torch.no_grad():
        for block in head.blocks:
            block.modulation.proj.weight.normal_(std=0.1)
            block.modulation.proj.bias.normal_(std=0.1)
        head.action_out.weight.normal_(std=0.1)

    context = torch.randn(B, 8, DM)
    actions = torch.randn(B, H, D)
    low = head(actions, context, torch.zeros(B))
    high = head(actions, context, torch.ones(B))
    assert not torch.allclose(low, high)


def test_euler_sampler_shape_and_step_count(head):
    context = torch.randn(B, 8, DM)
    chunk = head.sample(context, n_steps=10)
    assert chunk.shape == (B, H, D)


def test_euler_sampler_rejects_zero_steps(head):
    with pytest.raises(ValueError, match="n_steps must be >= 1"):
        head.sample(torch.randn(B, 8, DM), n_steps=0)


def test_sample_is_bitwise_deterministic_given_a_seed(head):
    """§8.3 needs reproducible rollouts, which needs a reproducible sampler."""
    context = torch.randn(B, 8, DM)

    g1 = torch.Generator().manual_seed(1234)
    g2 = torch.Generator().manual_seed(1234)
    first = head.sample(context, n_steps=10, generator=g1)
    second = head.sample(context, n_steps=10, generator=g2)
    assert torch.equal(first, second)


def test_sample_differs_across_seeds(head):
    context = torch.randn(B, 8, DM)
    a = head.sample(context, n_steps=10, generator=torch.Generator().manual_seed(1))
    b = head.sample(context, n_steps=10, generator=torch.Generator().manual_seed(2))
    assert not torch.equal(a, b)


def test_euler_integrates_a_known_constant_field(head):
    """With v == c, Euler from A_0 must land exactly at A_0 + c.

    Pins the integration schedule: N steps of size 1/N summing to a total time of
    exactly 1. An off-by-one in the loop or a wrong ds would show up here.
    """
    constant = 3.0

    class ConstantField(FlowHead):
        def forward(self, noisy_actions, context, s):  # noqa: D102
            return torch.full_like(noisy_actions, constant)

    field = ConstantField(d_action=D, horizon=H, d_model=DM, n_layers=1, n_heads=4)
    noise = torch.zeros(B, H, D)
    out = field.sample(torch.randn(B, 4, DM), n_steps=10, noise=noise)
    torch.testing.assert_close(out, torch.full_like(out, constant))

    # Also exact for a different step count, since total integration time is 1.
    out_100 = field.sample(torch.randn(B, 4, DM), n_steps=100, noise=noise)
    torch.testing.assert_close(out_100, torch.full_like(out_100, constant))


def test_euler_recovers_a_linear_target_exactly(head):
    """A perfectly trained field v = A_1 - A_0 integrates to A_1 for any N."""
    target = torch.randn(B, H, D)
    noise = torch.randn(B, H, D)

    class PerfectField(FlowHead):
        def forward(self, noisy_actions, context, s):  # noqa: D102
            return target - noise

    field = PerfectField(d_action=D, horizon=H, d_model=DM, n_layers=1, n_heads=4)
    out = field.sample(torch.randn(B, 4, DM), n_steps=10, noise=noise)
    torch.testing.assert_close(out, target, rtol=1e-5, atol=1e-6)
