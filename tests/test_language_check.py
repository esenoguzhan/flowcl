"""The §4.1 language-discriminability logic that gates ``seq_correlated``."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from flowcl.analysis.language_check import (
    MIN_RELATIVE_DIVERGENCE,
    MIN_SWAP_SUCCESS_DROP,
    LanguageCheckReport,
    SwapResult,
    instruction_sensitivity,
)
from flowcl.analysis.metrics import Estimate, paired_difference_ci, success_estimate
from flowcl.data.spec import ActionSpec, EmbodimentSpec, ObservationSpec
from flowcl.models.build import build_policy

MILK = "pick up the milk and place it in the basket"
SAUCE = "pick up the tomato sauce and place it in the basket"


@pytest.fixture(scope="module")
def spec() -> EmbodimentSpec:
    return EmbodimentSpec(
        name="libero_franka",
        observation=ObservationSpec(
            cameras=("agentview",), image_size=(128, 128), d_state=8
        ),
        action=ActionSpec(
            d_action=7,
            control_mode="osc_pose_delta",
            control_rate_hz=20.0,
            chunk_horizon=8,
            execute_k=4,
            already_normalized=True,
        ),
    )


def build_test_policy(spec):
    # §4.2 fixes d_model to 384-512 and the trunk to 6-8 layers, so this is already
    # the smallest legal policy; only the decoder and context budget are trimmed.
    return build_policy(
        {
            "d_model": 384,
            "n_trunk_layers": 6,
            "n_heads": 8,
            "n_decoder_layers": 2,
            "n_context_tokens": 4,
            "pretrained": False,
            "euler_steps": 2,
        },
        spec,
        pretrained=False,
    ).eval()


def wake_decoder(policy, seed: int = 0):
    """Undo the DiT-style zero-init so the decoder's residual branches are live.

    :class:`~flowcl.models.flow_head.AdaLNModulation` and ``action_out`` are
    zero-initialised, which makes an *untrained* head output exactly zero velocity for
    every input — see ``test_untrained_head_outputs_exactly_zero``. That is correct at
    initialisation but useless for testing whether conditioning reaches the output, so
    these tests give those layers small random weights, which is where a few optimiser
    steps put them anyway.
    """
    generator = torch.Generator().manual_seed(seed)
    for name, param in policy.named_parameters():
        if "modulation" in name or name.startswith("flow_head.action_out"):
            with torch.no_grad():
                param.copy_(
                    torch.randn(param.shape, generator=generator) * 0.05
                )
    return policy


@pytest.fixture(scope="module")
def policy(spec):
    return wake_decoder(build_test_policy(spec))


@pytest.fixture(scope="module")
def batch(spec):
    rng = np.random.default_rng(0)
    return {
        "images": {
            "agentview": torch.from_numpy(
                rng.integers(0, 255, size=(4, 128, 128, 3), dtype=np.uint8)
            )
        },
        "state": torch.zeros(4, 8),
        "language": [MILK] * 4,
    }


def test_untrained_head_outputs_exactly_zero(spec, batch):
    """Pins the zero-init property, because it looks exactly like a conditioning bug.

    Every AdaLN gate and ``action_out`` start at zero, so a freshly built head returns
    the input noise unchanged from ``sample()`` and zero velocity from ``forward()`` no
    matter what the observation or instruction is. Anyone debugging "the policy ignores
    language" on an untrained model will land here first, so the behaviour is asserted
    rather than left to be rediscovered.

    It also justifies :func:`instruction_sensitivity` refusing to report a divergence
    for such a policy: the divergence would be 0 while the chunk scale looked healthy
    (it is just the noise), reading as "language is ignored" when the real answer is
    "this policy is untrained".
    """
    fresh = build_test_policy(spec)
    context = fresh.encode_observation(dict(batch, language=[MILK] * 4))
    noise = torch.randn(4, spec.action.chunk_horizon, spec.d_action)

    velocity = fresh.flow_head(noise, context, torch.full((4,), 0.5))
    assert float(velocity.abs().max()) == 0.0
    # Integrating a zero field returns the noise untouched.
    torch.testing.assert_close(fresh.flow_head.sample(context, noise=noise), noise)

    with pytest.raises(ValueError, match="returned its input noise unchanged"):
        instruction_sensitivity(fresh, batch, MILK, SAUCE, seed=0)


def test_sensitivity_detects_a_live_language_pathway(policy, batch):
    """With live residual branches, two instructions must move the chunk."""
    result = instruction_sensitivity(policy, batch, MILK, SAUCE, seed=0)
    assert result.relative_divergence > 0.0
    assert result.n_samples == 4
    assert result.chunk_scale > 0.0


def test_sensitivity_is_zero_when_language_is_ignored(spec, batch, monkeypatch):
    """The check must actually fail for a language-blind policy.

    Without this, a divergence of 0 could never be distinguished from a broken
    measurement. Here the text encoder always returns the same instruction's
    embedding, so the two instructions are genuinely indistinguishable to the model.
    """
    blind = wake_decoder(build_test_policy(spec))
    real_encoder = blind.text_encoder

    class ConstantText(torch.nn.Module):
        def forward(self, strings):
            return real_encoder([MILK] * len(strings))

    monkeypatch.setattr(blind, "text_encoder", ConstantText())
    result = instruction_sensitivity(blind, batch, MILK, SAUCE, seed=0)

    assert result.relative_divergence == pytest.approx(0.0, abs=1e-7)
    assert not result.language_pathway_alive


def test_sensitivity_shares_noise_between_the_two_passes(policy, batch):
    """Without shared noise the sampler's randomness would swamp the effect.

    Two calls with the same seed must give the same divergence; if the noise were
    redrawn per instruction, repeated calls would not agree.
    """
    a = instruction_sensitivity(policy, batch, MILK, SAUCE, seed=3)
    b = instruction_sensitivity(policy, batch, MILK, SAUCE, seed=3)
    assert a.absolute_divergence == pytest.approx(b.absolute_divergence, rel=1e-9)


def test_sensitivity_rejects_comparing_an_instruction_with_itself(policy, batch):
    with pytest.raises(ValueError, match="two different instructions"):
        instruction_sensitivity(policy, batch, MILK, MILK, seed=0)


def test_divergence_threshold_is_documented():
    assert MIN_RELATIVE_DIVERGENCE > 0.0
    assert MIN_SWAP_SUCCESS_DROP == 0.15


# ---- swap verdict logic --------------------------------------------------------


def make_swap(correct: list[bool], swapped: list[bool]) -> SwapResult:
    return SwapResult(
        task_key="libero_object/t",
        correct_instruction=MILK,
        swapped_instruction=SAUCE,
        correct=success_estimate(correct, seed=0),
        swapped=success_estimate(swapped, seed=0),
        drop=paired_difference_ci(
            np.asarray(correct, dtype=np.float64),
            np.asarray(swapped, dtype=np.float64),
            seed=0,
        ),
        correct_successes=correct,
        swapped_successes=swapped,
    )


def test_swap_discriminates_when_the_wrong_instruction_costs_success():
    swap = make_swap([True] * 18 + [False] * 2, [True] * 3 + [False] * 17)
    assert swap.drop.value == pytest.approx(0.75)
    assert swap.discriminates


def test_swap_does_not_discriminate_when_the_instruction_makes_no_difference():
    """The failure this whole module exists to catch."""
    identical = [True] * 16 + [False] * 4
    swap = make_swap(identical, list(identical))
    assert swap.drop.value == pytest.approx(0.0)
    assert not swap.discriminates


def test_swap_requires_the_ci_to_exclude_zero():
    """A large point drop on noisy data is not evidence.

    Four rollouts with a 1/4 difference gives drop = 0.25 > the 0.15 threshold, but the
    paired interval includes zero, so the verdict must still be negative.
    """
    swap = make_swap([True, True, True, True], [True, True, True, False])
    assert swap.drop.value == pytest.approx(0.25)
    assert swap.drop.low <= 0.0
    assert not swap.discriminates


def test_report_requires_every_task_to_discriminate():
    """seq_correlated's premise breaks if even one task is not language-selected."""
    good = make_swap([True] * 18 + [False] * 2, [True] * 3 + [False] * 17)
    bad = make_swap([True] * 16 + [False] * 4, [True] * 16 + [False] * 4)

    assert LanguageCheckReport(task_keys=("a",), swaps=[good]).passed
    assert not LanguageCheckReport(task_keys=("a", "b"), swaps=[good, bad]).passed


def test_report_without_rollouts_cannot_pass():
    """The cheap divergence diagnostic alone is not a verdict."""
    report = LanguageCheckReport(task_keys=("a",))
    assert not report.passed
    assert "FAIL" in report.describe()


def test_report_explains_the_consequence_of_failure():
    bad = make_swap([True] * 16 + [False] * 4, [True] * 16 + [False] * 4)
    text = LanguageCheckReport(task_keys=("a",), swaps=[bad]).describe()
    assert "seq_correlated" in text


def test_estimate_construction_in_swap_is_consistent():
    """Guard against a drop whose CI does not contain it (§11 consistency)."""
    swap = make_swap([True] * 10 + [False] * 10, [False] * 20)
    assert isinstance(swap.drop, Estimate)
    assert swap.drop.low <= swap.drop.value <= swap.drop.high
