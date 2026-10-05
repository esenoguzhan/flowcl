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


# ---- token contrast and requested-goal swap rollouts --------------------------------------------


def test_token_contrast_locates_the_differing_word(spec):
    from flowcl.analysis.language_check import instruction_token_contrast

    fresh = build_test_policy(spec)
    with torch.no_grad():
        fresh.text_encoder([MILK, SAUCE])
    out = instruction_token_contrast(fresh, MILK, SAUCE)
    for view in ("raw", "projected"):
        assert out[view]["n_tokens"] == fresh.text_encoder.max_length
        assert out[view]["relative"] > 0
        # <|startoftext|> pick up the | milk/tomato ... : the first difference is token 4
        assert out[view]["first_differing_token"] == 4
        per = out[view]["per_token_relative"]
        assert len(per) == out[view]["n_tokens"] and per[:4] == [0.0] * 4 and per[4] > 0
    assert out["projected"]["n_tokens_differing"] <= out["projected"]["n_tokens"]
    with pytest.raises(ValueError, match="two different"):
        instruction_token_contrast(fresh, MILK, MILK)


class _SwapEnv:
    """Stands in for LiberoTaskEnv: records its rollouts, reports a requested goal on even episodes."""

    calls: list = []
    missing: list = []

    def __init__(self, suite, task_idx, spec, image_size):
        self.task_key = f"{suite}/t{task_idx}"

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def missing_goal_objects(self, goal):
        return list(self.missing)

    def rollout(self, policy, episode_idx, stats, run_id, cfg, language=None, alt_goal=None):
        from flowcl.envs.libero_env import RolloutResult

        _SwapEnv.calls.append((episode_idx, language, alt_goal is not None))
        swapped = language is not None
        return RolloutResult(success=not swapped, n_steps=600 if swapped else 50, task_key=self.task_key,
                             episode_idx=episode_idx, seed=1000 + episode_idx, n_replans=1,
                             alt_at_start=False if alt_goal else None,
                             alt_first_step=(30 if episode_idx % 2 == 0 else None) if alt_goal else None,
                             alt_at_end=(episode_idx % 2 == 0) if alt_goal else None)


def _swap(monkeypatch, **kwargs):
    import types

    import flowcl.analysis.language_check as lc
    import flowcl.envs.libero_env as le
    from flowcl.envs.libero_env import EvalConfig

    _SwapEnv.calls = []
    monkeypatch.setattr(lc, "LiberoTaskEnv", _SwapEnv)
    monkeypatch.setattr(le, "goal_state", lambda key: [["on", "bowl", key]])
    ref = types.SimpleNamespace(suite="libero_goal", task_idx=0, language="put the bowl on the plate",
                                task_key="libero_goal/plate")
    return lc.instruction_swap_rollouts(None, ref, "put the bowl on the stove", None, None, "ns",
                                        EvalConfig(n_episodes=4), progress=False, **kwargs)


def test_swap_rollouts_score_the_requested_goal_and_can_reuse_an_eval(monkeypatch):
    fresh = _swap(monkeypatch, requested_task="libero_goal/stove")
    assert [c for c in _SwapEnv.calls if c[1] is None] == [(i, None, False) for i in range(4)]
    assert fresh.correct_successes == [True] * 4 and fresh.swapped_successes == [False] * 4
    assert fresh.requested_reached == [True, False, True, False]
    assert fresh.requested_first_step == [30, None, 30, None]
    assert fresh.requested.value == 0.5 and fresh.seeds == [1000, 1001, 1002, 1003]
    assert not fresh.correct_reused and fresh.drop.value == 1.0

    reused = _swap(monkeypatch, requested_task="libero_goal/stove", correct_successes=[True, False, True, True])
    assert all(c[1] is not None for c in _SwapEnv.calls)          # only swapped rollouts ran
    assert reused.correct_reused and reused.correct_successes == [True, False, True, True]
    assert reused.drop.value == 0.75
    d = reused.as_dict()
    assert d["requested_estimate"]["value"] == 0.5 and d["requested_at_start"] == [False] * 4

    plain = _swap(monkeypatch)
    assert plain.requested is None and plain.requested_reached == [] and plain.as_dict()["requested_estimate"] is None


def test_swap_rollouts_refuse_an_unscorable_goal_and_a_short_reuse(monkeypatch):
    with pytest.raises(ValueError, match="episodes"):
        _swap(monkeypatch, correct_successes=[True])
    monkeypatch.setattr(_SwapEnv, "missing", ["flat_stove_1"])
    with pytest.raises(ValueError, match="cannot be scored"):
        _swap(monkeypatch, requested_task="libero_goal/stove")
