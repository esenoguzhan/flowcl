"""Language-discriminability check (§4.1), which gates every ``seq_correlated`` claim.

The problem it exists to catch: ``seq_correlated`` (§5) puts several ``libero_object``
tasks in sequence. Those tasks share one scene and one motion; only the *target object*
and the instruction differ. If the policy ignores language and picks the object from
pixels and habit alone, then "forgetting" on that curriculum is just the policy
overwriting one habit with another, and every conclusion drawn from it is about
something other than language-conditioned continual learning. Nothing in the loss curve
or the success rate of a single task reveals this.

Two measurements, cheap first:

1. :func:`instruction_sensitivity` — pure forward passes. From an identical observation
   and identical flow noise, predict the chunk under instruction A and under
   instruction B. If the predictions are numerically indistinguishable, the language
   pathway is dead and no rollout is needed to know it.
2. :func:`instruction_swap_rollouts` — rollouts on one task's scene and success
   predicate, under the correct instruction and under another task's instruction, on
   the *same* fixed initial states (§8.1). A policy that follows language succeeds far
   more often with the correct one. A flat difference means language is decorative.

Measurement 1 can pass while 2 fails: a policy can let language perturb its outputs
without letting it select the object. So 2 is the verdict and 1 is the diagnostic that
tells you *why*.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from flowcl.analysis.metrics import Estimate, paired_difference_ci, success_estimate
from flowcl.data.spec import EmbodimentSpec
from flowcl.data.stats import NormalizationStats
from flowcl.data.tasks import TaskRef
from flowcl.envs.libero_env import EvalConfig, LiberoTaskEnv

# Below this, the two instructions produce essentially the same chunk and the language
# pathway is not functioning. Expressed relative to the chunk's own scale so it is
# independent of action units.
MIN_RELATIVE_DIVERGENCE = 0.02

# The swapped instruction must cost at least this much success for language to be
# doing task selection rather than decoration.
MIN_SWAP_SUCCESS_DROP = 0.15


@dataclass
class SensitivityResult:
    """Output of :func:`instruction_sensitivity`."""

    relative_divergence: float
    absolute_divergence: float
    chunk_scale: float
    n_samples: int
    instruction_a: str
    instruction_b: str

    @property
    def language_pathway_alive(self) -> bool:
        return self.relative_divergence >= MIN_RELATIVE_DIVERGENCE

    def as_dict(self) -> dict:
        return {
            "relative_divergence": self.relative_divergence,
            "absolute_divergence": self.absolute_divergence,
            "chunk_scale": self.chunk_scale,
            "n_samples": self.n_samples,
            "instruction_a": self.instruction_a,
            "instruction_b": self.instruction_b,
            "threshold": MIN_RELATIVE_DIVERGENCE,
            "language_pathway_alive": self.language_pathway_alive,
        }


@torch.no_grad()
def instruction_sensitivity(
    policy,
    batch: dict,
    instruction_a: str,
    instruction_b: str,
    seed: int = 0,
    n_steps: int | None = None,
) -> SensitivityResult:
    """Do two instructions produce different action chunks from the same observation?

    The two forward passes share the *same* initial noise, so any difference is caused
    by the instruction alone. Without that, the flow sampler's own randomness would
    swamp the effect and the measurement would be meaningless.

    Args:
        batch: A collated batch. Its ``language`` field is overridden.
        seed: Fixes the shared noise.

    Returns:
        A :class:`SensitivityResult`. ``relative_divergence`` is the mean absolute
        difference between the two chunks divided by the mean absolute chunk value.
    """
    if instruction_a == instruction_b:
        raise ValueError(
            "instruction_sensitivity needs two different instructions; comparing an "
            "instruction with itself measures nothing"
        )

    device = next(policy.parameters()).device
    batch_size = batch["state"].shape[0]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(
        (batch_size, policy.horizon, policy.d_action),
        generator=generator,
        dtype=torch.float32,
    ).to(device)

    policy.eval()
    chunks = []
    for instruction in (instruction_a, instruction_b):
        scoped = dict(batch)
        scoped["language"] = [instruction] * batch_size
        chunks.append(policy.sample(scoped, n_steps=n_steps, noise=noise).float())

    # A dead decoder integrates a zero velocity field, so sample() hands back exactly
    # the noise it was given. Detect that explicitly: otherwise the divergence is 0
    # while the chunk scale looks healthy (it is just the noise), and the result reads
    # as "language is ignored" when the real answer is "this policy is untrained".
    if all(torch.equal(chunk, noise) for chunk in chunks):
        raise ValueError(
            "sample() returned its input noise unchanged, so the decoder predicted an "
            "identically zero velocity field. AdaLN gates and action_out are "
            "zero-initialised, so this is what an untrained policy does. Train before "
            "running the language check."
        )

    difference = float((chunks[0] - chunks[1]).abs().mean())
    scale = float(torch.cat(chunks).abs().mean())
    if scale <= 0.0:
        raise ValueError(
            "the policy predicted an all-zero action chunk, so instruction "
            "sensitivity is undefined"
        )

    return SensitivityResult(
        relative_divergence=difference / scale,
        absolute_divergence=difference,
        chunk_scale=scale,
        n_samples=batch_size,
        instruction_a=instruction_a,
        instruction_b=instruction_b,
    )


@dataclass
class SwapResult:
    """Per-task outcome of :func:`instruction_swap_rollouts`."""

    task_key: str
    correct_instruction: str
    swapped_instruction: str
    correct: Estimate
    swapped: Estimate
    drop: Estimate
    correct_successes: list[bool] = field(default_factory=list)
    swapped_successes: list[bool] = field(default_factory=list)

    @property
    def discriminates(self) -> bool:
        """Does the swap cost enough success to call language load-bearing?

        Uses the point estimate against :data:`MIN_SWAP_SUCCESS_DROP`, with the paired
        CI recorded alongside. A drop whose CI includes zero is not evidence of
        discriminability however large the point estimate, so that is required too.
        """
        return self.drop.value >= MIN_SWAP_SUCCESS_DROP and self.drop.low > 0.0

    def as_dict(self) -> dict:
        return {
            "task_key": self.task_key,
            "correct_instruction": self.correct_instruction,
            "swapped_instruction": self.swapped_instruction,
            "correct_success": self.correct.format_pp(),
            "swapped_success": self.swapped.format_pp(),
            "drop_pp": self.drop.format_pp(),
            "drop_ci_excludes_zero": self.drop.low > 0.0,
            "min_drop_threshold": MIN_SWAP_SUCCESS_DROP,
            "discriminates": self.discriminates,
            "correct_successes": [bool(s) for s in self.correct_successes],
            "swapped_successes": [bool(s) for s in self.swapped_successes],
        }


def instruction_swap_rollouts(
    policy,
    ref: TaskRef,
    swapped_instruction: str,
    spec: EmbodimentSpec,
    stats: NormalizationStats,
    run_id: str,
    cfg: EvalConfig,
    bootstrap: dict | None = None,
    progress: bool = True,
) -> SwapResult:
    """Roll out ``ref`` under its own instruction and under ``swapped_instruction``.

    Both arms use ``ref``'s scene, ``ref``'s success predicate and the same fixed
    initial states, so the difference is attributable to the instruction and the
    comparison is paired.
    """
    bootstrap = bootstrap or {}
    if swapped_instruction.strip().lower() == ref.language.strip().lower():
        raise ValueError(
            f"swapped instruction is identical to {ref.task_key}'s own instruction; "
            "the check would trivially show no difference"
        )

    correct_successes: list[bool] = []
    swapped_successes: list[bool] = []

    with LiberoTaskEnv(
        suite=ref.suite, task_idx=ref.task_idx, spec=spec, image_size=cfg.image_size
    ) as env:
        for episode_idx in range(cfg.n_episodes):
            correct = env.rollout(policy, episode_idx, stats, run_id, cfg)
            swapped = env.rollout(
                policy,
                episode_idx,
                stats,
                run_id,
                cfg,
                language=swapped_instruction,
            )
            correct_successes.append(correct.success)
            swapped_successes.append(swapped.success)
            if progress:
                print(
                    f"[flowcl] {ref.task_key} init {episode_idx + 1}/"
                    f"{cfg.n_episodes}: correct="
                    f"{'S' if correct.success else '.'} swapped="
                    f"{'S' if swapped.success else '.'}",
                    flush=True,
                )

    kwargs = {
        "seed": bootstrap.get("seed", 0),
        "n_bootstrap": bootstrap.get("n_resamples", 10000),
        "confidence": bootstrap.get("confidence", 0.95),
    }
    return SwapResult(
        task_key=ref.task_key,
        correct_instruction=ref.language,
        swapped_instruction=swapped_instruction,
        correct=success_estimate(correct_successes, **kwargs),
        swapped=success_estimate(swapped_successes, **kwargs),
        drop=paired_difference_ci(
            np.asarray(correct_successes, dtype=np.float64),
            np.asarray(swapped_successes, dtype=np.float64),
            **kwargs,
        ),
        correct_successes=correct_successes,
        swapped_successes=swapped_successes,
    )


@dataclass
class LanguageCheckReport:
    """The full §4.1 verdict over a pair of tasks."""

    task_keys: tuple[str, ...]
    sensitivity: list[SensitivityResult] = field(default_factory=list)
    swaps: list[SwapResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Every task must discriminate; one failure invalidates ``seq_correlated``.

        Requiring *all* tasks rather than *any* is the conservative reading, and the
        right one: ``seq_correlated``'s premise is that language distinguishes the
        tasks from each other, which a single non-discriminated task already breaks.
        """
        if not self.swaps:
            return False
        return all(swap.discriminates for swap in self.swaps)

    def as_dict(self) -> dict:
        return {
            "task_keys": list(self.task_keys),
            "passed": self.passed,
            "sensitivity": [s.as_dict() for s in self.sensitivity],
            "swaps": [s.as_dict() for s in self.swaps],
        }

    def describe(self) -> str:
        lines = [
            f"[language check] {'PASS' if self.passed else 'FAIL'} "
            f"over {list(self.task_keys)}"
        ]
        for s in self.sensitivity:
            lines.append(
                f"  chunk divergence {s.instruction_a[:30]!r} vs "
                f"{s.instruction_b[:30]!r}: {s.relative_divergence:.4f} "
                f"(alive={s.language_pathway_alive})"
            )
        for swap in self.swaps:
            lines.append(
                f"  {swap.task_key}: correct {swap.correct.format_pp()} vs swapped "
                f"{swap.swapped.format_pp()}, drop {swap.drop.format_pp()} "
                f"(discriminates={swap.discriminates})"
            )
        if not self.passed:
            lines.append(
                "  => do not draw seq_correlated conclusions: the policy is not "
                "selecting behaviour from the instruction (§4.1)."
            )
        return "\n".join(lines)
