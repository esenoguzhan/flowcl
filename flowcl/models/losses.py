"""Flow-matching loss with chunk masking.

Spec §4.3: masked MSE between the predicted velocity and ``A_1 - A_0``.
Spec §3.3: "masked steps contribute zero loss."

The normaliser is the number of *valid elements*, i.e.
``mask.sum() * d_action``. Two consequences worth being explicit about:

* Padded timesteps contribute exactly zero, not "approximately zero" — they are
  multiplied by a hard 0 before summing, never divided away afterwards.
* The value is a true mean over valid elements, so batches with different amounts of
  padding are comparable. This is also what makes the §7.5 equivalence check exact:
  summing per-bin gradients reproduces one mean-reduced backward only if every
  microbatch divides by the *same* total valid-element count, which is why
  :func:`flow_matching_loss` accepts an explicit ``normalizer``.
"""

from __future__ import annotations

import torch


def masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    normalizer: float | torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean squared error over valid chunk elements only.

    Args:
        prediction: ``(B, H, D)``.
        target: ``(B, H, D)``.
        mask: ``(B, H)`` with 1 for valid timesteps and 0 for right-padding.
        normalizer: Divisor for the summed squared error. Defaults to
            ``mask.sum() * D``, i.e. the mean over valid elements. Pass an explicit
            value to make several microbatches share one normaliser (§7.5).

    Returns:
        Scalar loss.
    """
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction shape {tuple(prediction.shape)} != target shape "
            f"{tuple(target.shape)}"
        )
    if prediction.ndim != 3:
        raise ValueError(
            f"expected (B, H, D) tensors, got {tuple(prediction.shape)}"
        )
    if mask.shape != prediction.shape[:2]:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} does not match (B, H) = "
            f"{tuple(prediction.shape[:2])}"
        )

    squared_error = (prediction - target) ** 2
    masked = squared_error * mask.unsqueeze(-1).to(squared_error.dtype)
    total = masked.sum()

    if normalizer is None:
        n_valid_elements = mask.sum() * prediction.shape[-1]
        if n_valid_elements == 0:
            raise ValueError(
                "every timestep in the batch is masked out, so the loss is undefined; "
                "this means the dataset produced an all-padding batch"
            )
        return total / n_valid_elements

    if float(normalizer) == 0.0:
        raise ValueError("normalizer must be non-zero")
    return total / normalizer


def flow_matching_loss(
    velocity_prediction: torch.Tensor,
    noise: torch.Tensor,
    target_actions: torch.Tensor,
    mask: torch.Tensor,
    normalizer: float | torch.Tensor | None = None,
) -> torch.Tensor:
    """§4.3 loss: masked MSE against ``A_1 - A_0``.

    Args:
        velocity_prediction: ``v_theta(A_s, o, s)``, shape ``(B, H, D)``.
        noise: ``A_0``, shape ``(B, H, D)``.
        target_actions: ``A_1``, the ground-truth chunk, shape ``(B, H, D)``.
        mask: ``(B, H)`` validity mask.
        normalizer: See :func:`masked_mse`.
    """
    target_velocity = target_actions - noise
    return masked_mse(velocity_prediction, target_velocity, mask, normalizer=normalizer)


def interpolate_actions(
    noise: torch.Tensor, target_actions: torch.Tensor, s: torch.Tensor
) -> torch.Tensor:
    """``A_s = (1 - s) A_0 + s A_1`` (§4.3).

    Args:
        noise: ``A_0``, ``(B, H, D)``.
        target_actions: ``A_1``, ``(B, H, D)``.
        s: ``(B,)`` flow times, broadcast over ``H`` and ``D``.
    """
    if noise.shape != target_actions.shape:
        raise ValueError(
            f"noise shape {tuple(noise.shape)} != target shape "
            f"{tuple(target_actions.shape)}"
        )
    if s.shape != (noise.shape[0],):
        raise ValueError(
            f"expected ({noise.shape[0]},) flow times, got {tuple(s.shape)}"
        )
    s_broadcast = s.view(-1, 1, 1)
    return (1.0 - s_broadcast) * noise + s_broadcast * target_actions


def valid_element_count(mask: torch.Tensor, d_action: int) -> torch.Tensor:
    """``mask.sum() * d_action`` — the normaliser §7.5 must share across bins."""
    return mask.sum() * d_action
