"""Training-time image augmentation: random shifts and colour jitter.

Applied by :func:`flowcl.train.trainer.train_one_task` to a batch's images after they reach
the device, only when the run's ``TrainConfig`` asks for it (every registered LIBERO run
leaves it off). It works on the policy's own input format, ``{camera: (B, H, W, 3) uint8}``,
and returns the same, so the policy's normalisation is untouched.

* **Shift** (DrQ-style): replicate-pad by ``shift_px`` and crop back at an integer offset
  drawn per sample and per camera, so content moves by up to ``shift_px`` pixels.
* **Colour**: brightness, contrast and saturation factors drawn uniformly in
  ``[1 - color, 1 + color]`` per sample and per camera, applied in that order.

Every draw comes from the caller's generator, so a run's augmentation stream is its own
and leaves the flow-time, noise and batch-order streams exactly as without augmentation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# ITU-R BT.601 luma, the usual grayscale for saturation/contrast jitter.
LUMA = (0.299, 0.587, 0.114)


def _gray(x: torch.Tensor) -> torch.Tensor:
    """``(B, H, W, 3)`` float -> ``(B, H, W, 1)`` luma."""
    weights = torch.tensor(LUMA, device=x.device, dtype=x.dtype)
    return (x * weights).sum(dim=-1, keepdim=True)


def _uniform(n: int, low: float, high: float, generator: torch.Generator, device) -> torch.Tensor:
    return low + (high - low) * torch.rand(n, generator=generator, device=device)


def shift(x: torch.Tensor, shift_px: int, generator: torch.Generator) -> torch.Tensor:
    """Random integer translation of ``(B, H, W, C)`` float images, edge pixels replicated."""
    b, h, w, _ = x.shape
    padded = F.pad(x.permute(0, 3, 1, 2), (shift_px,) * 4, mode="replicate").permute(0, 2, 3, 1)
    oy = torch.randint(0, 2 * shift_px + 1, (b,), generator=generator, device=x.device)
    ox = torch.randint(0, 2 * shift_px + 1, (b,), generator=generator, device=x.device)
    rows = (oy[:, None] + torch.arange(h, device=x.device))[:, :, None]
    cols = (ox[:, None] + torch.arange(w, device=x.device))[:, None, :]
    return padded[torch.arange(b, device=x.device)[:, None, None], rows, cols]


def jitter(x: torch.Tensor, color: float, generator: torch.Generator) -> torch.Tensor:
    """Brightness, contrast and saturation jitter of ``(B, H, W, 3)`` float images in [0, 255]."""
    b = x.shape[0]
    low, high = 1.0 - color, 1.0 + color
    brightness, contrast, saturation = (
        _uniform(b, low, high, generator, x.device)[:, None, None, None] for _ in range(3)
    )
    x = x * brightness
    mean = _gray(x).mean(dim=(1, 2, 3), keepdim=True)
    x = mean + contrast * (x - mean)
    gray = _gray(x)
    x = gray + saturation * (x - gray)
    return x.clamp(0.0, 255.0)


def augment_images(
    images: dict[str, torch.Tensor],
    shift_px: int,
    color: float,
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    """Shift and colour-jitter every camera's ``(B, H, W, 3)`` uint8 frames.

    With ``shift_px == 0`` and ``color == 0`` the input is returned as is and nothing is
    drawn from ``generator``.
    """
    if shift_px < 0:
        raise ValueError(f"shift_px must be >= 0, got {shift_px}")
    if not 0.0 <= color < 1.0:
        raise ValueError(f"color must lie in [0, 1), got {color}")
    if shift_px == 0 and color == 0.0:
        return images
    out = {}
    for camera, frames in images.items():
        if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != torch.uint8:
            raise ValueError(
                f"images[{camera!r}] must be (B, H, W, 3) uint8, got {tuple(frames.shape)} {frames.dtype}"
            )
        x = frames.float()
        if shift_px:
            x = shift(x, shift_px, generator)
        if color:
            x = jitter(x, color, generator)
        out[camera] = x.round().to(torch.uint8)
    return out
