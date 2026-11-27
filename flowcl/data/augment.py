"""Training-time image augmentation: random shifts, colour jitter and daylight.

Applied by :func:`flowcl.train.trainer.train_one_task` to a batch's images after they reach
the device, only when the run's ``TrainConfig`` asks for it (every registered LIBERO run
leaves it off). It works on the policy's own input format, ``{camera: (B, H, W, 3) uint8}``,
and returns the same, so the policy's normalisation is untouched.

* **Shift** (DrQ-style): replicate-pad by ``shift_px`` and crop back at an integer offset
  drawn per sample and per camera, so content moves by up to ``shift_px`` pixels.
* **Colour**: brightness, contrast and saturation factors drawn uniformly in
  ``[1 - color, 1 + color]`` per sample and per camera, applied in that order.
* **Daylight** (strength ``s``): the scene lit differently, as by daylight through a
  window against room light. Per sample and per camera, a colour temperature (red gain
  ``1 + t``, blue gain ``1 - t``, ``t`` in ``[-s/2, s/2]``) and a linear illumination ramp
  across the image in a random direction (one edge up to ``s`` brighter, the opposite one
  as much darker), then the colour jitter above at strength ``s``.

* **Shadow** (probability ``p`` per image): hard-edged light and shade, as sun through a window
  casts on the table (24 Nov: the sunlit table 1.5-1.8x its shadows, against 1.2x in the
  demos; daylight's ramps make no edges). Either a synthetic pattern (:func:`synthetic_light`:
  soft-edged bars inside a window patch, sometimes crossed by a second set, sometimes with
  blob shadows; shade 0.45-0.85 of the light, slightly blue; partly renormalised as a camera's
  auto-exposure would) or, for one camera and a share of the shadowed images, a real ratio map
  from a bank (flowcl.data.shadow_maps), randomly shifted, rotated, scaled and strengthened.
  Multiplicative, so black stays black (before daylight's contrast jitter, which lifts it).

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


def daylight(x: torch.Tensor, strength: float, generator: torch.Generator) -> torch.Tensor:
    """Colour temperature, an illumination ramp, then :func:`jitter`, all at ``strength``."""
    b, h, w, _ = x.shape
    temperature = _uniform(b, -strength / 2, strength / 2, generator, x.device)
    gains = torch.stack([1 + temperature, torch.ones_like(temperature), 1 - temperature], dim=-1)
    angle = _uniform(b, 0.0, 2 * torch.pi, generator, x.device)
    amplitude = _uniform(b, 0.0, strength, generator, x.device)
    cos, sin = torch.cos(angle), torch.sin(angle)
    cols = torch.linspace(-1.0, 1.0, w, device=x.device)[None, None, :]
    rows = torch.linspace(-1.0, 1.0, h, device=x.device)[None, :, None]
    # In [-1, 1]: the projection on the ramp's direction, scaled so a corner reaches +-1.
    ramp = (cos[:, None, None] * cols + sin[:, None, None] * rows) / (cos.abs() + sin.abs())[:, None, None]
    light = (1 + amplitude[:, None, None] * ramp)[..., None] * gains[:, None, None, :]
    return jitter(x * light, strength, generator)


def _soft_bars(proj: torch.Tensor, period, duty, offset, soft) -> torch.Tensor:
    """1 inside the bars of a square wave along ``proj`` (px), 0 between them, edges a logistic
    of width ``soft`` px. Every argument after ``proj`` is ``(B, 1, 1)``."""
    phase = torch.remainder(proj - offset, period)
    width = duty * period
    inside = torch.minimum(phase, width - phase)                     # > 0 inside a bar
    outside = torch.minimum(phase - width, period - phase)           # > 0 between bars
    signed = torch.where(phase < width, inside, -outside)
    return torch.sigmoid(signed / soft)


def synthetic_light(b: int, h: int, w: int, generator: torch.Generator, device) -> torch.Tensor:
    """``(B, H, W, 3)`` multiplicative light: sun through a window on a surface in shade.

    Per sample: a window patch (a soft half-plane covering 30-100 % of the frame) holding lit
    panes, i.e. bars of a square wave at a random angle (period 8-45 % of the frame, duty
    0.25-0.85), crossed by a second set in half the samples; in half the samples, blob shadows
    (thresholded coarse noise) inside the light. Edges are logistic, 0.5-4 px. Shade is
    0.45-0.85 of the light and slightly blue. Draws are the same whatever is used.
    """
    u = lambda lo, hi: _uniform(b, lo, hi, generator, device)[:, None, None]
    size = float(max(h, w))
    ys, xs = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32),
                            torch.arange(w, device=device, dtype=torch.float32), indexing="ij")
    xs, ys = xs[None], ys[None]
    soft = u(0.5, 4.0)

    def along(angle):
        return torch.cos(angle) * xs + torch.sin(angle) * ys

    a1 = u(0.0, torch.pi)
    lit = _soft_bars(along(a1), u(0.08, 0.45) * size, u(0.25, 0.85), u(0.0, size), soft)
    a2 = a1 + u(torch.pi / 3, 2 * torch.pi / 3)
    cross = _soft_bars(along(a2), u(0.08, 0.45) * size, u(0.5, 0.9), u(0.0, size), soft)
    use_cross = (u(0.0, 1.0) < 0.5).float()
    lit = lit * (use_cross * cross + (1 - use_cross))
    # the window patch: a soft half-plane through a point drawn so that 30-100 % of the frame is inside
    a3 = u(0.0, 2 * torch.pi)
    proj = along(a3)
    lo, hi = proj.amin(dim=(1, 2), keepdim=True), proj.amax(dim=(1, 2), keepdim=True)
    cut = lo + (hi - lo) * u(0.0, 0.7)
    lit = lit * torch.sigmoid((proj - cut) / soft)
    # blob shadows: coarse noise, upsampled and thresholded
    grid = torch.rand((b, 1, 6, 6), generator=generator, device=device)
    noise = F.interpolate(grid, size=(h, w), mode="bicubic", align_corners=False)[:, 0]
    blobs = torch.sigmoid((noise - u(0.45, 0.75)) * 255.0 / (soft * 16.0))
    use_blobs = (u(0.0, 1.0) < 0.5).float()
    lit = lit * (1 - use_blobs * blobs)
    shade = u(0.45, 0.85)[..., None]
    tint = u(0.0, 0.06)[..., None]
    shade_rgb = shade * torch.cat([1 - tint, torch.ones_like(tint), 1 + tint], dim=-1)
    light = lit[..., None] + (1 - lit[..., None]) * shade_rgb
    # auto-exposure: divide by a power 0-1 of the mean light
    mean = light.mean(dim=(1, 2, 3), keepdim=True)
    return light / mean ** u(0.0, 1.0)[..., None]


def warped_maps(bank: torch.Tensor, b: int, generator: torch.Generator) -> torch.Tensor:
    """``(B, H, W, 3)`` light from a bank of ratio maps ``(N, H, W, 3)``: a random map per sample,
    shifted up to 10 % of the frame, rotated up to 10 degrees, scaled 0.9-1.1 (edges repeated),
    and raised to a power 0.6-1.2 (weaker or stronger shadows)."""
    device = bank.device
    index = torch.randint(0, bank.shape[0], (b,), generator=generator, device=device)
    angle = _uniform(b, -torch.pi / 18, torch.pi / 18, generator, device)
    scale = _uniform(b, 0.9, 1.1, generator, device)
    tx, ty = (_uniform(b, -0.2, 0.2, generator, device) for _ in range(2))
    power = _uniform(b, 0.6, 1.2, generator, device)
    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    maps = bank[index].float().permute(0, 3, 1, 2)
    grid = F.affine_grid(theta, list(maps.shape), align_corners=False)
    warped = F.grid_sample(maps, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return warped.permute(0, 2, 3, 1).clamp_min(1e-3) ** power[:, None, None, None]


def shadow(
    x: torch.Tensor, p: float, generator: torch.Generator, bank: torch.Tensor | None = None,
    map_share: float = 0.0,
) -> torch.Tensor:
    """Shadows on ``(B, H, W, 3)`` float frames: with probability ``p`` each, a real map from
    ``bank`` (a ``map_share`` of those) or a synthetic pattern. Draws are the same whatever is
    chosen."""
    b, h, w, _ = x.shape
    pick = _uniform(b, 0.0, 1.0, generator, x.device)
    light = synthetic_light(b, h, w, generator, x.device)
    use_map = pick < p * map_share
    use_synth = (pick >= p * map_share) & (pick < p)
    if bank is not None:
        if tuple(bank.shape[1:3]) != (h, w):
            raise ValueError(f"shadow maps are {tuple(bank.shape[1:3])}, frames are {(h, w)}")
        mapped = warped_maps(bank, b, generator)
        light = torch.where(use_map[:, None, None, None], mapped, light)
    chosen = (use_map | use_synth) if bank is not None else use_synth
    light = torch.where(chosen[:, None, None, None], light, torch.ones_like(light))
    return (x * light).clamp(0.0, 255.0)


def augment_images(
    images: dict[str, torch.Tensor],
    shift_px: int,
    color: float,
    generator: torch.Generator,
    daylight_strength: float = 0.0,
    shadow_p: float = 0.0,
    shadow_maps: torch.Tensor | None = None,
    shadow_map_share: float = 0.0,
    shadow_cameras: tuple[str, ...] | None = None,
    shadow_map_camera: str = "front",
) -> dict[str, torch.Tensor]:
    """Shift, colour-jitter, relight and shade every camera's ``(B, H, W, 3)`` uint8 frames.

    With every setting 0 the input is returned as is and nothing is drawn from
    ``generator``; a setting at 0 draws nothing, so the draws of the others are unchanged.
    Shadows go on ``shadow_cameras`` (default every camera); ``shadow_maps`` only on
    ``shadow_map_camera``.
    """
    if shift_px < 0:
        raise ValueError(f"shift_px must be >= 0, got {shift_px}")
    if not 0.0 <= color < 1.0:
        raise ValueError(f"color must lie in [0, 1), got {color}")
    if not 0.0 <= daylight_strength < 1.0:
        raise ValueError(f"daylight_strength must lie in [0, 1), got {daylight_strength}")
    if not 0.0 <= shadow_p <= 1.0 or not 0.0 <= shadow_map_share <= 1.0:
        raise ValueError(f"shadow_p and shadow_map_share must lie in [0, 1], got {shadow_p}, {shadow_map_share}")
    if shadow_map_share and shadow_maps is None:
        raise ValueError("shadow_map_share needs shadow_maps")
    if shift_px == 0 and color == 0.0 and daylight_strength == 0.0 and shadow_p == 0.0:
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
        if daylight_strength:
            x = daylight(x, daylight_strength, generator)
        if shadow_p and (shadow_cameras is None or camera in shadow_cameras):
            on_maps = shadow_maps is not None and camera == shadow_map_camera
            x = shadow(x, shadow_p, generator, shadow_maps if on_maps else None,
                       shadow_map_share if on_maps else 0.0)
        out[camera] = x.round().to(torch.uint8)
    return out
