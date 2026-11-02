"""Training-time image augmentation: shapes, the identity at zero, exact shifts, jitter bounds,
daylight's smooth relighting."""

from __future__ import annotations

import pytest
import torch

from flowcl.data.augment import augment_images


def gen(seed=0):
    return torch.Generator().manual_seed(seed)


def frames(b=4, h=32, w=32, value=None, seed=0):
    if value is not None:
        return torch.full((b, h, w, 3), value, dtype=torch.uint8)
    return torch.randint(0, 256, (b, h, w, 3), generator=gen(seed), dtype=torch.uint8)


def test_zero_settings_are_the_identity_and_draw_nothing():
    images = {"front": frames(), "wrist": frames(seed=1)}
    g = gen()
    before = g.get_state()
    assert augment_images(images, 0, 0.0, g) is images
    assert augment_images(images, 0, 0.0, g, daylight_strength=0.0) is images
    assert torch.equal(g.get_state(), before)


def test_shapes_dtype_and_input_are_kept():
    images = {"front": frames(), "wrist": frames(seed=1)}
    copies = {k: v.clone() for k, v in images.items()}
    out = augment_images(images, 4, 0.2, gen())
    for cam in images:
        assert out[cam].shape == images[cam].shape and out[cam].dtype == torch.uint8
        assert torch.equal(images[cam], copies[cam])


def test_shift_moves_content_by_the_drawn_offset():
    b, h, w, p = 6, 20, 24, 3
    x = torch.zeros((b, h, w, 3), dtype=torch.uint8)
    x[:, 10, 12] = 255
    out = augment_images({"cam": x}, p, 0.0, gen(5))["cam"]
    replay = gen(5)
    oy = torch.randint(0, 2 * p + 1, (b,), generator=replay)
    ox = torch.randint(0, 2 * p + 1, (b,), generator=replay)
    for i in range(b):
        # out[i, r, c] = x[i, oy + r - p, ox + c - p]
        r, c = 10 - int(oy[i]) + p, 12 - int(ox[i]) + p
        assert out[i, r, c].tolist() == [255, 255, 255]
        assert int(out[i].sum()) == 255 * 3


def test_brightness_stays_within_its_band():
    # A uniform gray image: contrast and saturation leave it alone, brightness scales it.
    out = augment_images({"cam": frames(b=64, value=100)}, 0, 0.2, gen())["cam"]
    values = out[:, 0, 0, 0].float()
    assert (out == out[:, :1, :1, :1]).all()
    assert values.min() >= 80 and values.max() <= 120 and values.std() > 5


def test_same_seed_same_output_and_cameras_differ():
    images = {"front": frames(), "wrist": frames()}
    a = augment_images(images, 4, 0.2, gen(3))
    b = augment_images(images, 4, 0.2, gen(3))
    c = augment_images(images, 4, 0.2, gen(4))
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert not torch.equal(a["front"], c["front"])
    # each camera gets its own draws, so identical inputs come out differently
    assert not torch.equal(a["front"], a["wrist"])


def test_daylight_relights_a_flat_scene_with_a_smooth_ramp():
    # A uniform gray scene stays a plane in every channel: the ramp is linear across the
    # image and the jitter after it is affine per pixel. Both warm and cool light occur.
    b, h, w, s = 64, 16, 20, 0.3
    out = augment_images({"cam": frames(b=b, h=h, w=w, value=100)}, 0, 0.0, gen(2), daylight_strength=s)["cam"]
    y = out.double().reshape(b, h * w, 3)
    rows, cols = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    design = torch.stack([torch.ones(h * w), rows.flatten(), cols.flatten()], dim=1).double()
    fit = torch.linalg.lstsq(design.expand(b, -1, -1), y).solution
    assert (y - design @ fit).abs().max() <= 1.0  # rounding to uint8 only
    slope = fit[:, 1:].abs().amax(dim=(1, 2)) * max(h, w)
    assert (slope > 1).float().mean() >= 0.8 and 0 < out.min() and out.max() < 255
    warmth = out[..., 0].double().mean(dim=(1, 2)) - out[..., 2].double().mean(dim=(1, 2))
    assert (warmth > 2).any() and (warmth < -2).any()


def test_daylight_combines_with_shift_and_colour():
    images = {"front": frames(), "wrist": frames(seed=1)}
    plain = augment_images(images, 4, 0.2, gen(3))
    lit = augment_images(images, 4, 0.2, gen(3), daylight_strength=0.3)
    again = augment_images(images, 4, 0.2, gen(3), daylight_strength=0.3)
    assert all(torch.equal(lit[k], again[k]) for k in lit)
    assert not torch.equal(plain["front"], lit["front"])


@pytest.mark.parametrize("shift_px, color, daylight", [(-1, 0.0, 0.0), (0, -0.1, 0.0), (0, 1.0, 0.0),
                                                       (0, 0.0, -0.1), (0, 0.0, 1.0)])
def test_bad_settings_raise(shift_px, color, daylight):
    with pytest.raises(ValueError):
        augment_images({"cam": frames()}, shift_px, color, gen(), daylight_strength=daylight)
