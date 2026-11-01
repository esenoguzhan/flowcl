"""Training-time image augmentation: shapes, the identity at zero, exact shifts, jitter bounds."""

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


@pytest.mark.parametrize("shift_px, color", [(-1, 0.0), (0, -0.1), (0, 1.0)])
def test_bad_settings_raise(shift_px, color):
    with pytest.raises(ValueError):
        augment_images({"cam": frames()}, shift_px, color, gen())
