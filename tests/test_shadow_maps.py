"""Shadow maps for a fixed camera (flowcl.data.shadow_maps): ratios, support, filling, banks."""

from __future__ import annotations

import numpy as np

from flowcl.data.shadow_maps import load_bank, ratio_map, reference_frame, save_bank, support_mask


def scene(h=120, w=160, table=180):
    """A bright table on the right, a dark arm band across it, dark background on the left."""
    img = np.full((h, w, 3), 40, np.uint8)
    img[:, 50:] = table
    img[50:70, 50:130] = 20                                  # the arm at home
    return img


def test_the_reference_is_a_median_and_identical_frames_give_ratio_one():
    a, b = scene(), scene()
    b[10:20, 100:110] = (200, 30, 200)                       # an object in one frame only
    ref = reference_frame([a, a, b])
    assert np.array_equal(ref, a.astype(np.float32))
    support = support_mask(ref)
    r = ratio_map(scene(), ref, support)
    assert np.allclose(r, 1.0, atol=1e-3)


def test_a_shadow_on_the_table_shows_and_the_outside_stays_one():
    ref = scene().astype(np.float32)
    support = support_mask(ref)
    sun = scene()
    sun[:, 90:] = (sun[:, 90:].astype(np.float32) * 0.5).astype(np.uint8)    # shade the right of the table
    sun[50:70, 50:130] = 20
    r = ratio_map(sun, ref, support)
    assert abs(r[30, 140].mean() - 0.5) < 0.05             # in the shade, on the table
    assert abs(r[100, 70].mean() - 1.0) < 0.05             # lit table
    assert abs(r[30, 5].mean() - 1.0) < 0.02               # far outside the table: gain 1
    assert np.isfinite(r).all() and r.min() >= 0.25 and r.max() <= 2.0
    # the arm's band is filled smoothly from the table around it, with no hard edge
    assert np.abs(np.diff(r[40:80, 140, 0])).max() < 0.05


def test_objects_are_filled_and_the_size_is_the_policys():
    ref = scene().astype(np.float32)
    support = support_mask(ref)
    sun = scene()
    sun[20:30, 100:112] = (30, 160, 200)                     # a saturated plate, not light
    r = ratio_map(sun, ref, support)
    assert abs(r[25, 106].mean() - 1.0) < 0.05
    assert ratio_map(sun, ref, support, size=(48, 64)).shape == (48, 64, 3)


def test_bank_roundtrip(tmp_path):
    maps = [np.full((8, 8, 3), 0.5, np.float32), np.ones((8, 8, 3), np.float32)]
    sha = save_bank(tmp_path / "b.npz", maps, {"note": "x"})
    loaded, prov = load_bank(tmp_path / "b.npz")
    assert loaded.shape == (2, 8, 8, 3) and loaded.dtype == np.float16
    assert prov["note"] == "x" and prov["maps_sha256"] == sha
