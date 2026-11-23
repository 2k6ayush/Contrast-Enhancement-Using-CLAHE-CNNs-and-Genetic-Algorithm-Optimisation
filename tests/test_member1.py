import numpy as np
import pytest

from dataset import darken_image
from traditional import (
    apply_clahe, apply_histogram_equalization, calculate_contrast_improvement,
    calculate_edge_preservation, calculate_entropy, calculate_psnr, calculate_ssim, evaluate_all,
)


@pytest.fixture
def scene():
    rng = np.random.default_rng(1)
    base = np.zeros((128, 128, 3), np.uint8)
    base[:, :64] = 200
    base[32:96, 32:96] = (60, 120, 180)
    base = np.clip(base + rng.normal(0, 8, base.shape), 0, 255).astype(np.uint8)
    return base


@pytest.fixture
def dark(scene):
    return darken_image(scene, gamma=2.5, brightness=0.6)


def test_shape_dtype_preserved(dark):
    for out in (apply_histogram_equalization(dark), apply_clahe(dark)):
        assert out.shape == dark.shape and out.dtype == np.uint8


def test_enhancement_raises_brightness_and_contrast(dark):
    for out in (apply_histogram_equalization(dark), apply_clahe(dark, 3.0, 8)):
        assert out.mean() > dark.mean()
        assert calculate_contrast_improvement(dark, out) > 1.0


def test_clahe_clip_limit_monotonic(dark):
    weak = apply_clahe(dark, 1.0, 8)
    strong = apply_clahe(dark, 8.0, 8)
    assert strong.std() >= weak.std()


def test_clahe_tuple_grid_and_invalid_inputs(dark):
    assert apply_clahe(dark, 2.0, (4, 8)).shape == dark.shape
    with pytest.raises(ValueError):
        apply_clahe(dark, 0, 8)
    with pytest.raises(ValueError):
        apply_clahe(dark[:, :, 0])


def test_psnr_ssim_identical_and_ordering(scene, dark):
    assert calculate_ssim(scene, scene) == pytest.approx(1.0)
    assert calculate_psnr(scene, scene) > 90 or np.isinf(calculate_psnr(scene, scene))
    assert calculate_psnr(scene, scene) > calculate_psnr(dark, scene)
    assert calculate_ssim(scene, scene) > calculate_ssim(dark, scene)


def test_shape_mismatch_raises(scene):
    with pytest.raises(ValueError):
        calculate_psnr(scene, scene[:64])


def test_entropy_bounds():
    flat = np.full((32, 32, 3), 100, np.uint8)
    assert calculate_entropy(flat) == pytest.approx(0.0)
    ramp = np.tile(np.arange(256, dtype=np.uint8), (256, 1))
    ramp = np.stack([ramp] * 3, axis=2)
    assert calculate_entropy(ramp) == pytest.approx(8.0, abs=1e-6)


def test_edge_preservation_self_is_one(scene):
    assert calculate_edge_preservation(scene, scene) == pytest.approx(1.0)


def test_evaluate_all_keys(scene, dark):
    out = apply_clahe(dark)
    r = evaluate_all(dark, out, scene)
    assert {"PSNR", "SSIM", "Entropy", "Contrast Improvement", "Edge Preservation"} <= set(r)
    assert "PSNR" not in evaluate_all(dark, out)
