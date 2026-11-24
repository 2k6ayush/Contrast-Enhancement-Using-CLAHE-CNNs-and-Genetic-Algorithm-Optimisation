from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from cnn.dataset import PairedImageDataset, PairRecord, create_or_load_split_manifest
from cnn.inference import enhance_with_cnn, load_checkpoint
from cnn.losses import L1SSIMLoss, ssim
from cnn.model import build_model
from cnn.train import save_checkpoint


def test_model_odd_shape_forward_backward_and_finite_gradients():
    model = build_model(base_channels=4)
    inputs = torch.rand(2, 3, 31, 29)
    targets = torch.rand_like(inputs)
    output = model(inputs)
    assert output.shape == inputs.shape
    loss = L1SSIMLoss()(output, targets)
    loss.backward()
    assert torch.isfinite(loss)
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)


def test_ssim_identical_is_better_than_different():
    first = torch.rand(1, 3, 24, 25)
    identical = ssim(first, first)
    different = ssim(first, 1 - first)
    assert identical.item() == pytest.approx(1.0, abs=1e-5)
    assert identical > different
    assert L1SSIMLoss()(first, first).item() == pytest.approx(0.0, abs=1e-6)


def _write_pair(root: Path, name: str, low: np.ndarray, high: np.ndarray, subset="our485"):
    for label, array in (("low", low), ("high", high)):
        folder = root / subset / label
        folder.mkdir(parents=True, exist_ok=True)
        Image.fromarray(array).save(folder / name)


def test_paired_transform_uses_identical_crop_and_flips(tmp_path: Path):
    coordinates = np.arange(20 * 18 * 3, dtype=np.uint16).reshape(20, 18, 3) % 200
    low = coordinates.astype(np.uint8)
    high = (coordinates + 20).astype(np.uint8)
    _write_pair(tmp_path, "pair.png", low, high)
    record = PairRecord("pair.png", tmp_path / "our485/low/pair.png", tmp_path / "our485/high/pair.png")
    random.seed(7)
    item = PairedImageDataset([record], crop_size=12, augment=True)[0]
    difference = item["target"] - item["input"]
    assert item["input"].shape == (3, 12, 12)
    assert torch.allclose(difference, torch.full_like(difference, 20 / 255), atol=1e-6)


def test_manifest_keeps_exact_duplicates_together_and_excludes_eval15(tmp_path: Path):
    rng = np.random.default_rng(4)
    arrays = [rng.integers(0, 256, (12, 13, 3), dtype=np.uint8) for _ in range(5)]
    # Pair b is an exact duplicate of pair a in both domains.
    for name, array in zip(("a.png", "b.png", "c.png", "d.png", "e.png", "f.png"), [arrays[0], arrays[0], *arrays[1:]]):
        _write_pair(tmp_path, name, array, np.clip(array.astype(int) + 3, 0, 255).astype(np.uint8))
    test_array = rng.integers(0, 256, (12, 13, 3), dtype=np.uint8)
    _write_pair(tmp_path, "test.png", test_array, test_array, subset="eval15")
    manifest = create_or_load_split_manifest(tmp_path, tmp_path / "split.json", validation_fraction=0.33, seed=42)
    split_by_name = {item["name"]: item["split"] for item in manifest["pairs"]}
    assert split_by_name["a.png"] == split_by_name["b.png"]
    assert "test.png" not in split_by_name
    assert {item["split"] for item in manifest["pairs"]} == {"train", "validation"}
    assert manifest["test_overlap_check"]["exact_low_overlaps"] == []


def test_manifest_rejects_exact_test_leakage(tmp_path: Path):
    array = np.random.default_rng(8).integers(0, 256, (10, 11, 3), dtype=np.uint8)
    for index in range(3):
        current = (array.astype(np.uint16) + index).clip(0, 255).astype(np.uint8)
        _write_pair(tmp_path, f"{index}.png", current, current)
    _write_pair(tmp_path, "leak.png", array, 255 - array, subset="eval15")
    with pytest.raises(ValueError, match="duplicate leakage"):
        create_or_load_split_manifest(tmp_path, tmp_path / "split.json", 0.33, 1)


def test_checkpoint_round_trip_prediction_and_uint8_odd_inference(tmp_path: Path):
    torch.manual_seed(5)
    model = build_model(base_channels=4).eval()
    metadata = {"input_mode": "baseline", "test_only": True, "note": "not trained enhancement weights"}
    path = tmp_path / "tiny-test-only.pt"
    save_checkpoint(path, model, metadata, epoch=0)
    loaded, loaded_metadata, checkpoint = load_checkpoint(path, "cpu", "baseline", allow_test_only=True)
    tensor = torch.rand(1, 3, 17, 19)
    with torch.inference_mode():
        assert torch.equal(model(tensor), loaded(tensor))
    image = np.random.default_rng(2).integers(0, 256, (33, 35, 3), dtype=np.uint8)
    output = enhance_with_cnn(image, loaded)
    assert output.shape == image.shape
    assert output.dtype == np.uint8
    assert output.min() >= 0 and output.max() <= 255
    assert loaded_metadata["test_only"] is True
    assert checkpoint["format"] == "clahe-cnn-v1"


def test_checkpoint_rejects_wrong_input_mode(tmp_path: Path):
    model = build_model(base_channels=4)
    path = tmp_path / "refinement.pt"
    save_checkpoint(path, model, {"input_mode": "ga_clahe_refinement", "test_only": True}, epoch=0)
    with pytest.raises(ValueError, match="expected 'baseline'"):
        load_checkpoint(path, "cpu", "baseline", allow_test_only=True)


def test_checkpoint_rejects_test_only_weights_for_deployment(tmp_path: Path):
    model = build_model(base_channels=4)
    path = tmp_path / "test-only.pt"
    save_checkpoint(path, model, {"input_mode": "baseline", "test_only": True}, epoch=0)
    with pytest.raises(ValueError, match="test_only"):
        load_checkpoint(path, "cpu", "baseline")


def test_manifest_is_reused_not_rewritten(tmp_path: Path):
    rng = np.random.default_rng(1)
    for index in range(4):
        low = rng.integers(0, 256, (8, 9, 3), dtype=np.uint8)
        _write_pair(tmp_path, f"{index}.png", low, low)
    path = tmp_path / "split.json"
    first = create_or_load_split_manifest(tmp_path, path, 0.25, 9)
    second = create_or_load_split_manifest(tmp_path, path, 0.5, 999)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
