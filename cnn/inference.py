"""Checkpoint loading and RGB uint8 inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .model import build_model

CHECKPOINT_FORMAT = "clahe-cnn-v1"
VALID_INPUT_MODES = {"baseline", "ga_clahe_refinement"}


def _validate_rgb_uint8(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be a NumPy array, got {type(image).__name__}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must have shape HxWx3, got {image.shape}")
    if image.dtype != np.uint8:
        raise TypeError(f"image must have dtype uint8, got {image.dtype}")
    if image.shape[0] < 1 or image.shape[1] < 1:
        raise ValueError("image dimensions must be non-zero")


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    if device is None or str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return resolved


def validate_checkpoint(checkpoint: Any, expected_input_mode: str | None = None) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        raise ValueError("Unsupported checkpoint: expected the project checkpoint dictionary")
    required = {"format", "model_state", "architecture", "metadata"}
    missing = sorted(required - checkpoint.keys())
    if missing:
        raise ValueError(f"Checkpoint is missing required fields: {', '.join(missing)}")
    if checkpoint["format"] != CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint['format']!r}")
    metadata = checkpoint["metadata"]
    if not isinstance(metadata, dict) or metadata.get("input_mode") not in VALID_INPUT_MODES:
        raise ValueError("Checkpoint metadata must identify baseline or ga_clahe_refinement input_mode")
    if expected_input_mode and metadata["input_mode"] != expected_input_mode:
        raise ValueError(
            f"Checkpoint input_mode is {metadata['input_mode']!r}; expected {expected_input_mode!r}"
        )
    return checkpoint


def load_checkpoint(
    path: str | Path,
    device: str | torch.device | None = None,
    expected_input_mode: str | None = None,
    allow_test_only: bool = False,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    """Load a trained project checkpoint; raw state dictionaries are rejected."""
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    target_device = resolve_device(device)
    checkpoint = validate_checkpoint(
        torch.load(checkpoint_path, map_location=target_device, weights_only=False),
        expected_input_mode,
    )
    if checkpoint["metadata"].get("test_only") and not allow_test_only:
        raise ValueError(
            "Checkpoint is marked test_only and cannot be used as trained enhancement weights"
        )
    model = build_model(**checkpoint["architecture"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(target_device).eval()
    # Useful to UIs/evaluators; it never turns an untrained model into a trained one.
    model.checkpoint_metadata = dict(checkpoint["metadata"])
    model.checkpoint_path = str(checkpoint_path)
    return model, dict(checkpoint["metadata"]), checkpoint


def enhance_with_cnn(image: np.ndarray, model: torch.nn.Module) -> np.ndarray:
    """Enhance one RGB uint8 image while preserving its exact dimensions."""
    _validate_rgb_uint8(image)
    if model is None or not isinstance(model, torch.nn.Module):
        raise ValueError("A loaded PyTorch CNN model is required; no identity fallback is used")
    try:
        device = next(model.parameters()).device
    except StopIteration as exc:
        raise ValueError("The supplied model has no parameters") from exc
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.to(device=device, dtype=torch.float32).div_(255.0)
    model.eval()
    with torch.inference_mode():
        prediction = model(tensor)
    if prediction.shape != tensor.shape:
        raise RuntimeError(f"CNN returned shape {tuple(prediction.shape)}, expected {tuple(tensor.shape)}")
    if not torch.isfinite(prediction).all():
        raise RuntimeError("CNN output contains NaN or infinite values")
    output = prediction.clamp(0, 1).mul(255).round().to(torch.uint8)
    return output.squeeze(0).permute(1, 2, 0).cpu().numpy()
