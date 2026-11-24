"""Differentiable reconstruction losses with stable float32 SSIM."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _gaussian_window(size: int, sigma: float, channels: int, device, dtype) -> torch.Tensor:
    coordinates = torch.arange(size, device=device, dtype=dtype) - size // 2
    kernel_1d = torch.exp(-(coordinates.square()) / (2 * sigma * sigma))
    kernel_1d /= kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d.expand(channels, 1, size, size).contiguous()


def ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    data_range: float = 1.0,
) -> torch.Tensor:
    """Return mean SSIM for NCHW tensors in a common numeric range."""
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("SSIM expects prediction and target with identical NCHW shapes")
    prediction = prediction.float()
    target = target.float()
    size = min(window_size, prediction.shape[-2], prediction.shape[-1])
    if size % 2 == 0:
        size -= 1
    size = max(size, 1)
    window = _gaussian_window(size, 1.5, prediction.shape[1], prediction.device, prediction.dtype)
    padding = size // 2
    mu_x = F.conv2d(prediction, window, padding=padding, groups=prediction.shape[1])
    mu_y = F.conv2d(target, window, padding=padding, groups=target.shape[1])
    mu_x_sq, mu_y_sq, mu_xy = mu_x.square(), mu_y.square(), mu_x * mu_y
    sigma_x = F.conv2d(prediction.square(), window, padding=padding, groups=prediction.shape[1]) - mu_x_sq
    sigma_y = F.conv2d(target.square(), window, padding=padding, groups=target.shape[1]) - mu_y_sq
    sigma_xy = F.conv2d(prediction * target, window, padding=padding, groups=prediction.shape[1]) - mu_xy
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    score = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x_sq + mu_y_sq + c1) * (sigma_x + sigma_y + c2)
    )
    return score.mean()


class L1SSIMLoss(nn.Module):
    def __init__(self, l1_weight: float = 0.8, ssim_weight: float = 0.2) -> None:
        super().__init__()
        if l1_weight < 0 or ssim_weight < 0 or l1_weight + ssim_weight <= 0:
            raise ValueError("Loss weights must be non-negative and not both zero")
        total = l1_weight + ssim_weight
        self.l1_weight = l1_weight / total
        self.ssim_weight = ssim_weight / total

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Callers may use autocast; calculate the sensitive terms in float32.
        prediction_f = prediction.float()
        target_f = target.float()
        return self.l1_weight * F.l1_loss(prediction_f, target_f) + self.ssim_weight * (
            1.0 - ssim(prediction_f, target_f)
        )
