"""Compact U-Net used by both the baseline and refinement experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 3
    out_channels: int = 3
    base_channels: int = 32


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = min(8, out_channels)
        while out_channels % groups:
            groups -= 1
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, out_channels, 1)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat((self.reduce(x), skip), dim=1))


class CompactUNet(nn.Module):
    """Two-level U-Net that pads internally and returns the original H x W."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config.base_channels
        self.enc1 = ConvBlock(self.config.in_channels, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.bottleneck = ConvBlock(c * 2, c * 4)
        self.up2 = UpBlock(c * 4, c * 2, c * 2)
        self.up1 = UpBlock(c * 2, c, c)
        self.output = nn.Conv2d(c, self.config.out_channels, 1)
        self.pool = nn.MaxPool2d(2)

    @property
    def architecture_config(self) -> dict[str, int]:
        return asdict(self.config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.config.in_channels:
            raise ValueError(
                f"Expected NCHW tensor with {self.config.in_channels} channels, got {tuple(x.shape)}"
            )
        height, width = x.shape[-2:]
        pad_h = (-height) % 4
        pad_w = (-width) % 4
        if pad_h or pad_w:
            # Replicate is defined even for one-pixel-wide synthetic test inputs.
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        bottleneck = self.bottleneck(self.pool(e2))
        decoded = self.up1(self.up2(bottleneck, e2), e1)
        return torch.sigmoid(self.output(decoded))[..., :height, :width]


def build_model(
    in_channels: int = 3,
    out_channels: int = 3,
    base_channels: int = 32,
    **_: object,
) -> CompactUNet:
    """Build an untrained model. Loading trained weights is always explicit."""
    return CompactUNet(ModelConfig(in_channels, out_channels, base_channels))


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
