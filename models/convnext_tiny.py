# models/convnext_tiny.py
# ConvNeXt-Tiny (from-scratch, no timm) for binary classification (num_classes=1)
# Works with your training code: output logits shape (B, 1)

from __future__ import annotations

import torch
import torch.nn as nn


class DropPath(nn.Module):
    """Stochastic Depth (per-sample)."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or (not self.training):
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, device=x.device, dtype=x.dtype)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class LayerNorm2d(nn.Module):
    """
    LayerNorm for NCHW feature maps, normalizing over channel dimension.
    Implemented by permuting to NHWC, applying LN math, then permuting back.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        mean = x.mean(dim=-1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=-1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x * self.weight + self.bias
        return x.permute(0, 3, 1, 2)


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt block:
      DWConv 7x7 -> LN -> PWConv 1x1 (expand 4x) -> GELU -> PWConv 1x1 (project)
      + residual with DropPath
    """

    def __init__(self, dim: int, drop_path: float = 0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return shortcut + self.drop_path(x)


class Downsample(nn.Module):
    """Stage transition downsampling: LN -> Conv2d(2x2, stride=2)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.norm = LayerNorm2d(in_dim)
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        return self.conv(x)


class ConvNeXtTiny(nn.Module):
    """
    ConvNeXt-Tiny configuration:
      depths = [3, 3, 9, 3]
      dims   = [96, 192, 384, 768]
    """

    def __init__(self, num_classes: int = 1, drop_path_rate: float = 0.1):
        super().__init__()
        depths = [3, 3, 9, 3]
        dims = [96, 192, 384, 768]

        # Stem: patchify-like conv 4x4 stride 4
        self.stem = nn.Conv2d(3, dims[0], kernel_size=4, stride=4)

        # Stochastic depth schedule (linear)
        dp_rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        cur = 0

        layers = []
        for stage_idx in range(4):
            blocks = []
            for j in range(depths[stage_idx]):
                blocks.append(ConvNeXtBlock(dims[stage_idx], drop_path=dp_rates[cur + j]))
            cur += depths[stage_idx]
            layers.append(nn.Sequential(*blocks))

            if stage_idx < 3:
                layers.append(Downsample(dims[stage_idx], dims[stage_idx + 1]))

        self.stages = nn.ModuleList(layers)

        # Head: global average pooling -> LN -> Linear
        self.head_norm = nn.LayerNorm(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for layer in self.stages:
            x = layer(x)

        x = x.mean(dim=(2, 3))  # GAP: (B, C, H, W) -> (B, C)
        x = self.head_norm(x)
        x = self.head(x)        # logits: (B, num_classes)
        return x


def create_model(
    model_name: str = "convnext_tiny",
    pretrained: bool = False,
    num_classes: int = 1,
    drop_path_rate: float = 0.1,
) -> nn.Module:
    """
    Factory to match your previous `timm.create_model(...)` usage pattern.
    Note: `pretrained=True` is accepted but ignored (random init) in this from-scratch version.
    """
    _ = model_name, pretrained
    return ConvNeXtTiny(num_classes=num_classes, drop_path_rate=drop_path_rate)
