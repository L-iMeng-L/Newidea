"""
MALA-style dynamic convolution: spatial-adaptive multi-scale kernel fusion.

Reference: CFR-SAM (Multi-scale Adaptive Local Attention)
    - 3 parallel branches: Conv3×3, Conv5×5, Conv7×7
    - Lightweight spatial gate (SE-like) generates per-pixel softmax weights
    - Dense regions → small kernel (preserve boundaries)
    - Sparse regions → large kernel (expand receptive field)

Usage:
    mala = MALABlock(256, 256, kernels=[3,5,7])
    out = mala(x)  # same shape as x
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class GateSpatial(nn.Module):
    """Spatial attention gate: input → [B, K, H, W] softmax across kernel dim."""

    def __init__(self, in_ch: int, num_kernels: int = 3, reduction: int = 4):
        super().__init__()
        mid = max(in_ch // reduction, 8)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, num_kernels, 1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] → gate: [B, K, 1, 1]
        w = self.gate(x)
        # Spatial broadcast + softmax over kernel dim
        w = F.softmax(w, dim=1)  # [B, K, 1, 1] — per-image, NOT per-pixel, to save params
        return w


class GateSpatialPerPixel(nn.Module):
    """Per-pixel spatial gate: [B, C, H, W] → [B, K, H, W] softmax.

    Heavier but spatially adaptive — different regions can prefer different kernels.
    Uses depthwise conv to keep params low.
    """

    def __init__(self, in_ch: int, num_kernels: int = 3, reduction: int = 8):
        super().__init__()
        mid = max(in_ch // reduction, 8)
        self.gate = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.GELU(),
            nn.Conv2d(mid, num_kernels, 1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.gate(x), dim=1)  # [B, K, H, W]


class MALABlock(nn.Module):
    """Multi-scale Adaptive Local Attention block.

    Replaces a single Conv3×3 with 3 parallel convolution branches (3/5/7)
    fused by a learned spatial gate. Keeps the same input/output channels.

    Parameters
    ----------
    in_ch : int
        Input channels.
    out_ch : int
        Output channels.
    kernels : list of int
        Kernel sizes (must be odd). Default [3, 5, 7].
    groups : int
        Groups for the conv branches (depthwise when groups=in_ch).
        Default 1 = standard convolution.
    per_pixel_gate : bool
        If True, use per-pixel spatial gate (heavier but more adaptive).
        If False, use per-image gate (lighter, global decision).
    expansion : float
        Expansion ratio for gate hidden dim.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernels: List[int] = None,
        groups: int = 1,
        per_pixel_gate: bool = True,
        reduction: int = 8,
    ):
        super().__init__()
        if kernels is None:
            kernels = [3, 5, 7]
        self.kernels = kernels
        self.num_kernels = len(kernels)

        # Parallel conv branches
        self.branches = nn.ModuleList()
        for k in kernels:
            pad = k // 2
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, k, padding=pad, groups=groups, bias=False),
                    nn.BatchNorm2d(out_ch),
                )
            )

        # Spatial gate
        if per_pixel_gate:
            self.gate = GateSpatialPerPixel(in_ch, self.num_kernels, reduction)
        else:
            self.gate = GateSpatial(in_ch, self.num_kernels, reduction)

        # Post-fusion
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gate weights: [B, K, 1, 1] or [B, K, H, W]
        weights = self.gate(x)  # [B, K, ...]

        # Apply each branch
        outputs = []
        for i, branch in enumerate(self.branches):
            feat = branch(x)  # [B, C, H, W]
            w = weights[:, i:i + 1]  # [B, 1, ...]
            outputs.append(feat * w)

        # Fuse
        out = sum(outputs)
        return self.act(out)


class MALADecoderBlock(nn.Module):
    """DecoderBlock with MALA dynamic convolution after upsampling.

    Replaces standard Conv3×3 in the DecoderBlock with MALA (3/5/7 fusion).
    ConvTranspose2d for upsampling → MALA for refinement.

    Structure:
        Upsample (ConvTranspose2d or F.interpolate)
        → Concat(skip)
        → MalaBlock(in_ch+skip_ch, out_ch)  ← dynamic conv here
        → MalaBlock(out_ch, out_ch)          ← second dynamic conv
    """

    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        dropout: float = 0.1,
        upsample_mode: str = 'transpose',
        mala_kernels: List[int] = None,
        mala_per_pixel: bool = True,
    ):
        super().__init__()
        if mala_kernels is None:
            mala_kernels = [3, 5, 7]

        # Upsample
        if upsample_mode == 'transpose':
            self.upsample = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        else:
            self.upsample = None
            self._upsample_mode = upsample_mode

        # MALA conv layers (replace fixed Conv3×3)
        fused_in = in_ch + skip_ch
        self.conv1 = MALABlock(fused_in, out_ch, mala_kernels,
                               per_pixel_gate=mala_per_pixel)
        self.conv2 = MALABlock(out_ch, out_ch, mala_kernels,
                               per_pixel_gate=mala_per_pixel)

        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if self.upsample is not None:
            x = self.upsample(x)
        else:
            x = F.interpolate(x, size=skip.shape[-2:],
                              mode=self._upsample_mode, align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.drop(x)
