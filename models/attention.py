"""
Lightweight attention modules pluggable into segmentation heads.

- ECA (Efficient Channel Attention): 1D conv across channels, no dimensionality reduction.
- SpatialAttention: channel-pooled spatial map → single-channel attention.
- CBAM_Light: ECA → SpatialAttention, sequential.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ECA(nn.Module):
    """Efficient Channel Attention (Wang et al., CVPR 2020).

    Learns channel weights via a 1D conv of adaptively-sized kernel,
    avoiding the squeeze-and-excitation bottleneck (FC → ReLU → FC).
    """

    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        # Adaptive kernel size: k = |log2(C)/γ + b/γ|_odd
        t = int(abs((torch.log2(torch.tensor(channels, dtype=torch.float32)) + b) / gamma))
        k = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        y = x.mean(dim=[2, 3], keepdim=True)          # [B, C, 1, 1]  → GAP
        y = y.squeeze(-1).transpose(-1, -2)            # [B, 1, C]
        y = self.conv(y)                                # [B, 1, C]
        y = y.transpose(-1, -2).unsqueeze(-1)           # [B, C, 1, 1]
        return x * y.sigmoid()


class SpatialAttention(nn.Module):
    """Channel-pooled spatial attention (Woo et al., ECCV 2018).

    max-pool + avg-pool along channel dim → 2-channel map → 7×7 conv → sigmoid.
    Highlights *where* the informative regions are.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        avg = x.mean(dim=1, keepdim=True)              # [B, 1, H, W]
        max_ = x.max(dim=1, keepdim=True)[0]           # [B, 1, H, W]
        y = torch.cat([avg, max_], dim=1)              # [B, 2, H, W]
        y = self.conv(y).sigmoid()                      # [B, 1, H, W]
        return x * y


class CBAM_Light(nn.Module):
    """Lightweight CBAM: ECA (channel) → SpatialAttention (spatial).

    Optimised for small nuclei (10–17 px) — ECA's 1D conv kernel adapts to
    channel count, and the spatial attention uses a 7×7 conv for moderate
    receptive field.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channel_att = ECA(channels)
        self.spatial_att = SpatialAttention(kernel_size=7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x
