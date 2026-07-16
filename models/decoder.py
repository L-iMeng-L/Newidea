"""
ASPP + U-Net hybrid decoders for 3-head nuclei segmentation.

Architecture (per Design Doc §2):
    s3 (/32, 768ch) → Shared ASPP(dil[1,3,6,12,18]) → 256ch上下文
    Each decoder: ASPP_feat → DecoderBlock×3 + encoder skips → 64ch output

Also includes Selective Kernel (SK) fusion for multi-scale features (§5).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, Optional


# ==============================================================================
#  Building blocks
# ==============================================================================

class ConvBlock(nn.Module):
    """Conv3×3 → BN → GELU."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )
    def forward(self, x): return self.conv(x)


class SharedASPP(nn.Module):
    """DeepLabV3-style ASPP operating on the deepest encoder feature (s3, /32).

    dilations [1,3,6,12,18] + global avg pool → fuse to out_ch.
    Shared across all 3 decoders — computed once, reused.  (§2)
    """

    def __init__(self, in_ch: int, out_ch: int = 256,
                 dilations: Sequence[int] = (1, 3, 6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList()
        for d in dilations:
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
            ))
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )
        self.fuse = nn.Conv2d(out_ch * (len(dilations) + 1), out_ch, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        brs = [b(x) for b in self.branches]
        gap = F.interpolate(self.gap(x), size=size, mode='nearest')
        return self.fuse(torch.cat(brs + [gap], dim=1))


class DecoderBlock(nn.Module):
    """U-Net-style upsample block with skip connection + Dropout2d.  (§2, §9)

        x [in_ch] ──→ upsample×2 ──→ concat(skip) ──→ Conv×N ──→ Dropout ──→ [out_ch]

    upsample_mode:
        'nearest'   – F.interpolate (current behaviour, no extra params)
        'transpose' – ConvTranspose2d (learnable upsampling)
    n_convs: number of Conv3×3→BN→GELU blocks (default 2)
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.1,
                 upsample_mode: str = 'nearest', n_convs: int = 2):
        super().__init__()
        self.upsample_mode = upsample_mode
        if upsample_mode == 'transpose':
            self.upsample = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        else:
            self.upsample = None

        layers = []
        in_c = in_ch + skip_ch
        for i in range(n_convs):
            layers += [
                nn.Conv2d(in_c if i == 0 else out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
            ]
        self.conv = nn.Sequential(*layers)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if self.upsample is not None:
            x = self.upsample(x)
        else:
            x = F.interpolate(x, scale_factor=2, mode=self.upsample_mode)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return self.dropout(x)


# ==============================================================================
#  Selective Kernel Fusion  (§5)
# ==============================================================================

class SelectiveKernelFusion(nn.Module):
    """Per-position adaptive weighting of 4 encoder scales.

    Projects s0..s3 to embed_dim, then learns a 4-way soft-attention weight
    per spatial position via channel pooling → FC → softmax.
    """

    def __init__(self, enc_channels: Sequence[int], embed_dim: int = 256):
        super().__init__()
        self.proj0 = nn.Conv2d(enc_channels[0], embed_dim, 1, bias=False)
        self.proj1 = nn.Conv2d(enc_channels[1], embed_dim, 1, bias=False)
        self.proj2 = nn.Conv2d(enc_channels[2], embed_dim, 1, bias=False)
        self.proj3 = nn.Conv2d(enc_channels[3], embed_dim, 1, bias=False)

        self.attn_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(embed_dim, embed_dim // 4, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(embed_dim // 4, embed_dim * 4, 1, bias=False),
        )

    def forward(self, features: list, target_size) -> torch.Tensor:
        f0, f1, f2, f3 = features
        p0 = self.proj0(f0)
        p1 = F.interpolate(self.proj1(f1), size=target_size, mode='nearest')
        p2 = F.interpolate(self.proj2(f2), size=target_size, mode='nearest')
        p3 = F.interpolate(self.proj3(f3), size=target_size, mode='nearest')

        fused = p0 + p1 + p2 + p3
        B, C = fused.shape[:2]
        attn = self.attn_fc(fused).view(B, 4, C, 1, 1)
        attn = attn.softmax(dim=1)

        stacked = torch.stack([p0, p1, p2, p3], dim=1)  # [B, 4, C, H, W]
        return (attn * stacked).sum(dim=1)


# ==============================================================================
#  Per-branch U-Net decoders
# ==============================================================================

