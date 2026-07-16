"""
ConvNeXt encoder — supports V1 and V2, tiny/small/base/large variants.

Two loading modes:
    1. pretrained=True  → auto-detect local .pdparams (PaddlePaddle → PyTorch conversion)
    2. pretrained=False → timm random init (debug / quick test)

Weight directory:
    V1:  project_root/*.pdparams (legacy)
    V2:  project_root/paddle_model_22k_ft/*.pdparams (recommended)

ConvNeXt-V2 adds GRN (Global Response Normalization) after the MLP in each block.
This suppresses spurious high-activation features — particularly useful for H&E
pathology images where staining artefacts cause false-positive activation.

    ConvNeXt-V2 key mapping (Paddle → timm):
        downsample_layers.0.{idx}         → stem_{idx}
        downsample_layers.{s}.{idx}       → stages_{s}.downsample.{idx}
        stages.{s}.{b}.dwconv             → stages_{s}.blocks.{b}.conv_dw
        stages.{s}.{b}.norm               → stages_{s}.blocks.{b}.norm
        stages.{s}.{b}.pwconv1            → stages_{s}.blocks.{b}.mlp.fc1    (transpose)
        stages.{s}.{b}.pwconv2            → stages_{s}.blocks.{b}.mlp.fc2    (transpose)
        stages.{s}.{b}.grn.gamma          → stages_{s}.blocks.{b}.mlp.grn.weight (squeeze)
        stages.{s}.{b}.grn.beta           → stages_{s}.blocks.{b}.mlp.grn.bias   (squeeze)
"""
import os
import pickle
import re
from typing import Optional, Literal, Tuple

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── type aliases ──────────────────────────────────────────────────────────
ConvNeXtVersion = Literal["v1", "v2"]
ConvNeXtVariant = Literal["timm", "tiny", "small", "base", "large", "huge", "xlarge"]


# ==============================================================================
#  PaddlePaddle → PyTorch key mapping  (shared V1 & V2)
# ==============================================================================

def _paddle_to_torch_key(pp_key: str) -> Optional[str]:
    """
    Convert a PaddlePaddle param key to a timm PyTorch ConvNeXt key.

    Supports both V1 and V2 naming (V2 adds ``grn.*`` keys).
    Returns None for keys that should be skipped (head, final norm).
    """
    # Stem: downsample_layers.0.{idx}.{param} → stem_{idx}.{param}
    m = re.match(r'^downsample_layers\.0\.(\d)\.(.+)$', pp_key)
    if m:
        return f'stem_{m.group(1)}.{m.group(2)}'

    # Downsample layers 1/2/3: downsample_layers.{s}.{idx}.{param}
    #   → stages_{s}.downsample.{idx}.{param}
    m = re.match(r'^downsample_layers\.([123])\.([01])\.(.+)$', pp_key)
    if m:
        s, idx, rest = m.groups()
        return f'stages_{s}.downsample.{idx}.{rest}'

    # Stage blocks: stages.{s}.{b}.submodule → stages_{s}.blocks.{b}.submodule
    m = re.match(r'^stages\.(\d+)\.(\d+)\.(.+)$', pp_key)
    if m:
        s, b, rest = m.groups()
        rest = rest.replace('dwconv', 'conv_dw')
        rest = rest.replace('pwconv1', 'mlp.fc1')
        rest = rest.replace('pwconv2', 'mlp.fc2')
        # V2  GRN: grn.gamma → mlp.grn.weight  /  grn.beta → mlp.grn.bias
        rest = rest.replace('grn.gamma', 'mlp.grn.weight')
        rest = rest.replace('grn.beta', 'mlp.grn.bias')
        return f'stages_{s}.blocks.{b}.{rest}'

    # Final norm / head — not used in features_only mode
    if pp_key.startswith('norm.') or pp_key.startswith('head.'):
        return None

    return pp_key


def _paddle_to_torch_tensor(torch_key: str, pp_value: np.ndarray) -> torch.Tensor:
    """
    Convert a single PaddlePaddle weight to PyTorch format.

    Differences:
        - Paddle Linear weight: [in_features, out_features]
        - PyTorch Linear weight: [out_features, in_features]
        → transpose mlp.fc1 / mlp.fc2 weights
        - Paddle GRN gamma/beta: [1, 1, 1, C]  → PyTorch GRN: [C]
    """
    tensor = torch.from_numpy(pp_value.copy()).float()

    # Transpose linear layers
    if tensor.ndim == 2 and ('mlp.fc1' in torch_key or 'mlp.fc2' in torch_key):
        tensor = tensor.t()

    # Squeeze GRN params  (Paddle: [1,1,1,C] → timm: [C])
    if 'mlp.grn.weight' in torch_key or 'mlp.grn.bias' in torch_key:
        if tensor.ndim == 4:
            tensor = tensor.squeeze()

    return tensor


def load_paddle_weights(filepath: str, verbose: bool = True) -> dict:
    """Load a .pdparams file and convert to a timm-compatible state dict."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Local weights not found: {filepath}")

    with open(filepath, 'rb') as f:
        pp_state = pickle.load(f)

    pp_state.pop('StructuredToParameterName@@', None)

    torch_state = {}
    skipped = []
    for pp_key, pp_value in pp_state.items():
        torch_key = _paddle_to_torch_key(pp_key)
        if torch_key is None:
            skipped.append(pp_key)
            continue
        torch_state[torch_key] = _paddle_to_torch_tensor(torch_key, pp_value)

    if verbose:
        print(f"  Loaded {len(torch_state)} params from {os.path.basename(filepath)}")
        if skipped:
            print(f"  Skipped {len(skipped)} head/norm params")

    return torch_state


# ==============================================================================
#  Local weight registry
# ==============================================================================

# V1 names  (paddle_model_22k/)
_V1_DIR = 'paddle_model_22k'
_V1_WEIGHTS = {
    'tiny':   'convnext_tiny.pdparams',
    'small':  'convnext_small.pdparams',
    'base':   'convnext_base_22k_1k_224.pdparams',
    'large':  'convnext_large_22k_1k_224.pdparams',
    'xlarge': 'convnext_xlarge_22k_1k_224_ema.pdparams',
}
_V1_WEIGHTS_384 = {
    'base':   'convnext_base_22k_1k_384.pdparams',
    'large':  'convnext_large_22k_1k_384.pdparams',
    'xlarge': 'convnext_xlarge_22k_1k_384_ema.pdparams',
}

# V2 names  (paddle_model_22k_ft/)
# Note: small/huge exist in timm but may not have local .pdparams — fallback to timm pretrained
_V2_DIR = 'paddle_model_22k_ft'
_V2_WEIGHTS = {
    'tiny':  'convnextv2_tiny.pdparams',
    'nano':  'convnextv2_nano.pdparams',
    'small': 'convnextv2_small.pdparams',
    'base':  'convnextv2_base.pdparams',
    'large': 'convnextv2_large.pdparams',
    'huge':  'convnextv2_huge.pdparams',
}


def find_local_weights(
    variant: str,
    version: ConvNeXtVersion = "v2",
    search_dir: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Find local .pdparams for the given variant + version.

    Returns (path, version_string) or (None, None).
    """
    if search_dir is None:
        search_dir = os.path.dirname(os.path.abspath(__file__))
        search_dir = os.path.dirname(search_dir)  # project root

    if version == "v2":
        v2_dir = os.path.join(search_dir, _V2_DIR)
        if variant in _V2_WEIGHTS:
            path = os.path.join(v2_dir, _V2_WEIGHTS[variant])
            if os.path.exists(path):
                return path, "v2"
    else:
        v1_dir = os.path.join(search_dir, _V1_DIR)
        for img_size in [384, 224]:
            if img_size == 384 and variant in _V1_WEIGHTS_384:
                path = os.path.join(v1_dir, _V1_WEIGHTS_384[variant])
            elif variant in _V1_WEIGHTS:
                path = os.path.join(v1_dir, _V1_WEIGHTS[variant])
            else:
                continue
            if os.path.exists(path):
                return path, f"v1-{img_size}"

    return None, None


# ==============================================================================
#  ConvNeXt Encoder
# ==============================================================================

class ConvNeXtEncoder(nn.Module):
    """ConvNeXt (V1 or V2) encoder outputting 4 multi-scale feature maps.

    Feature map sizes (256×256 input):
        feat0: [B, C0, 64, 64]   1/4 resolution
        feat1: [B, C1, 32, 32]   1/8
        feat2: [B, C2, 16, 16]   1/16
        feat3: [B, C3,  8,  8]   1/32

    Channels by variant:
        nano:       [80,  160, 320,  640]
        tiny:       [96,  192, 384,  768]
        small:      [96,  192, 384,  768]
        base:       [128, 256, 512,  1024]
        large:      [192, 384, 768,  1536]
        huge:       [352, 704, 1408, 2816]
        xlarge:     [256, 512, 1024, 2048]
    """

    _V1_MAP = {
        "tiny": "convnext_tiny", "small": "convnext_small",
        "base": "convnext_base", "large": "convnext_large",
        "xlarge": "convnext_xlarge",
    }
    _V2_MAP = {
        "nano": "convnextv2_nano", "tiny": "convnextv2_tiny",
        "small": "convnextv2_small", "base": "convnextv2_base",
        "large": "convnextv2_large", "huge": "convnextv2_huge",
    }

    def __init__(
        self,
        variant: ConvNeXtVariant = "tiny",
        version: ConvNeXtVersion = "v2",
        pretrained: bool = True,
        local_weights: Optional[str] = None,
        out_indices: tuple = (0, 1, 2, 3),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.variant = variant
        self.version = version

        # Resolve timm model name
        model_map = self._V2_MAP if version == "v2" else self._V1_MAP
        if variant not in model_map:
            raise ValueError(
                f"Unknown {version} variant: {variant}. "
                f"Choose from {list(model_map.keys())}"
            )
        timm_name = model_map[variant]

        # ---- Auto-detect local weights ----
        if pretrained and local_weights is None:
            found_path, found_ver = find_local_weights(variant, version=version)
            if found_path:
                local_weights = found_path
                print(f"  Found local {found_ver} weights: {os.path.basename(found_path)}")

        # ---- Build model ----
        if local_weights and os.path.exists(local_weights):
            self.backbone = timm.create_model(
                timm_name, pretrained=False, features_only=True, out_indices=out_indices,
            )
            state_dict = load_paddle_weights(local_weights)
            missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"  Warning: {len(missing)} missing keys (head/aux, expected)")
            if unexpected:
                print(f"  Warning: {len(unexpected)} unexpected keys")
        elif pretrained:
            # Download from HuggingFace / torch hub
            self.backbone = timm.create_model(
                timm_name, pretrained=True, features_only=True, out_indices=out_indices,
            )
        else:
            self.backbone = timm.create_model(
                timm_name, pretrained=False, features_only=True, out_indices=out_indices,
            )

        self.channels = self.backbone.feature_info.channels()

        # Per-scale dropout (applied after each backbone output)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> list:
        """Returns [feat0, feat1, feat2, feat3] high→low resolution."""
        feats = self.backbone(x)
        return [self.dropout(f) for f in feats]


def create_encoder(
    variant: ConvNeXtVariant = "tiny",
    version: ConvNeXtVersion = "v2",
    pretrained: bool = True,
    local_weights: Optional[str] = None,
    dropout: float = 0.1,
) -> ConvNeXtEncoder:
    return ConvNeXtEncoder(
        variant=variant, version=version,
        pretrained=pretrained, local_weights=local_weights,
        dropout=dropout,
    )


# ==============================================================================
#  UNI2-h Encoder (ViT-based, frozen backbone + trainable FPN)
# ==============================================================================

class UNI2Encoder(nn.Module):
    """UNI2-h ViT encoder — multi-scale features from intermediate transformer blocks.

    - Builds ViT-L architecture locally (no network needed).
    - Loads pretrained weights from local file.
    - ViT backbone is frozen by default; only the per-scale projections are trainable.
    - Captures features from 4 evenly-spaced transformer blocks (layers 6, 12, 18, 24),
      each providing a different semantic level, directly aligned to U-Net decoder scales.
    - Input resized from 256×256 → 224×224.
    - Patch tokens reshaped to spatial grid [B, 1536, 16, 16].

    Block → scale mapping (24 blocks → 4 U-Net decoder stages):
        block_6  → f0: [B, 96,  64, 64]   1/4   (shallow: edges, textures)
        block_12 → f1: [B, 192, 32, 32]   1/8   (low-level shapes)
        block_18 → f2: [B, 384, 16, 16]   1/16  (mid-level semantics, native ViT res)
        block_24 → f3: [B, 768,  8,  8]   1/32  (deep: high-level semantics)
    """

    # Block indices for the 4 feature scales (1-indexed, same as timm)
    BLOCK_INDICES = [6, 12, 18, 24]

    def __init__(
        self,
        freeze: bool = True,
        local_weights: str = "/home/lwy/Newidea/pytorch_model.bin",
        dropout: float = 0.1,
    ):
        super().__init__()

        # -- Build UNI2-h backbone locally (no hf-hub, no network) --
        timm_kwargs = {
            'img_size': 224,
            'patch_size': 14,
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5,
            'embed_dim': 1536,
            'mlp_ratio': 2.66667 * 2,
            'num_classes': 0,
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked,
            'act_layer': nn.SiLU,
            'reg_tokens': 8,
            'dynamic_img_size': True,
        }
        self.vit = timm.models.vision_transformer.VisionTransformer(**timm_kwargs)

        # Load local weights
        if not os.path.exists(local_weights):
            raise FileNotFoundError(
                f"UNI2-h weights not found at {local_weights}. "
                f"Download from https://huggingface.co/MahmoodLab/UNI2-h"
            )
        print(f"  Loading UNI2-h weights from {local_weights} ...")
        state = torch.load(local_weights, map_location='cpu', weights_only=True)
        self.vit.load_state_dict(state, strict=True)
        self.vit.eval()

        if freeze:
            for p in self.vit.parameters():
                p.requires_grad = False
            print(f"  UNI2-h backbone frozen ({sum(p.numel() for p in self.vit.parameters())/1e6:.1f}M params)")

        self._vit_trainable = not freeze  # controls no_grad in forward()

        self.embed_dim = 1536
        self.grid_size = 224 // 14  # 16

        # -- Register hooks to capture intermediate block outputs --
        self._block_features = {}
        for idx in self.BLOCK_INDICES:
            block = self.vit.blocks[idx - 1]  # timm blocks are 0-indexed

            def make_hook(i):
                def hook(module, input, output):
                    self._block_features[i] = output
                return hook

            block.register_forward_hook(make_hook(idx))

        # -- Per-scale projection layers (trainable) --
        # Each block outputs [B, 265, 1536] → extract patch tokens → [B, 1536, 16, 16]
        # → project to target channels → resize to target spatial size.
        # Dropout2d after GELU prevents overfitting of the small trainable
        # projection layers to the frozen ViT features.
        #
        # f3 (block_24 → /32): deep semantics
        self.proj_f3 = nn.Sequential(
            nn.Conv2d(1536, 768, 1, bias=False),
            nn.BatchNorm2d(768),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )
        # f2 (block_18 → /16): mid-level, native ViT resolution
        self.proj_f2 = nn.Sequential(
            nn.Conv2d(1536, 384, 1, bias=False),
            nn.BatchNorm2d(384),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )
        # f1 (block_12 → /8): low-level shapes
        self.proj_f1 = nn.Sequential(
            nn.Conv2d(1536, 192, 1, bias=False),
            nn.BatchNorm2d(192),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )
        # f0 (block_6 → /4): shallow features
        self.proj_f0 = nn.Sequential(
            nn.Conv2d(1536, 96, 1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

        # Store output channels for decoder compatibility
        self.channels = [96, 192, 384, 768]

    def _extract_spatial(self, tokens: torch.Tensor) -> torch.Tensor:
        """Extract patch tokens from [cls|reg|patch] sequence → spatial grid."""
        # tokens: [B, 265, 1536] → remove cls(1) + reg(8) = first 9
        B = tokens.shape[0]
        patch = tokens[:, 9:, :]                                    # [B, 256, 1536]
        spatial = patch.transpose(1, 2).reshape(B, 1536, self.grid_size, self.grid_size)
        return spatial.contiguous()

    def set_vit_trainable(self, trainable: bool):
        """Enable/disable gradient computation for ViT backbone.

        When trainable=False, forward() wraps ViT in torch.no_grad()
        to save memory by not building the autograd graph.
        """
        self._vit_trainable = trainable

    def forward(self, x: torch.Tensor) -> list:
        """Returns [feat0, feat1, feat2, feat3] from 4 ViT block depths."""
        B, C, H, W = x.shape
        x_224 = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        # Clear previous hook captures
        self._block_features.clear()

        # ViT forward — hooks capture intermediate block outputs
        # no_grad when ViT is frozen (saves memory); full graph when trainable
        if self._vit_trainable:
            _ = self.vit.forward_features(x_224)
        else:
            with torch.no_grad():
                _ = self.vit.forward_features(x_224)

        # Extract features from each captured block
        # Block 6 → f0 (shallow)
        f0_tokens = self._block_features[6]
        f0_sp = self._extract_spatial(f0_tokens)                     # [B, 1536, 16, 16]
        feat0 = self.proj_f0(f0_sp)                                  # [B, 96, 16, 16]
        feat0 = F.interpolate(feat0, scale_factor=4, mode='bilinear',
                              align_corners=False)                   # [B, 96, 64, 64]

        # Block 12 → f1
        f1_tokens = self._block_features[12]
        f1_sp = self._extract_spatial(f1_tokens)                     # [B, 1536, 16, 16]
        feat1 = self.proj_f1(f1_sp)                                  # [B, 192, 16, 16]
        feat1 = F.interpolate(feat1, scale_factor=2, mode='bilinear',
                              align_corners=False)                   # [B, 192, 32, 32]

        # Block 18 → f2 (native resolution)
        f2_tokens = self._block_features[18]
        feat2 = self._extract_spatial(f2_tokens)                     # [B, 1536, 16, 16]
        feat2 = self.proj_f2(feat2)                                  # [B, 384, 16, 16]

        # Block 24 → f3 (deepest)
        f3_tokens = self._block_features[24]
        f3_sp = self._extract_spatial(f3_tokens)                     # [B, 1536, 16, 16]
        feat3 = self.proj_f3(f3_sp)                                  # [B, 768, 16, 16]
        feat3 = F.interpolate(feat3, scale_factor=0.5, mode='bilinear',
                              align_corners=False)                   # [B, 768, 8, 8]

        return [feat0, feat1, feat2, feat3]


def create_uni2_encoder(
    freeze: bool = True,
    local_weights: str = "/home/lwy/Newidea/pytorch_model.bin",
    dropout: float = 0.1,
) -> UNI2Encoder:
    """Create frozen UNI2-h encoder with trainable feature pyramid."""
    return UNI2Encoder(freeze=freeze, local_weights=local_weights, dropout=dropout)
