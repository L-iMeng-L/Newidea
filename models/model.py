"""
Three-head nuclei segmentation with switchable decoder architectures.

Decoder types:
    cellvit:   1 shared U-Net decoder → 3 lightweight heads
    unet:      3 independent U-Net decoders
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from .encoder import ConvNeXtEncoder, UNI2Encoder, ConvNeXtVariant, ConvNeXtVersion
from .decoder import SharedASPP, DecoderBlock, ConvBlock


class ConvNeXtSegmentor(nn.Module):
    """Encoder + switchable decoder."""

    def __init__(
        self,
        variant: ConvNeXtVariant = "tiny",
        version: ConvNeXtVersion = "v2",
        num_nc_classes: int = 5,
        pretrained: bool = True,
        decoder_type: str = "cellvit",     # "cellvit" | "unet" | "aspp_unet"
        aspp_out: int = 256,
        enc_dropout: float = 0.1,
        dec_dropout: float = 0.1,
        freeze_encoder: bool = False,
        full_unfreeze: bool = False,
        uni2_weights: str = "/home/lwy/Newidea/pytorch_model.bin",
        **kwargs,
    ):
        super().__init__()
        self.num_nc_classes = num_nc_classes
        self.decoder_type = decoder_type
        self._is_uni2 = (variant == "uni2-h")
        self._full_unfreeze = full_unfreeze
        self._nc_no_cbam = kwargs.get("nc_no_cbam", False)
        self._upsample_mode = kwargs.get("upsample_mode", "transpose")

        # ---- Shared backbone ----
        if self._is_uni2:
            self.encoder = UNI2Encoder(
                freeze=freeze_encoder if freeze_encoder else True,
                local_weights=uni2_weights,
                dropout=enc_dropout,
            )
            enc_ch = self.encoder.channels
        else:
            self.encoder = ConvNeXtEncoder(
                variant=variant, version=version, pretrained=pretrained,
                dropout=enc_dropout,
            )
            enc_ch = self.encoder.channels

        # ---- Bottleneck projection ----
        self.aspp = nn.Conv2d(enc_ch[3], aspp_out, 1, bias=False)

        if decoder_type == "shared_unet":
            self._build_shared_unet(enc_ch, num_nc_classes, aspp_out, dec_dropout)
        elif decoder_type == "shared_unet_mala":
            self._build_shared_unet_mala(enc_ch, num_nc_classes, aspp_out, dec_dropout)
        elif decoder_type == "unet3":
            self._build_unet3(enc_ch, num_nc_classes, aspp_out, dec_dropout)
        elif decoder_type == "unet3_mala":
            self._build_unet3_mala(enc_ch, num_nc_classes, aspp_out, dec_dropout)
        else:
            raise ValueError(f"Unknown decoder_type: {decoder_type}")

    # ------------------------------------------------------------------
    #  Shared U-Net: single decoder → 3 heads
    # ------------------------------------------------------------------
    def _build_shared_unet(self, enc_ch, num_classes, aspp_out, dropout):
        """Official CellViT decoder: single shared U-Net with 4 stages.

        Stage4 (/32→/16): Conv×3 → Deconv ↑2×          (deepest, 3 conv layers)
        Stage3 (/16→/8):  concat(f2) → Conv×2 → Deconv ↑2×
        Stage2 (/8→/4):   concat(f1) → Conv×2 → Deconv ↑2×
        Stage1 (/4):      concat(f0) → Conv×2           (final, no upsampling)
        → 3 lightweight output heads
        """
        from models.attention import CBAM_Light

        e0, e1, e2, e3 = enc_ch

        # ---- Stage4: deepest, /32 → /16, Conv×3 ----
        self.dec_stage4 = nn.Sequential(
            ConvBlock(aspp_out, 256),
            ConvBlock(256, 256),
            ConvBlock(256, 256),
        )
        self.dec_up4 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)

        # ---- Stage3: /16 → /8, concat(f2) → Conv×2 ----
        self.dec_stage3 = DecoderBlock(256, e2, 256, dropout,
                                       upsample_mode=self._upsample_mode, n_convs=2)

        # ---- Stage2: /8 → /4, concat(f1) → Conv×2 ----
        self.dec_stage2 = DecoderBlock(256, e1, 192, dropout,
                                       upsample_mode=self._upsample_mode, n_convs=2)

        # ---- Stage1: /4, concat(f0) → Conv×2 (no upsampling) ----
        self.dec_stage1 = nn.Sequential(
            nn.Conv2d(192 + e0, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        self.dec_dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        # ---- Shared output heads ----
        self.np_head = nn.Sequential(ConvBlock(128, 128), nn.Conv2d(128, 1, 1))
        if self._nc_no_cbam:
            self.nc_head = nn.Sequential(ConvBlock(128, 128),
                                         nn.Conv2d(128, num_classes, 1))
        else:
            self.nc_head = nn.Sequential(ConvBlock(128, 128), CBAM_Light(128),
                                         nn.Conv2d(128, num_classes, 1))
        self.hv_head = nn.Sequential(ConvBlock(128, 128), nn.Conv2d(128, 2, 1))

    def _build_shared_unet_mala(self, enc_ch, num_classes, aspp_out, dropout):
        """v2 shared decoder + MALA at /4 and /8 only (single-nucleus scales).

        mala_f0 (/4, 96ch):  12-28px 感受野 — 单核尺度 ✅
        mala_f1 (/8, 192ch): 24-56px 感受野 — 1~3核尺度 ✅
        /16 and /32: standard ConvBlock (too coarse for MALA, causes overfitting)
        """
        from models.attention import CBAM_Light
        from models.mala import MALABlock

        e0, e1, e2, e3 = enc_ch

        self.mala_f0 = MALABlock(e0, e0)          # /4,  96→96
        self.mala_f1 = MALABlock(e1, e1)          # /8,  192→192

        # Decoder stages (regular ConvBlocks, no MALA inside)
        self.dec_stage4 = nn.Sequential(
            ConvBlock(256, 256),
            ConvBlock(256, 256),
            ConvBlock(256, 256),
        )
        self.dec_up4 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.dec_stage3 = DecoderBlock(256, e2, 256, dropout,
                                       upsample_mode=self._upsample_mode, n_convs=2)
        self.dec_stage2 = DecoderBlock(256, e1, 192, dropout,
                                       upsample_mode=self._upsample_mode, n_convs=2)
        self.dec_stage1 = nn.Sequential(
            nn.Conv2d(192 + e0, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        self.dec_dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.np_head = nn.Sequential(ConvBlock(128, 128), nn.Conv2d(128, 1, 1))
        if self._nc_no_cbam:
            self.nc_head = nn.Sequential(ConvBlock(128, 128),
                                         nn.Conv2d(128, num_classes, 1))
        else:
            self.nc_head = nn.Sequential(ConvBlock(128, 128), CBAM_Light(128),
                                         nn.Conv2d(128, num_classes, 1))
        self.hv_head = nn.Sequential(ConvBlock(128, 128), nn.Conv2d(128, 2, 1))

    def _forward_shared_unet_mala(self, features, input_size):
        """v2 forward with MALA on /4 and /8 skip connections."""
        f0, f1, f2, f3 = features

        f0 = self.mala_f0(f0)                          # /4, MALA
        f1 = self.mala_f1(f1)                          # /8, MALA
        # f2 and f3: raw, no MALA (too coarse)
        x = self.aspp(f3)                              # /32, 256ch

        # Decoder
        x = self.dec_stage4(x)                         # Conv×3
        x = self.dec_up4(x)                            # Deconv → /16
        x = self.dec_stage3(x, f2)                     # concat(f2) → /8
        x = self.dec_stage2(x, f1)                     # concat(f1_mala) → /4
        if x.shape[-2:] != f0.shape[-2:]:
            x = F.interpolate(x, size=f0.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, f0], dim=1)                  # concat(f0_mala)
        x = self.dec_stage1(x)                         # Conv×2, 128ch
        x = self.dec_dropout(x)

        np = F.interpolate(self.np_head(x), size=input_size, mode='bilinear', align_corners=False)
        nc = F.interpolate(self.nc_head(x), size=input_size, mode='bilinear', align_corners=False)
        hv = F.interpolate(self.hv_head(x), size=input_size, mode='bilinear', align_corners=False)
        return {"np": np, "nc": nc, "hv": hv}

    def _forward_shared_unet(self, features, input_size):
        f0, f1, f2, f3 = features

        # Bottleneck: 1×1 projection
        x = self.aspp(f3)                              # /32, 256ch

        # Stage4: /32 → /16
        x = self.dec_stage4(x)                         # Conv×3
        x = self.dec_up4(x)                            # Deconv ↑2× → /16

        # Stage3: /16 → /8, concat f2
        x = self.dec_stage3(x, f2)                     # /8, 256ch

        # Stage2: /8 → /4, concat f1
        x = self.dec_stage2(x, f1)                     # /4, 192ch

        # Stage1: /4, concat f0
        if x.shape[-2:] != f0.shape[-2:]:
            x = F.interpolate(x, size=f0.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, f0], dim=1)                  # 192+96=288
        x = self.dec_stage1(x)                         # 128ch
        x = self.dec_dropout(x)

        # 3 output heads
        np = F.interpolate(self.np_head(x), size=input_size, mode='bilinear', align_corners=False)
        nc = F.interpolate(self.nc_head(x), size=input_size, mode='bilinear', align_corners=False)
        hv = F.interpolate(self.hv_head(x), size=input_size, mode='bilinear', align_corners=False)

        return {"np": np, "nc": nc, "hv": hv}

    # ------------------------------------------------------------------
    #  CellViT v3: 3 independent shared_unet-style decoders
    # ------------------------------------------------------------------
    @staticmethod
    def _build_branch(enc_ch, aspp_out, dropout, upsample_mode="transpose"):
        """Build one shared_unet-style decoder branch.

        Stage4 (/32→/16): ConvBlock×3 → ConvTranspose2d
        Stage3 (/16→/8):  DecoderBlock(f2)  → ConvTranspose2d
        Stage2 (/8→/4):   DecoderBlock(f1)  → ConvTranspose2d
        Stage1 (/4):      concat(f0) → Conv×2
        """
        e0, e1, e2, e3 = enc_ch

        stage4 = nn.Sequential(
            ConvBlock(aspp_out, 256),
            ConvBlock(256, 256),
            ConvBlock(256, 256),
        )
        up4 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        stage3 = DecoderBlock(256, e2, 256, dropout, upsample_mode=upsample_mode, n_convs=2)
        stage2 = DecoderBlock(256, e1, 192, dropout, upsample_mode=upsample_mode, n_convs=2)
        stage1 = nn.Sequential(
            nn.Conv2d(192 + e0, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        return nn.ModuleDict({
            "stage4": stage4, "up4": up4,
            "stage3": stage3, "stage2": stage2,
            "stage1": stage1, "drop": drop,
        })

    @staticmethod
    def _forward_branch(branch, features):
        f0, f1, f2, _ = features
        x = branch["stage4"](features[3])         # ConvBlock×3
        x = branch["up4"](x)                      # Deconv → /16
        x = branch["stage3"](x, f2)               # /8
        x = branch["stage2"](x, f1)               # /4
        if x.shape[-2:] != f0.shape[-2:]:
            x = F.interpolate(x, size=f0.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, f0], dim=1)
        x = branch["stage1"](x)
        return branch["drop"](x)

    def _build_unet3(self, enc_ch, num_classes, aspp_out, dropout):
        from models.attention import CBAM_Light

        self.np_branch = self._build_branch(enc_ch, aspp_out, dropout, upsample_mode=self._upsample_mode)
        self.nc_branch = self._build_branch(enc_ch, aspp_out, dropout, upsample_mode=self._upsample_mode)
        self.hv_branch = self._build_branch(enc_ch, aspp_out, dropout, upsample_mode=self._upsample_mode)

        self.np_head3 = nn.Conv2d(128, 1, 1)
        self.nc_cbam3 = CBAM_Light(128)
        self.nc_head3 = nn.Conv2d(128, num_classes, 1)
        self.hv_head3 = nn.Conv2d(128, 2, 1)

    def _forward_unet3(self, features, input_size):
        # Each branch starts from the same bottleneck but with its own weights
        features_np = (features[0], features[1], features[2], self.aspp(features[3]))
        features_nc = (features[0], features[1], features[2], self.aspp(features[3]))
        features_hv = (features[0], features[1], features[2], self.aspp(features[3]))

        np = self._forward_branch(self.np_branch, features_np)
        nc = self._forward_branch(self.nc_branch, features_nc)
        nc = self.nc_cbam3(nc)
        hv = self._forward_branch(self.hv_branch, features_hv)

        np = F.interpolate(self.np_head3(np), size=input_size, mode='bilinear', align_corners=False)
        nc = F.interpolate(self.nc_head3(nc), size=input_size, mode='bilinear', align_corners=False)
        hv = F.interpolate(self.hv_head3(hv), size=input_size, mode='bilinear', align_corners=False)

        return {"np": np, "nc": nc, "hv": hv}

    # ------------------------------------------------------------------
    #  CellViT v3 + MALA: dynamic conv in bottleneck + final refinement
    # ------------------------------------------------------------------
    @staticmethod
    def _build_branch_mala(enc_ch, aspp_out, dropout, upsample_mode="transpose"):
        """Same as _build_branch but MALABlock in stage4 bottleneck + stage1 refinement."""
        from models.mala import MALABlock

        e0, e1, e2, e3 = enc_ch

        # Stage4: MALA replaces first 2 ConvBlocks, keep last for transition
        stage4 = nn.Sequential(
            MALABlock(aspp_out, 256),
            MALABlock(256, 256),
            ConvBlock(256, 256),
        )
        up4 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        stage3 = DecoderBlock(256, e2, 256, dropout, upsample_mode=upsample_mode, n_convs=2)
        stage2 = DecoderBlock(256, e1, 192, dropout, upsample_mode=upsample_mode, n_convs=2)

        # Stage1: MALA replaces 2 Conv2d layers
        stage1 = nn.Sequential(
            MALABlock(192 + e0, 128),
            MALABlock(128, 128),
        )
        drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        return nn.ModuleDict({
            "stage4": stage4, "up4": up4,
            "stage3": stage3, "stage2": stage2,
            "stage1": stage1, "drop": drop,
        })

    def _build_unet3_mala(self, enc_ch, num_classes, aspp_out, dropout):
        from models.attention import CBAM_Light

        self.np_branch = self._build_branch_mala(enc_ch, aspp_out, dropout, upsample_mode=self._upsample_mode)
        self.nc_branch = self._build_branch_mala(enc_ch, aspp_out, dropout, upsample_mode=self._upsample_mode)
        self.hv_branch = self._build_branch_mala(enc_ch, aspp_out, dropout, upsample_mode=self._upsample_mode)

        self.np_head3 = nn.Conv2d(128, 1, 1)
        self.nc_cbam3 = CBAM_Light(128)
        self.nc_head3 = nn.Conv2d(128, num_classes, 1)
        self.hv_head3 = nn.Conv2d(128, 2, 1)

    # ------------------------------------------------------------------
    #  Freeze / Unfreeze encoder
    # ------------------------------------------------------------------
    def freeze_encoder(self):
        if self._is_uni2:
            vit_params = list(self.encoder.vit.parameters())
            for p in vit_params:
                p.requires_grad = False
            self.encoder.set_vit_trainable(False)
            frozen = sum(p.numel() for p in vit_params)
            proj_trainable = sum(
                p.numel() for p in self.encoder.parameters() if p.requires_grad)
            print(f"  UNI2-h ViT frozen ({frozen/1e6:.1f}M), "
                  f"projections trainable ({proj_trainable/1e6:.2f}M)")
        else:
            for p in self.encoder.parameters():
                p.requires_grad = False
            frozen = sum(p.numel() for p in self.encoder.parameters())
            print(f"  Encoder frozen ({frozen/1e6:.1f}M params)")

    def unfreeze_encoder(self):
        if self._is_uni2:
            if self._full_unfreeze:
                for p in self.encoder.vit.parameters():
                    p.requires_grad = True
                self.encoder.set_vit_trainable(True)
                trainable = sum(
                    p.numel() for p in self.encoder.parameters() if p.requires_grad)
                print(f"  UNI2-h ViT fully unfrozen ({trainable/1e6:.1f}M trainable)")
            else:
                for p in self.encoder.vit.parameters():
                    p.requires_grad = False
                self.encoder.set_vit_trainable(False)
                frozen = sum(
                    p.numel() for p in self.encoder.parameters() if not p.requires_grad)
                trainable = sum(
                    p.numel() for p in self.encoder.parameters() if p.requires_grad)
                print(f"  UNI2-h: ViT frozen ({frozen/1e6:.1f}M), "
                      f"projections trainable ({trainable/1e6:.2f}M)")
            for p in self.encoder.proj_f0.parameters(): p.requires_grad = True
            for p in self.encoder.proj_f1.parameters(): p.requires_grad = True
            for p in self.encoder.proj_f2.parameters(): p.requires_grad = True
            for p in self.encoder.proj_f3.parameters(): p.requires_grad = True
        else:
            for p in self.encoder.parameters():
                p.requires_grad = True
            trainable = sum(p.numel() for p in self.encoder.parameters())
            print(f"  Encoder unfrozen ({trainable/1e6:.1f}M params)")

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, return_enc_features: bool = False
                ) -> Dict[str, torch.Tensor]:
        input_size = x.shape[-2:]
        features = self.encoder(x)
        if self.decoder_type == "shared_unet":
            outputs = self._forward_shared_unet(features, input_size)
        elif self.decoder_type == "shared_unet_mala":
            outputs = self._forward_shared_unet_mala(features, input_size)
        elif self.decoder_type in ("unet3", "unet3_mala"):
            outputs = self._forward_unet3(features, input_size)
        else:
            raise RuntimeError(f"Unknown decoder_type: {self.decoder_type}")

        if return_enc_features:
            return outputs, features
        return outputs

    def get_param_groups(self, lr: float, backbone_lr_mult: float = 0.05):
        all_enc = list(self.encoder.parameters())
        groups = []
        if all_enc:
            groups.append({"params": all_enc, "lr": lr * backbone_lr_mult, "name": "encoder"})
        groups.append({"params": self.aspp.parameters(), "lr": lr, "name": "aspp"})
        if self.decoder_type in ("unet3", "unet3_mala"):
            groups += [
                {"params": self.np_branch.parameters(),  "lr": lr, "name": "np_branch"},
                {"params": self.nc_branch.parameters(),  "lr": lr, "name": "nc_branch"},
                {"params": self.hv_branch.parameters(),  "lr": lr, "name": "hv_branch"},
                {"params": self.np_head3.parameters(),   "lr": lr, "name": "np_branch"},
                {"params": self.nc_cbam3.parameters(),   "lr": lr, "name": "nc_branch"},
                {"params": self.nc_head3.parameters(),   "lr": lr, "name": "nc_branch"},
                {"params": self.hv_head3.parameters(),   "lr": lr, "name": "hv_branch"},
            ]
        elif self.decoder_type == "shared_unet":
            groups += [
                {"params": self.dec_stage4.parameters(),  "lr": lr, "name": "decoder"},
                {"params": self.dec_up4.parameters(),     "lr": lr, "name": "decoder"},
                {"params": self.dec_stage3.parameters(),  "lr": lr, "name": "decoder"},
                {"params": self.dec_stage2.parameters(),  "lr": lr, "name": "decoder"},
                {"params": self.dec_stage1.parameters(),  "lr": lr, "name": "decoder"},
                {"params": self.dec_dropout.parameters(), "lr": lr, "name": "decoder"},
                {"params": self.np_head.parameters(),     "lr": lr, "name": "np_head"},
                {"params": self.nc_head.parameters(),     "lr": lr, "name": "nc_head"},
                {"params": self.hv_head.parameters(),     "lr": lr, "name": "hv_head"},
            ]
        elif self.decoder_type == "shared_unet_mala":
            groups += [
                {"params": self.mala_f0.parameters(),     "lr": lr, "name": "mala"},
                {"params": self.mala_f1.parameters(),     "lr": lr, "name": "mala"},
                {"params": self.dec_stage4.parameters(),  "lr": lr, "name": "decoder"},
                {"params": self.dec_up4.parameters(),     "lr": lr, "name": "decoder"},
                {"params": self.dec_stage3.parameters(),  "lr": lr, "name": "decoder"},
                {"params": self.dec_stage2.parameters(),  "lr": lr, "name": "decoder"},
                {"params": self.dec_stage1.parameters(),  "lr": lr, "name": "decoder"},
                {"params": self.dec_dropout.parameters(), "lr": lr, "name": "decoder"},
                {"params": self.np_head.parameters(),     "lr": lr, "name": "np_head"},
                {"params": self.nc_head.parameters(),     "lr": lr, "name": "nc_head"},
                {"params": self.hv_head.parameters(),     "lr": lr, "name": "hv_head"},
            ]
        else:
            raise RuntimeError(f"Unknown decoder_type: {self.decoder_type}")
        return groups


def create_model(
    variant: ConvNeXtVariant = "tiny",
    version: ConvNeXtVersion = "v2",
    num_nc_classes: int = 5,
    pretrained: bool = True,
    decoder_type: str = "cellvit",
    freeze_encoder: bool = False,
    full_unfreeze: bool = False,
    uni2_weights: str = "/home/lwy/Newidea/pytorch_model.bin",
    **kwargs,
) -> ConvNeXtSegmentor:
    return ConvNeXtSegmentor(
        variant=variant, version=version,
        num_nc_classes=num_nc_classes,
        pretrained=pretrained, decoder_type=decoder_type,
        freeze_encoder=freeze_encoder,
        full_unfreeze=full_unfreeze,
        uni2_weights=uni2_weights,
        **kwargs,
    )
