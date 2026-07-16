"""
Distillation losses for UNI2-h (teacher) → ConvNeXt (student) knowledge transfer.

Three distillation channels:
    1. Encoder features:   MSE between corresponding f0..f3 feature maps
    2. Output KD:          KL divergence on NP/NC logits + MSE on HV
    3. (optional) Decoder: CosineSimilarity on intermediate decoder stages

Usage:
    kd_loss_fn = DistillationLoss(temperature=4.0, alpha=0.9, ...)
    total, loss_dict = kd_loss_fn(
        s_outputs, t_outputs, s_enc_feats, t_enc_feats, student_sup_loss
    )
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple


class DistillationLoss(nn.Module):
    """Combined distillation loss for UNI2-h → ConvNeXt KD.

    Loss = L_supervised + λ_enc * L_enc_feat + λ_np * L_kd_np + λ_nc * L_kd_nc + λ_hv * L_kd_hv

    Parameters
    ----------
    temperature : float
        Softmax temperature for KL divergence (higher = softer teacher).
    alpha : float
        KD loss weight vs supervised loss (0=sup only, 1=KD only).
        Final weight = alpha for output KD, enc_weight for encoder features.
    enc_weight : float
        Weight for encoder feature alignment loss.
    np_weight, nc_weight, hv_weight : float
        Per-head KD weights (relative to each other within alpha).
    """

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.9,
        enc_weight: float = 0.1,
        np_weight: float = 1.0,
        nc_weight: float = 1.0,
        hv_weight: float = 1.0,
        enc_layer_weights: Tuple[float, ...] = (0.1, 0.2, 0.3, 0.4),
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.enc_weight = enc_weight
        self.np_weight = np_weight
        self.nc_weight = nc_weight
        self.hv_weight = hv_weight
        self.enc_layer_weights = enc_layer_weights  # f0..f3 per-layer weight

    @staticmethod
    def _kl_div(p_student: torch.Tensor, p_teacher: torch.Tensor) -> torch.Tensor:
        """KL divergence: KL(teacher || student) = sum(t * log(t/s))."""
        p_t = p_teacher.clamp(1e-7, 1 - 1e-7)
        p_s = p_student.clamp(1e-7, 1 - 1e-7)
        return (p_t * (p_t.log() - p_s.log())).sum(dim=1).mean()

    def _kd_logits(self, s_logits: torch.Tensor, t_logits: torch.Tensor) -> torch.Tensor:
        """Softened KL divergence on raw logits (applies sigmoid for binary, softmax for multi)."""
        if s_logits.shape[1] == 1:
            # Binary classification (NP): apply sigmoid
            s_p = torch.sigmoid(s_logits / self.temperature)
            t_p = torch.sigmoid(t_logits.detach() / self.temperature)
            return self._kl_div(s_p, t_p)
        else:
            # Multi-class (NC): apply softmax
            s_p = F.softmax(s_logits / self.temperature, dim=1)
            t_p = F.softmax(t_logits.detach() / self.temperature, dim=1)
            # Per-pixel KL, average over batch
            kl = t_p * (t_p.log() - s_p.log())
            return kl.sum(dim=1).mean()

    def forward(
        self,
        student_outputs: Dict[str, torch.Tensor],
        teacher_outputs: Dict[str, torch.Tensor],
        student_enc_feats: List[torch.Tensor],
        teacher_enc_feats: List[torch.Tensor],
        student_sup_loss: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined distillation + supervised loss.

        Parameters
        ----------
        student_outputs : {"np": [B,1,H,W], "nc": [B,C,H,W], "hv": [B,2,H,W]}
        teacher_outputs : same shapes (detached internally)
        student_enc_feats : [f0, f1, f2, f3] from ConvNeXt
        teacher_enc_feats : [f0, f1, f2, f3] from UNI2-h
        student_sup_loss : scalar, original supervised loss on student outputs

        Returns
        -------
        total_loss : scalar tensor
        loss_dict  : {"enc_feat", "kd_np", "kd_nc", "kd_hv", "total"}
        """
        # ---- Encoder feature alignment (MSE) ----
        enc_losses = []
        for i, (sf, tf) in enumerate(zip(student_enc_feats, teacher_enc_feats)):
            # Both should have same spatial size; if not, interpolate student to match teacher
            if sf.shape[-2:] != tf.shape[-2:]:
                sf = F.interpolate(sf, size=tf.shape[-2:], mode='bilinear', align_corners=False)
            enc_losses.append(F.mse_loss(sf, tf.detach()) * self.enc_layer_weights[i])
        enc_loss = sum(enc_losses)

        # ---- Output KD ----
        kd_np = self._kd_logits(student_outputs["np"], teacher_outputs["np"])
        kd_nc = self._kd_logits(student_outputs["nc"], teacher_outputs["nc"])
        kd_hv = F.mse_loss(student_outputs["hv"], teacher_outputs["hv"].detach())

        # ---- Combined loss ----
        kd_loss = (self.np_weight * kd_np +
                   self.nc_weight * kd_nc +
                   self.hv_weight * kd_hv)

        total_loss = (1 - self.alpha) * student_sup_loss + \
                     self.alpha * kd_loss + \
                     self.enc_weight * enc_loss

        loss_dict = {
            "enc_feat": enc_loss.item(),
            "kd_np": kd_np.item(),
            "kd_nc": kd_nc.item(),
            "kd_hv": kd_hv.item(),
            "supervised": student_sup_loss.item(),
            "total": total_loss.item(),
        }
        return total_loss, loss_dict
