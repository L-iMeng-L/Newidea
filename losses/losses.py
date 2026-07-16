"""
Losses for 3-head (NP + HV + NC) nuclei segmentation.

NP:  asym (AsymmetricLoss + OHEM top-50%) | dice (BinaryDiceLoss)
     [+ optional clDice] [+ optional SizePrior]
HV:  MSE + MSGE (Sobel gradient MSE) + Focus Mask
NC:  focal (FocalLoss) | dice (MultiClassDiceLoss) | ce (CrossEntropyLoss)
     [+ optional ClassBalancedWeight]

Gate: SpatialConsistencyLoss  (from models.gate)

Total: L = λ_np·L_np + λ_hv·L_hv + λ_nc·L_nc + λ_gate·L_gate
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ==============================================================================
#  Soft skeletonization
# ==============================================================================

def soft_skel(x: torch.Tensor, iters: int = 5) -> torch.Tensor:
    """Differentiable soft skeletonization via iterative min/max pooling."""
    if x.dim() == 3:
        x, squeeze_back = x.unsqueeze(1), True
    else:
        squeeze_back = False

    orig_dtype = x.dtype
    if x.dtype == torch.float16:
        x = x.float()

    skel = torch.zeros_like(x)
    remaining = x.clone()
    for _ in range(iters):
        min_pool = -F.max_pool2d(-remaining, kernel_size=3, stride=1, padding=1)
        opened = F.max_pool2d(min_pool, kernel_size=3, stride=1, padding=1)
        delta = F.relu(remaining - opened)
        skel = skel + delta
        remaining = remaining - delta

    skel = skel.to(orig_dtype)
    if squeeze_back:
        skel = skel.squeeze(1)
    return skel


def soft_dice(y_pred: torch.Tensor, y_true: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    intersection = (y_pred * y_true).sum(dim=(2, 3))
    union = y_pred.sum(dim=(2, 3)) + y_true.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean()


def cl_dice(y_pred: torch.Tensor, y_true: torch.Tensor,
            iters: int = 5, smooth: float = 1.0) -> torch.Tensor:
    skel_pred = soft_skel(y_pred, iters=iters)
    skel_true = soft_skel(y_true, iters=iters)
    t_prec = (skel_true * y_pred).sum(dim=(2, 3))
    t_prec = (t_prec + smooth) / (skel_true.sum(dim=(2, 3)) + smooth)
    t_sens = (skel_pred * y_true).sum(dim=(2, 3))
    t_sens = (t_sens + smooth) / (skel_pred.sum(dim=(2, 3)) + smooth)
    cldice = 2.0 * t_prec * t_sens / (t_prec + t_sens + 1e-7)
    return cldice.mean()


# ==============================================================================
#  Binary Dice Loss (NP branch)
# ==============================================================================

class BinaryDiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation.

    Dice = 2|P∩G| / (|P|+|G|),  loss = 1 - Dice.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  [B, 1, H, W] raw logits
            targets: [B, 1, H, W] binary {0,1}

        Returns:
            scalar loss
        """
        prob = torch.sigmoid(logits)
        inter = (prob * targets).sum()
        union = prob.sum() + targets.sum()
        dice = (2.0 * inter + self.smooth) / (union + self.smooth)
        return 1.0 - dice

    def per_pixel(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Per-pixel proxy for OHEM compatibility — returns BCE loss."""
        return F.binary_cross_entropy_with_logits(logits, targets, reduction='none')


# ==============================================================================
#  Asymmetric Loss  (§4.1, from HistoNeXt)
# ==============================================================================

class AsymmetricLoss(nn.Module):
    """Asymmetric focal loss for extreme foreground/background imbalance.

    gamma_neg >> gamma_pos → easy negatives are heavily down-weighted
    while easy positives get only mild suppression.
    """

    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 0.5,
                 clip: float = 0.05, eps: float = 1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  [B, 1, H, W] raw logits
            targets: [B, 1, H, W] binary {0,1}

        Returns:
            scalar loss
        """
        prob = torch.sigmoid(logits)
        prob = prob.clamp(self.clip, 1.0 - self.clip)

        # Positive (foreground) and negative (background) terms
        pos_term = (1 - prob) ** self.gamma_pos * torch.log(prob + self.eps)
        neg_term = prob ** self.gamma_neg * torch.log(1 - prob + self.eps)

        loss = -targets * pos_term - (1 - targets) * neg_term
        return loss.mean()

    def per_pixel(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Same as forward but returns per-pixel loss (no mean)."""
        prob = torch.sigmoid(logits)
        prob = prob.clamp(self.clip, 1.0 - self.clip)
        pos_term = (1 - prob) ** self.gamma_pos * torch.log(prob + self.eps)
        neg_term = prob ** self.gamma_neg * torch.log(1 - prob + self.eps)
        return -targets * pos_term - (1 - targets) * neg_term


# ==============================================================================
#  MSGE (Multi-Scale Gradient Error) + Focus Mask  (CellViT official)
# ==============================================================================

def _gradient_kernel_2d(size: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """CellViT gradient direction field kernel.

    kernel_h = h / (h² + v² + ε)   — x-component of normalised gradient direction
    kernel_v = v / (h² + v² + ε)   — y-component

    For HV displacement maps, this measures spatial consistency of the
    centroid-pointing vector field, not intensity edges (unlike Sobel).
    """
    assert size % 2 == 1, f"Kernel size must be odd, got {size}"
    r = torch.arange(-size // 2 + 1, size // 2 + 1, dtype=torch.float32, device=device)
    h, v = torch.meshgrid(r, r, indexing="ij")
    denom = h * h + v * v + 1.0e-15
    return h / denom, v / denom


def msge_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
              ksize: int = 5, scales: Tuple[int, ...] = None) -> torch.Tensor:
    """Multi-Scale Gradient Error — CellViT gradient direction field consistency.

    Computes gradient field MSE at multiple kernel scales (3, 5, 7) and averages.
    Focus mask restricts loss to nuclear pixels only.

    Args:
        pred:   [B, 2, H, W] predicted HV
        target: [B, 2, H, W] ground-truth HV
        mask:   [B, 1, H, W] binary nucleus mask
        ksize:  Single kernel size (used if scales is None, for backward compat)
        scales: Tuple of kernel sizes for multi-scale (e.g. (3, 5, 7))

    Returns:
        scalar loss
    """
    if scales is None:
        scales = (ksize,)

    m = mask.expand(-1, 2, -1, -1)
    total_loss = 0.0

    for s in scales:
        kh, kv = _gradient_kernel_2d(s, pred.device)
        # Expand to 2-channel groups: each channel gets its own kernel copy
        kh = kh.view(1, 1, s, s).expand(2, 1, s, s)
        kv = kv.view(1, 1, s, s).expand(2, 1, s, s)
        p = s // 2

        grad_pred_h = F.conv2d(pred, kh, groups=2, padding=p)
        grad_pred_v = F.conv2d(pred, kv, groups=2, padding=p)
        grad_target_h = F.conv2d(target, kh, groups=2, padding=p)
        grad_target_v = F.conv2d(target, kv, groups=2, padding=p)

        loss_h = F.mse_loss(grad_pred_h * m, grad_target_h * m)
        loss_v = F.mse_loss(grad_pred_v * m, grad_target_v * m)
        total_loss += loss_h + loss_v

    return total_loss / len(scales)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
               eps: float = 1e-8) -> torch.Tensor:
    """MSE computed only on masked (nuclear) pixels."""
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    if mask.shape[1] == 1:
        mask = mask.expand(-1, 2, -1, -1)
    diff = (pred - target) ** 2
    return (diff * mask).sum() / (mask.sum() + eps)


# ==============================================================================
#  Focal Tversky Loss  (CellViT NP loss)
# ==============================================================================

class FocalTverskyLoss(nn.Module):
    """Focal Tversky loss for binary segmentation (CellViT).

    Tversky index:    TI = (TP + ε) / (TP + α·FP + β·FN + ε)
    Focal mechanism:  loss = (1 - TI)^γ

    α > β  →  penalise FP more than FN (precision-leaning)
    α < β  →  penalise FN more than FP (recall-leaning)
    α = β  →  equivalent to Dice (α=β=0.5 gives Dice)
    γ > 1  →  suppress easy regions, focus on hard ones

    CellViT defaults: α=0.7, β=0.3, γ=4/3 (≈1.333)
    """

    def __init__(self, alpha: float = 0.7, beta: float = 0.3,
                 gamma: float = 4/3, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  [B, 1, H, W] raw logits
            targets: [B, 1, H, W] binary {0,1}

        Returns:
            scalar loss
        """
        prob = torch.sigmoid(logits)
        prob = prob.reshape(-1)
        targets = targets.reshape(-1)

        tp = (prob * targets).sum()
        fp = (prob * (1.0 - targets)).sum()
        fn = ((1.0 - prob) * targets).sum()

        ti = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return (1.0 - ti) ** self.gamma

    def per_pixel(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Per-pixel proxy for OHEM compatibility — returns BCE loss."""
        return F.binary_cross_entropy_with_logits(logits, targets, reduction='none')


# ==============================================================================
#  Focal Loss  (§4.3, from HistoNeXt)
# ==============================================================================

class FocalLoss(nn.Module):
    """Multi-class focal loss with ignore_index support."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, ignore_index: int = 255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                class_weights: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            logits:        [B, C, H, W]
            targets:       [B, H, W] class indices, with ignore_index for bg
            class_weights: [C] per-class importance weights (from ClassBalancedWeight)

        Returns:
            scalar loss
        """
        B, C, H, W = logits.shape
        logits = logits.permute(0, 2, 3, 1).reshape(-1, C)  # [N, C]
        targets = targets.reshape(-1)                         # [N]

        # Remove ignore pixels
        valid = targets != self.ignore_index
        if valid.sum() == 0:
            return torch.tensor(0.0, device=logits.device)

        logits = logits[valid]
        targets = targets[valid]

        log_prob = F.log_softmax(logits, dim=-1)
        prob = log_prob.exp()
        ce = -log_prob[range(len(targets)), targets]

        p_t = prob[range(len(targets)), targets]
        focal_weight = (1 - p_t) ** self.gamma

        # Class-balanced importance weights (HoVer-NeXt)
        if class_weights is not None:
            cw = class_weights.to(targets.device)[targets]
            focal_weight = focal_weight * cw

        if self.alpha is not None:
            alpha_t = torch.full_like(targets, self.alpha, dtype=torch.float32)
            focal_weight = alpha_t * focal_weight

        return (focal_weight * ce).mean()


# ==============================================================================
#  Multi-class Dice Loss
# ==============================================================================

class MultiClassDiceLoss(nn.Module):
    """Soft Dice loss for multi-class segmentation.

    Computes Dice per class (excluding ignored pixels), then averages.
    Smooth term prevents division by zero.
    """

    def __init__(self, smooth: float = 1.0, ignore_index: int = 255):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                class_weights: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            logits:        [B, C, H, W]
            targets:       [B, H, W] class indices, with ignore_index for bg
            class_weights: [C]  (unused; kept for API compatibility)

        Returns:
            scalar loss (1 - mean Dice)
        """
        B, C = logits.shape[:2]
        prob = F.softmax(logits, dim=1)                      # [B, C, H, W]
        targets_onehot = F.one_hot(targets.clamp(0, C - 1), C) \
                           .permute(0, 3, 1, 2).float()      # [B, C, H, W]

        # Mask out ignored pixels
        mask = (targets != self.ignore_index).unsqueeze(1).float()  # [B, 1, H, W]
        prob = prob * mask
        targets_onehot = targets_onehot * mask

        inter = (prob * targets_onehot).sum(dim=(0, 2, 3))   # [C]
        union = (prob + targets_onehot).sum(dim=(0, 2, 3))   # [C]

        dice = (2.0 * inter + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


# ==============================================================================
#  Plain Cross-Entropy Loss
# ==============================================================================

class CELoss(nn.Module):
    """Standard cross-entropy with optional class weights and ignore_index."""

    def __init__(self, ignore_index: int = 255):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                class_weights: torch.Tensor = None) -> torch.Tensor:
        B, C, H, W = logits.shape
        logits = logits.permute(0, 2, 3, 1).reshape(-1, C)
        targets = targets.reshape(-1)

        valid = targets != self.ignore_index
        if valid.sum() == 0:
            return torch.tensor(0.0, device=logits.device)

        logits = logits[valid]
        targets = targets[valid]

        w = class_weights.to(logits.device) if class_weights is not None else None
        return F.cross_entropy(logits, targets, weight=w)


# ==============================================================================
#  Class-based Importance Sampling  (HoVer-NeXt, §4.3)
# ==============================================================================

class ClassBalancedWeight(nn.Module):
    """Pixel-level class-balanced loss weighting with EMA-tracked distribution.

    HoVer-NeXt formulation:
        p_c^t     = EMA of class c pixel fraction over batches
        w_c       = (1 / p_c)^gamma   — inverse frequency with exponential smoothing
        L_weighted= -1/N Σ w_ci · log(ŷ_i,ci)

    gamma=0 → uniform (no reweighting).
    gamma=1 → linear inverse-frequency.
    gamma=2+ → heavy boost for rare classes (recommended for PanNuke).

    The EMA adapts to batch-to-batch distribution drift, avoiding stale
    global statistics.
    """

    def __init__(self, num_classes: int = 5, gamma: float = 1.0,
                 ema_alpha: float = 0.99, eps: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.ema_alpha = ema_alpha
        self.eps = eps
        self.register_buffer('p_ema', torch.ones(num_classes) / num_classes)
        self.register_buffer('weights', torch.ones(num_classes))
        self._initialised = False

    def update(self, targets: torch.Tensor):
        """Update EMA class distribution from current batch targets.

        Args:
            targets: [B, H, W] valid class indices (0..C-1), background already ignored
        """
        device = targets.device
        if self.p_ema.device != device:
            self.p_ema = self.p_ema.to(device)

        batch_counts = torch.zeros(self.num_classes, device=device)
        for c in range(self.num_classes):
            batch_counts[c] = (targets == c).sum().float()

        total = batch_counts.sum()
        if total == 0:
            return

        p_batch = batch_counts / total

        if not self._initialised:
            self.p_ema.copy_(p_batch)
            self._initialised = True
        else:
            self.p_ema.copy_(self.ema_alpha * self.p_ema + (1 - self.ema_alpha) * p_batch)

        # Compute weights: (1 / p_c)^gamma, normalised
        w = (1.0 / (self.p_ema + self.eps)) ** self.gamma
        self.weights.copy_(w / w.sum())  # normalise so sum=1

    def get_weights(self) -> torch.Tensor:
        """Returns per-class weights [C]."""
        return self.weights


# ==============================================================================
#  OHEM  (§4.4)
# ==============================================================================

def ohem_loss(loss_per_pixel: torch.Tensor, keep_ratio: float = 0.5) -> torch.Tensor:
    """Keep gradient only on the top-keep_ratio hardest pixels.

    e.g. keep_ratio=0.5 → keeps the top 50% pixels with highest loss.
    """
    k = max(1, int((1.0 - keep_ratio) * loss_per_pixel.numel()))
    k = min(k, loss_per_pixel.numel() - 1)  # ensure at least 1 pixel kept
    threshold = torch.kthvalue(loss_per_pixel.flatten(), k).values
    mask = (loss_per_pixel >= threshold).float()
    return (loss_per_pixel * mask).sum() / mask.sum().clamp(min=1)


# ==============================================================================
#  Size Prior (kept as optional)
# ==============================================================================

class SizePriorLoss(nn.Module):
    """TV loss + area ratio constraint."""

    def __init__(self, tv_weight: float = 1.0, min_area_ratio: float = 0.03,
                 max_area_ratio: float = 0.45):
        super().__init__()
        self.tv_weight = tv_weight
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio

    def forward(self, np_prob: torch.Tensor) -> tuple:
        grad_x = torch.abs(np_prob[..., :-1] - np_prob[..., 1:])
        grad_y = torch.abs(np_prob[..., :-1, :] - np_prob[..., 1:, :])
        loss_tv = grad_x.mean() + grad_y.mean()
        fg_ratio = np_prob.mean(dim=[1, 2, 3])
        loss_area = (F.relu(self.min_area_ratio - fg_ratio) +
                     F.relu(fg_ratio - self.max_area_ratio)).mean()
        total = self.tv_weight * loss_tv + 0.5 * loss_area
        return total, {"size_tv": loss_tv.item(), "size_area": loss_area.item()}


# ==============================================================================
#  Combined NC Loss (Focal + Dice)
# ==============================================================================

class MultiClassFocalTverskyLoss(nn.Module):
    """Multi-class Focal Tversky loss (applied per-class, averaged).

    For each class c: treat as binary problem with softmax prob[:,c] vs onehot[:,c].
    TI_c = (TP + ε) / (TP + α·FP + β·FN + ε)
    loss = mean_c (1 - TI_c)^γ
    """

    def __init__(self, alpha: float = 0.7, beta: float = 0.3,
                 gamma: float = 4/3, smooth: float = 1e-6,
                 ignore_index: int = 255):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """logits: [B,C,H,W], targets: [B,H,W] class indices"""
        B, C = logits.shape[:2]
        prob = F.softmax(logits, dim=1)                        # [B, C, H, W]
        targets_onehot = F.one_hot(targets.clamp(0, C - 1), C) \
                           .permute(0, 3, 1, 2).float()        # [B, C, H, W]

        mask = (targets != self.ignore_index).unsqueeze(1).float()
        prob = prob * mask
        targets_onehot = targets_onehot * mask

        loss = 0.0
        for c in range(C):
            p = prob[:, c].reshape(-1)
            t = targets_onehot[:, c].reshape(-1)
            tp = (p * t).sum()
            fp = (p * (1.0 - t)).sum()
            fn = ((1.0 - p) * t).sum()
            ti = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
            loss = loss + (1.0 - ti) ** self.gamma
        return loss / C


class FocalDiceLoss(nn.Module):
    """FocalLoss + MultiClassDiceLoss, summed with equal weight."""

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0,
                 dice_weight: float = 1.0, ignore_index: int = 255):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma, ignore_index=ignore_index)
        self.dice = MultiClassDiceLoss(ignore_index=ignore_index)
        self.dice_weight = dice_weight

    def forward(self, logits, targets, class_weights=None):
        loss_f = self.focal(logits, targets, class_weights)
        loss_d = self.dice(logits, targets)
        return loss_f + self.dice_weight * loss_d


class FocalTverskyDiceBCELoss(nn.Module):
    """Official CellViT NC loss: FocalTversky + Dice + BCE.

    L = wt_ft * L_FT + wt_dice * L_Dice + wt_bce * L_BCE
    """

    def __init__(self, alpha: float = 0.7, beta: float = 0.3, gamma: float = 4/3,
                 ft_weight: float = 0.5, dice_weight: float = 0.2,
                 bce_weight: float = 0.5, ignore_index: int = 255):
        super().__init__()
        self.ft = MultiClassFocalTverskyLoss(alpha=alpha, beta=beta, gamma=gamma,
                                              ignore_index=ignore_index)
        self.dice = MultiClassDiceLoss(ignore_index=ignore_index)
        self.ft_weight = ft_weight
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.ignore_index = ignore_index

    def forward(self, logits, targets, class_weights=None):
        loss_ft = self.ft(logits, targets)
        loss_dice = self.dice(logits, targets)

        B, C, H, W = logits.shape
        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)
        targets_flat = targets.reshape(-1)
        valid = targets_flat != self.ignore_index
        if valid.sum() > 0:
            loss_bce = F.binary_cross_entropy_with_logits(
                logits_flat[valid], F.one_hot(targets_flat[valid], C).float())
        else:
            loss_bce = torch.tensor(0.0, device=logits.device)

        return self.ft_weight * loss_ft + self.dice_weight * loss_dice + self.bce_weight * loss_bce


# ==============================================================================
#  Combined Loss (NP + HV + NC + Gate)
# ==============================================================================

class CombinedLoss(nn.Module):
    """
    P0-design losses for NP / HV / NC + gate regularisation.

        L = loss_np + loss_hv + loss_nc + λ_gate * L_gate_smooth
    """

    def __init__(
        self,
        num_nc_classes: int = 5,
        # NP
        np_loss: str = "asym",              # "asym" | "dice" | "bce+dice" | "ce+dice" | "ft" | "ft+dice"
        np_weight: float = 2.0,             # NP contribution (NP:NC:HV = 2:2:1)
        asym_gamma_neg: float = 4.0,
        asym_gamma_pos: float = 0.5,
        np_ohem: float = 0.5,           # keep_ratio for OHEM, 0 = disabled
        np_cl_dice_weight: float = 0.0, # optional, default off
        np_bce_weight: float = 1.0,      # BCE weight for bce+dice / ce+dice
        np_dice_weight: float = 1.0,     # Dice weight for bce+dice / ce+dice / ft+dice
        np_ft_alpha: float = 0.7,        # FT α (FP weight)
        np_ft_beta: float = 0.3,         # FT β (FN weight)
        np_ft_gamma: float = 4/3,        # FT γ (focal exponent)
        # HV
        hv_mse_weight: float = 1.0,
        hv_msge_weight: float = 0.5,
        hv_loss_weight: float = 1.0,    # total HV weight (NP:NC:HV = 2:2:1)
        # NC
        nc_loss: str = "focal",         # "focal" | "dice" | "ce"
        nc_focal_alpha: float = 1.0,
        nc_focal_gamma: float = 2.0,
        nc_dice_weight: float = 1.0,   # Dice weight in focal+dice / ft+dice+bce
        nc_ft_alpha: float = 0.7,       # FT α (FP weight) for NC ft+dice+bce
        nc_ft_beta: float = 0.3,        # FT β (FN weight) for NC ft+dice+bce
        nc_ft_gamma: float = 4/3,       # FT γ (focal exponent) for NC ft+dice+bce
        nc_ft_weight: float = 0.5,      # FT weight for NC ft+dice+bce
        nc_bce_weight: float = 0.5,     # BCE weight for NC ft+dice+bce
        nc_weight: float = 2.0,         # NC weight (NP:NC:HV = 2:2:1)
        # Class-balanced importance sampling (HoVer-NeXt)
        cb_gamma: float = 1.5,           # inverse-frequency exponent, 0=off
        cb_ema_alpha: float = 0.99,     # EMA decay for class distribution
        # Size prior (optional)
        use_size_prior: bool = False,
        tv_weight: float = 1.0,
        cldice_iters: int = 5,
    ):
        super().__init__()
        self.num_nc_classes = num_nc_classes

        # NP — select loss type
        self.np_loss_type = np_loss
        if np_loss == "dice":
            self.np_loss_fn = BinaryDiceLoss()
            self.np_ohem = 0
        elif np_loss in ("bce+dice", "ce+dice"):
            self.np_loss_fn = None     # handled inline in forward()
            self.np_ohem = 0
            self.np_bce_weight = np_bce_weight
            self.np_dice_weight = np_dice_weight
        elif np_loss == "ft":
            self.np_loss_fn = FocalTverskyLoss(
                alpha=np_ft_alpha, beta=np_ft_beta, gamma=np_ft_gamma)
            self.np_ohem = 0           # FT has built-in focal
        elif np_loss == "ft+dice":
            self.np_loss_fn = FocalTverskyLoss(
                alpha=np_ft_alpha, beta=np_ft_beta, gamma=np_ft_gamma)
            self.np_ohem = 0
            self.np_dice_weight = np_dice_weight
        else:
            self.np_loss_fn = AsymmetricLoss(gamma_neg=asym_gamma_neg, gamma_pos=asym_gamma_pos)
            self.np_ohem = np_ohem
        self.np_weight = np_weight
        self.np_cl_dice_weight = np_cl_dice_weight
        self.cldice_iters = cldice_iters

        # HV
        self.hv_mse_weight = hv_mse_weight
        self.hv_msge_weight = hv_msge_weight
        self.hv_loss_weight = hv_loss_weight

        # NC — select loss type
        self.nc_loss_type = nc_loss
        if nc_loss == "dice":
            self.nc_loss_fn = MultiClassDiceLoss(ignore_index=255)
        elif nc_loss == "ce":
            self.nc_loss_fn = CELoss(ignore_index=255)
        elif nc_loss == "focal+dice":
            self.nc_loss_fn = FocalDiceLoss(alpha=nc_focal_alpha, gamma=nc_focal_gamma,
                                            dice_weight=nc_dice_weight, ignore_index=255)
        elif nc_loss == "ft+dice+bce":
            self.nc_loss_fn = FocalTverskyDiceBCELoss(
                alpha=nc_ft_alpha, beta=nc_ft_beta, gamma=nc_ft_gamma,
                ft_weight=nc_ft_weight, dice_weight=nc_dice_weight,
                bce_weight=nc_bce_weight, ignore_index=255)
        else:
            self.nc_loss_fn = FocalLoss(alpha=nc_focal_alpha, gamma=nc_focal_gamma,
                                        ignore_index=255)
        self.nc_weight = nc_weight

        # Class-balanced importance sampling (HoVer-NeXt)
        self.cb_gamma = cb_gamma
        self.cb_active = cb_gamma > 0
        if self.cb_active:
            self.cb_weight = ClassBalancedWeight(
                num_classes=num_nc_classes, gamma=cb_gamma, ema_alpha=cb_ema_alpha,
            )

        # Size prior (optional)
        self.use_size_prior = use_size_prior
        if use_size_prior:
            self.size_prior = SizePriorLoss(tv_weight=tv_weight)

    def forward(self, outputs: dict, batch: dict) -> tuple:
        """
        Args:
            outputs: {'np': [B,1,H,W], 'nc': [B,C,H,W], 'hv': [B,2,H,W],
                      'gate': [B,1,H,W] or None}
            batch:   dict with 'mask', 'np_gt', 'hv_gt'

        Returns:
            (total_loss, loss_dict)
        """
        np_logits = outputs["np"]
        nc_logits = outputs["nc"]
        hv_pred = outputs["hv"]

        targets = batch["mask"].long()
        np_gt = batch["np_gt"]
        hv_gt = batch["hv_gt"]
        device = np_logits.device
        loss_dict = {}

        # ================================================================
        #  NP: asym / dice / bce+dice / ce+dice / ft / ft+dice + OHEM + clDice + SizePrior
        # ================================================================
        if self.np_loss_type in ("bce+dice", "ce+dice"):
            # BCE/CE + Dice: binary cross-entropy + soft dice
            np_prob = torch.sigmoid(np_logits)
            loss_np_bce = F.binary_cross_entropy_with_logits(
                np_logits, np_gt, reduction='mean')
            loss_np_dice = 1.0 - soft_dice(np_prob, np_gt)
            loss_np = (self.np_bce_weight * loss_np_bce +
                       self.np_dice_weight * loss_np_dice)
            loss_dict["np_bce"] = loss_np_bce.item()
            loss_dict["np_dice"] = loss_np_dice.item()
        elif self.np_loss_type == "ft+dice":
            # FocalTversky + Dice (CellViT NP loss)
            loss_np_ft = self.np_loss_fn(np_logits, np_gt)
            np_prob = torch.sigmoid(np_logits)
            loss_np_dice = 1.0 - soft_dice(np_prob, np_gt)
            loss_np = loss_np_ft + self.np_dice_weight * loss_np_dice
            loss_dict["np_ft"] = loss_np_ft.item()
            loss_dict["np_dice"] = loss_np_dice.item()
        elif self.np_ohem > 0 and self.training:
            # Per-pixel loss, then keep top-k hardest (training only)
            px_loss = self.np_loss_fn.per_pixel(np_logits, np_gt)
            loss_np_asym = ohem_loss(px_loss, keep_ratio=self.np_ohem)
            # Optional clDice
            if self.np_cl_dice_weight > 0:
                np_prob = torch.sigmoid(np_logits)
                loss_np_cldice = 1.0 - cl_dice(np_prob, np_gt, iters=self.cldice_iters)
                loss_np = loss_np_asym + self.np_cl_dice_weight * loss_np_cldice
                loss_dict["np_cldice"] = loss_np_cldice.item()
            else:
                loss_np = loss_np_asym
            loss_dict["np_base"] = loss_np_asym.item()
        else:
            loss_np_asym = self.np_loss_fn(np_logits, np_gt)
            # Optional clDice
            if self.np_cl_dice_weight > 0:
                np_prob = torch.sigmoid(np_logits)
                loss_np_cldice = 1.0 - cl_dice(np_prob, np_gt, iters=self.cldice_iters)
                loss_np = loss_np_asym + self.np_cl_dice_weight * loss_np_cldice
                loss_dict["np_cldice"] = loss_np_cldice.item()
            else:
                loss_np = loss_np_asym
            loss_dict["np_base"] = loss_np_asym.item()

        # Optional SizePrior
        if self.use_size_prior:
            np_prob = torch.sigmoid(np_logits)
            loss_sp, sp_dict = self.size_prior(np_prob)
            loss_np = loss_np + 0.1 * loss_sp
            loss_dict["np_size"] = loss_sp.item()

        loss_dict["np_total"] = loss_np.item()

        # ================================================================
        #  HV: MSE + MSGE + Focus Mask
        # ================================================================
        loss_hv_mse = F.mse_loss(hv_pred, hv_gt)  # CellViT: all pixels, bg→0
        loss_hv_msge = msge_loss(hv_pred, hv_gt, np_gt, scales=(3, 5, 7))
        loss_hv = (self.hv_mse_weight * loss_hv_mse +
                   self.hv_msge_weight * loss_hv_msge)

        loss_dict["hv_mse"] = loss_hv_mse.item()
        loss_dict["hv_msge"] = loss_hv_msge.item()
        loss_dict["hv_total"] = loss_hv.item()

        # ================================================================
        #  NC: focal / dice / ce + Class-balanced importance sampling
        # ================================================================
        nc_targets = targets.clone()
        nc_targets[targets >= self.num_nc_classes] = 255  # bg → ignore

        # Update class-balanced weights from valid NC targets
        # Dice loss is class-agnostic by design, skip CBWeight
        if self.cb_active and self.training and self.nc_loss_type != "dice":
            valid = nc_targets[nc_targets != 255]
            if valid.numel() > 0:
                self.cb_weight.update(valid)
            class_w = self.cb_weight.get_weights()
        else:
            class_w = None

        loss_nc = self.nc_loss_fn(nc_logits, nc_targets, class_weights=class_w)
        loss_dict["nc_total"] = loss_nc.item()

        # ================================================================
        #  Total
        # ================================================================
        total = (self.np_weight * loss_np +
                 self.hv_loss_weight * loss_hv +
                 self.nc_weight * loss_nc)
        if torch.isnan(total):
            safe = loss_np if not torch.isnan(loss_np) else torch.tensor(0.0, device=device)
            total = safe

        loss_dict["total"] = total.item()
        return total, loss_dict
