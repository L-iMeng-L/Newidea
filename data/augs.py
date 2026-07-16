"""
Data augmentation for nuclei segmentation on H&E pathology images.

All transforms operate on paired (image, mask) tensors [C,H,W] and [H,W].
Uses only numpy/torch operations — no extra dependencies.
"""
import random
import numpy as np
import torch
import torch.nn.functional as F


class Compose:
    """Compose multiple augmentations with independent probabilities."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


class RandomFlip:
    """Random horizontal and vertical flips (p=0.5 each, independent)."""

    def __init__(self, p_hflip=0.5, p_vflip=0.5):
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip

    def __call__(self, image, mask):
        if random.random() < self.p_hflip:
            image = torch.flip(image, dims=[-1])
            mask = torch.flip(mask, dims=[-1])
        if random.random() < self.p_vflip:
            image = torch.flip(image, dims=[-2])
            mask = torch.flip(mask, dims=[-2])
        return image, mask


class RandomRotate90:
    """Random rotation by 0, 90, 180, or 270 degrees."""

    def __call__(self, image, mask):
        k = random.randint(0, 3)
        if k > 0:
            image = torch.rot90(image, k, dims=[-2, -1])
            mask = torch.rot90(mask, k, dims=[-2, -1])
        return image, mask


class RandomBrightnessContrast:
    """
    Random brightness/contrast jitter (image only).
    Simulates H&E staining variation across labs and scanners.
    """

    def __init__(self, brightness=0.2, contrast=0.2):
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, image, mask):
        # image: [3, H, W], normalized with ImageNet stats
        # Apply to denormalized image for physically meaningful color jitter
        if random.random() < 0.5:
            # Brightness
            delta = random.uniform(-self.brightness, self.brightness)
            image = image + delta
        if random.random() < 0.5:
            # Contrast: scale around mean
            alpha = 1.0 + random.uniform(-self.contrast, self.contrast)
            mean_val = image.mean(dim=[-2, -1], keepdim=True)
            image = (image - mean_val) * alpha + mean_val
        return image, mask


class RandomAffine:
    """
    Small random affine transform: rotation (±15°) + scale (0.9~1.1) + translation (±5%).

    Uses grid_sample for differentiable-like behavior, but works fine in numpy preprocessing.
    Actually implemented with torch for simplicity.
    """

    def __init__(self, degrees=15, scale=(0.9, 1.1), translate=(0.05, 0.05), p=0.5):
        self.degrees = degrees
        self.scale = scale
        self.translate = translate
        self.p = p

    def __call__(self, image, mask):
        if random.random() > self.p:
            return image, mask

        _, H, W = image.shape
        angle = random.uniform(-self.degrees, self.degrees)
        sc = random.uniform(*self.scale)
        dx = random.uniform(-self.translate[0], self.translate[0]) * W
        dy = random.uniform(-self.translate[1], self.translate[1]) * H

        theta = np.radians(angle)
        cos_a, sin_a = np.cos(theta), np.sin(theta)
        a, b, c, d = cos_a / sc, sin_a / sc, -sin_a / sc, cos_a / sc
        tx_norm, ty_norm = 2.0 * dx / W, 2.0 * dy / H

        affine = torch.tensor([[a, b, tx_norm], [c, d, ty_norm]], dtype=torch.float32)
        grid = F.affine_grid(affine.unsqueeze(0), image.unsqueeze(0).size(), align_corners=False)

        image = F.grid_sample(image.unsqueeze(0), grid, mode='bilinear',
                              padding_mode='reflection', align_corners=False).squeeze(0)

        # Mask: supports both [H,W] (semantic) and [C,H,W] (multi-channel)
        mask_shape = mask.shape
        mask_4d = mask.unsqueeze(0) if mask.dim() == 2 else mask.unsqueeze(0)
        # mask_4d is now [1, H, W] or [1, C, H, W]
        mask_out = F.grid_sample(mask_4d.float(), grid, mode='nearest',
                                 padding_mode='zeros', align_corners=False).squeeze(0)
        if len(mask_shape) == 2:
            mask_out = mask_out.squeeze(0).long()
        else:
            mask_out = mask_out.long()

        return image, mask_out


class RandomHueSaturation:
    """
    Random hue/saturation/value shift as proxy for stain variation.
    """

    def __init__(self, hue=0.05, saturation=0.1, value=0.1, p=0.5):
        self.hue = hue
        self.saturation = saturation
        self.value = value
        self.p = p

    def __call__(self, image, mask):
        if random.random() > self.p:
            return image, mask

        image = image.clone()
        h_shift = random.uniform(-self.hue, self.hue)
        s_scale = 1.0 + random.uniform(-self.saturation, self.saturation)
        v_scale = 1.0 + random.uniform(-self.value, self.value)
        r, g, b = image[0], image[1], image[2]
        image[0] = r + h_shift * (g - r) * 0.5
        image[1] = g * s_scale
        image[2] = b * v_scale
        return image, mask


class RandomGaussianBlur:
    """Random Gaussian blur to simulate out-of-focus / low-magnification scans.

    Applies in denormalised [0,1] space for physically correct blur.
    """

    def __init__(self, ksize_min=3, ksize_max=7, p=0.3):
        self.ksize_min = ksize_min
        self.ksize_max = ksize_max
        self.p = p

    def __call__(self, image, mask):
        if random.random() > self.p:
            return image, mask

        ksize = random.randrange(self.ksize_min, self.ksize_max + 1, 2)
        # Apply per-channel (torch GaussianBlur needs 4D, use unfold trick or scipy)
        import scipy.ndimage as ndi
        img_np = image.permute(1, 2, 0).numpy()
        blurred = ndi.gaussian_filter(img_np, sigma=(ksize / 6.0, ksize / 6.0, 0),
                                      mode='reflect')
        return torch.from_numpy(blurred).permute(2, 0, 1).float(), mask


class RandomElasticTransform:
    """Elastic deformation to simulate tissue stretching / compression.

    Uses a coarse deformation field (alpha, sigma) applied via grid_sample.
    """

    def __init__(self, alpha=30, sigma=5, p=0.3):
        self.alpha = alpha
        self.sigma = sigma
        self.p = p

    def __call__(self, image, mask):
        if random.random() > self.p:
            return image, mask

        import scipy.ndimage as ndi
        _, H, W = image.shape

        # Generate smooth random displacement field
        dx = ndi.gaussian_filter(np.random.randn(H, W) * self.alpha, self.sigma,
                                 mode='reflect')
        dy = ndi.gaussian_filter(np.random.randn(H, W) * self.alpha, self.sigma,
                                 mode='reflect')

        # Build grid_sample grid
        y, x = torch.meshgrid(torch.arange(H, dtype=torch.float32),
                               torch.arange(W, dtype=torch.float32), indexing='ij')
        grid_x = (x + torch.from_numpy(dx).float()) / (W - 1) * 2.0 - 1.0
        grid_y = (y + torch.from_numpy(dy).float()) / (H - 1) * 2.0 - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # [1, H, W, 2]

        image = F.grid_sample(image.unsqueeze(0), grid, mode='bilinear',
                              padding_mode='reflection', align_corners=True).squeeze(0)
        # Mask: [H,W] or [C,H,W] → [1, C, H, W]
        if mask.dim() == 2:
            mask_4d = mask.unsqueeze(0).unsqueeze(0).float()   # [1, 1, H, W]
        else:
            mask_4d = mask.unsqueeze(0).float()                 # [1, C, H, W]
        mask_out = F.grid_sample(mask_4d, grid, mode='nearest',
                                 padding_mode='zeros', align_corners=True).squeeze(0)
        if len(mask.shape) == 2:
            mask_out = mask_out.squeeze(0).long()
        else:
            mask_out = mask_out.long()

        return image, mask_out


def get_train_augmentations(heavy: bool = False) -> Compose:
    """
    Build training augmentation pipeline.

    Args:
        heavy: If True (strong augmentation, inspired by Simple Copy-Paste paper):
               large affine, strong color jitter, blur, elastic deformation.
    """
    transforms = [
        RandomFlip(p_hflip=0.5, p_vflip=0.5),
        RandomRotate90(),
        RandomBrightnessContrast(brightness=0.3, contrast=0.3),
    ]
    if heavy:
        transforms += [
            # Aggressive affine: nuclei are rotation-invariant, scale varies a lot
            RandomAffine(degrees=45, scale=(0.7, 1.3), translate=(0.15, 0.15), p=0.7),
            # Stronger stain variation
            RandomHueSaturation(hue=0.08, saturation=0.25, value=0.25, p=0.5),
            # Simulate out-of-focus / low-magnification
            RandomGaussianBlur(ksize_min=3, ksize_max=7, p=0.3),
            # Simulate tissue stretching/compression
            RandomElasticTransform(alpha=25, sigma=5, p=0.3),
        ]
    return Compose(transforms)
