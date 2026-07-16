"""
PanNuke dataset loader for 3-head (NP + NC + HV) nuclei segmentation.

Reads preprocessed data from:
    processed/FoldX/
        images/*.png       — RGB images (256×256)
        hover/*.npz        — GT with keys: np_map, hv_map, inst_map, type_map

Pre-loads all data into RAM at init time for fast per-batch access.
Uses float16 for HV maps and int16 for instance/type maps to reduce memory.
"""
import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, List, Dict

from .augs import get_train_augmentations


CLASS_NAMES = ["neoplastic", "inflammatory", "connective", "dead", "epithelial"]
NUM_CLASSES = 5
IMAGENET_MEAN_NP = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD_NP = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class PanNukeDataset(Dataset):
    """Preprocessed PanNuke dataset. All data pre-loaded into RAM for speed."""

    def __init__(
        self,
        data_root: str = "/home/lwy/dataset/PanNuke/processed",
        folds: Tuple[int, ...] = (1, 2),
        augment: bool = False,
        heavy_aug: bool = False,
    ):
        self.augment = augment
        self.aug_pipeline = get_train_augmentations(heavy=heavy_aug) if augment else None

        # ---- Collect file pairs ----
        sample_paths: List[Tuple[Path, Path]] = []
        for fold in folds:
            if fold == 0:
                # Flat structure: data_root/images/ + data_root/hover/
                img_dir = Path(data_root) / "images"
                gt_dir  = Path(data_root) / "hover"
            else:
                img_dir = Path(data_root) / f"Fold{fold}" / "images"
                gt_dir  = Path(data_root) / f"Fold{fold}" / "hover"
            if not img_dir.exists():
                raise FileNotFoundError(f"Image dir not found: {img_dir}")
            for img_path in sorted(img_dir.glob("*.png")):
                gt_path = gt_dir / (img_path.stem + ".npz")
                if gt_path.exists():
                    sample_paths.append((img_path, gt_path))

        # ---- Pre-load all data into RAM ----
        print(f"Pre-loading {len(sample_paths)} samples into RAM ...")
        self._images: List[np.ndarray] = []
        self._np_maps: List[np.ndarray] = []
        self._hv_maps: List[np.ndarray] = []
        self._inst_maps: List[np.ndarray] = []
        self._type_maps: List[np.ndarray] = []

        total_type = np.zeros(6, dtype=np.int64)
        for img_path, gt_path in sample_paths:
            # Image: keep as uint8 to save RAM
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self._images.append(img)

            # GT: load from npz
            gt = np.load(gt_path)
            np_map = gt["np_map"].astype(np.uint8)       # [H, W] 0/1
            hv_map = gt["hv_map"].astype(np.float16)     # [H, W, 2] → fp16
            inst_map = gt["inst_map"].astype(np.int16)   # [H, W] → int16
            type_map = gt["type_map"].astype(np.int16)   # [H, W] → int16

            self._np_maps.append(np_map)
            self._hv_maps.append(hv_map)
            self._inst_maps.append(inst_map)
            self._type_maps.append(type_map)

            # Accumulate class stats
            for c in range(6):
                total_type[c] += (type_map == c).sum()

        # ---- Stats ----
        self._print_stats(total_type)

    def _print_stats(self, total):
        print(f"Loaded {len(self)} samples")
        total_px = total.sum()
        if total_px == 0:
            return
        print("  Class distribution:")
        for i, name in enumerate(CLASS_NAMES):
            print(f"    {i+1} ({name:>15s}): {100*total[i+1]/total_px:5.2f}%")
        print(f"    0 ({'background':>15s}): {100*total[0]/total_px:5.2f}%")

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img = self._images[idx]  # uint8 [H, W, 3]

        # Normalise image (numpy — faster than torch)
        img_np = img.astype(np.float32) / 255.0
        img_np = (img_np - IMAGENET_MEAN_NP.reshape(1, 1, 3)) / IMAGENET_STD_NP.reshape(1, 1, 3)
        img_t = torch.from_numpy(img_np).permute(2, 0, 1)

        # GT: restore precision → tensor
        np_map = torch.from_numpy(self._np_maps[idx]).unsqueeze(0).float()
        hv_map = torch.from_numpy(self._hv_maps[idx].astype(np.float32)).permute(2, 0, 1)
        inst_t = torch.from_numpy(self._inst_maps[idx].astype(np.int32))
        type_map = self._type_maps[idx].astype(np.int32)

        # Semantic mask: 1..5=fg → 0..4, 0=bg → 5
        semantic = type_map.copy()
        fg = semantic > 0
        semantic[fg] = semantic[fg] - 1   # fg: 1..5 → 0..4
        semantic[~fg] = 5                 # bg: 0   → 5
        mask_t = torch.from_numpy(semantic)

        # Augmentation
        if self.aug_pipeline is not None:
            combined = torch.stack([mask_t.float(), inst_t.float()], dim=0)
            img_t, combined = self.aug_pipeline(img_t, combined)
            mask_t = combined[0].long()
            inst_t = combined[1].long()
            np_map = (inst_t > 0).float().unsqueeze(0)
            # Rebuild HV from augmented instance map (geometry transforms
            # would otherwise leave stale HV coordinates)
            hv_map = self._rebuild_hv(inst_t)

        return {
            "image": img_t.float(),
            "mask": mask_t,                # [H, W] 0..4=fg, 5=bg
            "np_gt": np_map,              # [1, H, W] binary
            "hv_gt": hv_map,              # [2, H, W] HV offsets
            "inst_gt": inst_t,            # [H, W] instance IDs
            "idx": idx,
        }

    @staticmethod
    def _rebuild_hv(inst_t: torch.Tensor) -> torch.Tensor:
        """Rebuild HV map from augmented instance labels."""
        import cv2
        inst_np = inst_t.cpu().numpy()
        H, W = inst_np.shape
        hv = np.zeros((2, H, W), dtype=np.float32)
        for iid in np.unique(inst_np):
            if iid == 0:
                continue
            mask = (inst_np == iid).astype(np.uint8)
            if mask.sum() < 4:
                continue
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            if not rows.any():
                continue
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            rmax, cmax = rmax + 1, cmax + 1
            rmin, cmin = max(rmin - 2, 0), max(cmin - 2, 0)
            rmax, cmax = min(rmax + 2, H), min(cmax + 2, W)

            crop = mask[rmin:rmax, cmin:cmax]
            if crop.shape[0] < 2 or crop.shape[1] < 2:
                continue
            moments = cv2.moments(crop)
            if moments["m00"] == 0:
                continue
            cx = int(moments["m10"] / moments["m00"] + 0.5)
            cy = int(moments["m01"] / moments["m00"] + 0.5)

            ys, xs = np.mgrid[0:crop.shape[0], 0:crop.shape[1]]
            xo = (xs - cx).astype(np.float32)
            yo = (ys - cy).astype(np.float32)
            xo[crop == 0] = 0
            yo[crop == 0] = 0

            xn = xo[xo < 0]
            if len(xn) > 0: xo[xo < 0] /= -xn.min()
            xp = xo[xo > 0]
            if len(xp) > 0: xo[xo > 0] /= xp.max()
            yn = yo[yo < 0]
            if len(yn) > 0: yo[yo < 0] /= -yn.min()
            yp = yo[yo > 0]
            if len(yp) > 0: yo[yo > 0] /= yp.max()

            hv[0, rmin:rmax, cmin:cmax][crop > 0] = xo[crop > 0]
            hv[1, rmin:rmax, cmin:cmax][crop > 0] = yo[crop > 0]

        return torch.from_numpy(hv)


def get_pannuke_loaders(
    data_root: str = "/home/lwy/dataset/PanNuke/processed",
    train_folds: Tuple[int, ...] = (1, 2),
    val_folds: Tuple[int, ...] = (3,),
    batch_size: int = 128,
    num_workers: int = 16,
    augment: bool = True,
    heavy_aug: bool = False,
    balance_sample: bool = False,
    seed: int = 114514,
) -> Tuple[DataLoader, DataLoader]:

    def worker_init_fn(worker_id):
        np.random.seed(seed + worker_id)
        random.seed(seed + worker_id)

    train_dataset = PanNukeDataset(
        data_root=data_root, folds=train_folds,
        augment=augment, heavy_aug=heavy_aug,
    )
    val_dataset = PanNukeDataset(
        data_root=data_root, folds=val_folds,
        augment=False,
    )

    # ---- Class-balanced importance sampling (HoVer-NeXt) ----
    # Weights are inverse-instance-frequency per image:
    #   w_img = Σ_c (1 / count_c)  for each class c present in the image
    # Dead cells (0.06% of all instances) → 1/count ≈ massive boost
    if balance_sample:
        from torch.utils.data import WeightedRandomSampler

        # Count instances per class across the entire training set
        class_inst_counts = np.zeros(6, dtype=np.int64)  # 1..5=fg, 0=bg (unused)
        for i in range(len(train_dataset)):
            inst = train_dataset._inst_maps[i]
            tp = train_dataset._type_maps[i]
            for iid in np.unique(inst):
                if iid == 0:
                    continue
                cls = int(tp[inst == iid][0])
                if 1 <= cls <= 5:
                    class_inst_counts[cls] += 1

        # Inverse frequency (add small epsilon to avoid division by zero)
        inv_freq = {c: 1.0 / max(class_inst_counts[c], 1) for c in range(1, 6)}
        # Normalise so max boost = 1.0 (neoplastic gets ~0, dead gets ~1)
        max_inv = max(inv_freq.values())
        inv_freq = {c: v / max_inv for c, v in inv_freq.items()}

        # Per-image weight = sum of inverse frequencies of classes present
        weights = []
        for i in range(len(train_dataset)):
            inst = train_dataset._inst_maps[i]
            tp = train_dataset._type_maps[i]
            classes_present = set()
            for iid in np.unique(inst):
                if iid == 0:
                    continue
                cls = int(tp[inst == iid][0])
                if 1 <= cls <= 5:
                    classes_present.add(cls)
            w = sum(inv_freq.get(c, 0.0) for c in classes_present) + 0.01  # min baseline
            weights.append(w)

        print(f"  Class-balanced sampling enabled")
        print(f"    Instance counts: neo={class_inst_counts[1]} infl={class_inst_counts[2]} "
              f"conn={class_inst_counts[3]} dead={class_inst_counts[4]} epi={class_inst_counts[5]}")
        print(f"    Sample weights:  neo={inv_freq[1]:.4f} infl={inv_freq[2]:.4f} "
              f"conn={inv_freq[3]:.4f} dead={inv_freq[4]:.4f} epi={inv_freq[5]:.4f}")

        g = torch.Generator()
        g.manual_seed(seed)
        sampler = WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True,
            generator=g,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=True, drop_last=True,
            worker_init_fn=worker_init_fn, generator=g,
            persistent_workers=True if num_workers > 0 else False,
        )
    else:
        g = torch.Generator()
        g.manual_seed(seed)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True,
            worker_init_fn=worker_init_fn, generator=g,
            persistent_workers=True if num_workers > 0 else False,
        )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        worker_init_fn=worker_init_fn,
        persistent_workers=True if num_workers > 0 else False,
    )
    return train_loader, val_loader
