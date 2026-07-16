#!/usr/bin/env python3
"""
Inference + visualization: 5 composite figures (original | GT | NP | HV | result).

Usage:
    python infer_viz.py --checkpoint output/15run/hv_cvt/best.pth --output ./viz/
"""
import argparse, os, sys
from pathlib import Path
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import create_model
from utils.postprocess import process_np_hv, classify_instances
from data.pannuke import PanNukeDataset

CLASS_COLORS = {
    0: (0, 0, 0),
    1: (220, 20, 60),       # neoplastic: crimson
    2: (255, 127, 0),        # inflammatory: orange
    3: (0, 160, 0),          # connective: green
    4: (180, 0, 180),        # dead: purple
    5: (0, 140, 200),        # epithelial: teal
}
CLASS_NAMES = ["neoplastic", "inflammatory", "connective", "dead", "epithelial"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--decoder", type=str, default="unet3")
    p.add_argument("--output", type=str, default="./viz/")
    p.add_argument("--data_root", type=str, default="/home/lwy/dataset/PanNuke/processed")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--num_images", type=int, default=5)
    p.add_argument("--start_idx", type=int, default=0)
    # PP params (best for hv_cvt)
    p.add_argument("--np_thresh", type=float, default=0.45)
    p.add_argument("--min_area", type=int, default=20)
    p.add_argument("--energy_thresh", type=float, default=0.4)
    p.add_argument("--sobel_ksize", type=int, default=23)
    p.add_argument("--marker_ksize", type=int, default=1)
    return p.parse_args()


def colorize_class_map(class_map):
    """Class map [H,W] (0=bg, 1..5=fg) -> RGB image."""
    h, w = class_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in CLASS_COLORS.items():
        rgb[class_map == cls_id] = color
    return rgb


def colorize_instance_map(inst_map):
    """Instance map -> random-colour RGB."""
    h, w = inst_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for iid in np.unique(inst_map):
        if iid == 0:
            continue
        np.random.seed(int(iid) * 37)
        color = np.random.randint(0, 256, 3, dtype=np.uint8)
        rgb[inst_map == iid] = color
    return rgb


def make_overlay(image, class_map, alpha=0.5):
    color_mask = colorize_class_map(class_map)
    return cv2.addWeighted(image, 1.0 - alpha, color_mask, alpha, 0)


def create_composite(orig, gt_mask_rgb, np_map, hv_map, pred_overlay, inst_rgb, save_path):
    """Create 1×6 composite: [orig | GT | NP | HV | pred | instances]."""
    fig, axes = plt.subplots(1, 6, figsize=(30, 5.5))

    axes[0].imshow(orig)
    axes[0].set_title("Original", fontsize=11); axes[0].axis("off")

    axes[1].imshow(gt_mask_rgb)
    axes[1].set_title("Ground Truth", fontsize=11); axes[1].axis("off")

    im2 = axes[2].imshow(np_map, cmap="hot", vmin=0, vmax=1)
    axes[2].set_title("NP Probability", fontsize=11); axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    axes[3].imshow(hv_map, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[3].set_title("HV (horizontal)", fontsize=11); axes[3].axis("off")

    axes[4].imshow(pred_overlay)
    axes[4].set_title("Prediction", fontsize=11); axes[4].axis("off")

    axes[5].imshow(inst_rgb)
    axes[5].set_title(f"Instances (n={len(np.unique(inst_rgb))//3})", fontsize=11)
    axes[5].axis("off")

    plt.tight_layout(pad=1.0)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output, exist_ok=True)

    # ---- Load model ----
    print(f"Loading: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = create_model(
        variant="uni2-h", num_nc_classes=5, pretrained=False,
        decoder_type=args.decoder, enc_dropout=0.5, dec_dropout=0.2,
        freeze_encoder=True, full_unfreeze=True,
    )
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    print(f"  Epoch {ckpt.get('epoch','?')}  mPQ={ckpt.get('best_metric','?'):.4f}")

    # ---- Load 5 images from PanNuke fold3 ----
    dataset = PanNukeDataset(data_root=args.data_root, folds=(3,), augment=False)
    total = len(dataset)
    indices = list(range(args.start_idx, min(args.start_idx + args.num_images, total)))
    print(f"Using images: {indices}  (fold3, total={total})")

    for idx in indices:
        sample = dataset[idx]
        image_t = sample["image"].unsqueeze(0).to(device)
        mask = sample["mask"].numpy()           # 0..4=fg, 5=bg
        inst_gt = sample["inst_gt"].numpy()
        orig_hw = image_t.shape[-2:]

        # ---- GT visual ----
        # mask: 0..4=fg (neo,infl,conn,dead,epi), 5=bg
        gt_viz = np.zeros_like(mask, dtype=np.int32)
        for c in range(5):
            gt_viz[mask == c] = c + 1            # 0..4 → 1..5, bg stays 0
        gt_rgb = colorize_class_map(gt_viz)

        # ---- Inference ----
        with torch.no_grad():
            out = model(image_t)
        np_prob = torch.sigmoid(out["np"]).cpu().numpy()[0, 0]
        hv = out["hv"].cpu().numpy()[0]
        nc = out["nc"].cpu().numpy()[0]

        # ---- Postprocess ----
        nc_argmax = np.argmax(nc, axis=0)
        inst_map = process_np_hv(
            np_prob, hv,
            np_thresh=args.np_thresh, min_area=args.min_area,
            sobel_ksize=args.sobel_ksize, energy_thresh=args.energy_thresh,
            marker_ksize=args.marker_ksize,
        )
        inst_info = classify_instances(inst_map, nc_argmax)
        class_map = np.zeros_like(inst_map, dtype=np.int32)
        for iid, info in inst_info.items():
            class_map[inst_map == iid] = info["type"] + 1    # 1-indexed

        # ---- HV viz: show horizontal component ----
        hv_h = hv[0]

        # ---- Original image ----
        orig_np = image_t.cpu().numpy()[0]
        orig_np = orig_np * np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1) + np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        orig_np = np.clip(orig_np * 255, 0, 255).astype(np.uint8).transpose(1, 2, 0)

        # ---- Instance map (random colours) ----
        inst_rgb = colorize_instance_map(inst_map)

        # ---- Prediction overlay ----
        pred_overlay = make_overlay(orig_np, class_map)

        # ---- Save composite ----
        save_path = os.path.join(args.output, f"img_{idx:04d}.png")
        create_composite(orig_np, gt_rgb, np_prob, hv_h, pred_overlay, inst_rgb, save_path)

    print(f"\nDone! {len(indices)} images saved to {args.output}/")


if __name__ == "__main__":
    main()
