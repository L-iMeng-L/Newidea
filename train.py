#!/usr/bin/env python3
"""
Training script for 3-head (NP + HV + NC) nuclei segmentation.

Usage:
    python train.py --encoder uni2-h --decoder cellvit --full_unfreeze
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import random
import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
plt.style.use("seaborn-v0_8-whitegrid")  # once at import, not per-plot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import create_model
from losses import CombinedLoss
from data import get_pannuke_loaders
from data.pannuke import PanNukeDataset
from utils.evaluate import CLASS_NAMES

NUM_NC_CLASSES = 5


# ==============================================================================
#  Arguments
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Train 3-head ConvNeXt-V2 on PanNuke")
    p.add_argument("--encoder", type=str, default="tiny",
                   choices=["tiny", "small", "base", "large", "uni2-h"])
    p.add_argument("--version", type=str, default="v2", choices=["v1", "v2"])
    p.add_argument("--decoder", type=str, default="shared_unet",
                   choices=["shared_unet", "shared_unet_mala", "unet3", "unet3_mala"],
                   help="shared_unet:shared decoder | shared_unet_mala:+MALA | unet3:3 indep | unet3_mala:3 indep + MALA")
    p.add_argument("--batch_size", type=int, default=96)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--backbone_lr_mult", type=float, default=0.1,
                   help="Encoder LR = lr * backbone_lr_mult (0.1 = encoder LR 1e-5 when lr=1e-4)")
    p.add_argument("--weight_decay", type=float, default=5e-3)
    p.add_argument("--aspp_out", type=int, default=256)
    p.add_argument("--enc_dropout", type=float, default=0.5,
                   help="Dropout rate for encoder (ConvNeXt output / UNI2 projections)")
    p.add_argument("--dec_dropout", type=float, default=0.2,
                   help="Dropout rate for decoder blocks")
    # Loss
    p.add_argument("--np_loss", type=str, default="asym",
                   choices=["asym", "dice", "bce+dice", "ce+dice", "ft", "ft+dice"],
                   help="NP loss type: asym | dice | bce+dice | ce+dice | ft | ft+dice")
    p.add_argument("--np_weight", type=float, default=2.0,
                   help="NP loss weight (NP:NC:HV = 2:3:1)")
    p.add_argument("--asym_gamma_neg", type=float, default=4.0)
    p.add_argument("--asym_gamma_pos", type=float, default=0.5)
    p.add_argument("--np_ohem", type=float, default=0.5, help="OHEM keep ratio (0=off, 0.5=top 50%%)")
    p.add_argument("--np_cldice_weight", type=float, default=0.0)
    p.add_argument("--np_bce_weight", type=float, default=1.0,
                   help="BCE weight for bce+dice / ce+dice NP loss")
    p.add_argument("--np_dice_weight", type=float, default=1.0,
                   help="Dice weight for bce+dice / ce+dice / ft+dice NP loss")
    p.add_argument("--np_ft_alpha", type=float, default=0.5,
                   help="FocalTversky α (FP weight, higher=more precision)")
    p.add_argument("--np_ft_beta", type=float, default=0.5,
                   help="FocalTversky β (FN weight, higher=more recall)")
    p.add_argument("--np_ft_gamma", type=float, default=1.333,
                   help="FocalTversky γ (focal exponent, >1=focus on hard regions)")
    p.add_argument("--hv_mse_weight", type=float, default=1.0)
    p.add_argument("--hv_msge_weight", type=float, default=0.5)
    p.add_argument("--hv_loss_weight", type=float, default=1.0)
    p.add_argument("--nc_loss", type=str, default="focal",
                   choices=["focal", "dice", "ce", "focal+dice", "ft+dice+bce"],
                   help="NC loss type: focal | dice | ce | focal+dice | ft+dice+bce")
    p.add_argument("--nc_focal_alpha", type=float, default=1.0)
    p.add_argument("--nc_focal_gamma", type=float, default=2.0)
    p.add_argument("--nc_dice_weight", type=float, default=1.0,
                   help="Dice weight in focal+dice / ft+dice+bce")
    p.add_argument("--nc_ft_alpha", type=float, default=0.7,
                   help="FT α (FP weight) for nc_loss=ft+dice+bce")
    p.add_argument("--nc_ft_beta", type=float, default=0.3,
                   help="FT β (FN weight) for nc_loss=ft+dice+bce")
    p.add_argument("--nc_ft_gamma", type=float, default=1.333,
                   help="FT γ (focal exponent) for nc_loss=ft+dice+bce")
    p.add_argument("--nc_ft_weight", type=float, default=0.5,
                   help="FT weight for nc_loss=ft+dice+bce")
    p.add_argument("--nc_bce_weight", type=float, default=0.5,
                   help="BCE weight for nc_loss=ft+dice+bce")
    p.add_argument("--nc_weight", type=float, default=2.0,
                   help="NC loss weight (NP:NC:HV = 2:2:1)")
    p.add_argument("--use_size_prior", action="store_true", default=False)
    # Hardware
    p.add_argument("--no_pretrained", action="store_true", default=False)
    p.add_argument("--freeze_encoder", action="store_true", default=False,
                   help="Freeze encoder backbone (auto-enabled for uni2-h)")
    p.add_argument("--full_unfreeze", action="store_true", default=False,
                   help="UNI2-h: full ViT unfreeze in Phase 2 (CellViT-style, needs large VRAM)")
    p.add_argument("--no_cbam", action="store_true", default=False,
                   help="Remove CBAM from NC head (ablation)")
    p.add_argument("--upsample_mode", type=str, default="transpose",
                   choices=["transpose", "bilinear", "nearest"],
                   help="Decoder upsampling mode")
    p.add_argument("--freeze_epochs", type=int, default=15,
                   help="Epochs to keep encoder frozen before unfreezing")
    p.add_argument("--uni2_weights", type=str, default="/home/lwy/Newidea/pytorch_model.bin",
                   help="Path to local UNI2-h weights (.bin)")
    p.add_argument("--device", type=str, default="cuda:1")
    p.add_argument("--num_workers", type=int, default=12)
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--no_fp16", action="store_false", dest="fp16")
    # Paths
    p.add_argument("--data_root", type=str, default="/home/lwy/dataset/PanNuke/processed")
    p.add_argument("--extend_data", type=str, default=None,
                   help="Path to extend dataset (flat images/ + hover/, fold=0)")
    p.add_argument("--output_dir", type=str, default="./output")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--resume", type=str, default=None)
    # Fold
    p.add_argument("--val_fold", type=int, default=3, choices=[1, 2, 3])
    p.add_argument("--no_aug", action="store_true", default=False,
                   help="Disable all augmentations")
    p.add_argument("--heavy_aug", action="store_true", default=True,
                   help="Enable heavy augmentations (affine + HSV jitter, default: on)")
    p.add_argument("--balance_sample", action="store_true", default=False,
                   help="Image-level weighted sampling (legacy, use --cb_gamma)")
    p.add_argument("--cb_gamma", type=float, default=0.0,
                   help="HoVer-NeXt FocalCE class-balanced exponent (0=off)")
    p.add_argument("--cb_ema_alpha", type=float, default=0.99,
                   help="EMA decay for class distribution tracking")
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--val_interval", type=int, default=1)
    p.add_argument("--log_interval", type=int, default=50)
    # Eval
    p.add_argument("--eval_np_thresh", type=float, default=0.5)
    p.add_argument("--eval_min_area", type=int, default=10)
    p.add_argument("--eval_match_iou", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=114514,
                   help="Random seed for reproducibility")
    return p.parse_args()


# ==============================================================================
#  History & curves
# ==============================================================================

from utils.history import HistoryTracker
from utils.plotting import plot_curves


def _nc_loss_label(args):
    """Human-readable NC loss description for config.json."""
    if args.nc_loss == "dice":
        return "MultiClassDiceLoss"
    elif args.nc_loss == "ce":
        return "CrossEntropyLoss"
    elif args.nc_loss == "focal+dice":
        return f"FocalLoss+MultiClassDiceLoss(alpha={args.nc_focal_alpha},gamma={args.nc_focal_gamma})"
    elif args.nc_loss == "ft+dice+bce":
        return f"FT(α={args.nc_ft_alpha},β={args.nc_ft_beta},γ={args.nc_ft_gamma})+Dice(w={args.nc_dice_weight})+BCE(w={args.nc_bce_weight})"
    else:
        return f"FocalLoss(alpha={args.nc_focal_alpha},gamma={args.nc_focal_gamma})"


# ==============================================================================
#  Fast validation (IoU only)
# ==============================================================================

@torch.no_grad()
def validate_fast(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    np_inter_sum = 0
    np_union_sum = 0
    fg_inters = np.zeros(5, dtype=np.int64)
    fg_unions = np.zeros(5, dtype=np.int64)

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        batch_dev = {
            "mask": batch["mask"].to(device, non_blocking=True),
            "np_gt": batch["np_gt"].to(device, non_blocking=True),
            "hv_gt": batch["hv_gt"].to(device, non_blocking=True),
        }

        outputs = model(images)
        loss, loss_dict = criterion(outputs, batch_dev)
        total_loss += loss.item()

        np_prob = torch.sigmoid(outputs["np"])
        np_pred = (np_prob > 0.5).squeeze(1)
        np_gt_bin = batch_dev["mask"] < 5
        np_inter_sum += (np_pred & np_gt_bin).sum().item()
        np_union_sum += (np_pred | np_gt_bin).sum().item()

        nc_pred = outputs["nc"].argmax(dim=1)
        fg_mask = batch_dev["mask"] < 5  # exclude bg (5) — NC head has no bg class
        for c in range(5):
            pc = (nc_pred == c) & fg_mask
            tc = batch_dev["mask"] == c
            fg_inters[c] += (pc & tc).sum().item()
            fg_unions[c] += (pc | tc).sum().item()

    n = len(loader)
    avg_loss = total_loss / n
    avg_np_iou = np_inter_sum / (np_union_sum + 1e-8)
    per_class_iou = [float(fg_inters[c]) / (fg_unions[c] + 1e-8) for c in range(5)]
    fg_miou = np.mean(per_class_iou)
    return avg_loss, avg_np_iou, fg_miou, per_class_iou


def print_val_summary(epoch, avg_loss, np_iou, fg_miou, per_class_iou):
    print(f"--- Validation (Epoch {epoch}) ---")
    print(f"  Loss: {avg_loss:.4f}  |  NP IoU: {np_iou:.4f}  |  FG mIoU: {fg_miou:.4f}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"    {name:>14}: IoU={per_class_iou[i]:.4f}")


# ==============================================================================
#  Per-epoch PQ validation + best-model selection
# ==============================================================================

@torch.no_grad()
def validate_pq(model, loader, criterion, device, args):
    """Validate with full PQ metrics. Returns (val_loss, np_iou, fg_miou,
    per_class_iou, mPQ_Tiss, bPQ_Tiss)."""
    from utils.evaluate import aggregate_metrics, load_tissue_types, evaluate_parallel

    model.eval()
    total_loss = 0.0
    np_inter_sum = 0
    np_union_sum = 0
    fg_inters = np.zeros(5, dtype=np.int64)
    fg_unions = np.zeros(5, dtype=np.int64)
    tissue = load_tissue_types(args.data_root, args.val_fold)

    pq_inputs = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        batch_dev = {
            "mask": batch["mask"].to(device, non_blocking=True),
            "np_gt": batch["np_gt"].to(device, non_blocking=True),
            "hv_gt": batch["hv_gt"].to(device, non_blocking=True),
        }

        outputs = model(images)
        loss, _ = criterion(outputs, batch_dev)
        total_loss += loss.item()

        # --- Fast IoU ---
        np_prob = torch.sigmoid(outputs["np"])
        np_pred = (np_prob > 0.5).squeeze(1)
        np_gt_bin = batch_dev["mask"] < 5
        np_inter_sum += (np_pred & np_gt_bin).sum().item()
        np_union_sum += (np_pred | np_gt_bin).sum().item()

        nc_pred = outputs["nc"].argmax(dim=1)
        fg_mask = batch_dev["mask"] < 5
        for c in range(5):
            pc = (nc_pred == c) & fg_mask
            tc = batch_dev["mask"] == c
            fg_inters[c] += (pc & tc).sum().item()
            fg_unions[c] += (pc | tc).sum().item()

        np_prob_np = np_prob.cpu().numpy()
        hv_np = outputs["hv"].cpu().numpy()
        nc_logits_np = outputs["nc"].cpu().numpy()
        masks = batch["mask"].cpu().numpy()
        inst_gts = batch["inst_gt"].cpu().numpy()

        for b in range(images.shape[0]):
            pq_inputs.append((
                np_prob_np[b, 0], hv_np[b], nc_logits_np[b],
                masks[b], inst_gts[b],
                args.eval_np_thresh, args.eval_min_area,
                args.eval_match_iou,
            ))

    # Parallel post-processing + PQ evaluation
    pq_results = evaluate_parallel(pq_inputs)

    n = len(loader)
    avg_loss = total_loss / n
    avg_np_iou = np_inter_sum / (np_union_sum + 1e-8)
    per_class_iou = [float(fg_inters[c]) / (fg_unions[c] + 1e-8) for c in range(5)]
    fg_miou = np.mean(per_class_iou)
    pq_metrics = aggregate_metrics(pq_results, tissue_types=tissue)
    return (avg_loss, avg_np_iou, fg_miou, per_class_iou,
            pq_metrics["mPQ_Tiss"], pq_metrics["bPQ_Tiss"])


# ==============================================================================
#  Training epoch
# ==============================================================================

def train_one_epoch(model, loader, criterion, optimizer, scaler, epoch, args, device):
    model.train()
    total_loss = 0.0
    total_np = 0.0
    total_hv = 0.0
    total_nc = 0.0
    total_gate = 0.0

    for batch_idx, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        batch_dev = {
            "mask": batch["mask"].to(device, non_blocking=True),
            "np_gt": batch["np_gt"].to(device, non_blocking=True),
            "hv_gt": batch["hv_gt"].to(device, non_blocking=True),
        }

        optimizer.zero_grad()

        if scaler is not None:
            with autocast(device_type=device.type):
                outputs = model(images)
                loss, loss_dict = criterion(outputs, batch_dev)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss, loss_dict = criterion(outputs, batch_dev)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        total_np += loss_dict["np_total"]
        total_hv += loss_dict["hv_total"]
        total_nc += loss_dict["nc_total"]
        if batch_idx % args.log_interval == 0:
            print(f"  Epoch {epoch:3d} | Batch {batch_idx:4d}/{len(loader)} | "
                  f"Loss: {loss.item():.4f} | NP: {loss_dict['np_total']:.4f} | "
                  f"HV: {loss_dict['hv_total']:.4f} | NC: {loss_dict['nc_total']:.4f}")

    n = len(loader)
    return total_loss / n, total_np / n, total_hv / n, total_nc / n, total_gate / n


# ==============================================================================
#  Checkpoint
# ==============================================================================

def save_checkpoint(model, optimizer, scaler, scheduler, history, epoch, best_metric, path):
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler else None,
        "scheduler": scheduler.state_dict(),
        "best_metric": best_metric,
        "history": history.records,
    }, path)


# ==============================================================================
#  Main
# ==============================================================================

def main():
    args = parse_args()
    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Seed: {args.seed}")
    arch_label = {"shared_unet": "SharedUNet", "shared_unet_mala": "SharedUNet+MALA",
                  "unet3": "Unet3", "unet3_mala": "Unet3+MALA"}.get(args.decoder, args.decoder)
    enc_label = "UNI2-h" if args.encoder == "uni2-h" else f"ConvNeXt-{args.version.upper()}-{args.encoder}"
    size_label = " + SizePrior" if args.use_size_prior else ""
    print(f"Model: {enc_label} + {arch_label}{size_label}")

    run_name = args.run_name or f"convnext_{args.version}_{args.encoder}_gate_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = Path(args.output_dir) / run_name
    for d in [run_dir, run_dir / "curves"]:
        d.mkdir(parents=True, exist_ok=True)

    history = HistoryTracker()
    print(f"\nRun dir: {run_dir}/")

    # ---- Data ----
    val_fold = args.val_fold
    train_folds = tuple(f for f in [1, 2, 3] if f != val_fold)

    # If extend dataset, load original folds + extend (fold 0)
    if args.extend_data:
        # Extend: flat directory (fold=0), training only — no validation split
        ext_train_ds = PanNukeDataset(
            data_root=args.extend_data, folds=(0,),
            augment=not args.no_aug, heavy_aug=args.heavy_aug,
        )
        orig_loader = get_pannuke_loaders(
            data_root=args.data_root, train_folds=train_folds, val_folds=(val_fold,),
            batch_size=args.batch_size, num_workers=args.num_workers,
            augment=not args.no_aug, heavy_aug=args.heavy_aug,
            balance_sample=args.balance_sample, seed=args.seed,
        )
        from torch.utils.data import ConcatDataset
        train_dataset = ConcatDataset([ext_train_ds, orig_loader[0].dataset])
        from torch.utils.data import DataLoader
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers,
                                  pin_memory=True, drop_last=True)
        val_loader = orig_loader[1]
        print(f"Folds: train=extend(0)+{train_folds}, val=({val_fold},)")
        print(f"  Extend: {len(ext_train_ds)} + Original: {len(orig_loader[0].dataset)} = {len(train_dataset)} total")
    else:
        print(f"Folds: train={train_folds}, val=({val_fold},)")
        train_loader, val_loader = get_pannuke_loaders(
            data_root=args.data_root, train_folds=train_folds, val_folds=(val_fold,),
            batch_size=args.batch_size, num_workers=args.num_workers,
            augment=not args.no_aug, heavy_aug=args.heavy_aug,
            balance_sample=args.balance_sample, seed=args.seed,
        )

    # ---- Model ----
    freeze_enc = args.freeze_encoder or (args.encoder == "uni2-h")
    if args.encoder == "uni2-h":
        print(f"\nBuilding UNI2-h (frozen) + {arch_label}{size_label}...")
    else:
        print(f"\nBuilding {enc_label} + {arch_label}{size_label}...")
    model = create_model(
        variant=args.encoder, version=args.version,
        num_nc_classes=NUM_NC_CLASSES, pretrained=not args.no_pretrained,
        decoder_type=args.decoder, aspp_out=args.aspp_out,
        enc_dropout=args.enc_dropout, dec_dropout=args.dec_dropout,
        freeze_encoder=freeze_enc,
        full_unfreeze=args.full_unfreeze,
        uni2_weights=args.uni2_weights,
        nc_no_cbam=args.no_cbam,
        upsample_mode=args.upsample_mode,
    )
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Total params: {n_params:.2f}M")

    # ---- Loss ----
    criterion = CombinedLoss(
        num_nc_classes=NUM_NC_CLASSES,
        np_loss=args.np_loss,
        np_weight=args.np_weight,
        asym_gamma_neg=args.asym_gamma_neg,
        asym_gamma_pos=args.asym_gamma_pos,
        np_ohem=args.np_ohem,
        np_cl_dice_weight=args.np_cldice_weight,
        np_bce_weight=args.np_bce_weight,
        np_dice_weight=args.np_dice_weight,
        np_ft_alpha=args.np_ft_alpha,
        np_ft_beta=args.np_ft_beta,
        np_ft_gamma=args.np_ft_gamma,
        hv_mse_weight=args.hv_mse_weight,
        hv_msge_weight=args.hv_msge_weight,
        hv_loss_weight=args.hv_loss_weight,
        nc_loss=args.nc_loss,
        nc_focal_alpha=args.nc_focal_alpha,
        nc_focal_gamma=args.nc_focal_gamma,
        nc_dice_weight=args.nc_dice_weight,
        nc_ft_alpha=args.nc_ft_alpha,
        nc_ft_beta=args.nc_ft_beta,
        nc_ft_gamma=args.nc_ft_gamma,
        nc_ft_weight=args.nc_ft_weight,
        nc_bce_weight=args.nc_bce_weight,
        nc_weight=args.nc_weight,
        cb_gamma=args.cb_gamma,
        cb_ema_alpha=args.cb_ema_alpha,
        use_size_prior=args.use_size_prior,
    )

    # ---- Resume or fresh start ----
    # --epochs N = train for N epochs (from scratch) or N more epochs (from resume)
    start_epoch, best_mPQ = 0, 0.0
    no_improve = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt.get("epoch", -1) + 1
        print(f"Resume from epoch {ckpt.get('epoch', '?')}, train {args.epochs} more → "
              f"epoch {start_epoch}-{start_epoch + args.epochs - 1}")
        if "history" in ckpt:
            for k, v in ckpt["history"].items():
                history.records[k] = v
            if "val/loss" in history.records:
                history.epochs = list(range(0, len(history.records["val/loss"]) * args.val_interval, args.val_interval))
        best_mPQ = ckpt.get("best_metric", 0.0)
        print(f"  best_mPQ from ckpt: {best_mPQ:.4f}")

    if args.resume:
        # Keep the checkpoint's freeze state — don't re-freeze
        if start_epoch > args.freeze_epochs:
            model.unfreeze_encoder()
            print("  (resume from Phase 2, ViT unfrozen)")
        # else: stays as loaded (frozen or unfrozen depending on ckpt)
    elif start_epoch > args.freeze_epochs:
        model.unfreeze_encoder()
    else:
        model.freeze_encoder()

    # ---- Optimizer ----
    param_groups = model.get_param_groups(lr=args.lr, backbone_lr_mult=args.backbone_lr_mult)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    # ---- Mixed precision ----
    scaler = GradScaler(device=device.type) if args.fp16 and device.type == "cuda" else None

    # Load optimizer/scaler states BEFORE creating scheduler
    if args.resume:
        if ckpt.get("optimizer"):
            optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scaler") and scaler:
            scaler.load_state_dict(ckpt["scaler"])

    # ---- LR scheduler: cosine → plateau ----
    # First 'cosine_epochs' epochs: cosine decay from args.lr to 1e-6.
    # Remaining epochs: LR stays at 1e-6 (plateau).
    cosine_epochs = min(args.epochs, 200)
    scheduler = CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=1e-6)

    # ---- Config ----
    with open(run_dir / "config.json", "w") as f:
        json.dump({
            "encoder": args.encoder, "version": args.version,
            "architecture": args.decoder,
            "full_unfreeze": args.full_unfreeze,
            "train_folds": list(train_folds), "val_fold": val_fold,
            "batch_size": args.batch_size, "epochs": args.epochs,
            "lr": args.lr, "weight_decay": args.weight_decay,
            "enc_dropout": args.enc_dropout,
            "dec_dropout": args.dec_dropout,
            "loss_weights": {
                "np_weight": args.np_weight,
                "nc_weight": args.nc_weight,
                "hv_loss_weight": args.hv_loss_weight,
            },
            "loss": {
                "np": "BinaryDiceLoss" if args.np_loss == "dice" else "AsymmetricLoss",
                "np_ohem": args.np_ohem,
                "np_cldice_weight": args.np_cldice_weight,
                "hv": f"MSE({args.hv_mse_weight})+MSGE({args.hv_msge_weight})",
                "hv_loss_weight": args.hv_loss_weight,
                "nc": _nc_loss_label(args),
                "nc_weight": args.nc_weight,
                "size_prior": args.use_size_prior,
            },
            "total_params_M": round(n_params, 2),
        }, f, indent=2)

    # ==========================================================================
    #  Training loop
    # ==========================================================================
    encoder_unfrozen = False  # track for Phase 1 → 2 transition

    end_epoch = start_epoch + args.epochs

    print(f"\n{'='*60}")
    print(f"Training epoch {start_epoch}-{end_epoch-1} ({args.epochs} epochs) | "
          f"Phase 1: frozen enc (0-{args.freeze_epochs-1}) →"
          f" Phase 2: fine-tune ({args.freeze_epochs}+)")
    print(f"LR: {args.lr} → {1e-6} (cosine, no warmup)")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, end_epoch):
        # ---- Phase 1 → 2: unfreeze encoder ----
        if epoch == args.freeze_epochs and not encoder_unfrozen:
            print(f"\n>>> Epoch {epoch}: Unfreezing encoder for Phase 2 fine-tuning <<<\n")
            model.unfreeze_encoder()
            encoder_unfrozen = True

        avg_loss, avg_np, avg_hv, avg_nc, avg_gate = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, epoch, args, device,
        )
        history.log_train(epoch, avg_loss, avg_np, avg_hv, avg_nc, avg_gate,
                         optimizer.param_groups[0]["lr"], 0)

        if epoch % args.val_interval == 0 or epoch == end_epoch - 1:
            val_loss, np_iou, fg_miou, per_class_iou, mPQ_Tiss, bPQ_Tiss = validate_pq(
                model, val_loader, criterion, device, args,
            )
            history.log_val(epoch, val_loss, np_iou, fg_miou, per_class_iou)
            history.records.setdefault("val/mPQ_Tiss", []).append(mPQ_Tiss)
            history.records.setdefault("val/bPQ_Tiss", []).append(bPQ_Tiss)
            print_val_summary(epoch, val_loss, np_iou, fg_miou, per_class_iou)
            print(f"  mPQ_Tiss: {mPQ_Tiss:.4f}  |  bPQ_Tiss: {bPQ_Tiss:.4f}")

            if mPQ_Tiss > best_mPQ:
                best_mPQ = mPQ_Tiss
                no_improve = 0
                save_checkpoint(model, optimizer, scaler, scheduler, history, epoch,
                               float(best_mPQ), str(run_dir / "best.pth"))
                print(f"  => New best (mPQ_Tiss: {best_mPQ:.4f})")
            else:
                no_improve += 1

            save_checkpoint(model, optimizer, scaler, scheduler, history, epoch,
                           float(best_mPQ), str(run_dir / "last.pth"))
            history.save_json(str(run_dir / "history.json"))
            plot_curves(history, run_dir, epoch)

            if no_improve >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

        scheduler.step()
        # After cosine_epochs, clamp LR to eta_min=1e-6 (plateau phase)
        if epoch >= cosine_epochs:
            for pg in optimizer.param_groups:
                pg['lr'] = 1e-6

    # ==========================================================================
    #  Final PQ evaluation
    # ==========================================================================
    print(f"\n{'='*60}")
    print(f"Training done. Best mPQ_Tiss: {best_mPQ:.4f}")

    best_path = str(run_dir / "best.pth")
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        model.eval()
        print(f"Loaded best model (epoch {ckpt.get('epoch', '?')}), running PQ eval...")

    from utils.evaluate import save_metrics, evaluate_final
    pq_metrics = evaluate_final(model, val_loader, device,
                                data_root=args.data_root, val_fold=val_fold,
                                eval_np_thresh=args.eval_np_thresh,
                                eval_min_area=args.eval_min_area,
                                eval_match_iou=args.eval_match_iou)
    save_metrics(pq_metrics, str(run_dir), f"fold{val_fold}")

    print(f"\nOutput: {run_dir}/")


if __name__ == "__main__":
    main()
