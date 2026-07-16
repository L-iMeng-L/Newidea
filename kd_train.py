#!/usr/bin/env python3
"""
Knowledge distillation training: UNI2-h (teacher) → ConvNeXt (student).

Teacher:  UNI2-h + unet3  (~700M) — frozen, eval mode
Student:  ConvNeXt-Tiny + shared_unet  (~35M) — trained end-to-end

Loss = L_supervised(student_gt) + α * L_kd(student, teacher) + λ_enc * L_enc(student, teacher)

Usage:
    python kd_train.py \
        --teacher_ckpt output/15run/hv2_hvcvt/best.pth \
        --student_encoder tiny \
        --epochs 300 --batch_size 32 \
        --device cuda:0 --output_dir output/kd_run
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import create_model
from data import get_pannuke_loaders
from losses.losses import CombinedLoss
from losses.kd_losses import DistillationLoss

NUM_NC_CLASSES = 5


# ==============================================================================
#  Command line
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="KD: UNI2-h → ConvNeXt")

    # ---- Teacher ----
    p.add_argument("--teacher_ckpt", type=str, required=True,
                   help="Path to teacher checkpoint")
    p.add_argument("--teacher_decoder", type=str, default="shared_unet",
                   choices=["shared_unet", "unet3", "shared_unet_mala"])

    # ---- Student ----
    p.add_argument("--student_encoder", type=str, default="tiny",
                   choices=["tiny", "small", "base", "large"],
                   help="ConvNeXt variant for student")
    p.add_argument("--student_version", type=str, default="v2",
                   choices=["v1", "v2"])
    p.add_argument("--student_decoder", type=str, default="shared_unet",
                   choices=["shared_unet"])

    # ---- KD hyperparams ----
    p.add_argument("--kd_temperature", type=float, default=4.0)
    p.add_argument("--kd_alpha", type=float, default=0.9,
                   help="KD loss weight (0=supervised only, 1=KD only)")
    p.add_argument("--kd_enc_weight", type=float, default=0.1,
                   help="Encoder feature alignment weight")
    p.add_argument("--kd_np_weight", type=float, default=1.0)
    p.add_argument("--kd_nc_weight", type=float, default=1.0)
    p.add_argument("--kd_hv_weight", type=float, default=1.0)

    # ---- Training ----
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=5e-3)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--num_workers", type=int, default=12)
    p.add_argument("--fp16", action="store_true", default=True)

    # ---- Data ----
    p.add_argument("--data_root", type=str, default="/home/lwy/dataset/PanNuke/processed")
    p.add_argument("--val_fold", type=int, default=3, choices=[1, 2, 3])
    p.add_argument("--extend_data", type=str, default=None)

    # ---- Loss (supervised part for student) ----
    p.add_argument("--np_loss", type=str, default="ft+dice")
    p.add_argument("--np_weight", type=float, default=2.0)
    p.add_argument("--np_ft_alpha", type=float, default=0.5)
    p.add_argument("--np_ft_beta", type=float, default=0.5)
    p.add_argument("--np_ft_gamma", type=float, default=1.333)
    p.add_argument("--np_dice_weight", type=float, default=1.0)
    p.add_argument("--np_bce_weight", type=float, default=1.0)
    p.add_argument("--np_ohem", type=float, default=0.5)
    p.add_argument("--np_cldice_weight", type=float, default=0.0)
    p.add_argument("--asym_gamma_neg", type=float, default=4.0)
    p.add_argument("--asym_gamma_pos", type=float, default=0.5)

    p.add_argument("--hv_mse_weight", type=float, default=1.0)
    p.add_argument("--hv_msge_weight", type=float, default=0.5)
    p.add_argument("--hv_loss_weight", type=float, default=1.0)

    p.add_argument("--nc_loss", type=str, default="focal+dice")
    p.add_argument("--nc_weight", type=float, default=2.0)
    p.add_argument("--nc_focal_alpha", type=float, default=1.0)
    p.add_argument("--nc_focal_gamma", type=float, default=2.0)
    p.add_argument("--nc_dice_weight", type=float, default=1.0)

    p.add_argument("--cb_gamma", type=float, default=0.0)
    p.add_argument("--cb_ema_alpha", type=float, default=0.99)
    p.add_argument("--use_size_prior", action="store_true", default=False)

    # ---- Misc ----
    p.add_argument("--output_dir", type=str, default="./output/kd")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--val_interval", type=int, default=1)
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--patience", type=int, default=50)
    # Eval
    p.add_argument("--eval_np_thresh", type=float, default=0.5)
    p.add_argument("--eval_min_area", type=int, default=10)
    p.add_argument("--eval_match_iou", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=114514)
    p.add_argument("--enc_dropout", type=float, default=0.5)
    p.add_argument("--dec_dropout", type=float, default=0.2)
    p.add_argument("--aspp_out", type=int, default=256)
    p.add_argument("--no_aug", action="store_true", default=False)
    p.add_argument("--heavy_aug", action="store_true", default=True)
    p.add_argument("--balance_sample", action="store_true", default=False)

    return p.parse_args()


# ==============================================================================
#  Training loop
# ==============================================================================

def train_one_epoch(model, teacher_model, loader, criterion, kd_criterion,
                    optimizer, scaler, epoch, args, device):
    model.train()
    teacher_model.eval()

    total_loss = 0.0
    kd_losses = {"enc_feat": 0.0, "kd_np": 0.0, "kd_nc": 0.0, "kd_hv": 0.0, "supervised": 0.0}

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
                # Student forward with encoder features
                s_outputs, s_enc = model(images, return_enc_features=True)

                # Teacher forward (no grad)
                with torch.no_grad():
                    t_outputs, t_enc = teacher_model(images, return_enc_features=True)

                # Supervised loss
                sup_loss, sup_dict = criterion(s_outputs, batch_dev)

                # KD loss
                kd_loss, kd_dict = kd_criterion(
                    s_outputs, t_outputs, s_enc, t_enc, sup_loss,
                )

            scaler.scale(kd_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            s_outputs, s_enc = model(images, return_enc_features=True)

            with torch.no_grad():
                t_outputs, t_enc = teacher_model(images, return_enc_features=True)

            sup_loss, sup_dict = criterion(s_outputs, batch_dev)
            kd_loss, kd_dict = kd_criterion(
                s_outputs, t_outputs, s_enc, t_enc, sup_loss,
            )

            kd_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += kd_loss.item()
        for k in kd_losses:
            kd_losses[k] += kd_dict.get(k, 0.0)

        if batch_idx % args.log_interval == 0:
            print(f"  Epoch {epoch:3d} | Batch {batch_idx:4d}/{len(loader)} | "
                  f"KD: {kd_loss.item():.4f} | Sup: {sup_loss.item():.4f} | "
                  f"NP: {kd_dict['kd_np']:.4f} | NC: {kd_dict['kd_nc']:.4f} | HV: {kd_dict['kd_hv']:.4f}")

    n = len(loader)
    return (total_loss / n,
            kd_losses["supervised"] / n,
            kd_losses["enc_feat"] / n,
            kd_losses["kd_np"] / n,
            kd_losses["kd_hv"] / n,
            kd_losses["kd_nc"] / n)


# ==============================================================================
#  Validation (same as train.py — supervised only, no KD overhead)
# ==============================================================================

@torch.no_grad()
def validate_pq(model, loader, criterion, device, args):
    from utils.evaluate import aggregate_metrics, load_tissue_types, evaluate_parallel

    model.eval()
    total_loss = 0.0
    pq_inputs = []
    tissue = load_tissue_types(args.data_root, args.val_fold)

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

        np_probs = torch.sigmoid(outputs["np"]).cpu().numpy()
        hv_maps = outputs["hv"].cpu().numpy()
        nc_logits = outputs["nc"].cpu().numpy()
        masks = batch["mask"].cpu().numpy()
        inst_gts = batch["inst_gt"].cpu().numpy()

        for b in range(images.shape[0]):
            pq_inputs.append((
                np_probs[b, 0], hv_maps[b], nc_logits[b],
                masks[b], inst_gts[b],
                args.eval_np_thresh, args.eval_min_area,
                args.eval_match_iou,
            ))

    pq_results = evaluate_parallel(pq_inputs)
    agg = aggregate_metrics(pq_results, tissue_types=tissue)
    return total_loss / len(loader), agg["mPQ_Tiss"], agg["bPQ_Tiss"]


# ==============================================================================
#  Checkpointing
# ==============================================================================

def save_checkpoint(model, optimizer, scaler, scheduler, epoch, best_metric, path):
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler else None,
        "scheduler": scheduler.state_dict(),
        "best_metric": best_metric,
    }, path)


# ==============================================================================
#  Main
# ==============================================================================

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ---- Output directory ----
    run_name = args.run_name or f"kd_{args.student_encoder}_v2"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}/")

    # ---- Data ----
    train_folds = tuple(f for f in (1, 2, 3) if f != args.val_fold)
    val_fold = args.val_fold

    if args.extend_data:
        from torch.utils.data import ConcatDataset, DataLoader
        from data.pannuke import PanNukeDataset
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
        train_dataset = ConcatDataset([ext_train_ds, orig_loader[0].dataset])
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

    # ---- Teacher (UNI2-h + unet3, frozen) ----
    print(f"\n[Teacher] UNI2-h + {args.teacher_decoder}")
    teacher = create_model(
        variant="uni2-h", num_nc_classes=NUM_NC_CLASSES, pretrained=False,
        decoder_type=args.teacher_decoder,
        enc_dropout=args.enc_dropout, dec_dropout=args.dec_dropout,
        aspp_out=args.aspp_out,
        freeze_encoder=True, full_unfreeze=True,
    )
    t_ckpt = torch.load(args.teacher_ckpt, map_location=device, weights_only=False)
    teacher.load_state_dict(t_ckpt["model"])
    teacher = teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    t_params = sum(p.numel() for p in teacher.parameters())
    print(f"  Teacher: {t_params/1e6:.1f}M params (frozen)")

    # ---- Student (ConvNeXt + shared_unet, trainable) ----
    print(f"\n[Student] ConvNeXt-{args.student_encoder} + {args.student_decoder}")
    student = create_model(
        variant=args.student_encoder, version=args.student_version,
        num_nc_classes=NUM_NC_CLASSES, pretrained=True,
        decoder_type=args.student_decoder,
        enc_dropout=args.enc_dropout, dec_dropout=args.dec_dropout,
        aspp_out=args.aspp_out,
        freeze_encoder=False, full_unfreeze=False,
    )
    student = student.to(device)
    s_params = sum(p.numel() for p in student.parameters())
    print(f"  Student: {s_params/1e6:.1f}M params (trainable)")
    print(f"  Compression: {t_params/s_params:.1f}×")

    # ---- Loss ----
    criterion = CombinedLoss(
        num_nc_classes=NUM_NC_CLASSES,
        np_loss=args.np_loss, np_weight=args.np_weight,
        asym_gamma_neg=args.asym_gamma_neg,
        asym_gamma_pos=args.asym_gamma_pos,
        np_ohem=args.np_ohem, np_cl_dice_weight=args.np_cldice_weight,
        np_bce_weight=args.np_bce_weight,
        np_dice_weight=args.np_dice_weight,
        np_ft_alpha=args.np_ft_alpha,
        np_ft_beta=args.np_ft_beta,
        np_ft_gamma=args.np_ft_gamma,
        hv_mse_weight=args.hv_mse_weight,
        hv_msge_weight=args.hv_msge_weight,
        hv_loss_weight=args.hv_loss_weight,
        nc_loss=args.nc_loss, nc_focal_alpha=args.nc_focal_alpha,
        nc_focal_gamma=args.nc_focal_gamma,
        nc_dice_weight=args.nc_dice_weight,
        nc_weight=args.nc_weight,
        cb_gamma=args.cb_gamma, cb_ema_alpha=args.cb_ema_alpha,
        use_size_prior=args.use_size_prior,
    )

    kd_criterion = DistillationLoss(
        temperature=args.kd_temperature,
        alpha=args.kd_alpha,
        enc_weight=args.kd_enc_weight,
        np_weight=args.kd_np_weight,
        nc_weight=args.kd_nc_weight,
        hv_weight=args.kd_hv_weight,
    )

    # ---- Resume ----
    start_epoch, best_mPQ = 0, 0.0
    no_improve = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        student.load_state_dict(ckpt["model"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_mPQ = ckpt.get("best_metric", 0.0)
        print(f"Resume at epoch {start_epoch}, best mPQ: {best_mPQ:.4f}")

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler(device=device.type) if args.fp16 and device.type == "cuda" else None

    if args.resume:
        if ckpt.get("optimizer"):
            optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scaler") and scaler:
            scaler.load_state_dict(ckpt["scaler"])
        if ckpt.get("scheduler"):
            scheduler.load_state_dict(ckpt["scheduler"])

    # ---- Config ----
    with open(run_dir / "config.json", "w") as f:
        json.dump({
            "teacher_ckpt": args.teacher_ckpt,
            "teacher_decoder": args.teacher_decoder,
            "student_encoder": args.student_encoder,
            "student_decoder": args.student_decoder,
            "student_params_m": s_params / 1e6,
            "compression_ratio": t_params / s_params,
            "kd_temperature": args.kd_temperature,
            "kd_alpha": args.kd_alpha,
            "kd_enc_weight": args.kd_enc_weight,
        }, f, indent=2)

    # ---- Training loop ----
    print(f"\n{'='*60}")
    print(f"KD training: {args.epochs} epochs, LR={args.lr}→1e-6 (cosine)")
    print(f"Temperature={args.kd_temperature}, α={args.kd_alpha}, enc_wt={args.kd_enc_weight}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs):
        avg_loss, avg_sup, avg_enc, avg_np_kd, avg_hv_kd, avg_nc_kd = train_one_epoch(
            student, teacher, train_loader, criterion, kd_criterion,
            optimizer, scaler, epoch, args, device,
        )
        print(f"Epoch {epoch:3d} | KD loss: {avg_loss:.4f} | "
              f"Sup: {avg_sup:.4f} | Enc: {avg_enc:.4f} | "
              f"KD_NP: {avg_np_kd:.4f} KD_HV: {avg_hv_kd:.4f} KD_NC: {avg_nc_kd:.4f}")

        if epoch % args.val_interval == 0 or epoch == args.epochs - 1:
            val_loss, mPQ_Tiss, bPQ_Tiss = validate_pq(
                student, val_loader, criterion, device, args,
            )
            print(f"  Val Loss: {val_loss:.4f} | mPQ_Tiss: {mPQ_Tiss:.4f} | bPQ_Tiss: {bPQ_Tiss:.4f}")

            if mPQ_Tiss > best_mPQ:
                best_mPQ = mPQ_Tiss
                no_improve = 0
                save_checkpoint(student, optimizer, scaler, scheduler, epoch,
                               float(best_mPQ), str(run_dir / "best.pth"))
                print(f"  => New best (mPQ_Tiss: {best_mPQ:.4f})")
            else:
                no_improve += 1

            save_checkpoint(student, optimizer, scaler, scheduler, epoch,
                           float(best_mPQ), str(run_dir / "last.pth"))

            if no_improve >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

        scheduler.step()

    # ---- Final eval ----
    print(f"\n{'='*60}")
    print(f"KD training done. Best mPQ_Tiss: {best_mPQ:.4f}")

    best_path = str(run_dir / "best.pth")
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        student.load_state_dict(ckpt["model"])
        student.eval()
        print(f"Loaded best model (epoch {ckpt.get('epoch', '?')}), running PQ eval...")

    from utils.evaluate import save_metrics, evaluate_final
    pq_metrics = evaluate_final(student, val_loader, device,
                                data_root=args.data_root, val_fold=val_fold,
                                eval_np_thresh=args.eval_np_thresh,
                                eval_min_area=args.eval_min_area,
                                eval_match_iou=args.eval_match_iou)
    save_metrics(pq_metrics, str(run_dir), f"fold{val_fold}")
    print(f"\nOutput: {run_dir}/")


if __name__ == "__main__":
    main()
