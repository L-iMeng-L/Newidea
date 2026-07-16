#!/usr/bin/env python3
"""
MoNuSeg 2018 external validation — zero-shot, detection-only.

Parses Aperio XML annotations → instance map → evaluate binary PQ + DICE + AJI.
No classification (MoNuSeg has no class labels).

Usage:
    python eval_monuseg.py --checkpoint best.pth --encoder uni2-h --device cuda:0
"""

import argparse, os, sys, time, xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import create_model
from eval_consep import (
    predict_image, compute_aji, IMAGENET_MEAN, IMAGENET_STD, PATCH_SIZE,
)
from utils.postprocess import postprocess_nuclei
from utils.evaluate import remap_label, get_fast_pq


def parse_xml_annotations(xml_path):
    """Parse Aperio ImageScope XML → per-nucleus polygon vertices."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    nuclei = []
    for annotation in root.findall('.//Annotation'):
        for region in annotation.findall('.//Region'):
            vertices = []
            for v in region.findall('.//Vertex'):
                vertices.append((float(v.get('X')), float(v.get('Y'))))
            if vertices:
                nuclei.append(vertices)
    return nuclei


def xml_to_inst_map(nuclei, h=1000, w=1000):
    """Convert polygon vertex lists → instance map [H,W] int32."""
    inst_map = np.zeros((h, w), dtype=np.int32)
    for i, poly in enumerate(nuclei, 1):
        pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(inst_map, [pts], i)
    return inst_map


def compute_dice(gt, pred):
    inter = (gt & pred).sum()
    total = gt.sum() + pred.sum()
    return 2 * inter / total if total > 0 else 0.0


@torch.no_grad()
def predict_monuseg(model, image_np, device, pp_kwargs, n_tta=0):
    if n_tta <= 1:
        np_prob, hv_map, nc_logits = predict_image(model, image_np, device)
    else:
        np_sum, hv_sum, nc_sum = None, None, None
        for r in range(n_tta):
            rot = np.rot90(image_np, k=r, axes=(-2, -1)) if r else image_np
            np_p, hv_p, nc_p = predict_image(model, rot, device)
            if r:
                np_p = np.rot90(np_p, k=4-r, axes=(-2, -1))
                # Rotate HV vector field back
                hv_h = hv_p[0].copy(); hv_v = hv_p[1].copy()
                for _ in range(r): hv_h, hv_v = -hv_v, hv_h
                hv_p = np.stack([hv_h, hv_v])
                nc_p = np.rot90(nc_p, k=4-r, axes=(-2, -1))
            if np_sum is None: np_sum, hv_sum, nc_sum = np_p, hv_p, nc_p
            else: np_sum += np_p; hv_sum += hv_p; nc_sum += nc_p
        np_prob, hv_map, nc_logits = np_sum/n_tta, hv_sum/n_tta, nc_sum/n_tta

    inst_map, _, _ = postprocess_nuclei(
        np_prob, hv_map, np.argmax(nc_logits, axis=0), **pp_kwargs)
    return inst_map


def parse_args():
    p = argparse.ArgumentParser(description="Zero-shot MoNuSeg evaluation")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--encoder", type=str, default="uni2-h",
                   choices=["uni2-h", "tiny", "small", "base", "large"])
    p.add_argument("--decoder", type=str, default="shared_unet")
    p.add_argument("--data_root", type=str, default="/home/lwy/dataset/MoNuSegTestData")
    p.add_argument("--output", type=str, default="./monuseg_results")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--np_thresh", type=float, default=0.50)
    p.add_argument("--min_area", type=int, default=10)
    p.add_argument("--energy_thresh", type=float, default=0.30)
    p.add_argument("--sobel_ksize", type=int, default=21)
    p.add_argument("--marker_ksize", type=int, default=3)
    p.add_argument("--match_iou", type=float, default=0.5)
    p.add_argument("--stride", type=int, default=128)
    p.add_argument("--margin", type=int, default=64)
    p.add_argument("--tta", type=int, default=0,
                   help="TTA views: 4=4 rotations averaged")
    return p.parse_args()


def main():
    # Set module-level stride/margin before importing eval_consep functions
    import eval_consep
    args = parse_args()
    eval_consep._STRIDE = args.stride
    eval_consep._MARGIN = args.margin
    os.makedirs(args.output, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    is_uni2 = (args.encoder == "uni2-h")
    model = create_model(
        variant=args.encoder, num_nc_classes=5,
        pretrained=(not is_uni2),
        decoder_type=args.decoder,
        enc_dropout=0.5, dec_dropout=0.2,
        freeze_encoder=is_uni2, full_unfreeze=is_uni2,
    )
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  {args.encoder} + {args.decoder}  |  {n_params:.1f}M  |  epoch {ckpt.get('epoch', '?')}")

    # Find images
    data_dir = Path(args.data_root)
    image_files = sorted(data_dir.glob("*.tif"))
    print(f"\nFound {len(image_files)} images", flush=True)

    pp_kwargs = dict(
        np_thresh=args.np_thresh, min_area=args.min_area,
        energy_thresh=args.energy_thresh, sobel_ksize=args.sobel_ksize,
        marker_ksize=args.marker_ksize,
    )

    all_metrics = []
    t0 = time.time()

    for idx, img_path in enumerate(image_files):
        img_id = img_path.stem
        xml_path = data_dir / f"{img_id}.xml"

        if not xml_path.exists():
            print(f"  [{idx+1}/{len(image_files)}] {img_id} — no XML, skipping", flush=True)
            continue

        # GT from XML
        nuclei = parse_xml_annotations(xml_path)
        gt_inst = xml_to_inst_map(nuclei)
        print(f"  [{idx+1}/{len(image_files)}] {img_id}: GT={gt_inst.max()} nuclei, ", end="", flush=True)

        # Load + infer
        image = np.array(Image.open(img_path).convert("RGB")).astype(np.float32)
        image /= np.float32(255.0)
        image = (image - IMAGENET_MEAN.reshape(1,1,3)) / IMAGENET_STD.reshape(1,1,3)
        image_np = image.transpose(2, 0, 1).copy()

        inst_map = predict_monuseg(model, image_np, device, pp_kwargs, n_tta=args.tta)
        print(f"Pred={inst_map.max()}, ", end="", flush=True)

        # Binary evaluation
        gt_bin = gt_inst > 0
        pred_bin = inst_map > 0
        dice = compute_dice(gt_bin, pred_bin)
        aji = compute_aji(gt_inst, inst_map)

        if gt_inst.max() > 0 and inst_map.max() > 0:
            [dq, sq, pq], _ = get_fast_pq(
                remap_label(gt_inst), remap_label(inst_map), args.match_iou)
        else:
            dq, sq, pq = np.nan, np.nan, np.nan

        print(f"bPQ={pq:.3f} DICE={dice:.3f} AJI={aji:.3f}", flush=True)

        all_metrics.append({
            "image": img_id, "bPQ": pq, "DQ": dq, "SQ": sq,
            "DICE": dice, "AJI": aji, "n_gt": int(gt_inst.max()), "n_pred": int(inst_map.max()),
        })

    elapsed = time.time() - t0
    n = max(len(all_metrics), 1)
    print(f"\nDone in {elapsed:.0f}s ({elapsed/n:.1f}s/image)")

    # Aggregate
    bPQ = np.nanmean([m["bPQ"] for m in all_metrics])
    dq = np.nanmean([m["DQ"] for m in all_metrics])
    sq = np.nanmean([m["SQ"] for m in all_metrics])
    dice = np.mean([m["DICE"] for m in all_metrics])
    aji  = np.mean([m["AJI"] for m in all_metrics])
    total_gt = sum(m["n_gt"] for m in all_metrics)
    total_pred = sum(m["n_pred"] for m in all_metrics)

    print(f"\n{'='*55}")
    print(f"  MoNuSeg External Validation (zero-shot)")
    print(f"  {'-'*45}")
    print(f"  Images:      {len(all_metrics)}")
    print(f"  GT nuclei:   {total_gt}")
    print(f"  Pred nuclei: {total_pred}")
    print(f"  {'-'*45}")
    print(f"  DICE:        {dice:.4f}")
    print(f"  AJI:         {aji:.4f}")
    print(f"  bPQ:         {bPQ:.4f}  (DQ={dq:.4f}, SQ={sq:.4f})")
    print(f"  F1 det:      {dq:.4f}  (DQ = F1_det)")
    print(f"{'='*55}")

    # Save
    with open(os.path.join(args.output, "monuseg_summary.txt"), "w") as f:
        f.write(f"MoNuSeg External Validation (zero-shot)\n")
        f.write(f"Model: {args.checkpoint}\n")
        f.write(f"Encoder: {args.encoder}  |  Decoder: {args.decoder}\n")
        f.write(f"PP: np_thresh={args.np_thresh} min_area={args.min_area}\n")
        f.write(f"{'='*50}\n")
        f.write(f"Images:       {len(all_metrics)}\n")
        f.write(f"GT nuclei:    {total_gt}\n")
        f.write(f"Pred nuclei:  {total_pred}\n")
        f.write(f"DICE:         {dice:.4f}\n")
        f.write(f"AJI:          {aji:.4f}\n")
        f.write(f"bPQ:          {bPQ:.4f}  (DQ={dq:.4f}, SQ={sq:.4f})\n")
        f.write(f"F1 detection: {dq:.4f}\n")
    print(f"\nSummary → {args.output}/monuseg_summary.txt")


if __name__ == "__main__":
    main()
