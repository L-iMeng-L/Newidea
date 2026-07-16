#!/usr/bin/env python3
"""
Inference benchmark — HoVer-Net / HoVer-UNet style reporting.

Metrics (all on PanNuke 256×256 @ 0.25 mpp, single GPU, no mixed precision):
    - GMacs, #Params
    - Pure inference FPS (10 warmup + 100 tiles, fixed batch)
    - TTA4 / TTA16 throughput
    - Post-processing throughput (watershed + classify)
    - Full pipeline: inference + TTA + postprocess
    - WSI end-to-end estimate (load → screen → tile → TTA → stitch → post → save)
    - Per-area throughput (s/mm² @ 0.25 mpp)

TTA4 = original + 3 rotations = 4 branches, average logits.
TTA16 = 8 rot×flip + 8 transpose×rot×flip = 16 branches, average logits.
"""
import argparse, time, sys, os
import numpy as np
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import create_model
from utils.postprocess import postprocess_nuclei

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--encoder", type=str, default="tiny",
                   choices=["uni2-h", "tiny", "small", "base"])
    p.add_argument("--decoder", type=str, default="shared_unet")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--input_size", type=int, default=256)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--wsi_tiles", type=int, default=15000,
                   help="Estimated #tiles per WSI (0.25 mpp, ~100k pix, 10% tissue)")
    return p.parse_args()

# ---------------------------------------------------------------------------
# GMACs estimation
# ---------------------------------------------------------------------------
def count_gmacs(model, input_size=256):
    try:
        from fvcore.nn import FlopCountAnalysis
        x = torch.randn(1, 3, input_size, input_size)
        return FlopCountAnalysis(model, x).total() / 1e9
    except ImportError:
        pass
    try:
        from thop import profile
        x = torch.randn(1, 3, input_size, input_size).to(next(model.parameters()).device)
        macs, _ = profile(model, inputs=(x,), verbose=False)
        return macs / 1e9
    except ImportError:
        pass
    return None

# ---------------------------------------------------------------------------
# TTA transforms
# ---------------------------------------------------------------------------
def tta_transform(x, idx):
    """idx 0-15:  0-3 rot, 4-7 hflip+rot, 8-11 transp+rot, 12-15 transp+hflip+rot"""
    if isinstance(x, np.ndarray):
        if idx >= 8:
            x = x.transpose(0, 1, 3, 2) if x.ndim == 4 else x.transpose(0, 2, 1)
            idx -= 8
        if idx >= 4:
            x = np.flip(x, axis=-1) if x.ndim == 4 else np.flip(x, axis=-1)
            idx -= 4
        return np.rot90(x, k=idx, axes=(-2, -1)) if x.ndim == 4 else np.rot90(x, k=idx, axes=(-2, -1))

    # torch path
    x = x.clone()
    if idx >= 8:
        x = torch.transpose(x, -2, -1); idx -= 8
    if idx >= 4:
        x = torch.flip(x, dims=[-1]); idx -= 4
    return torch.rot90(x, k=idx, dims=[-2, -1])

def tta_inverse_np(np_prob, hv_map, nc_logits, idx):
    """Reverse TTA on numpy arrays (postprocess input)."""
    p, h, n = np_prob.copy(), hv_map.copy(), nc_logits.copy()
    transposed = (idx >= 8)
    if transposed: idx -= 8
    flipped = (idx >= 4)
    if flipped: idx -= 4

    # np_prob: [H,W]
    for _ in range((4 - idx) % 4):
        p = np.rot90(p, k=1, axes=(-2, -1))
    if flipped:
        p = np.flip(p, axis=-1)
    if transposed:
        p = p.T

    # hv_map: [2, H, W] — rotate vectors
    hv_h, hv_v = h[0].copy(), h[1].copy()
    if flipped:
        hv_h = -hv_h
    for _ in range(idx):
        hv_h, hv_v = hv_v, -hv_h  # forward rotation transform
    # Reverse rotation
    for _ in range(idx):
        hv_h, hv_v = -hv_v, hv_h
    if flipped:
        hv_h = -hv_h
    if transposed:
        hv_h, hv_v = hv_v, hv_h
        hv_h = hv_h.T; hv_v = hv_v.T
        p = p.T
    else:
        pass  # hv_h, hv_v already correct shape

    h[0], h[1] = hv_h, hv_v

    # nc_logits: [C, H, W]
    for _ in range((4 - idx) % 4):
        n = np.rot90(n, k=1, axes=(-2, -1))
    if flipped:
        n = np.flip(n, axis=-1)
    if transposed:
        n = n.transpose(0, 2, 1)

    return p, h, n

# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def bench_inference(model, images, n_tta, warmup, iters, device):
    """Return (elapsed_s, ms_per_batch, patches_per_sec)."""
    model.eval()
    fn = (lambda x: tta_forward(model, x, n_tta)) if n_tta else (lambda x: model(x))

    for _ in range(warmup):
        fn(images)
    if device != "cpu":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        fn(images)
    if device != "cpu":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    total = images.shape[0] * iters
    return total, elapsed, elapsed / iters * 1000, total / elapsed

def tta_forward(model, images, n_tta):
    """TTA forward: average logits across n_tta branches."""
    keys = ["np", "nc", "hv"]
    accum = {k: 0.0 for k in keys}
    for t in range(n_tta):
        x = images if t == 0 else tta_transform(images, t)
        out = model(x)
        if t == 0:
            for k in keys:
                accum[k] = out[k]
        else:
            for k in keys:
                corrected = tta_inverse_torch(out[k], t, is_hv=(k == "hv"))
                accum[k] = accum[k] + corrected
    return {k: v / n_tta for k, v in accum.items()}

def tta_inverse_torch(tensor, idx, is_hv=False):
    """Reverse TTA on torch tensor."""
    out = tensor.clone()
    transposed = (idx >= 8)
    if transposed: idx -= 8
    flipped = (idx >= 4)
    if flipped: idx -= 4

    if is_hv:
        hv_h, hv_v = out[:, 0], out[:, 1]
        if flipped: hv_h = -hv_h
        for _ in range(idx): hv_h, hv_v = hv_v, -hv_h
        for _ in range(idx): hv_h, hv_v = -hv_v, hv_h
        if flipped: hv_h = -hv_h
        if transposed: hv_h, hv_v = hv_v, hv_h; out = torch.transpose(out, -2, -1)
        out[:, 0], out[:, 1] = hv_h, hv_v
    else:
        out = torch.rot90(out, k=(4 - idx) % 4, dims=[-2, -1])
        if flipped: out = torch.flip(out, dims=[-1])
        if transposed: out = torch.transpose(out, -2, -1)
    return out

def bench_postprocess(np_probs, hv_maps, nc_logits, np_thresh=0.5, min_area=10):
    """Time postprocessing on a batch of numpy outputs."""
    B = np_probs.shape[0]
    t0 = time.perf_counter()
    for b in range(B):
        postprocess_nuclei(np_probs[b, 0], hv_maps[b], nc_logits[b],
                           np_thresh=np_thresh, min_area=min_area)
    return time.perf_counter() - t0

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    is_uni2 = (args.encoder == "uni2-h")
    model = create_model(
        variant=args.encoder, num_nc_classes=5,
        pretrained=(not is_uni2),
        decoder_type=args.decoder, enc_dropout=0.5, dec_dropout=0.2,
        freeze_encoder=is_uni2, full_unfreeze=is_uni2,
    )
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()

    n_params = sum(p.numel() for p in model.parameters())
    gmacs = count_gmacs(model, args.input_size)

    print(f"\n{'='*65}")
    print(f"  {args.encoder} + {args.decoder}")
    print(f"  {'-'*50}")
    print(f"  Params:      {n_params/1e6:8.1f} M")
    if gmacs:
        print(f"  GMacs:       {gmacs:8.1f}  (per {args.input_size}x{args.input_size} patch)")
    print(f"  Resolution:  0.25 mpp (PanNuke standard)")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  GPU:         {args.device}, fp32, single GPU")
    print(f"{'='*65}")

    # Prepare fixed input batch
    images = torch.randn(args.batch_size, 3, args.input_size, args.input_size,
                         device=device)

    # ---- Pure inference FPS (no TTA) ----
    total_img, elapsed, ms_batch, fps = bench_inference(model, images, 0, args.warmup, args.iters, device)
    ms_patch = ms_batch / args.batch_size * 1000

    print(f"\n  [Pure Inference FPS]  ({args.warmup} warmup + {args.iters} iters × bs={args.batch_size})")
    print(f"  {'─'*40}")
    print(f"  No TTA:        {fps:.0f} patches/s   ({ms_batch:.1f} ms/batch,  {ms_patch/1000:.1f} ms/patch)")

    # ---- TTA (small model only; unnecessary for UNI2-h) ----
    if not is_uni2:
        total4, ela4, ms4, fps4 = bench_inference(model, images, 4, 5, args.iters // 2, device)
        print(f"  TTA4:          {fps4:.0f} patches/s   (slowdown {fps/fps4:.1f}x)")

        total16, ela16, ms16, fps16 = bench_inference(model, images, 16, 3, args.iters // 4, device)
        print(f"  TTA16:         {fps16:.0f} patches/s   (slowdown {fps/fps16:.1f}x)")
    else:
        fps4 = fps16 = 0

    # ---- Post-processing throughput ----
    np_batch = np.random.rand(args.batch_size, 1, 256, 256).astype(np.float32)
    hv_batch = np.random.randn(args.batch_size, 2, 256, 256).astype(np.float32)
    nc_batch = np.random.randn(args.batch_size, 5, 256, 256).astype(np.float32)

    pp_warm = sum(bench_postprocess(np_batch, hv_batch, nc_batch) for _ in range(5))
    pp_total = sum(bench_postprocess(np_batch, hv_batch, nc_batch) for _ in range(20))
    pp_ms_batch = pp_total / 20 * 1000
    pp_ms_patch = pp_ms_batch / args.batch_size
    pp_fps = args.batch_size / (pp_total / 20)

    print(f"\n  [Post-processing (Watershed)]  (batch={args.batch_size}, avg of 20 runs)")
    print(f"  {'─'*40}")
    print(f"  Throughput:    {pp_fps:.0f} patches/s   ({pp_ms_patch:.1f} ms/patch)")

    # ---- Full pipeline (inference + postprocess) ----
    full_fps_no = 1.0 / (1.0 / fps + 1.0 / pp_fps)

    print(f"\n  [Full Pipeline FPS]  (inference + postprocess)")
    print(f"  {'─'*40}")
    print(f"  No TTA:        {full_fps_no:.0f} patches/s")

    if not is_uni2:
        full_fps_t4  = 1.0 / (1.0 / fps4 + 1.0 / pp_fps)
        full_fps_t16 = 1.0 / (1.0 / fps16 + 1.0 / pp_fps)
        print(f"  TTA4:          {full_fps_t4:.0f} patches/s")
        print(f"  TTA16:         {full_fps_t16:.0f} patches/s")

    # ---- WSI end-to-end estimate ----
    n_tiles = args.wsi_tiles
    wsi_area_mm2 = (256 * 0.00025) ** 2 * n_tiles  # mm²

    rows = [("No TTA ", full_fps_no)]
    if not is_uni2:
        rows += [("TTA4   ", full_fps_t4), ("TTA16  ", full_fps_t16)]

    print(f"\n  [WSI End-to-End]  ({n_tiles:,} tiles/WSI, 0.25 mpp, ~{wsi_area_mm2:.0f} mm² tissue)")
    print(f"  WSI = load → tile (256×256 stride=256) → model → stitch → postprocess → save")
    print(f"  {'─'*40}")
    for tag, f_pipeline in rows:
        secs = n_tiles / f_pipeline
        mins = secs / 60.0
        per_mm2 = secs / wsi_area_mm2

        if mins >= 1:
            print(f"  {tag}  {mins:6.1f} min/WSI   ({per_mm2:.2f} s/mm²)")
        else:
            print(f"  {tag}  {secs:6.0f} sec/WSI  ({per_mm2:.2f} s/mm²)")

    print(f"\n  (WSI load/stitch/save overhead ≈5-10%, not included)")
    print(f"{'='*65}\n")

if __name__ == "__main__":
    main()
