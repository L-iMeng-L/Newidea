#!/bin/bash
# MoNuSeg 3-fold eval — 256+64 overlap (stride=192, margin=32)
set -euo pipefail

OUTDIR="./monuseg_results/overlap_3fold"
PP1="--np_thresh 0.9 --min_area 20 --energy_thresh 0.5 --sobel_ksize 15 --marker_ksize 7"
PP2="--np_thresh 0.55 --min_area 19 --energy_thresh 0.4 --sobel_ksize 19 --marker_ksize 3"
PP3="--np_thresh 0.6 --min_area 14 --energy_thresh 0.5 --sobel_ksize 25 --marker_ksize 7"
CKPT="./output/run_sharedunet_3fold"

mkdir -p "$OUTDIR"

echo "=== fold1 (cuda:0) ==="
python -u eval_monuseg.py --checkpoint "$CKPT/fold1/best.pth" \
  --encoder uni2-h --decoder shared_unet --device cuda:0 \
  --output "$OUTDIR/fold1" $PP1 --stride 192 --margin 32

echo "=== fold2 (cuda:1) ==="
python -u eval_monuseg.py --checkpoint "$CKPT/fold2/best.pth" \
  --encoder uni2-h --decoder shared_unet --device cuda:1 \
  --output "$OUTDIR/fold2" $PP2 --stride 192 --margin 32

echo "=== fold3 (cuda:2) ==="
python -u eval_monuseg.py --checkpoint "$CKPT/fold3/best.pth" \
  --encoder uni2-h --decoder shared_unet --device cuda:2 \
  --output "$OUTDIR/fold3" $PP3 --stride 192 --margin 32

echo "Done. Results → $OUTDIR/"
python3 << 'PYEOF'
import re, numpy as np
bps, dices, ajis = [], [], []
for f in [1,2,3]:
    with open(f"./monuseg_results/overlap_3fold/fold{f}/monuseg_summary.txt") as fh: t=fh.read()
    b=re.search(r'bPQ:\s+([\d.]+)',t); d=re.search(r'DICE:\s+([\d.]+)',t); a=re.search(r'AJI:\s+([\d.]+)',t)
    if b: bps.append(float(b.group(1)))
    if d: dices.append(float(d.group(1)))
    if a: ajis.append(float(a.group(1)))
    print(f"  fold{f}: bPQ={b.group(1) if b else '?'} DICE={d.group(1) if d else '?'} AJI={a.group(1) if a else '?'}")
if bps: print(f"  3-fold: bPQ={np.mean(bps):.4f}±{np.std(bps):.4f}  DICE={np.mean(dices):.4f}±{np.std(dices):.4f}  AJI={np.mean(ajis):.4f}±{np.std(ajis):.4f}")
PYEOF
