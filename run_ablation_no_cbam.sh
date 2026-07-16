#!/bin/bash
# run_ablation_no_cbam.sh — Ablation: no CBAM in NC head, UNI2-h + shared_unet
# fold1 only, cuda:2, no PP search
# ==============================================================================
set -euo pipefail

OUTDIR="./output/ablation_no_cbam"
EPOCHS=300
UNI2_WTS="/home/lwy/Newidea/pytorch_model.bin"
SEED=114514
GPU=3
VAL_FOLD=1
NAME="fold${VAL_FOLD}"

mkdir -p "$OUTDIR"

echo "===== Ablation: NO CBAM (fold1, cuda:2) ====="
echo ""
echo "============================================================"
echo "  [No CBAM]  val=fold${VAL_FOLD}  GPU=${GPU}  TRAIN from scratch"
echo "  $(date)"
echo "============================================================"

mkdir -p "$OUTDIR/${NAME}"
python -u train.py \
    --epochs $EPOCHS --batch_size 48 --val_fold $VAL_FOLD \
    --device "cuda:${GPU}" --output_dir "$OUTDIR" --run_name "$NAME" \
    --encoder uni2-h --decoder shared_unet --full_unfreeze \
    --heavy_aug --backbone_lr_mult 0.1 --enc_dropout 0.5 --dec_dropout 0.2 \
    --balance_sample --uni2_weights "$UNI2_WTS" --no_cbam \
    --cb_gamma 1.5 --np_loss ft+dice --np_ft_alpha 0.5 --nc_loss focal+dice \
    --np_weight 2.0 --hv_loss_weight 2.0 --nc_weight 2.0 \
    --hv_mse_weight 2.5 --hv_msge_weight 8.0 \
    --seed $SEED --val_interval 1 --patience 50 \
    --freeze_epochs 100 \
    > "$OUTDIR/${NAME}/train.log" 2>&1
echo "  Log -> $OUTDIR/${NAME}/train.log"
echo ""
echo "===== Done: $(date) ====="
