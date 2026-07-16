#!/bin/bash
# Ablation: uniform sampling vs class-balanced sampling
# fold1 only, cuda:3
set -euo pipefail
OUTDIR="./output/ablation_no_cbsample"
GPU=2
VAL_FOLD=1
NAME="fold${VAL_FOLD}"

mkdir -p "$OUTDIR/${NAME}"
echo "===== Ablation: NO class-balanced sampling fold1 cuda:3 ====="
python -u train.py \
    --epochs 300 --batch_size 48 --val_fold $VAL_FOLD \
    --device "cuda:${GPU}" --output_dir "$OUTDIR" --run_name "$NAME" \
    --encoder uni2-h --decoder shared_unet --full_unfreeze \
    --heavy_aug --backbone_lr_mult 0.1 --enc_dropout 0.5 --dec_dropout 0.2 \
    --uni2_weights "/home/lwy/Newidea/pytorch_model.bin" \
    --np_loss ft+dice --np_ft_alpha 0.5 --nc_loss focal+dice \
    --np_weight 2.0 --hv_loss_weight 2.0 --nc_weight 2.0 \
    --hv_mse_weight 2.5 --hv_msge_weight 8.0 \
    --seed 114514 --val_interval 1 --patience 50 --freeze_epochs 100 \
    > "$OUTDIR/${NAME}/train.log" 2>&1
echo "Done → $OUTDIR/${NAME}/train.log"
