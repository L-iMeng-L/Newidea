#!/bin/bash
# run_scratch_tiny.sh — ConvNeXt-Tiny + shared_unet from scratch, no KD, 3-fold
# cuda:3 sequential
# ==============================================================================
set -euo pipefail

cleanup() {
    echo "Stopping all training jobs..."
    jobs -p | xargs -r kill 2>/dev/null
    wait 2>/dev/null
    echo "All stopped."
}
trap cleanup EXIT INT TERM

OUTDIR="./output/scratch_tiny_3fold"
EPOCHS=300
SEED=114514
GPU=3   # on cuda:3

mkdir -p "$OUTDIR"


run_fold() {
    local val_fold=$1
    local name="fold${val_fold}"
    local last_ckpt="$OUTDIR/${name}/last.pth"
    local cur_epochs=$EPOCHS
    local resume_flag=""

    echo ""
    echo "============================================================"
    echo "  [Scratch Tiny]  val=fold${val_fold}  GPU=${GPU}"

    if [ -f "$last_ckpt" ]; then
        local done_epochs=$(python -c "import torch; print(torch.load('$last_ckpt',map_location='cpu',weights_only=False).get('epoch',0))")
        if [ "$done_epochs" -ge "$EPOCHS" ]; then
            echo "  SKIP: epoch ${done_epochs}/${EPOCHS} done"
            
            return 0
        fi
        cur_epochs=$((EPOCHS - done_epochs))
        resume_flag="--resume $last_ckpt"
        echo "  RESUME: epoch ${done_epochs}→${EPOCHS}  (train ${cur_epochs} more)"
    else
        echo "  TRAIN from scratch (${EPOCHS} epochs, no frozen encoder)"
    fi

    echo "  $(date)"
    echo "============================================================"
    mkdir -p "$OUTDIR/${name}"
    python -u train.py \
        --epochs $cur_epochs --batch_size 48 --val_fold $val_fold \
        --device "cuda:${GPU}" --output_dir "$OUTDIR" --run_name "$name" \
        --encoder tiny --decoder shared_unet \
        --heavy_aug --backbone_lr_mult 0.1 --enc_dropout 0.5 --dec_dropout 0.2 \
        --balance_sample \
        --cb_gamma 1.5 --np_loss ft+dice --np_ft_alpha 0.5 --nc_loss focal+dice \
        --np_weight 2.0 --hv_loss_weight 2.0 --nc_weight 2.0 \
        --hv_mse_weight 2.5 --hv_msge_weight 8.0 \
        --seed $SEED --val_interval 1 --patience 50 \
        --freeze_epochs 0 \
        $resume_flag > "$OUTDIR/${name}/train.log" 2>&1
    echo "  Log -> $OUTDIR/${name}/train.log"

    
}

# ================================================================
# 3-fold from scratch
# ================================================================
echo "===== Scratch Tiny 3-fold (cuda:3 sequential) ====="
for f in 1 2 3; do
    run_fold $f
    sleep 30
done
echo ""
echo "All folds done: $(date)"


