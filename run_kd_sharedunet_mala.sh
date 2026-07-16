#!/bin/bash
# run_kd_sharedunet_mala_tiny.sh — KD: sharedunet_mala teacher → ConvNeXt-Tiny, 3-fold
# cuda:2 sequential
# ==============================================================================
set -euo pipefail

cleanup() {
    echo "Stopping all training jobs..."
    jobs -p | xargs -r kill 2>/dev/null
    wait 2>/dev/null
    echo "All stopped."
}
trap cleanup EXIT INT TERM

OUTDIR="./output/kd_sharedunet_mala_3fold"
TEACHER_DIR="./output/sharedunet_3fold_mala"
EPOCHS=300
SEED=114514
GPU=2   # on cuda:2

mkdir -p "$OUTDIR"


run_fold() {
    local val_fold=$1
    local name="fold${val_fold}"
    local teacher_ckpt="$TEACHER_DIR/${name}/best.pth"
    local last_ckpt="$OUTDIR/${name}/last.pth"
    local cur_epochs=$EPOCHS
    local resume_flag=""

    echo ""
    echo "============================================================"
    echo "  [KD sharedunet_mala→Tiny]  val=fold${val_fold}  GPU=${GPU}"

    if [ ! -f "$teacher_ckpt" ]; then
        echo "  SKIP: teacher checkpoint not found: $teacher_ckpt"
        return 1
    fi

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
        echo "  TRAIN from scratch (${EPOCHS} epochs)"
    fi

    echo "  Teacher: $teacher_ckpt"
    echo "  $(date)"
    echo "============================================================"
    mkdir -p "$OUTDIR/${name}"
    python -u kd_train.py \
        --teacher_ckpt "$teacher_ckpt" --teacher_decoder shared_unet_mala \
        --student_encoder tiny --student_decoder shared_unet \
        --epochs $cur_epochs --batch_size 32 --val_fold $val_fold \
        --device "cuda:${GPU}" --output_dir "$OUTDIR" --run_name "$name" \
        --kd_temperature 1.0 --kd_alpha 0.5 --kd_enc_weight 0.0 \
        --lr 1e-4 --weight_decay 5e-3 --heavy_aug --balance_sample \
        --np_loss ft+dice --nc_loss focal+dice \
        --np_weight 2.0 --hv_loss_weight 2.0 --nc_weight 2.0 \
        --hv_mse_weight 2.5 --hv_msge_weight 8.0 \
        --seed $SEED --val_interval 1 --patience 50 \
        --enc_dropout 0.5 --dec_dropout 0.2 \
        $resume_flag > "$OUTDIR/${name}/train.log" 2>&1
    echo "  Log -> $OUTDIR/${name}/train.log"

    
}

# ================================================================
# 3-fold KD
# ================================================================
echo "===== KD (sharedunet_mala→Tiny) 3-fold (cuda:2 sequential) ====="
for f in 1 2 3; do
    run_fold $f
    sleep 30
done
echo ""
echo "All folds done: $(date)"


