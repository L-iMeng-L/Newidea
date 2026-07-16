#!/bin/bash
# run18_3fold.sh — 3-fold CV: shared_unet (no MALA), 2:2:2, 300 epochs
# cuda:0 sequential
# ==============================================================================
set -euo pipefail

cleanup() {
    echo "Stopping all training jobs..."
    jobs -p | xargs -r kill 2>/dev/null
    wait 2>/dev/null
    echo "All stopped."
}
trap cleanup EXIT INT TERM

OUTDIR="./output/sharedunet_3fold"
EPOCHS=300                # target total epochs
UNI2_WTS="/home/lwy/Newidea/pytorch_model.bin"
SEED=114514
GPU=0   # run18 on cuda:0
DECODER="shared_unet"

mkdir -p "$OUTDIR"

run_fold() {
    local val_fold=$1
    local name="fold${val_fold}"
    local last_ckpt="$OUTDIR/${name}/last.pth"
    local cur_epochs=$EPOCHS
    local resume_flag=""

    echo ""
    echo "============================================================"
    echo "  [3-fold]  val=fold${val_fold}  GPU=${GPU}"

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

    echo "  $(date)"
    echo "============================================================"
    mkdir -p "$OUTDIR/${name}"
    python -u train.py \
        --epochs $cur_epochs --batch_size 48 --val_fold $val_fold \
        --device "cuda:${GPU}" --output_dir "$OUTDIR" --run_name "$name" \
        --encoder uni2-h --decoder "$DECODER" --full_unfreeze \
        --heavy_aug --backbone_lr_mult 0.1 --enc_dropout 0.5 --dec_dropout 0.2 \
        --balance_sample --uni2_weights "$UNI2_WTS" \
        --cb_gamma 1.5 --np_loss ft+dice --np_ft_alpha 0.5 --nc_loss focal+dice \
        --np_weight 2.0 --hv_loss_weight 2.0 --nc_weight 2.0 \
        --hv_mse_weight 2.5 --hv_msge_weight 8.0 \
        --seed $SEED --val_interval 1 --patience 50 \
        --freeze_epochs 100 \
        $resume_flag > "$OUTDIR/${name}/train.log" 2>&1
    echo "  Log -> $OUTDIR/${name}/train.log"
}

# ================================================================
# Train + PP search per fold (sequential on cuda:0)
# ================================================================
echo "===== 3-fold CV (cuda:0 sequential) ====="
for f in 2 3; do
    run_fold $f
    sleep 50
done
echo ""
echo "All folds done: $(date)"
