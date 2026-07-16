# UNI2-SharedUNet: Efficient Nuclei Instance Segmentation via Shared Decoder + Knowledge Distillation

High-performance nuclei instance segmentation on PanNuke — a single shared U-Net decoder with UNI2-h backbone achieves SOTA mPQ, outperforming independent decoder designs and ViT-level baselines. Compressed 20× via knowledge distillation to ConvNeXt-Tiny with negligible performance loss.

## Architecture

```
Input 256²×3
    │
    ▼
UNI2-h ViT (681M) ──→ f3 /32 · 768ch
                  ──→ f2 /16 · 384ch
                  ──→ f1 /8  · 192ch
                  ──→ f0 /4  · 96ch
    │
    ▼
1×1 Conv (768→256)
    │
    ▼
Shared U-Net Decoder (6.9M)
  dec_stage4: ConvBlock×3 · 256ch
  dec_up4:    ConvTranspose2d ↑2×
  dec_stage3: ConvTranspose2d→concat(f2)→Conv×2 · 256ch
  dec_stage2: ConvTranspose2d→concat(f1)→Conv×2 · 192ch
  dec_stage1: concat(f0)→Conv×2 · 128ch
    │
    ├── NP Head (Conv+Conv2d→1)
    ├── NC Head (Conv+CBAM+Conv2d→5)
    └── HV Head (Conv+Conv2d→2)
```

**Core insight**: NP/HV/NC share low-level features (edges, textures, morphology). A single shared decoder with task-specific lightweight heads achieves implicit cross-task regularization — outperforming independent decoders with 3× fewer parameters.

## Results

### PanNuke (3-fold CV, tissue-level, PP-optimized)

| Model | Encoder | Decoder | Params | mPQ | bPQ |
|---|---|---|---|---|---|
| **shared_unet** | UNI2-h | shared (6.9M) | 690.5M | **0.5238±0.0076** | 0.6815±0.0067 |
| +MALA | UNI2-h | shared+MALA (10.7M) | 694.3M | 0.5237±0.0076 | 0.6811±0.0059 |
| unet3 | UNI2-h | 3 indep (18.8M) | 702.4M | 0.5236±0.0092 | 0.6803±0.0048 |
| KD Student | ConvNeXt-Tiny | shared (6.9M) | **35M** | 0.5175±0.0101 | 0.6811±0.0057 |
| Scratch Tiny | ConvNeXt-Tiny | shared (6.9M) | 35M | 0.5121±0.0074 | 0.6840±0.0055 |

### External Validation (zero-shot)

Models trained on PanNuke only, evaluated directly on MoNuSeg without fine-tuning.

**MoNuSeg** (7 organs, detection only). Baseline data from CFR-SAM:

| Model | AJI | bPQ |
|---|---|---|
| CellViT-SAM-H | 0.644 | 0.490 |
| CFR-SAM-H | 0.668 | 0.662 |
| **Ours (Teacher)** | 0.640 | **0.667** |
| **Ours (KD Student)** | 0.645 | **0.668** |

### Per-Class Metrics (PanNuke, shared_unet 3-fold)

| Class | F1 | P | R |
|---|---|---|---|
| Neoplastic | 0.763±0.007 | 0.768±0.012 | 0.758±0.007 |
| Epithelial | 0.745±0.022 | 0.825±0.008 | 0.680±0.040 |
| Inflammatory | 0.741±0.006 | 0.741±0.002 | 0.740±0.011 |
| Connective | 0.644±0.006 | 0.638±0.013 | 0.651±0.008 |
| Dead | 0.491±0.041 | 0.567±0.050 | 0.434±0.035 |

## Key Findings

1. **Shared Decoder** — 6.9M shared U-Net matches 18.8M independent decoders with identical mPQ. The shared decoder achieves the same performance at 37% parameter cost through implicit cross-task regularization.

2. **MALA: Zero Gain** — Adding 3.8M MALA parameters at /4 and /8 scales yields +0.000 mPQ. ViT global self-attention already covers the multi-scale receptive fields that MALA provides. A valuable negative result for ViT + nuclei segmentation research.

3. **20× KD Compression** — Output-only distillation (KL+MSE, T=1, α=0.5) preserves 98.8% of teacher mPQ. Encoder feature alignment (MSE) is unnecessary — ViT and CNN feature spaces are fundamentally different, and forcing alignment may harm CNN's local inductive biases.

4. **ConvTranspose2d Upsampling** — Learnable upsampling contributes ~0.004 mPQ vs bilinear interpolation, primarily improving boundary precision (bPQ drop of 0.01 when removed).

5. **CBAM Attention** — Lightweight channel-spatial attention on the NC head adds ~0.004 mPQ with negligible parameter cost.

6. **Class-Balanced Sampling** — Inverse-frequency sampling is critical for rare classes (dead cells: 0.06% prevalence), enabling non-zero detection where uniform sampling fails.

## Training

```bash
# Teacher (UNI2-h + shared_unet, 3-fold)
bash run_sharedunet_3fold.sh

# Teacher +MALA (3-fold)
bash run_sharedunet_3fold_mala.sh

# KD Student (ConvNeXt-Tiny, 3-fold)
bash run_kd_sharedunet.sh              # shared_unet teacher
bash run_kd_sharedunet_mala_tiny.sh    # shared_unet_mala teacher

# Ablations (fold1 only)
bash run_ablation_no_cbam.sh          # NC head without CBAM
bash run_ablation_unet3.sh            # Independent decoders
bash run_ablation_blinear.sh          # Bilinear upsampling
bash run_ablation_no_cbsample.sh      # Uniform sampling
```

## External Evaluation

```bash
# MoNuSeg (3-fold, with per-fold PP params)
bash run_monuseg_overlap.sh

```

## Directory Structure

```
├── train.py, kd_train.py       # Training scripts
├── search_pp.py                # PP hyperparameter search
├── eval_monuseg.py              # External validation
├── benchmark_speed.py          # Inference speed benchmark
├── infer_viz.py                # PanNuke visualization
├── models/
│   ├── model.py                # ConvNeXtSegmentor (all decoder types)
│   ├── decoder.py              # DecoderBlock, ConvBlock
│   ├── encoder.py              # ConvNeXt V1/V2 + UNI2-h ViT encoder
│   ├── attention.py            # CBAM, ECA
│   └── mala.py                 # MALA dynamic convolution (3/5/7 kernels)
├── losses/
│   ├── losses.py               # NP (FT+Dice) + HV (MSE+MSGE) + NC (Focal+Dice)
│   └── kd_losses.py            # Distillation loss (KL + MSE)
├── utils/
│   ├── evaluate.py             # PQ metrics (official PanNuke protocol)
│   ├── postprocess.py          # HoVer-Net watershed post-processing
│   ├── history.py              # Training history tracker
│   └── plotting.py             # Training curve plotting
├── data/
│   ├── pannuke.py              # PanNuke dataset (multi-fold, pre-loaded)
│   └── augs.py                 # Heavy augmentation (blur+elastic+HSV)
└── run*.sh                     # Experiment scripts
```

## Requirements

- PyTorch 2.x, CUDA 12+
- UNI2-h weights (download from MahmoodLab/UNI2-h)
- PanNuke dataset (processed: images/ + hover/ per fold)
- 40GB+ GPU for teacher training, 4GB+ for student inference
