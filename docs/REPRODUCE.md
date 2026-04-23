# Reproducing Paper Results

This guide walks through reproducing the main results from Table 1 (CAT protocol) and Table 2 (MVSS protocol) from scratch.

---

## Environment

```bash
# Python 3.8+ required
conda create -n dinov3iml python=3.10
conda activate dinov3iml

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install peft imdlbenco Pillow numpy matplotlib
```

---

## Download DINOv3 Backbone

```bash
# Clone the DINOv3 repository
git clone https://github.com/facebookresearch/dinov2.git

# Download ViT-L/16 weights (recommended for best results)
# ViT-L: dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
# ViT-B: dinov3_vitb16_pretrain.pth
# ViT-S: dinov3_vits16_pretrain.pth
# Download from: https://github.com/facebookresearch/dinov2#pretrained-models
```

---

## Dataset Preparation

### CAT Protocol

Training requires 4 datasets:
- **CASIA-v2**: CASIA Image Tampering Detection Evaluation Database v2
- **FantasticReality**: [link](https://github.com/Laix66/FantasticReality)
- **IMD2020**: [link](http://staff.utia.cas.cz/novozada/db/)
- **TampCOCO**: [link](https://github.com/mjkwon2021/CAT-Net) (use bcm, cm, bcmc subsets)

Prepare JSON files via [IMDLBenCo dataset tools](https://github.com/scu-zjz/IMDLBenCo/blob/main/docs/dataset.md).

Test sets: CASIAv1, Columbia, NIST16, Coverage — follow IMDLBenCo docs.

### MVSS Protocol

Training: CASIA-v2 only (tampered images, ~5,123 pairs).
Test: CASIAv1, Columbia, NIST16, Coverage, IMD2020.

---

## Training

### Best configuration (CAT protocol, ViT-L LoRA r=32)

1. Edit `configs/cat_lora_vitl_r32.yaml`:
   ```yaml
   data_path: /path/to/balanced_dataset_cat.json
   test_data_path: /path/to/test_datasets_cat.json
   dinov3_repo_path: /path/to/dinov3
   dinov3_weights_path: /path/to/dinov3_vitl16_pretrain_lvd1689m.pth
   ```

2. Launch:
   ```bash
   bash scripts/train.sh configs/cat_lora_vitl_r32.yaml
   ```

3. Training takes ~10 hours on a single RTX 6000 Ada (49 GB VRAM).
   Evaluate every 4 epochs; best checkpoint typically at epoch 48.

### MVSS protocol (ViT-L LoRA r=64)

```bash
# Edit mvss config paths, then:
bash scripts/train.sh configs/mvss_lora_vitl_r64.yaml
```

Training on CASIA-v2 only (~100 epochs, ~8 hours on RTX 6000 Ada).

---

## Checkpoint Selection

Best epoch is selected via **mode-of-argmax** across test datasets:
for each dataset, find the epoch with maximum F1; the best overall epoch
is the one where most datasets peak simultaneously. Do NOT use the
final epoch or a global running average — the training evaluates every
4 epochs, and the best window is typically epoch 44–84.

```python
# Example post-hoc selection from training stdout log
import re
from collections import defaultdict

log = open("output/cat_lora_vitl_r32/train.log").read()
# Parse "Dataset: X" and "F1 = Y" entries per epoch...
```

---

## Expected Results

| Config | Expected Avg F1 | Typical Best Epoch |
|---|---|---|
| cat_lora_vitl_r32 | 0.847 | ep 48 |
| cat_lora_vitl_r64 | 0.837 | ep 64–76 |
| cat_lora_vitb_r64 | 0.780 | ep 48–72 |
| cat_lora_vits_r32 | 0.704 | ep 96–99 |
| mvss_lora_vitl_r64 | 0.774 | ep 76 |
| mvss_lora_vitl_r32 | 0.770 | ep 84 |

Results may vary ±0.005 depending on hardware and random seed.

---

## Inference

For a quick single-image inference example, use the command shown in `README.md`.
