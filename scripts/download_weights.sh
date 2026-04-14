#!/usr/bin/env bash
# Download pretrained checkpoints for DINOv3-IML.
#
# Usage:
#   bash scripts/download_weights.sh          # download all
#   bash scripts/download_weights.sh cat_vitl # download specific checkpoint
#
# Requires: gdown (pip install gdown)
#
set -euo pipefail

mkdir -p checkpoints

echo "Installing gdown if not present..."
pip install -q gdown

# ============================================================
# Pretrained DINOv3-IML checkpoints (Google Drive)
# ============================================================
declare -A CKPT_IDS=(
    # CAT protocol
    ["cat_vitl_lora_r32"]="PLACEHOLDER_GDRIVE_ID"   # ViT-L LoRA r=32, avg F1=0.847
    ["cat_vitl_lora_r64"]="PLACEHOLDER_GDRIVE_ID"   # ViT-L LoRA r=64, avg F1=0.837
    ["cat_vitb_lora_r64"]="PLACEHOLDER_GDRIVE_ID"   # ViT-B LoRA r=64, avg F1=0.780
    ["cat_vits_lora_r32"]="PLACEHOLDER_GDRIVE_ID"   # ViT-S LoRA r=32, avg F1=0.704
    # MVSS protocol
    ["mvss_vitl_lora_r64"]="PLACEHOLDER_GDRIVE_ID"  # ViT-L LoRA r=64, avg F1=0.774
    ["mvss_vitl_lora_r32"]="PLACEHOLDER_GDRIVE_ID"  # ViT-L LoRA r=32, avg F1=0.770
)

TARGET="${1:-all}"

for name in "${!CKPT_IDS[@]}"; do
    if [ "$TARGET" != "all" ] && [ "$TARGET" != "$name" ]; then
        continue
    fi
    id="${CKPT_IDS[$name]}"
    out="checkpoints/${name}.pth"
    if [ "$id" = "PLACEHOLDER_GDRIVE_ID" ]; then
        echo "  [skip] $name — download ID not yet set (update this script after paper release)"
        continue
    fi
    if [ -f "$out" ]; then
        echo "  [skip] $out already exists"
        continue
    fi
    echo "  Downloading $name → $out ..."
    gdown "https://drive.google.com/uc?id=${id}" -O "$out"
done

echo ""
echo "Done. Checkpoints saved to ./checkpoints/"
echo ""
echo "For DINOv3 backbone weights, see the DINOv3 repository:"
echo "  https://github.com/facebookresearch/dinov2"
