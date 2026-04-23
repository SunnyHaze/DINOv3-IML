#!/usr/bin/env bash
# DINOv3-IML training launcher
#
# Usage:
#   bash scripts/train.sh configs/cat_lora_vitl_r32.yaml
#
# Requirements:
#   pip install imdlbenco peft torch
#
# Before running:
#   1. Edit the config YAML to set data_path, test_data_path,
#      dinov3_repo_path, and dinov3_weights_path for your system.
#   2. Ensure the DINOv3 backbone weights are downloaded.
#
set -euo pipefail

CONFIG="${1:-configs/cat_lora_vitl_r32.yaml}"

if [ ! -f "$CONFIG" ]; then
    echo "Error: config file not found: $CONFIG"
    exit 1
fi

# Parse key fields from YAML (requires python3)
MODEL=$(python3 -c "import yaml; d=yaml.safe_load(open('$CONFIG')); print(d['model'])")
OUTPUT_DIR="output/$(basename ${CONFIG%.yaml})"
MASTER_PORT="${MASTER_PORT:-29500}"
NPROC="${NPROC:-1}"

echo "========================================"
echo "Config:     $CONFIG"
echo "Model:      $MODEL"
echo "Output:     $OUTPUT_DIR"
echo "GPUs:       $NPROC"
echo "========================================"

mkdir -p "$OUTPUT_DIR"

# Local training entry point registers this repo's custom models first.
torchrun \
    --nproc_per_node="$NPROC" \
    --master_port="$MASTER_PORT" \
    train.py \
    $(python3 -c "
import shlex
import yaml
d = yaml.safe_load(open('$CONFIG'))
skip = {'model', 'data_path', 'test_data_path', 'dinov3_repo_path', 'dinov3_weights_path'}
for k, v in d.items():
    if k not in skip:
        if isinstance(v, bool):
            if v:
                print(f'--{k}', end=' ')
        else:
            print(f'--{k} {shlex.quote(str(v))}', end=' ')
") \
    --model "$MODEL" \
    --data_path "$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['data_path'])")" \
    --test_data_path "$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['test_data_path'])")" \
    --dinov3_repo_path "$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['dinov3_repo_path'])")" \
    --dinov3_weights_path "$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['dinov3_weights_path'])")" \
    --output_dir "$OUTPUT_DIR" \
    --find_unused_parameters \
    2>&1 | tee "$OUTPUT_DIR/train.log"

echo "Training complete. Results in: $OUTPUT_DIR"
