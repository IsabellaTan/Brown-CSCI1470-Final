# bash scripts/train_baseline.sh data/tiny-imagenet-200 resnet50
# bash scripts/train_baseline.sh /data/tiny-imagenet-200 vit-small


#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/train_baseline.sh /path/to/tiny-imagenet-200 [resnet50|vit-small]
DATA_ROOT="${1:-}"
MODEL="${2:-resnet50}"

if [[ -z "$DATA_ROOT" ]]; then
  echo "Usage: bash scripts/train_baseline.sh /path/to/tiny-imagenet-200 [resnet50|vit-small]"
  exit 1
fi

if [[ ! -d "$DATA_ROOT/train" || ! -d "$DATA_ROOT/val" ]]; then
  echo "Error: dataset path is invalid: $DATA_ROOT"
  echo "Expected folders: $DATA_ROOT/train and $DATA_ROOT/val"
  if [[ -d "data/tiny-imagenet-200/train" && -d "data/tiny-imagenet-200/val" ]]; then
    echo "Hint: dataset exists at: data/tiny-imagenet-200"
    echo "Try: bash scripts/train_baseline.sh data/tiny-imagenet-200 ${MODEL}"
  fi
  exit 1
fi

if [[ "$MODEL" == "vit-small" ]]; then
  OPTIMIZER="adamw"
  LR="5e-4"
  WD="0.05"
  OUT="outputs/vit_small_baseline"
else
  OPTIMIZER="sgd"
  LR="0.1"
  WD="1e-4"
  OUT="outputs/resnet50_baseline"
fi

python scripts/train_baseline.py \
  --model "$MODEL" \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUT" \
  --optimizer "$OPTIMIZER" \
  --lr "$LR" \
  --weight-decay "$WD" \
  --batch-size 64 \
  --epochs 1 \
  --num-workers 2 \
  --amp
