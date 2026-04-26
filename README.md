先跑在主目录跑python scripts/download_tiny_imagenet.py
下载数据集后，在train_baseline.sh中选择训练参数，然后跑bash scripts/train_baseline.sh data/tiny-imagenet-200 vit-small 或者 bash scripts/train_baseline.sh data/tiny-imagenet-200 resnet50

# Curriculum-Learning-for-CNNs-and-Transformers

This repository now includes a **baseline supervised pretraining script** for Tiny-ImageNet using:

- `ResNet50`
- `ViT-Small (vit_small_patch16_224)`

This baseline is intended as a clean starting point before adding curriculum learning strategies.

## 1) Install dependencies

```bash
pip install -r requirements.txt
```

## 2) Expected Tiny-ImageNet layout

The script expects an extracted `tiny-imagenet-200` directory with the default structure:

```text
tiny-imagenet-200/
  train/
    n01443537/
      images/*.JPEG
    ...
  val/
    images/*.JPEG
    val_annotations.txt
```

## 3) Train a ResNet50 baseline

```bash
python scripts/train_baseline.py \
  --model resnet50 \
  --data-root /path/to/tiny-imagenet-200 \
  --output-dir outputs/resnet50_baseline \
  --optimizer sgd \
  --lr 0.1 \
  --weight-decay 1e-4 \
  --batch-size 128 \
  --epochs 100 \
  --amp
```

## 4) Train a ViT-Small baseline

```bash
python scripts/train_baseline.py \
  --model vit-small \
  --data-root /path/to/tiny-imagenet-200 \
  --output-dir outputs/vit_small_baseline \
  --optimizer adamw \
  --lr 5e-4 \
  --weight-decay 0.05 \
  --batch-size 128 \
  --epochs 100 \
  --amp
```

## 5) Outputs

Each run saves the following files under `--output-dir`:

- `last.pt`: checkpoint from the latest epoch
- `best.pt`: checkpoint with highest validation accuracy
- `train_history.json`: per-epoch training/validation loss, accuracy, and LR

## 6) Why this version is more reliable

- Validation labels are mapped using the **train split class index mapping**, preventing class-index mismatches.
- Input directory checks are explicit and fail fast when data layout is wrong.
- Optimizer choice is configurable (`adamw` or `sgd`) so CNN and ViT baselines can use better defaults.

## 7) Next step: curriculum learning

You can now layer curriculum learning on top of this baseline by:

1. Defining per-sample difficulty scores (confidence, loss, or external heuristics).
2. Scheduling sampling from easy-to-hard across epochs.
3. Keeping everything else fixed to compare baseline vs curriculum fairly.
