import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import timm
import torch
import torch.nn as nn
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms


# 一些基本配置参数
@dataclass
class TrainConfig:
    model_name: str
    data_root: str
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    momentum: float
    optimizer: str
    image_size: int
    seed: int


# 固定随机种子（保证结果稳定）
def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# 数据预处理（图像增强 + 标准化）
def build_transforms(image_size):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])

    return train_transform, val_transform


# 构建模型（ResNet 或 ViT）
def build_model(model_name, num_classes):
    if model_name == "resnet50":
        return models.resnet50(weights=None, num_classes=num_classes)

    if model_name == "vit-small":
        return timm.create_model(
            "vit_small_patch16_224",
            pretrained=False,
            num_classes=num_classes
        )


# 构建优化器
def build_optimizer(model, cfg):
    if cfg.optimizer == "adamw":
        return AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    return SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay
    )


# 单轮训练
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss, total_correct, total_samples = 0, 0, 0

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * targets.size(0)
        total_correct += (logits.argmax(1) == targets).sum().item()
        total_samples += targets.size(0)

    return total_loss / total_samples, total_correct / total_samples


# 验证集评估
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss, total_correct, total_samples = 0, 0, 0

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)

        logits = model(images)
        loss = criterion(logits, targets)

        total_loss += loss.item() * targets.size(0)
        total_correct += (logits.argmax(1) == targets).sum().item()
        total_samples += targets.size(0)

    return total_loss / total_samples, total_correct / total_samples


# Curriculum Learning 核心
# 控制每一轮用多少数据（从简单到困难）
def get_subset_ratio(epoch, total_epochs):
    if epoch < total_epochs * 0.25:
        return 0.25   # 前25% easiest
    elif epoch < total_epochs * 0.5:
        return 0.5
    else:
        return 1.0   # 最后全部数据



# 主函数
def main(args):
    cfg = TrainConfig(
    model_name=args.model,
    data_root=args.data_root,
    epochs=args.epochs,
    batch_size=args.batch_size,
    lr=args.lr,
    weight_decay=args.weight_decay,
    momentum=args.momentum,
    optimizer=args.optimizer,
    image_size=args.image_size,
    seed=args.seed,)
    seed_everything(cfg.seed)

    warmup_epochs = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据加载
    train_transform, val_transform = build_transforms(cfg.image_size)

    train_set = datasets.ImageFolder(Path(cfg.data_root) / "train", transform=train_transform)
    from train_baseline import TinyImageNetValDataset
    val_set = TinyImageNetValDataset(
        val_dir=str(Path(cfg.data_root) / "val"),
        class_to_idx=train_set.class_to_idx,
        transform=val_transform,
    )

    # 模型
    model = build_model(cfg.model_name, 200).to(device)

    optimizer = build_optimizer(model, cfg)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    criterion = nn.CrossEntropyLoss()
    history = []


    # Step 0: warmup训练
    print("Warmup training...")

    full_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True
    )

    for epoch in range(warmup_epochs):
        train_loss, train_acc = train_one_epoch(
            model, full_loader, criterion, optimizer, device
        )
        print(f"[Warmup {epoch+1}] train_acc={train_acc:.4f}")


    # Step 1: 计算每个样本的loss（用于判断难度）
    print("Computing sample difficulty...")

    temp_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=False)

    sample_losses = []

    model.eval()
    with torch.no_grad():
        for images, targets in temp_loader:
            images, targets = images.to(device), targets.to(device)

            logits = model(images)

            # 不取平均，每个样本一个loss
            loss = nn.CrossEntropyLoss(reduction='none')(logits, targets)

            sample_losses.extend(loss.cpu().tolist())


    # Step 2: 按loss排序（easy → hard）
    indices = list(range(len(train_set)))
    indices_sorted = sorted(indices, key=lambda i: sample_losses[i])

    # Step 3: 训练循环（逐步增加数据)
    for epoch in range(1, cfg.epochs + 1):

        ratio = get_subset_ratio(epoch, cfg.epochs)
        subset_size = int(len(train_set) * ratio)

        # 取前 subset_size 个（最简单的）
        subset_indices = indices_sorted[:subset_size]
        subset = Subset(train_set, subset_indices)

        # 注意：不能shuffle（否则课程学习失效）
        train_loader = DataLoader(subset, batch_size=cfg.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=cfg.batch_size)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step()
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": scheduler.get_last_lr()[0],
            "subset_ratio": ratio
        })

        print(f"[Epoch {epoch}] subset={ratio:.1f} "
              f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}")
    
    import json
    with open(f"results_{cfg.model_name}_bs{cfg.batch_size}.json", "w") as f:
        json.dump(history, f, indent=2)
    print("Training finished!")


# 参数解析
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--data-root", required=True)

    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=64)

    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--optimizer", default="sgd")

    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--momentum", type=float, default=0.9)

    
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())




# RESNET: batch size=64
#python scripts/train_curriculum_loss_subset.py \
#  --model resnet50 \
#  --data-root data/tiny-imagenet-200

# RESNET: batch size=128
#python scripts/train_curriculum_loss_subset.py \
#  --model resnet50 \
#  --data-root data/tiny-imagenet-200 \
#  --batch-size 128



# ViT: batch size=64
#python scripts/train_curriculum_loss_subset.py \
#  --model vit-small \
#  --data-root data/tiny-imagenet-200 \
#  --optimizer adamw \
#  --lr 5e-4\
#  --weight-decay 0.05

# ViT: batch size=128
#python scripts/train_curriculum_loss_subset.py \
#  --model vit-small \
#  --data-root data/tiny-imagenet-200 \
#  --optimizer adamw \
#  --lr 5e-4 \
#  --weight-decay 0.05\
#  --batch-size 128