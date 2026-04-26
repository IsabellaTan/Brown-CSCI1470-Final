import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import wandb

import timm
import torch
import torch.nn as nn
from PIL import Image
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from tqdm.auto import tqdm


@dataclass
class TrainConfig:
    model_name: str
    data_root: str
    output_dir: str
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    momentum: float
    optimizer: str
    num_workers: int
    image_size: int
    seed: int
    amp: bool
    progress: bool
    wandb_project: str
    wandb_run_name: str
    wandb_disable: bool


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Keep deterministic=False for speed. Toggle if strict reproducibility is needed.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


class TinyImageNetValDataset(Dataset):
    """Tiny-ImageNet validation split loader based on val_annotations.txt."""

    def __init__(self, val_dir: str, class_to_idx: Dict[str, int], transform=None) -> None:
        self.transform = transform
        self.class_to_idx = class_to_idx

        val_root = Path(val_dir)
        image_dir = val_root / "images"
        annotation_path = val_root / "val_annotations.txt"

        if not image_dir.exists() or not annotation_path.exists():
            raise FileNotFoundError(
                f"Expected both '{image_dir}' and '{annotation_path}' to exist."
            )

        self.samples: List[Tuple[Path, int]] = []
        with annotation_path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                image_name, class_name = parts[0], parts[1]

                if class_name not in self.class_to_idx:
                    raise KeyError(
                        f"Class '{class_name}' in validation annotations is not present in train class_to_idx."
                    )

                self.samples.append((image_dir / image_name, self.class_to_idx[class_name]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, target = self.samples[idx]

        if not image_path.exists():
            raise FileNotFoundError(f"Missing validation image: {image_path}")

        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return image, target


def build_transforms(image_size: int):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.15)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )

    return train_transform, val_transform


def build_model(model_name: str, num_classes: int) -> nn.Module:
    model_name = model_name.lower()

    if model_name == "resnet50":
        return models.resnet50(weights=None, num_classes=num_classes)

    if model_name in {"vit-small", "vit_small", "vit_small_patch16_224"}:
        return timm.create_model(
            "vit_small_patch16_224",
            pretrained=False,
            num_classes=num_classes,
        )

    raise ValueError("Unsupported model_name. Use 'resnet50' or 'vit-small'.")


def build_optimizer(model: nn.Module, cfg: TrainConfig):
    if cfg.optimizer == "adamw":
        return AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    if cfg.optimizer == "sgd":
        return SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )

    raise ValueError("Unsupported optimizer. Use 'adamw' or 'sgd'.")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        total_loss += loss.item() * targets.size(0)
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_samples += targets.size(0)

    return total_loss / total_samples, total_correct / total_samples


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    epoch: int,
    total_epochs: int,
    show_progress: bool,
):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    iterator = loader
    if show_progress and tqdm is not None:
        iterator = tqdm(loader, desc=f"Train {epoch}/{total_epochs}", leave=False)

    for images, targets in iterator:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * targets.size(0)
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_samples += targets.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate_with_progress(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    show_progress: bool,
):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    iterator = loader
    if show_progress and tqdm is not None:
        iterator = tqdm(loader, desc=f"Val   {epoch}/{total_epochs}", leave=False)

    for images, targets in iterator:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        total_loss += loss.item() * targets.size(0)
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_samples += targets.size(0)

    return total_loss / total_samples, total_correct / total_samples


def maybe_init_wandb(
    cfg: TrainConfig,
    train_set: datasets.ImageFolder,
    val_set: Dataset,
    device: torch.device,
) -> bool:
    if cfg.wandb_disable:
        print("W&B disabled by --wandb-disable.")
        return False

    if wandb is None:
        print("wandb is not installed. Continuing without W&B logging.")
        return False

    auto_run_name = f"{cfg.model_name}_ep{cfg.epochs}_bs{cfg.batch_size}_lr{cfg.lr:g}"
    run_name = cfg.wandb_run_name or auto_run_name
    data_info = {
        "dataset_name": "tiny-imagenet-200",
        "data_root": str(Path(cfg.data_root).resolve()),
        "num_train_samples": len(train_set),
        "num_val_samples": len(val_set),
        "num_classes": len(train_set.classes),
        "device": str(device),
    }
    run_config = {**asdict(cfg), **data_info}

    try:
        wandb.init(
            project=cfg.wandb_project,
            name=run_name,
            config=run_config,
            dir=cfg.output_dir,
        )
        print(f"W&B initialized: project={cfg.wandb_project}, run={run_name}")
        return True
    except Exception as exc:
        print(f"W&B init failed: {exc}. Continuing without W&B.")
        return False


def main(args):
    cfg = TrainConfig(
        model_name=args.model,
        data_root=args.data_root,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        optimizer=args.optimizer,
        num_workers=args.num_workers,
        image_size=args.image_size,
        seed=args.seed,
        amp=args.amp,
        progress=not args.no_progress,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_disable=args.wandb_disable,
    )

    data_root = Path(cfg.data_root)
    train_path = data_root / "train"
    val_path = data_root / "val"

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"Expected Tiny-ImageNet root with 'train/' and 'val/' at: {data_root}"
        )

    seed_everything(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_transform, val_transform = build_transforms(cfg.image_size)

    train_set = datasets.ImageFolder(train_path, transform=train_transform)
    if len(train_set.classes) != 200:
        raise RuntimeError(
            f"Expected 200 classes in Tiny-ImageNet train split, got {len(train_set.classes)}."
        )

    val_set = TinyImageNetValDataset(
        val_dir=str(val_path),
        class_to_idx=train_set.class_to_idx,
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg.model_name, num_classes=200).to(device)

    wandb_enabled = maybe_init_wandb(cfg, train_set, val_set, device)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, cfg)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device=device.type, enabled=use_amp)

    if cfg.progress and tqdm is None:
        print("tqdm is not installed. Falling back to epoch-only logs.")

    best_acc = 0.0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            epoch=epoch,
            total_epochs=cfg.epochs,
            show_progress=cfg.progress,
        )

        val_loss, val_acc = evaluate_with_progress(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            total_epochs=cfg.epochs,
            show_progress=cfg.progress,
        )
        scheduler.step()

        epoch_log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(epoch_log)
        print(
            f"Epoch {epoch:03d}/{cfg.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"lr={scheduler.get_last_lr()[0]:.6g}"
        )
        print(json.dumps(epoch_log, ensure_ascii=False))

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
            "config": asdict(cfg),
            "history": history,
        }
        torch.save(checkpoint, output_dir / "last.pt")

        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint["best_acc"] = best_acc
            torch.save(checkpoint, output_dir / "best.pt")

        if wandb_enabled:
            wandb.log({**epoch_log, "best_acc": best_acc})

    with (output_dir / "train_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print(f"Finished training. Best val acc: {best_acc:.4f}")

    if wandb_enabled:
        wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tiny-ImageNet baseline pretraining for ResNet50 and ViT-Small"
    )
    parser.add_argument("--model", type=str, default="resnet50", choices=["resnet50", "vit-small"])
    parser.add_argument("--data-root", type=str, required=True, help="Path to tiny-imagenet-200")
    parser.add_argument("--output-dir", type=str, default="outputs/baseline")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision training on CUDA.")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="tiny_imagenet_baseline",
        help="W&B project name.",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default="",
        help="W&B run name. If empty, uses: model_ep{epochs}_bs{batch_size}_lr{lr}.",
    )
    parser.add_argument(
        "--wandb-disable",
        action="store_true",
        help="Disable W&B logging.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
