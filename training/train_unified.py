"""
Train EfficientNet-B0 on merged unified dataset splits.

Expected dataset layout:
  TestData/merged_dataset/
    train/<class_name>/*.jpg
    val/<class_name>/*.jpg
    test/<class_name>/*.jpg

Usage:
  /home/pranam/Downloads/AgriVision/.venv/bin/python training/train_unified.py
  /home/pranam/Downloads/AgriVision/.venv/bin/python training/train_unified.py --eval
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "TestData" / "merged_dataset"
WEIGHTS_DIR = ROOT / "backend" / "model" / "weights"
BEST_MODEL = WEIGHTS_DIR / "best_model.pth"
CLASS_NAMES_JSON = WEIGHTS_DIR / "class_names.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train model on merged unified dataset")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--eval", action="store_true", help="Only evaluate best_model.pth")
    return parser.parse_args()


def build_transforms(img_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose(
        [
            transforms.Resize((img_size + 24, img_size + 24)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return train_tf, eval_tf


def create_loaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    train_tf, eval_tf = build_transforms(args.img_size)

    train_dir = args.dataset_root / "train"
    val_dir = args.dataset_root / "val"
    test_dir = args.dataset_root / "test"

    if not train_dir.exists() or not val_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(f"Expected split folders at: {args.dataset_root}")

    train_ds = datasets.ImageFolder(str(train_dir), transform=train_tf)
    class_to_idx = train_ds.class_to_idx

    val_ds = datasets.ImageFolder(str(val_dir), transform=eval_tf)
    test_ds = datasets.ImageFolder(str(test_dir), transform=eval_tf)

    if val_ds.class_to_idx != class_to_idx or test_ds.class_to_idx != class_to_idx:
        raise ValueError("Class folders differ across train/val/test splits")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_loader, class_to_idx


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> Tuple[float, float]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    preds_all = []
    labels_all = []

    with torch.set_grad_enabled(training):
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(imgs)
            loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds_all.extend(logits.argmax(dim=1).detach().cpu().tolist())
            labels_all.extend(labels.detach().cpu().tolist())

    avg_loss = total_loss / max(1, len(loader.dataset))
    macro_f1 = f1_score(labels_all, preds_all, average="macro", zero_division=0)
    return avg_loss, macro_f1


def evaluate_and_report(
    model: torch.nn.Module,
    loader: DataLoader,
    class_names: list[str],
    device: torch.device,
) -> float:
    model.eval()
    preds = []
    labels = []

    with torch.no_grad():
        for imgs, y in loader:
            logits = model(imgs.to(device, non_blocking=True))
            preds.extend(logits.argmax(dim=1).cpu().tolist())
            labels.extend(y.tolist())

    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    print(f"Final Test Macro F1: {f1:.4f}")
    print(classification_report(labels, preds, target_names=class_names, zero_division=0))
    return f1


def main() -> None:
    args = parse_args()
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, test_loader, class_to_idx = create_loaders(args)
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    CLASS_NAMES_JSON.write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    print(f"Saved classes to: {CLASS_NAMES_JSON}")
    print(f"Classes: {len(class_names)}")

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    if args.eval:
        if not BEST_MODEL.exists():
            raise FileNotFoundError(f"Model not found: {BEST_MODEL}")
        model.load_state_dict(torch.load(BEST_MODEL, map_location=device))
        evaluate_and_report(model, test_loader, class_names, device)
        return

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_f1 = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_f1 = run_epoch(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_f1={train_f1:.4f} | "
            f"val_loss={val_loss:.4f} val_f1={val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), BEST_MODEL)
            print(f"  Saved best model (val_f1={best_f1:.4f}) -> {BEST_MODEL}")

    print(f"Best Validation F1: {best_f1:.4f}")
    if BEST_MODEL.exists():
        model.load_state_dict(torch.load(BEST_MODEL, map_location=device))
    evaluate_and_report(model, test_loader, class_names, device)


if __name__ == "__main__":
    main()
