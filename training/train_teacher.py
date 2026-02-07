"""
Teacher fine-tuning: training loop with logging (loss, accuracy, time) and best checkpoint save.
Evaluation: load best checkpoint and report test-set classification results.
"""

import csv
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import get_teacher_vit


def train_teacher(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 10,
    lr: float = 1e-4,
    log_path: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
) -> nn.Module:
    """
    Fine-tune teacher ViT on train_loader, validate on val_loader.
    Logs epoch, time, loss, accuracy to CSV. Saves best checkpoint by val accuracy.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    num_classes = train_loader.dataset.num_classes

    best_val_acc = 0.0
    if checkpoint_dir:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        best_path = Path(checkpoint_dir) / "teacher_best.pt"

    rows = []
    for epoch in range(epochs):
        t0 = time.perf_counter()
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        wall_time = time.perf_counter() - t0
        row = {
            "epoch": epoch + 1,
            "wall_time_s": round(wall_time, 2),
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
        }
        rows.append(row)
        print(
            f"Epoch {epoch+1}/{epochs} "
            f"time={wall_time:.1f}s "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc and checkpoint_dir:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "val_acc": val_acc,
                    "num_classes": num_classes,
                },
                best_path,
            )
            print(f"  -> saved best ({val_acc:.4f})")

    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)

    return model


def load_teacher_checkpoint(checkpoint_path: str, device: torch.device) -> nn.Module:
    """Load teacher model from saved checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    num_classes = ckpt["num_classes"]
    model = get_teacher_vit(num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device)


def evaluate_teacher(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    class_names: list,
) -> Dict:
    """Evaluate teacher on test set. Returns dict with test_acc and per_class_acc."""
    model.eval()
    correct = 0
    total = 0
    num_classes = len(class_names)
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            for c in range(num_classes):
                mask = labels == c
                class_total[c] += mask.sum().item()
                class_correct[c] += ((preds == labels) & mask).sum().item()
    test_acc = correct / total
    per_class_acc = [
        class_correct[c] / class_total[c] if class_total[c] > 0 else 0
        for c in range(num_classes)
    ]
    return {"test_acc": test_acc, "per_class_acc": per_class_acc, "class_names": class_names}
