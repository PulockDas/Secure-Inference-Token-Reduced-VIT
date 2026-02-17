"""
Knowledge distillation: train student with KL (temperature) + optional hard-label loss.
Logging and checkpoints are saved so training curves can be used for plots.
"""

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import get_student_vit


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 4.0,
) -> torch.Tensor:
    """
    Temperature-scaled KD: KL(softmax(z_s/T), softmax(z_t/T)) * (T*T).
    Default T=4.0 (configurable). Do not squash or clip logits before KD so gradients
    flow correctly and the student can match the teacher distribution.
    """
    s_soft = F.log_softmax(student_logits / temperature, dim=1)
    t_soft = F.softmax(teacher_logits / temperature, dim=1)
    kl = F.kl_div(s_soft, t_soft, reduction="batchmean")
    return kl * (temperature * temperature)


def train_student(
    student: nn.Module,
    teacher: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 30,
    lr: float = 1e-4,
    temperature: float = 4.0,
    alpha: float = 0.2,
    use_hard_labels: bool = True,
    log_path: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    run_config: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    """
    Train student via knowledge distillation. Teacher is frozen.
    Loss = alpha * soft_loss (KL with temperature) + (1 - alpha) * hard_loss (CE) when use_hard_labels.

    Saves per-epoch metrics to CSV (for plots) and best student checkpoint.
    run_config is saved as JSON alongside logs for reproducibility.
    """
    student = student.to(device)
    teacher = teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(student.parameters(), lr=lr)
    num_classes = train_loader.dataset.num_classes
    ce = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    if checkpoint_dir:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    # CSV columns: all metrics needed for plotting + logits std for KD verification
    fieldnames = [
        "epoch", "wall_time_s",
        "train_loss", "train_soft_loss", "train_hard_loss", "train_acc",
        "val_loss", "val_soft_loss", "val_hard_loss", "val_acc",
        "student_logits_std", "teacher_logits_std", "teacher_std_over_T",
    ]
    rows: list = []

    for epoch in range(epochs):
        t0 = time.perf_counter()
        student.train()
        train_loss = 0.0
        train_soft = 0.0
        train_hard = 0.0
        train_correct = 0
        train_total = 0
        train_batches = 0
        # Accumulate for logits std (verify student_std ≈ teacher_std / T)
        sum_s, sum_sq_s = 0.0, 0.0
        sum_t, sum_sq_t = 0.0, 0.0
        logits_count = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                teacher_logits = teacher(images)
            student_logits = student(images)

            soft_loss = distillation_loss(student_logits, teacher_logits, temperature)
            if use_hard_labels:
                hard_loss = ce(student_logits, labels)
                loss = alpha * soft_loss + (1.0 - alpha) * hard_loss
            else:
                hard_loss = torch.tensor(0.0, device=device)
                loss = soft_loss

            # Skip step if loss is invalid (avoids corrupting weights with NaN/Inf)
            if not torch.isfinite(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            # KD stability: clip gradients so updates don't explode and corrupt the student.
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_soft += soft_loss.item()
            train_hard += hard_loss.item() if use_hard_labels else 0.0
            preds = student_logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            train_batches += 1
            # Accumulate for logits std (no grad needed)
            with torch.no_grad():
                n_el = student_logits.numel()
                sum_s += student_logits.sum().item()
                sum_sq_s += (student_logits ** 2).sum().item()
                sum_t += teacher_logits.sum().item()
                sum_sq_t += (teacher_logits ** 2).sum().item()
                logits_count += n_el

        n = max(1, train_batches)
        train_loss /= n
        train_soft /= n
        train_hard /= n if use_hard_labels else 1.0
        train_acc = train_correct / max(1, train_total)

        # Validation
        student.eval()
        val_loss = 0.0
        val_soft = 0.0
        val_hard = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                teacher_logits = teacher(images)
                student_logits = student(images)
                soft_loss = distillation_loss(student_logits, teacher_logits, temperature)
                hard_loss = ce(student_logits, labels)
                if use_hard_labels:
                    loss = alpha * soft_loss + (1.0 - alpha) * hard_loss
                else:
                    loss = soft_loss
                val_loss += loss.item()
                val_soft += soft_loss.item()
                val_hard += hard_loss.item()
                preds = student_logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        nv = len(val_loader)
        val_loss /= nv
        val_soft /= nv
        val_hard /= nv
        val_acc = val_correct / val_total

        # Logits std: verify student_std ≈ teacher_std / T (temperature-scaled KD)
        if logits_count > 0:
            mean_s = sum_s / logits_count
            mean_t = sum_t / logits_count
            var_s = (sum_sq_s / logits_count) - (mean_s * mean_s)
            var_t = (sum_sq_t / logits_count) - (mean_t * mean_t)
            std_s = (var_s ** 0.5) if var_s > 0 else 0.0
            std_t = (var_t ** 0.5) if var_t > 0 else 0.0
            teacher_std_over_T = std_t / temperature if temperature != 0 else 0.0
        else:
            std_s = std_t = teacher_std_over_T = 0.0

        wall_time = time.perf_counter() - t0
        row = {
            "epoch": epoch + 1,
            "wall_time_s": round(wall_time, 2),
            "train_loss": round(train_loss, 6),
            "train_soft_loss": round(train_soft, 6),
            "train_hard_loss": round(train_hard, 6),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 6),
            "val_soft_loss": round(val_soft, 6),
            "val_hard_loss": round(val_hard, 6),
            "val_acc": round(val_acc, 4),
            "student_logits_std": round(std_s, 4),
            "teacher_logits_std": round(std_t, 4),
            "teacher_std_over_T": round(teacher_std_over_T, 4),
        }
        rows.append(row)
        print(
            f"Epoch {epoch+1}/{epochs} time={wall_time:.1f}s "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        print(
            f"  logits_std: student={std_s:.4f} teacher={std_t:.4f}  "
            f"(verify student_std ≈ teacher_std/T={teacher_std_over_T:.4f})"
        )

        if val_acc > best_val_acc and checkpoint_dir:
            best_val_acc = val_acc
            ckpt_path = Path(checkpoint_dir) / "student_best.pt"
            state = student.state_dict()
            ckpt = {
                "student_state_dict": state,
                "epoch": epoch + 1,
                "val_acc": val_acc,
                "num_classes": num_classes,
                "norm_mode": "layernorm" if "blocks.0.norm1.weight" in state else "affine",
            }
            if run_config is not None:
                ckpt["num_output_tokens"] = run_config.get("num_output_tokens", 97)
                ckpt["embed_dim"] = run_config.get("embed_dim", 384)
                ckpt["depth"] = run_config.get("depth", 6)
                ckpt["num_heads"] = run_config.get("num_heads", 6)
            torch.save(ckpt, ckpt_path)
            print(f"  -> saved best ({val_acc:.4f})")

    if log_path:
        with open(log_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    if run_config is not None and (log_path or checkpoint_dir):
        base = Path(log_path).parent if log_path else Path(checkpoint_dir)
        base.mkdir(parents=True, exist_ok=True)
        k = run_config.get("num_output_tokens")
        name = f"distill_run_config_K{k}.json" if k is not None else "distill_run_config.json"
        config_path = base / name
        with open(config_path, "w") as f:
            json.dump(run_config, f, indent=2)

    return student


def load_student_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    num_output_tokens: int = 97,
    embed_dim: int = 384,
    depth: int = 6,
    num_heads: int = 6,
    norm_mode: Optional[str] = None,
) -> nn.Module:
    """Load student from saved checkpoint. Uses checkpoint's config when available."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    num_classes = ckpt["num_classes"]
    norm_mode = norm_mode or ckpt.get("norm_mode", "layernorm")
    num_output_tokens = ckpt.get("num_output_tokens", num_output_tokens)
    embed_dim = ckpt.get("embed_dim", embed_dim)
    depth = ckpt.get("depth", depth)
    num_heads = ckpt.get("num_heads", num_heads)
    student = get_student_vit(
        num_classes=num_classes,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        num_output_tokens=num_output_tokens,
        norm_mode=norm_mode,
    )
    student.load_state_dict(ckpt["student_state_dict"])
    return student.to(device)
