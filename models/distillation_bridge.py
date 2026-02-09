"""
Training-only bridge: projects student features to teacher dimension for distillation.
NOT part of the inference graph — use only when computing distillation loss.
When saving the student for deployment, save only the student state_dict; do not save the bridge.

Usage (training):
  student_feat = student(x)           # (B, N, student_dim), e.g. 384
  teacher_feat = teacher(x)           # (B, N, teacher_dim), e.g. 768
  student_for_loss = bridge(student_feat)  # (B, N, 768)
  loss = F.mse_loss(student_for_loss, teacher_feat.detach())

Saving for inference:
  torch.save(student.state_dict(), "student.pt")   # do NOT include bridge
"""

import torch
import torch.nn as nn


def student_embed_dim(teacher_embed_dim: int) -> int:
    """Student embed dim as N//2 of teacher (e.g. 768 → 384)."""
    return teacher_embed_dim // 2


class DistillationBridge(nn.Module):
    """
    Linear(student_dim → teacher_dim). For training only.

    Use when matching student intermediate features to teacher for distillation.
    Student runs in smaller dim (e.g. teacher_embed_dim // 2); teacher is unchanged.
    At inference, only the student (small dim) runs; this module must not be loaded.
    """

    def __init__(self, student_embed_dim: int, teacher_embed_dim: int, bias: bool = True):
        super().__init__()
        self.student_embed_dim = student_embed_dim
        self.teacher_embed_dim = teacher_embed_dim
        self.proj = nn.Linear(student_embed_dim, teacher_embed_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, student_embed_dim) — student features
        returns: (B, N, teacher_embed_dim) — for comparison with teacher features
        """
        return self.proj(x)


def build_bridge_for_teacher(teacher_embed_dim: int, use_half_dim: bool = True) -> DistillationBridge:
    """
    Build a bridge so student (smaller dim) can be distilled against teacher.

    Args:
        teacher_embed_dim: Teacher feature dim (e.g. 768).
        use_half_dim: If True, student_embed_dim = teacher_embed_dim // 2.

    Returns:
        Bridge: student_embed_dim → teacher_embed_dim. Training-only; do not save for inference.
    """
    student_dim = student_embed_dim(teacher_embed_dim) if use_half_dim else teacher_embed_dim
    return DistillationBridge(student_dim, teacher_embed_dim)
