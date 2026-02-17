"""
Forward pass test for Student ViT. Run on PC: python tests/test_student.py
"""

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Import without pulling teacher (timm)
import importlib.util
_spec = importlib.util.spec_from_file_location("student", PROJECT_ROOT / "models" / "student.py")
# student.py imports from .activations, .he_attention, .norm, .token_reduction -> need package
# So we need to load from models package; that will pull teacher. Use models.student after adding path.
from models.student import StudentViT, get_student_vit, PatchEmbed, StudentBlock


def test_patch_embed():
    """PatchEmbed: (B, 3, 224, 224) -> (B, 196, embed_dim)."""
    embed = PatchEmbed(img_size=224, patch_size=16, embed_dim=384)
    x = torch.randn(2, 3, 224, 224)
    out = embed(x)
    assert out.shape == (2, 196, 384)


def test_student_forward_shape():
    """StudentViT forward: (B, 3, 224, 224) -> (B, num_classes)."""
    model = get_student_vit(num_classes=5, num_output_tokens=97, norm_mode="layernorm")
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    assert logits.shape == (2, 5)


def test_student_forward_backward():
    """Forward and backward pass."""
    model = get_student_vit(num_classes=5, num_output_tokens=49, norm_mode="none")
    x = torch.randn(2, 3, 224, 224, requires_grad=True)
    logits = model(x)
    loss = logits.sum()
    loss.backward()
    assert x.grad is not None


def test_student_k_values():
    """Forward works for K in {197, 97, 49, 25}."""
    for K in [197, 97, 49, 25]:
        model = get_student_vit(num_classes=5, num_output_tokens=K, embed_dim=384)
        x = torch.randn(1, 3, 224, 224)
        logits = model(x)
        assert logits.shape == (1, 5), f"K={K} failed"


def test_student_norm_modes():
    """Both norm_mode 'none' and 'layernorm' run without error."""
    x = torch.randn(1, 3, 224, 224)
    for mode in ("none", "layernorm"):
        model = get_student_vit(num_classes=5, num_output_tokens=97, norm_mode=mode)
        logits = model(x)
        assert logits.shape == (1, 5)


if __name__ == "__main__":
    test_patch_embed()
    print("test_patch_embed passed")
    test_student_forward_shape()
    print("test_student_forward_shape passed")
    test_student_forward_backward()
    print("test_student_forward_backward passed")
    test_student_k_values()
    print("test_student_k_values passed")
    test_student_norm_modes()
    print("test_student_norm_modes passed")
    print("All student tests passed.")
