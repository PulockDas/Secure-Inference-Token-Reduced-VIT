"""
Unit tests for TokenReduction module.
- Dummy tensors: shape, forward, backward
- Real images: patch embed + reduction, verify output shape
"""

import sys
from pathlib import Path

import torch

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.token_reduction import TokenReduction


def test_shape_forward_backward():
    """Dummy tensors: verify (B, N, D) -> (B, K, D), forward and backward pass."""
    B, N, D = 2, 197, 768
    K = 97
    module = TokenReduction(dim=D, num_output_tokens=K)
    x = torch.randn(B, N, D, requires_grad=True)
    out = module(x)
    assert out.shape == (B, K, D), f"Expected (B,K,D)={(B,K,D)}, got {out.shape}"
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "Backward failed"


def test_passthrough():
    """K >= N: passthrough, no reduction."""
    B, N, D = 2, 197, 768
    module = TokenReduction(dim=D, num_output_tokens=197)
    x = torch.randn(B, N, D)
    out = module(x)
    assert out.shape == (B, 197, D)
    assert torch.allclose(out, x)


def test_cls_only():
    """K=1: only CLS token."""
    B, N, D = 2, 197, 768
    module = TokenReduction(dim=D, num_output_tokens=1)
    x = torch.randn(B, N, D)
    out = module(x)
    assert out.shape == (B, 1, D)
    assert torch.allclose(out, x[:, :1, :])


def test_multi_k():
    """Various K values: 97, 49, 25."""
    B, N, D = 2, 197, 768
    for K in [97, 49, 25]:
        module = TokenReduction(dim=D, num_output_tokens=K)
        x = torch.randn(B, N, D)
        out = module(x)
        assert out.shape == (B, K, D), f"K={K}: expected (B,K,D), got {out.shape}"
        # CLS preserved
        assert torch.allclose(out[:, :1, :], x[:, :1, :])


def test_real_images():
    """Real images: ViT patch embed + TokenReduction, verify output shape."""
    import timm

    B, D = 2, 768
    K = 97
    img_size = 224
    num_patches = (img_size // 16) ** 2
    N = 1 + num_patches  # CLS + patches

    # Patch embed + CLS from timm ViT-B/16
    vit = timm.create_model("vit_base_patch16_224", pretrained=False)
    patch_embed, cls_token = vit.patch_embed, vit.cls_token

    images = torch.randn(B, 3, img_size, img_size)
    patches = patch_embed(images)  # (B, num_patches, D)
    cls = cls_token.expand(B, -1, -1)
    tokens = torch.cat([cls, patches], dim=1)  # (B, N, D)
    assert tokens.shape == (B, N, D)

    reduction = TokenReduction(dim=D, num_output_tokens=K)
    out = reduction(tokens)
    assert out.shape == (B, K, D), f"Expected (B,K,D)={(B,K,D)}, got {out.shape}"
    assert torch.allclose(out[:, :1, :], tokens[:, :1, :]), "CLS not preserved"


if __name__ == "__main__":
    test_shape_forward_backward()
    print("test_shape_forward_backward passed")
    test_passthrough()
    print("test_passthrough passed")
    test_cls_only()
    print("test_cls_only passed")
    test_multi_k()
    print("test_multi_k passed")
    test_real_images()
    print("test_real_images passed")
    print("All tests passed.")
