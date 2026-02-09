"""
Tests for modular norm (none / affine / layernorm). Run on PC: python tests/test_norm.py
"""

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import importlib.util
_spec = importlib.util.spec_from_file_location("norm", PROJECT_ROOT / "models" / "norm.py")
_norm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_norm)
AffineNorm = _norm.AffineNorm
create_norm = _norm.create_norm


def test_none_identity():
    """none: output equals input, shape (B,N,D)."""
    norm = create_norm("none", dim=384)
    x = torch.randn(2, 97, 384)
    out = norm(x)
    assert out.shape == x.shape
    assert torch.allclose(out, x)


def test_affine_shape_and_backward():
    """affine: (B,N,D) -> (B,N,D), backward."""
    norm = create_norm("affine", dim=384)
    x = torch.randn(2, 97, 384, requires_grad=True)
    out = norm(x)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None


def test_affine_init_near_identity():
    """AffineNorm init: gamma=1, beta=0 so residual is preserved at start."""
    norm = create_norm("affine", dim=64)
    x = torch.randn(4, 10, 64)
    out = norm(x)
    assert torch.allclose(out, x, atol=1e-5), "Affine init should be identity"


def test_layernorm_option():
    """layernorm mode returns LayerNorm, shape and backward work."""
    norm = create_norm("layernorm", dim=384)
    assert isinstance(norm, torch.nn.LayerNorm)
    x = torch.randn(2, 97, 384, requires_grad=True)
    out = norm(x)
    assert out.shape == x.shape
    out.sum().backward()


def test_residual_structure():
    """Residual x + sublayer(norm(x)) stays valid for none and affine."""
    B, N, D = 2, 10, 64
    for mode in ("none", "affine"):
        norm = create_norm(mode, D)
        x = torch.randn(B, N, D, requires_grad=True)
        normed = norm(x)
        # Simulate sublayer (e.g. attention): just 0.1 * normed
        sub = 0.1 * normed
        out = x + sub
        assert out.shape == (B, N, D)
        out.sum().backward()
        assert x.grad is not None


if __name__ == "__main__":
    test_none_identity()
    print("test_none_identity passed")
    test_affine_shape_and_backward()
    print("test_affine_shape_and_backward passed")
    test_affine_init_near_identity()
    print("test_affine_init_near_identity passed")
    test_layernorm_option()
    print("test_layernorm_option passed")
    test_residual_structure()
    print("test_residual_structure passed")
    print("All norm tests passed.")
