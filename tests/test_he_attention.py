"""
Unit tests for HE-friendly attention.
- Shape, forward, backward; compatibility with token reduction output (B, K, D).
"""

import sys
from pathlib import Path

import torch

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.he_attention import HEAttention


def test_shape_forward_backward():
    """(B, N, D) -> (B, N, D), forward and backward."""
    B, N, D = 2, 97, 768
    attn = HEAttention(embed_dim=D, num_heads=1)
    x = torch.randn(B, N, D, requires_grad=True)
    out = attn(x, residual=True)
    assert out.shape == (B, N, D), f"Expected (B,N,D)={(B,N,D)}, got {out.shape}"
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "Backward failed"


def test_with_residual_off():
    """Residual=False: output is only attention output."""
    B, N, D = 2, 49, 384
    attn = HEAttention(embed_dim=D, num_heads=1)
    x = torch.randn(B, N, D)
    out = attn(x, residual=False)
    assert out.shape == (B, N, D)
    # With residual=False, output is not x + something
    assert not torch.allclose(out, x)


def test_multi_head():
    """Multi-head: same shape, divisible embed_dim."""
    B, N, D = 2, 25, 384
    attn = HEAttention(embed_dim=D, num_heads=6)
    x = torch.randn(B, N, D)
    out = attn(x, residual=True)
    assert out.shape == (B, N, D)


def test_compatible_with_token_reduction_k():
    """Output of token reduction (B, K, D) is valid input; K in {197, 97, 49, 25}."""
    B, D = 2, 768
    attn = HEAttention(embed_dim=D, num_heads=12)
    for K in [197, 97, 49, 25]:
        x = torch.randn(B, K, D)
        out = attn(x, residual=True)
        assert out.shape == (B, K, D), f"K={K}: expected (B,K,D), got {out.shape}"


def test_no_nan_inf():
    """Stability: no NaN/Inf for typical input scale."""
    B, N, D = 4, 97, 768
    attn = HEAttention(embed_dim=D, num_heads=12)
    x = torch.randn(B, N, D) * 0.1
    out = attn(x, residual=True)
    assert torch.isfinite(out).all(), "Output contained NaN or Inf"


if __name__ == "__main__":
    test_shape_forward_backward()
    print("test_shape_forward_backward passed")
    test_with_residual_off()
    print("test_with_residual_off passed")
    test_multi_head()
    print("test_multi_head passed")
    test_compatible_with_token_reduction_k()
    print("test_compatible_with_token_reduction_k passed")
    test_no_nan_inf()
    print("test_no_nan_inf passed")
    print("All tests passed.")
