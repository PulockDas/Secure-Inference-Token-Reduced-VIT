"""
Modular norm for student transformer blocks: no norm, affine-only, or LayerNorm.
LayerNorm uses mean/variance and division (not HE-friendly). Affine-only is scale + shift (HE-friendly).
Residual connections stay valid: out = x + sublayer(norm(x)); norm is (B,N,D)->(B,N,D).
"""

from typing import Literal

import torch
import torch.nn as nn


class AffineNorm(nn.Module):
    """
    Per-dimension scale and shift only: out = gamma * x + beta.
    No mean/variance or division — HE-friendly (only + and *).
    gamma and beta are learnable (nn.Parameter); we only init to 1 and 0 so the
    layer starts as identity; they are updated by backprop during training.
    Use this in place of LayerNorm when targeting encrypted inference.
    """

    def __init__(self, dim: int):
        super().__init__()
        # Learnable; init 1/0 so initial forward is identity
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, N, D) * (D,) + (D,) -> (B, N, D)
        return x * self.gamma + self.beta


class _IdentityNorm(nn.Module):
    """No normalization; pass-through so residual is x + sublayer(x)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def create_norm(
    mode: Literal["none", "affine", "layernorm"],
    dim: int,
    eps: float = 1e-6,
) -> nn.Module:
    """
    Factory for drop-in norm in student transformer blocks.

    Design choice:
    - none: No norm. Residual = x + sublayer(x). Simplest; no extra params or ops.
    - affine: HE-friendly scale/shift only. Residual = x + sublayer(affine(x)).
      Init preserves residual at start of training.
    - layernorm: Full LayerNorm (for plaintext ablation only; not HE-friendly).

    Args:
        mode: "none" | "affine" | "layernorm"
        dim: Token dimension D (last dim of (B, N, D)).
        eps: Only for layernorm (numerical stability).

    Returns:
        Module that maps (B, N, D) -> (B, N, D).
    """
    if mode == "none":
        return _IdentityNorm()
    if mode == "affine":
        return AffineNorm(dim)
    if mode == "layernorm":
        return nn.LayerNorm(dim, eps=eps)
    raise ValueError(f"Unknown norm mode: {mode!r}. Use 'none', 'affine', or 'layernorm'.")
