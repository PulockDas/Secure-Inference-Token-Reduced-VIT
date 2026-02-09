"""
HE-friendly attention: no softmax, exp, or division; only addition and multiplication.
Compatible with token reduction output (B, K, D). Drop-in for student ViT.
"""

import math
import torch
import torch.nn as nn


class HEAttention(nn.Module):
    """
    Attention using only linear combinations and matrix products.
    Replaces softmax(QK^T/sqrt(d)) @ V with: scale * (Q K^T) @ V.
    Scale = 1/sqrt(d_k) is a constant (multiplication), so HE-friendly.

    Input/Output: (B, N, D) — e.g. (B, K, D) after token reduction.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 1,
        bias: bool = True,
    ):
        """
        Args:
            embed_dim: Token dimension D (must be divisible by num_heads).
            num_heads: Number of heads (each head uses embed_dim // num_heads).
            bias: Use bias in Q,K,V and out projections (addition is HE-friendly).
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.scale = self.head_dim ** -0.5  # constant; applied as multiplication

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        residual: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) token sequence (e.g. after token reduction).
            residual: If True, add input to output (residual connection).

        Returns:
            (B, N, D) same shape as input.
        """
        B, N, D = x.shape
        assert D == self.embed_dim

        q = self.q_proj(x)  # (B, N, D)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Scale Q and K for stability (constant multiplication only)
        q = q * self.scale
        k = k * self.scale

        # Reshape for multi-head: (B, N, D) -> (B, N, num_heads, head_dim) -> (B, num_heads, N, head_dim)
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Scores: (B, num_heads, N, head_dim) @ (B, num_heads, head_dim, N) -> (B, num_heads, N, N)
        # No softmax — HE-friendly linear attention
        scores = torch.matmul(q, k.transpose(-2, -1))
        # Output: (B, num_heads, N, N) @ (B, num_heads, N, head_dim) -> (B, num_heads, N, head_dim)
        out = torch.matmul(scores, v)

        # (B, num_heads, N, head_dim) -> (B, N, D)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)

        if residual:
            out = out + x
        return out
