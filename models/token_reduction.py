"""
Learnable token reduction module for student ViT.
Input (B, N, D) -> output (B, K, D). Preserves CLS token.
Uses K-1 learnable queries that cross-attend to patch tokens; fully differentiable.
"""

import math

import torch
import torch.nn as nn


class TokenReduction(nn.Module):
    """
    Reduces N tokens to K tokens by preserving CLS and cross-attending K-1 learnable
    queries to patch tokens. Fully differentiable; no hard pruning.

    Input: (B, N, D) - N = 1 + num_patches (CLS + patches)
    Output: (B, K, D) - K <= N
    """

    def __init__(self, dim: int, num_output_tokens: int):
        """
        Args:
            dim: Token embedding dimension D.
            num_output_tokens: Target number of output tokens K (e.g. 97, 49, 25, 197).
        """
        super().__init__()
        self.dim = dim
        self.num_output_tokens = num_output_tokens
        # K-1 learnable query vectors for cross-attention to patch tokens
        num_queries = max(0, num_output_tokens - 1)
        self.queries = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)
        self.scale = dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) - N = 1 + num_patches

        Returns:
            (B, K, D) - K = num_output_tokens
        """
        B, N, D = x.shape
        K = self.num_output_tokens

        if K >= N:
            # No reduction: passthrough (e.g. K=197, N=197)
            return x[:, :K, :]  # pad or truncate if K > N

        if K == 1:
            # Only CLS
            return x[:, :1, :]

        cls = x[:, :1, :]  # (B, 1, D)
        patches = x[:, 1:, :]  # (B, N-1, D)

        # Cross-attention: K-1 queries attend to patches
        q = self.queries.expand(B, -1, -1)  # (B, K-1, D)
        attn = torch.matmul(q, patches.transpose(-2, -1)) * self.scale  # (B, K-1, N-1)
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, patches)  # (B, K-1, D)

        return torch.cat([cls, out], dim=1)  # (B, K, D)
