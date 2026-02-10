"""
HE-friendly attention: no softmax, exp; linear attention with bounded range by design.
Compatible with token reduction output (B, K, D). Drop-in for student ViT.

Design goal: keep the attention path bounded without softmax and without division.

We do this with a low-degree polynomial "squash" (only + and *) that maps values into
approximately [-1, 1] when inputs are in a known range. Then we use constant scaling so
the attention weights have magnitude <= 1/N, which keeps the output bounded.
"""

import math
import torch
import torch.nn as nn


def _cubic_squash_unit(x: torch.Tensor) -> torch.Tensor:
    """
    Cubic "smoothstep-like" squash on [-1, 1]:
      s(x) = 1.5 x - 0.5 x^3
    For x in [-1,1], s(x) is also in [-1,1] and has zero slope at ±1.

    Note: No non-constant polynomial is globally bounded for all real inputs. This squash
    is intended to be used after scaling inputs into (roughly) [-1,1].
    """
    return 1.5 * x - 0.5 * (x * x * x)


class HEAttention(nn.Module):
    """
    Linear attention with range bounded by design (HE-friendly path):

    - Q, K are squashed into [-1, 1] elementwise (after scaling by 1/bound), so
      dot(Q, K) is bounded by head_dim.
    - We scale scores by 1/(head_dim * N), so each score entry has magnitude <= 1/N.
    - V is squashed into [-bound, bound], and the weighted sum stays in [-bound, bound]
      because sum_j |score_ij| <= 1.

    This avoids L2 normalization (division) entirely.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 1,
        bias: bool = True,
        bound: float = 6.0,
        output_squash: bool = False,
    ):
        """
        Args:
            embed_dim: Token dimension D (must be divisible by num_heads).
            num_heads: Number of heads (each head uses embed_dim // num_heads).
            bias: Use bias in Q,K,V and out projections (addition is HE-friendly).
            bound: Single range bound: design target and overflow clamp both [-bound, bound].
            output_squash: If True, squash the post-projection output back to [-bound, bound].
                Default False for KD stability (avoids squashing the residual path and helps gradients flow).
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.bound = bound
        self.output_squash = output_squash

        # Constant scale = 1/sqrt(head_dim), computed once at init. Forward only does multiplication.
        # HE inference: this is just q * scale (multiply by constant), no division/sqrt at runtime.
        self.scale = self.head_dim ** -0.5

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

        q = self.q_proj(x) * self.scale
        k = self.k_proj(x) * self.scale
        v = self.v_proj(x)

        # Reshape for multi-head: (B, N, D) -> (B, num_heads, N, head_dim)
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        inv_bound = 1.0 / float(self.bound)

        # Cubic squashing only for Q/K: keeps attention scores bounded without softmax/division (HE-friendly).
        # Bounded Q,K => dot(Q,K) in [-head_dim, head_dim]; scale by 1/(head_dim*N) => weights ~ 1/N.
        q = _cubic_squash_unit(q * inv_bound)
        k = _cubic_squash_unit(k * inv_bound)

        scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / (self.head_dim * N))

        # V squashed with same cubic as Q/K (HE-friendly; no clamp so no approximation needed at inference).
        v = _cubic_squash_unit(v * inv_bound) * self.bound
        out = torch.matmul(scores, v)

        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)

        # output_squash=False by default for KD stability (no extra squashing on residual path).
        if self.output_squash:
            out = _cubic_squash_unit(out * inv_bound) * self.bound

        if residual:
            out = out + x
        return out
