"""
Student ViT: token reduction + HE-friendly attention + polynomial GELU.
Smaller than teacher; trained via distillation.
"""

from typing import Literal

import torch
import torch.nn as nn

from .activations import PolynomialGELU
from .he_attention import HEAttention
from .norm import create_norm
from .token_reduction import TokenReduction


class PatchEmbed(nn.Module):
    """ViT-style patch embedding: (B, 3, H, W) -> (B, num_patches, embed_dim)."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 384,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, 3, H, W) -> (B, embed_dim, h, w) -> (B, embed_dim, num_patches) -> (B, num_patches, embed_dim)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class StudentBlock(nn.Module):
    """
    One transformer block: pre-norm -> HEAttention -> residual;
    pre-norm -> MLP (Linear -> PolynomialGELU -> Linear) -> residual.

    Norm roles (no redundant stacking):
    - Pre-norm (affine/none): standard for residuals; affine = scale+shift only (HE-friendly).
    - L2 inside HEAttention: only place we need it, to bound attention by design.
    - No RunningNorm in MLP: keeps block light; PolynomialGELU has its own input clip.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        norm_mode: Literal["none", "affine", "layernorm"] = "affine",
        residual_scale: float = 1,
    ):
        super().__init__()
        self.norm1 = create_norm(norm_mode, embed_dim)
        # HE-friendly attention: cubic squashing on Q/K only; output_squash=False for KD stability.
        self.attn = HEAttention(embed_dim, num_heads=num_heads, bound=6.0, output_squash=False)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.norm2 = create_norm(norm_mode, embed_dim)
        # MLP: Linear -> PolynomialGELU -> Linear (no extra cubic squashing; helps KD stability).
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            PolynomialGELU(),
            nn.Linear(mlp_hidden, embed_dim),
        )
        # Scale residual branches so activations don't drift upward with depth.
        # Constant multiplication is HE-friendly.
        self.residual_scale = residual_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.residual_scale * self.attn(self.norm1(x), residual=False)
        x = x + self.residual_scale * self.mlp(self.norm2(x))
        return x


class StudentViT(nn.Module):
    """
    Student ViT: patch embed -> CLS + patches -> token reduction (K) -> stacked blocks -> CLS -> head.
    """

    def __init__(
        self,
        num_classes: int,
        embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        num_output_tokens: int = 97,
        norm_mode: Literal["none", "affine", "layernorm"] = "affine",
        img_size: int = 224,
        patch_size: int = 16,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, embed_dim) * 0.02)
        self.token_reduction = TokenReduction(embed_dim, num_output_tokens)
        self.blocks = nn.ModuleList([
            StudentBlock(embed_dim, num_heads, norm_mode=norm_mode)
            for _ in range(depth)
        ])
        self.norm = create_norm(norm_mode, embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.token_reduction(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        cls_final = x[:, 0]
        return self.head(cls_final)


def get_student_vit(
    num_classes: int,
    embed_dim: int = 384,
    depth: int = 6,
    num_heads: int = 6,
    num_output_tokens: int = 97,
    norm_mode: Literal["none", "affine", "layernorm"] = "affine",
    img_size: int = 224,
    patch_size: int = 16,
) -> StudentViT:
    """Build student ViT with given config."""
    return StudentViT(
        num_classes=num_classes,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        num_output_tokens=num_output_tokens,
        norm_mode=norm_mode,
        img_size=img_size,
        patch_size=patch_size,
    )
