"""
Student Vision Transformer: HE-friendly, token reduction, polynomial activations.
No LayerNorm (use none or affine via create_norm). For distillation and encrypted inference.

Norm choice: Use "affine" for training (learnable scale/shift helps stability and accuracy).
Use "none" for minimal HE inference (fewer ops); accuracy may drop slightly.
"""

from typing import Literal

import torch
import torch.nn as nn

from .activations import PolynomialGELU
from .he_attention import HEAttention
from .norm import create_norm
from .token_reduction import TokenReduction


class PatchEmbed(nn.Module):
    """Patch embedding: (B, 3, H, W) -> (B, num_patches, embed_dim). ViT-style 16x16 patches."""

    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, 3, H, W) -> (B, embed_dim, h, w) -> (B, embed_dim, num_patches) -> (B, num_patches, embed_dim)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class StudentBlock(nn.Module):
    """
    One transformer block: pre-norm HE attention + pre-norm FFN (polynomial GELU).
    Residual: x = x + attn(norm(x)); x = x + ffn(norm(x)).
    No LayerNorm: norm is either identity or affine (create_norm).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        norm_mode: Literal["none", "affine"] = "affine",
    ):
        super().__init__()
        self.norm1 = create_norm(norm_mode, embed_dim)
        self.attn = HEAttention(embed_dim, num_heads=num_heads)
        self.norm2 = create_norm(norm_mode, embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            PolynomialGELU(),
            nn.Linear(mlp_hidden, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # HEAttention with residual=False; we add residual here so pre-norm is correct
        x = x + self.attn(self.norm1(x), residual=False)
        x = x + self.mlp(self.norm2(x))
        return x


class StudentViT(nn.Module):
    """
    Student Vision Transformer: HE-friendly, smaller than teacher.
    - Patch embed -> CLS + patches -> token reduction -> blocks -> head.
    - Uses HE attention (no softmax), PolynomialGELU, no LayerNorm (none or affine).
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 384,
        num_heads: int = 6,
        depth: int = 6,
        num_output_tokens: int = 97,
        num_classes: int = 5,
        mlp_ratio: float = 4.0,
        norm_mode: Literal["none", "affine"] = "affine",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_output_tokens = num_output_tokens
        self.patch_embed = PatchEmbed(img_size, patch_size, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        self.token_reduction = TokenReduction(embed_dim, num_output_tokens)
        self.blocks = nn.ModuleList([
            StudentBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio, norm_mode=norm_mode)
            for _ in range(depth)
        ])
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, 3, 224, 224) -> (B, num_patches, embed_dim)
        x = self.patch_embed(x)
        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, 1 + num_patches, embed_dim)
        x = self.token_reduction(x)  # (B, K, embed_dim)
        for block in self.blocks:
            x = block(x)
        logits = self.head(x[:, 0])  # CLS token
        return logits


def get_student_vit(
    num_classes: int = 5,
    embed_dim: int = 384,
    depth: int = 6,
    num_heads: int = 6,
    num_output_tokens: int = 97,
    norm_mode: Literal["none", "affine"] = "affine",
    img_size: int = 224,
) -> StudentViT:
    """
    Build student ViT. Default embed_dim=384 (teacher 768//2); use norm_mode "affine" for
    training (recommended), "none" for minimal HE inference.
    """
    return StudentViT(
        img_size=img_size,
        patch_size=16,
        embed_dim=embed_dim,
        num_heads=num_heads,
        depth=depth,
        num_output_tokens=num_output_tokens,
        num_classes=num_classes,
        norm_mode=norm_mode,
    )
