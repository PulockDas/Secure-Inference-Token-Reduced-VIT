"""
Modular norm for student transformer blocks: no norm, affine-only, or LayerNorm.
LayerNorm uses mean/variance and division (not HE-friendly). Affine-only is scale + shift (HE-friendly).
Residual connections stay valid: out = x + sublayer(norm(x)); norm is (B,N,D)->(B,N,D).

Static/Frozen LayerNorm: for HE inference, replace LayerNorm with y = scale * x + bias where
scale and bias are precomputed from calibration (μ_fixed, σ_fixed^2) and the trained gamma/beta.
"""

from typing import Literal, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


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


class StaticLayerNorm(nn.Module):
    """
    HE-friendly LayerNorm replacement: y = scale * x + bias.
    scale and bias are computed once from calibration (μ_fixed, σ_fixed^2) and the
    trained LayerNorm's gamma/beta, so forward uses only + and * (no division at inference).
    """

    def __init__(self, scale: torch.Tensor, bias: torch.Tensor):
        super().__init__()
        assert scale.shape == bias.shape and scale.dim() == 1
        self.register_buffer("scale", scale.clone().detach())
        self.register_buffer("bias", bias.clone().detach())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.bias


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


def _compute_static_ln_scale_bias(
    gamma: torch.Tensor,
    beta: torch.Tensor,
    mu: torch.Tensor,
    var: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """From LN params and calibration (μ, σ^2): scale = gamma/sqrt(σ^2+eps), bias = beta - μ*scale."""
    std = (var + eps).sqrt()
    scale = gamma / std
    bias = beta - mu * scale
    return scale, bias


def replace_layernorm_with_static(
    model: nn.Module,
    calibration_loader: DataLoader,
    device: torch.device,
    num_calibration_batches: int = 16,
    eps: float = 1e-6,
) -> List[str]:
    """
    Replace all nn.LayerNorm in model with StaticLayerNorm using calibration data.

    Runs calibration batches through the model, captures (input, output) pairs for
    each LayerNorm, and for every feature dimension solves a 1D least-squares fit
    y ≈ scale * x + bias over the collected data. The fitted (scale, bias) are
    then frozen into StaticLayerNorm so that inference uses only + and * while
    closely mimicking the original LayerNorm on the calibration distribution.

    Returns:
        List of replaced module names (e.g. ["blocks.0.norm1", "blocks.0.norm2", ...]).
    """
    # Collect (name, module) for all LayerNorms
    ln_names: List[str] = []
    ln_modules: List[nn.LayerNorm] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            ln_names.append(name)
            ln_modules.append(module)

    if not ln_names:
        return []

    # Capture inputs and outputs for each LayerNorm via forward hooks
    inputs_per_ln: List[List[torch.Tensor]] = [[] for _ in ln_names]
    outputs_per_ln: List[List[torch.Tensor]] = [[] for _ in ln_names]
    handles = []

    def make_hook(idx: int):
        def hook(module: nn.Module, inp: Tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
            x = inp[0] if isinstance(inp[0], torch.Tensor) else inp[0][0]
            inputs_per_ln[idx].append(x.detach())
            outputs_per_ln[idx].append(out.detach())
        return hook

    for idx, (name, module) in enumerate(zip(ln_names, ln_modules)):
        handles.append(module.register_forward_hook(make_hook(idx)))

    model.eval()
    batch_count = 0
    with torch.no_grad():
        for batch in calibration_loader:
            if batch_count >= num_calibration_batches:
                break
            images = batch[0].to(device)
            model(images)
            batch_count += 1

    for h in handles:
        h.remove()

    # For each LayerNorm: concat captured inputs/outputs and fit per-feature affine map
    replaced: List[str] = []
    tiny = 1e-8
    for idx, name in enumerate(ln_names):
        ln = ln_modules[idx]
        xs = inputs_per_ln[idx]
        ys = outputs_per_ln[idx]
        if not xs or not ys:
            continue

        all_x = torch.cat(xs, dim=0)  # (B_total, N, D)
        all_y = torch.cat(ys, dim=0)  # (B_total, N, D)
        B_total, N, D = all_x.shape

        x_flat = all_x.view(B_total * N, D)
        y_flat = all_y.view(B_total * N, D)

        # Linear regression y ≈ scale * x + bias (per feature)
        mu_x = x_flat.mean(dim=0)
        mu_y = y_flat.mean(dim=0)
        x_centered = x_flat - mu_x
        y_centered = y_flat - mu_y

        var_x = (x_centered * x_centered).mean(dim=0)
        cov_xy = (x_centered * y_centered).mean(dim=0)

        scale = cov_xy / (var_x + eps)
        # Guard against near-constant inputs
        mask_bad = var_x < tiny
        if mask_bad.any():
            scale = scale.clone()
            scale[mask_bad] = 1.0

        bias = mu_y - scale * mu_x

        static_ln = StaticLayerNorm(scale, bias)
        static_ln.to(device=device)

        parent_name = ".".join(name.split(".")[:-1])
        child_name = name.split(".")[-1]
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, static_ln)
        replaced.append(name)

    return replaced
