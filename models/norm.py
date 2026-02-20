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

    Runs calibration batches through the model, captures inputs to each LayerNorm,
    computes μ_fixed and σ_fixed^2 per layer, then precomputes scale and bias so
    LN becomes y = scale * x + bias. Replaces layers in place.

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

    # Capture inputs to each LayerNorm via forward hooks
    captured: List[List[torch.Tensor]] = [[] for _ in ln_names]
    handles = []

    def make_hook(idx: int):
        def hook(module: nn.Module, inp: Tuple[torch.Tensor, ...]) -> None:
            # Pre-hook: only (module, args); no output yet
            x = inp[0] if isinstance(inp[0], torch.Tensor) else inp[0][0]
            captured[idx].append(x.detach())
        return hook

    for idx, (name, module) in enumerate(zip(ln_names, ln_modules)):
        handles.append(module.register_forward_pre_hook(make_hook(idx)))

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

    # For each LayerNorm: concat captured inputs, compute μ and σ^2 over (batch, seq), then scale/bias
    replaced: List[str] = []
    for idx, name in enumerate(ln_names):
        ln = ln_modules[idx]
        inputs_list = captured[idx]
        if not inputs_list:
            continue
        # (B, N, D) each; concat along batch
        all_x = torch.cat(inputs_list, dim=0)
        # μ, var over dims (0, 1) -> (D,)
        mu = all_x.mean(dim=(0, 1))
        var = all_x.var(dim=(0, 1), unbiased=False)
        gamma = ln.weight.detach()
        beta = ln.bias.detach()
        scale, bias = _compute_static_ln_scale_bias(gamma, beta, mu, var, eps=eps)
        static_ln = StaticLayerNorm(scale, bias)
        static_ln.to(device=device)

        parent_name = ".".join(name.split(".")[:-1])
        child_name = name.split(".")[-1]
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, static_ln)
        replaced.append(name)

    return replaced
