"""
Modular norm for student transformer blocks: no norm, affine-only, or LayerNorm.
LayerNorm uses mean/variance and division (not HE-friendly). Affine-only is scale + shift (HE-friendly).
Residual connections stay valid: out = x + sublayer(norm(x)); norm is (B,N,D)->(B,N,D).

LayerNorm approximation for HE inference: replace LayerNorm with a +/*-only approximation
(polynomial init + Newton rsqrt) so the encrypted circuit avoids division/sqrt.
"""

from typing import Literal, List, Optional, Tuple

import math
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


class ApproxLayerNorm(nn.Module):
    """
    HE-friendly LayerNorm approximation using only + and * at inference time.

    We compute per-token mean/variance over the feature dimension, then use a
    scaled Newton iteration to approximate 1/sqrt(a) where a = var + eps:

        a_scaled = a / m
        y_0 = 1
        y_{k+1} = y_k * (1.5 - 0.5 * a_scaled * y_k^2)
        inv_sqrt(a) ≈ y_K * inv_sqrt_m

    Here m = E[a] from calibration; inv_m = 1/m and inv_sqrt_m = 1/sqrt(m)
    are precomputed in plaintext. At inference, the forward uses only + and *.
    """

    def __init__(
        self,
        gamma: torch.Tensor,
        beta: torch.Tensor,
        inv_m: torch.Tensor,
        inv_sqrt_m: torch.Tensor,
        eps: float,
        iters: int = 3,
    ):
        super().__init__()
        assert gamma.shape == beta.shape and gamma.dim() == 1
        self.register_buffer("gamma", gamma.clone().detach())
        self.register_buffer("beta", beta.clone().detach())

        inv_m = inv_m.clone().detach()
        inv_sqrt_m = inv_sqrt_m.clone().detach()
        if inv_m.dim() == 0:
            inv_m = inv_m.view(1, 1, 1)
        if inv_sqrt_m.dim() == 0:
            inv_sqrt_m = inv_sqrt_m.view(1, 1, 1)
        self.register_buffer("inv_m", inv_m)
        self.register_buffer("inv_sqrt_m", inv_sqrt_m)

        self.eps = float(eps)
        self.iters = int(iters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        c = x - mean
        var = (c * c).mean(dim=-1, keepdim=True)
        a = var + self.eps

        a_scaled = a * self.inv_m
        # During training, clamp the scaled variance to avoid exploding/vanishing
        # Newton iterations (no clamp in eval/inference to keep HE behavior intact).
        if self.training:
            a_scaled = torch.clamp(a_scaled, 1e-3, 1e3)

        z = a_scaled
        # First-order Taylor init of 1/sqrt(z) around z=1: y0 ≈ 1 - 0.5*(z-1)
        y = 1.0 - 0.5 * (z - 1.0)
        for _ in range(self.iters):
            y = y * (1.5 - 0.5 * z * y * y)

        inv_sqrt_a = y * self.inv_sqrt_m
        x_norm = c * inv_sqrt_a
        return x_norm * self.gamma + self.beta


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




def replace_layernorm_with_approx(
    model: nn.Module,
    calibration_loader: DataLoader,
    device: torch.device,
    num_calibration_batches: int = 16,
    eps: float = 1e-6,
    iters: int = 6,
) -> List[str]:
    """
    Replace all nn.LayerNorm with ApproxLayerNorm using calibration data.

    Calibration estimates the typical per-token a = var + eps (over the feature
    dim) at the input to each LayerNorm. From this we derive a scalar m = E[a],
    and precompute inv_m = 1/m and inv_sqrt_m = 1/sqrt(m) used in a scaled
    Newton iteration. Replacement is then done in-place.
    """
    ln_names: List[str] = []
    ln_modules: List[nn.LayerNorm] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            ln_names.append(name)
            ln_modules.append(module)

    if not ln_names:
        return []

    sum_a = [0.0 for _ in ln_names]
    count_a = [0 for _ in ln_names]
    sample_x: List[Optional[torch.Tensor]] = [None for _ in ln_names]
    handles = []

    def make_hook(idx: int):
        def hook(module: nn.Module, inp: Tuple[torch.Tensor, ...]) -> None:
            x = inp[0] if isinstance(inp[0], torch.Tensor) else inp[0][0]
            with torch.no_grad():
                mean = x.mean(dim=-1, keepdim=True)
                c = x - mean
                var = (c * c).mean(dim=-1, keepdim=True)
                a = var + eps  # (B, N, 1)
                sum_a[idx] += float(a.sum().detach().cpu().item())
                count_a[idx] += int(a.numel())
                if sample_x[idx] is None:
                    sample_x[idx] = x.detach()[:1].to(device)

        return hook

    for idx, module in enumerate(ln_modules):
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

    replaced: List[str] = []
    did_diag = False
    for idx, name in enumerate(ln_names):
        ln = ln_modules[idx]
        if count_a[idx] <= 0:
            continue

        m = sum_a[idx] / float(count_a[idx])
        inv_m = torch.tensor(1.0 / m, device=device)
        inv_sqrt_m = torch.tensor(1.0 / math.sqrt(m), device=device)

        approx_ln = ApproxLayerNorm(
            gamma=ln.weight.detach(),
            beta=ln.bias.detach(),
            inv_m=inv_m,
            inv_sqrt_m=inv_sqrt_m,
            eps=ln.eps if hasattr(ln, "eps") else eps,
            iters=iters,
        ).to(device=device)

        print(f"[ApproxLN] layer '{name}': mean(var+eps) m = {m:.6e}")

        if not did_diag and sample_x[idx] is not None:
            x_sample = sample_x[idx]
            with torch.no_grad():
                y_true = ln(x_sample.to(device))
                y_apx = approx_ln(x_sample.to(device))
                rel_err = (
                    (y_true - y_apx).pow(2).mean().sqrt()
                    / (y_true.pow(2).mean().sqrt() + 1e-8)
                ).item()
                print(
                    f"[ApproxLN] diagnostic for layer '{name}': "
                    f"rel_err={rel_err:.6e}, "
                    f"true_std={y_true.std().item():.6e}, "
                    f"approx_std={y_apx.std().item():.6e}"
                )
            did_diag = True

        parent_name = ".".join(name.split(".")[:-1])
        child_name = name.split(".")[-1]
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, approx_ln)
        replaced.append(name)

    return replaced
