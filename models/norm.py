"""
Modular norm for student transformer blocks: no norm, affine-only, or LayerNorm.
LayerNorm uses mean/variance and division (not HE-friendly). Affine-only is scale + shift (HE-friendly).
Residual connections stay valid: out = x + sublayer(norm(x)); norm is (B,N,D)->(B,N,D).

LayerNorm approximation for HE inference: replace LayerNorm with a +/*-only approximation
(polynomial init + Newton rsqrt) so the encrypted circuit avoids division/sqrt.
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


class ApproxLayerNorm(nn.Module):
    """
    HE-friendly LayerNorm approximation using only + and * at inference time.

    We compute per-token mean/variance over the feature dimension (addition/multiplication
    plus a constant multiply for division by D), then approximate inv_sqrt(var + eps)
    with a small number of Newton-Raphson iterations:

        y_{k+1} = y_k * (1.5 - 0.5 * a * y_k^2),   where a = var + eps

    This avoids division/sqrt in the forward pass while closely matching LayerNorm
    if the initialization is calibrated well.
    """

    def __init__(
        self,
        gamma: torch.Tensor,
        beta: torch.Tensor,
        inv_sqrt_init: Optional[torch.Tensor] = None,
        inv_sqrt_poly_coeffs: Optional[torch.Tensor] = None,
        eps: float = 1e-6,
        iters: int = 2,
    ):
        super().__init__()
        assert gamma.shape == beta.shape and gamma.dim() == 1
        self.register_buffer("gamma", gamma.clone().detach())
        self.register_buffer("beta", beta.clone().detach())

        if inv_sqrt_poly_coeffs is None and inv_sqrt_init is None:
            raise ValueError("Provide either inv_sqrt_init or inv_sqrt_poly_coeffs.")

        if inv_sqrt_poly_coeffs is not None:
            coeffs = inv_sqrt_poly_coeffs.clone().detach()
            if coeffs.dim() != 1:
                raise ValueError("inv_sqrt_poly_coeffs must be 1D.")
            self.register_buffer("inv_sqrt_poly_coeffs", coeffs)
        else:
            self.register_buffer("inv_sqrt_poly_coeffs", torch.empty(0))

        if inv_sqrt_init is not None:
            inv_sqrt_init = inv_sqrt_init.clone().detach()
            if inv_sqrt_init.dim() == 0:
                inv_sqrt_init = inv_sqrt_init.view(1, 1, 1)
            self.register_buffer("inv_sqrt_init", inv_sqrt_init)
        else:
            self.register_buffer("inv_sqrt_init", torch.empty(0))

        self.eps = float(eps)
        self.iters = int(iters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        centered = x - mean
        var = (centered * centered).mean(dim=-1, keepdim=True)
        a = var + self.eps

        if self.inv_sqrt_poly_coeffs.numel() > 0:
            coeffs = self.inv_sqrt_poly_coeffs
            y = coeffs[-1]
            for c in reversed(coeffs[:-1]):
                y = y * a + c
        else:
            y = self.inv_sqrt_init
        for _ in range(self.iters):
            y = y * (1.5 - 0.5 * a * y * y)

        x_norm = centered * y
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
    iters: int = 2,
) -> List[str]:
    """
    Replace all nn.LayerNorm with ApproxLayerNorm using calibration data.

    Calibration estimates the typical per-token variance (over the feature dim) at the
    input to each LayerNorm. We use that to choose an initialization for the Newton
    inv_sqrt iterations. Replacement is then done in-place.
    """
    ln_names: List[str] = []
    ln_modules: List[nn.LayerNorm] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            ln_names.append(name)
            ln_modules.append(module)

    if not ln_names:
        return []

    # Fit a small polynomial y0(a) ≈ 1/sqrt(a) for a = var + eps (per token) to
    # get a good Newton initialization without non-HE ops at inference.
    # Degree-2 least squares using accumulated moments:
    # A = [[Σ1,  Σa,   Σa² ],
    #      [Σa, Σa²,  Σa³ ],
    #      [Σa²,Σa³,  Σa⁴ ]],  b = [Σt, Σt·a, Σt·a²],  t = 1/sqrt(a)
    s0 = [0.0 for _ in ln_names]
    s1 = [0.0 for _ in ln_names]
    s2 = [0.0 for _ in ln_names]
    s3 = [0.0 for _ in ln_names]
    s4 = [0.0 for _ in ln_names]
    b0 = [0.0 for _ in ln_names]
    b1 = [0.0 for _ in ln_names]
    b2 = [0.0 for _ in ln_names]
    handles = []

    def make_hook(idx: int):
        def hook(module: nn.Module, inp: Tuple[torch.Tensor, ...]) -> None:
            x = inp[0] if isinstance(inp[0], torch.Tensor) else inp[0][0]
            # Per-token variance over feature dim
            with torch.no_grad():
                m = x.mean(dim=-1, keepdim=True)
                c = x - m
                a = (c * c).mean(dim=-1) + eps  # (B, N)
                t = a.rsqrt()  # plaintext calibration only

                a1 = a
                a2 = a1 * a1
                a3 = a2 * a1
                a4 = a2 * a2

                s0[idx] += float(a1.numel())
                s1[idx] += float(a1.sum().detach().cpu().item())
                s2[idx] += float(a2.sum().detach().cpu().item())
                s3[idx] += float(a3.sum().detach().cpu().item())
                s4[idx] += float(a4.sum().detach().cpu().item())

                b0[idx] += float(t.sum().detach().cpu().item())
                b1[idx] += float((t * a1).sum().detach().cpu().item())
                b2[idx] += float((t * a2).sum().detach().cpu().item())

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
    for idx, name in enumerate(ln_names):
        ln = ln_modules[idx]
        if s0[idx] <= 0:
            continue

        A = torch.tensor(
            [
                [s0[idx], s1[idx], s2[idx]],
                [s1[idx], s2[idx], s3[idx]],
                [s2[idx], s3[idx], s4[idx]],
            ],
            dtype=torch.float64,
        )
        bb = torch.tensor([b0[idx], b1[idx], b2[idx]], dtype=torch.float64)
        ridge = (1e-6 * s0[idx]) if s0[idx] > 0 else 1e-6
        A = A + torch.eye(3, dtype=torch.float64) * ridge
        coeffs = torch.linalg.solve(A, bb).to(dtype=torch.float32, device=device)

        approx_ln = ApproxLayerNorm(
            gamma=ln.weight.detach(),
            beta=ln.bias.detach(),
            inv_sqrt_poly_coeffs=coeffs,
            eps=eps,
            iters=iters,
        ).to(device=device)

        parent_name = ".".join(name.split(".")[:-1])
        child_name = name.split(".")[-1]
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, approx_ln)
        replaced.append(name)

    return replaced
