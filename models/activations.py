"""
HE-friendly activations: PolynomialGELU (no exp/erf) and RunningNorm (add/mul only at inference).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CubicSquash(nn.Module):
    """
    HE-friendly bounded-ish activation using only + and *.

    We use the cubic polynomial:
      s(u) = 1.5u - 0.5u^3
    which maps [-1, 1] -> [-1, 1] smoothly.

    Usage: scale inputs into roughly [-1,1] via u = x / bound (constant multiply),
    apply s(u), then rescale back to [-bound, bound].

    Note: no non-constant polynomial is globally bounded on R; this is intended to
    operate when its input has already been kept in a reasonable numeric range.
    """

    def __init__(self, bound: float = 6.0):
        super().__init__()
        self.bound = float(bound)
        self.inv_bound = 1.0 / self.bound

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x * self.inv_bound
        u3 = u * u * u
        s = 1.5 * u - 0.5 * u3
        return s * self.bound


def fit_gelu_polynomial(
    degree: int = 5,
    low: float = -4.0,
    high: float = 4.0,
    num_points: int = 2001,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Fit a polynomial of given degree to GELU(x) on [low, high].
    Returns coefficients [c0, c1, ..., c_degree] for c0 + c1*x + ... + c_degree*x^degree.
    """
    x = torch.linspace(low, high, num_points, device=device, dtype=torch.float32)
    y = F.gelu(x)
    # Design matrix: columns 1, x, x^2, ...
    powers = x.unsqueeze(1).pow(torch.arange(degree + 1, device=x.device, dtype=x.dtype))
    result = torch.linalg.lstsq(
        powers, y.unsqueeze(1), rcond=None,
    )
    coeffs = result[0]
    return coeffs.squeeze(1)


def default_gelu_poly_coefficients(device: Optional[torch.device] = None) -> torch.Tensor:
    """Least-squares fit to GELU on [-4, 4] for degree-5 polynomial."""
    return fit_gelu_polynomial(degree=5, low=-4.0, high=4.0, device=device)


class PolynomialGELU(nn.Module):
    """
    GELU approximation using a polynomial (HE-friendly: only + and *).
    Input is clamped to [-clip_val, clip_val] so the polynomial does not explode for large |x|.
    """

    def __init__(
        self,
        degree: int = 5,
        coefficients: Optional[torch.Tensor] = None,
        clip_val: float = 6.0,
    ):
        """
        Args:
            degree: Polynomial degree when coefficients is None.
            coefficients: If set, use these (length degree+1); else use default_gelu_poly_coefficients().
            clip_val: Clamp input to [-clip_val, clip_val] for stability (avoids explosion/NaN).
        """
        super().__init__()
        if coefficients is not None:
            self.coefficients = nn.Parameter(coefficients.clone().detach(), requires_grad=False)
        else:
            self.coefficients = nn.Parameter(
                default_gelu_poly_coefficients().clone().detach(),
                requires_grad=False,
            )
        self.clip_val = clip_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, -self.clip_val, self.clip_val)
        # Horner-style: c0 + x*(c1 + x*(c2 + ...))
        c = self.coefficients.to(x.dtype)
        out = c[-1].expand_as(x) if x.dim() > 0 else c[-1]
        for i in range(len(c) - 2, -1, -1):
            out = out * x + c[i]
        return out


class RunningNorm(nn.Module):
    """
    Normalization using running mean and inv-std (add and multiply only at inference; HE-friendly).
    Training: update running stats. Inference: out = (x - running_mean) * running_inv_std (+ optional affine).
    """

    def __init__(self, num_features: int, momentum: float = 0.1, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            with torch.no_grad():
                # x: (B, N, D) or (B, D)
                dim = list(range(x.dim() - 1))
                m = x.mean(dim=dim)
                v = x.var(dim=dim, unbiased=False) + self.eps
                self.running_mean.lerp_(m, self.momentum)
                self.running_var.lerp_(v, self.momentum)
        inv_std = (self.running_var + self.eps).rsqrt()
        return (x - self.running_mean) * inv_std
