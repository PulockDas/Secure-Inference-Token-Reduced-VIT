"""
HE-friendly activation approximations: only addition and multiplication.
Polynomial GELU approximates GELU(x) = 0.5*x*(1+erf(x/sqrt(2))) with a polynomial.
RunningNorm: normalize by running mean and inv_std (only + and * in forward at inference).
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class RunningNorm(nn.Module):
    """
    Normalize using running mean and inv_std: out = (x - mean) * inv_std.
    Both positive and negative values allowed (zero mean, unit variance scale).
    Forward uses only subtraction and multiplication (HE-friendly at inference).
    Running stats updated only in training; at inference we use fixed buffers.
    """

    def __init__(self, dim: int, momentum: float = 0.1, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.momentum = momentum
        self.eps = eps
        self.register_buffer("running_mean", torch.zeros(dim))
        self.register_buffer("running_inv_std", torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim). Stats over all dims except last.
        if self.training and x.requires_grad:
            with torch.no_grad():
                flat = x.reshape(-1, self.dim)
                batch_mean = flat.mean(dim=0)
                batch_std = flat.std(dim=0) + self.eps
                batch_inv_std = 1.0 / batch_std
                self.running_mean.mul_(self.momentum).add_(batch_mean, alpha=1 - self.momentum)
                self.running_inv_std.mul_(self.momentum).add_(batch_inv_std, alpha=1 - self.momentum)
        return (x - self.running_mean) * self.running_inv_std


def fit_gelu_polynomial(
    degree: int = 5,
    x_min: float = -4.0,
    x_max: float = 4.0,
    num_points: int = 2000,
) -> torch.Tensor:
    """
    Least-squares polynomial fit to GELU on [x_min, x_max].
    Returns coefficients [c0, c1, ..., c_degree] for sum_i c_i * x^i.
    """
    x = torch.linspace(x_min, x_max, num_points, dtype=torch.float64)
    y = torch.nn.functional.gelu(x)
    # Vandermonde: X[i, j] = x[i]**j
    X = x.unsqueeze(1).pow(torch.arange(degree + 1, dtype=torch.float64, device=x.device))
    # (X^T X) c = X^T y  =>  c = (X^T X)^{-1} X^T y
    XtX = X.T @ X
    Xty = X.T @ y.unsqueeze(1)
    coeffs = torch.linalg.solve(XtX, Xty).squeeze(1)
    return coeffs.to(torch.float32)


def default_gelu_poly_coefficients(degree: int = 5) -> torch.Tensor:
    """
    Default polynomial coefficients approximating GELU on [-4, 4].
    From least-squares fit (Taylor diverges for large |x|). HE-friendly: only + and *.
    """
    return fit_gelu_polynomial(degree=degree)


class PolynomialGELU(nn.Module):
    """
    Polynomial approximation of GELU (only + and *). Coefficients fitted on [-4, 4].
    Use RunningNorm before this in the graph so input is normalized from real stats (no hardcoding).
    """

    def __init__(
        self,
        degree: int = 5,
        coefficients: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        if coefficients is not None:
            c = coefficients.view(-1).clone().detach()
        else:
            c = default_gelu_poly_coefficients(degree)
        self.register_buffer("_coeffs", c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = self._coeffs
        out = torch.zeros_like(x)
        for i in range(c.numel()):
            out = out + c[i] * x.pow(i)
        return out
