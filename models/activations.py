"""
HE-friendly activation approximations: only addition and multiplication.
Polynomial GELU approximates GELU(x) = 0.5*x*(1+erf(x/sqrt(2))) with a polynomial.
"""

import math
from typing import Optional

import torch
import torch.nn as nn


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
    Polynomial approximation of GELU for HE-friendly inference (only + and *).
    Forward: out = sum_i coeffs[i] * x^i.
    Default coefficients approximate GELU on [-4, 4]; usable elsewhere with some error.
    """

    def __init__(
        self,
        degree: int = 5,
        coefficients: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            degree: Polynomial degree (only used if coefficients is None).
            coefficients: 1D tensor [c0, c1, ..., c_degree]. If None, use default GELU approx.
        """
        super().__init__()
        if coefficients is not None:
            c = coefficients.view(-1).clone().detach()
        else:
            c = default_gelu_poly_coefficients(degree)
        self.register_buffer("_coeffs", c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """out = c0 + c1*x + c2*x^2 + ... (only + and *)."""
        c = self._coeffs
        out = torch.zeros_like(x)
        for i in range(c.numel()):
            out = out + c[i] * x.pow(i)
        return out
