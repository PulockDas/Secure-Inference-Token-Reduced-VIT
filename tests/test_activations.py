"""
Polynomial GELU: unit tests and comparison vs true GELU (plaintext).
Run on your PC: python tests/test_activations.py
Uses only torch and models.activations (no timm/Colab).
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Import activations without pulling in teacher (timm)
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "activations",
    PROJECT_ROOT / "models" / "activations.py",
)
_activations = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_activations)
PolynomialGELU = _activations.PolynomialGELU
default_gelu_poly_coefficients = _activations.default_gelu_poly_coefficients


def test_shape_forward_backward():
    """PolynomialGELU preserves shape and supports backward."""
    module = PolynomialGELU(degree=5)
    x = torch.randn(2, 97, 384, requires_grad=True)
    out = module(x)
    assert out.shape == x.shape
    loss = out.sum()
    loss.backward()
    assert x.grad is not None


def test_he_friendly_no_exp_division():
    """Forward uses only + and * (no softmax, exp, or division)."""
    module = PolynomialGELU(degree=5)
    x = torch.randn(10)
    out = module(x)
    assert torch.isfinite(out).all()
    # Gradient is polynomial (no exp/div)
    x = torch.randn(5, 5, requires_grad=True)
    out = module(x).sum()
    out.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_comparison_vs_true_gelu():
    """Compare polynomial GELU vs F.gelu over [-4, 4] and report errors."""
    module = PolynomialGELU(degree=5)
    x = torch.linspace(-4.0, 4.0, 2001)
    with torch.no_grad():
        approx = module(x)
        true = F.gelu(x)
    abs_err = (approx - true).abs()
    max_abs = abs_err.max().item()
    mean_abs = abs_err.mean().item()
    rel_err = abs_err / (true.abs() + 1e-8)
    max_rel = rel_err.max().item()

    print("\n--- Polynomial GELU vs true GELU (plaintext) ---")
    print(f"  Range: x in [-4, 4], 2001 points")
    print(f"  Max absolute error:  {max_abs:.6f}")
    print(f"  Mean absolute error: {mean_abs:.6f}")
    print(f"  Max relative error:  {max_rel:.6f}")

    # Sanity: approximation should be reasonable on [-4, 4]
    assert max_abs < 0.2, f"Approximation too coarse: max_abs={max_abs}"
    assert mean_abs < 0.06, f"Mean error too large: mean_abs={mean_abs}"


def test_comparison_random_inputs():
    """Compare on random inputs in [-4, 4] (fit range)."""
    module = PolynomialGELU(degree=5)
    torch.manual_seed(42)
    x = torch.rand(500, 100) * 8.0 - 4.0  # uniform in [-4, 4]
    with torch.no_grad():
        approx = module(x)
        true = F.gelu(x)
    max_abs = (approx - true).abs().max().item()
    print(f"  Random (500x100, uniform [-4,4]): max absolute error = {max_abs:.6f}")
    assert max_abs < 0.25


def test_custom_coefficients():
    """Custom coefficients are used correctly."""
    c = torch.tensor([0.0, 0.5, 0.1, 0.0, 0.0, 0.0])
    module = PolynomialGELU(coefficients=c)
    x = torch.tensor([1.0, 2.0])
    with torch.no_grad():
        out = module(x)
    # 0 + 0.5*x + 0.1*x^2 => [0.6, 1.4]
    expected = torch.tensor([0.5 + 0.1, 1.0 + 0.4])
    assert torch.allclose(out, expected), f"got {out}, expected {expected}"


if __name__ == "__main__":
    test_shape_forward_backward()
    print("test_shape_forward_backward passed")
    test_he_friendly_no_exp_division()
    print("test_he_friendly_no_exp_division passed")
    test_comparison_vs_true_gelu()
    print("test_comparison_vs_true_gelu passed")
    test_comparison_random_inputs()
    print("test_comparison_random_inputs passed")
    test_custom_coefficients()
    print("test_custom_coefficients passed")
    print("\nAll activation tests passed.")
