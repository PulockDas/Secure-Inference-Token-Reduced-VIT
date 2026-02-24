from .activations import PolynomialGELU, RunningNorm, default_gelu_poly_coefficients, fit_gelu_polynomial
from .distillation_bridge import DistillationBridge, build_bridge_for_teacher, student_embed_dim
from .he_attention import HEAttention
from .norm import AffineNorm, ApproxLayerNorm, create_norm, replace_layernorm_with_approx
from .student import StudentViT, get_student_vit
from .token_reduction import TokenReduction


def get_teacher_vit(num_classes: int, pretrained: bool = True):
    """Lazy import so student tests can run without timm."""
    from .teacher import get_teacher_vit as _get_teacher_vit
    return _get_teacher_vit(num_classes=num_classes, pretrained=pretrained)


__all__ = [
    "get_teacher_vit",
    "TokenReduction",
    "HEAttention",
    "PolynomialGELU",
    "RunningNorm",
    "default_gelu_poly_coefficients",
    "fit_gelu_polynomial",
    "DistillationBridge",
    "build_bridge_for_teacher",
    "student_embed_dim",
    "AffineNorm",
    "ApproxLayerNorm",
    "create_norm",
    "replace_layernorm_with_approx",
    "StudentViT",
    "get_student_vit",
]
