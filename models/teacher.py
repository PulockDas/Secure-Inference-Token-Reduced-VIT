"""
Teacher Vision Transformer: pretrained ViT-B/16 @ 224, classification head replaced for num_classes.
Architecture unchanged.
"""

import timm
import torch.nn as nn


def get_teacher_vit(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Load pretrained ViT-B/16 @ 224 from timm and replace classification head.
    Architecture unchanged; only the final linear layer is replaced.
    """
    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return model
