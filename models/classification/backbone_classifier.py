"""
Generic pretrained backbone classifier for skin lesion classification.
Supports EfficientNet and ConvNeXt families from timm.
"""

import torch
import torch.nn as nn
import timm


class BackboneClassifier(nn.Module):
    """Pretrained timm backbone + dropout head."""

    def __init__(
        self,
        num_classes: int = 7,
        backbone: str = "efficientnet_b0",
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        feat_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, num_classes),
        )
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)
