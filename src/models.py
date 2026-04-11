from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch.nn as nn
from torchvision import models


class CNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        conv_channels: Sequence[int] = (32, 64, 128, 256),
        hidden_dim: int = 128,
        feature_dropout: float = 0.4,
        classifier_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if not conv_channels:
            raise ValueError("conv_channels must contain at least one channel width.")
        if any(channel <= 0 for channel in conv_channels):
            raise ValueError("conv_channels must contain positive integers.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if not 0.0 <= feature_dropout < 1.0:
            raise ValueError("feature_dropout must be in the range [0.0, 1.0).")
        if not 0.0 <= classifier_dropout < 1.0:
            raise ValueError("classifier_dropout must be in the range [0.0, 1.0).")

        feature_layers: list[nn.Module] = []
        in_channels = 3
        for out_channels in conv_channels:
            feature_layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),
                ]
            )
            in_channels = out_channels

        self.features = nn.Sequential(*feature_layers)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=feature_dropout),
            nn.Linear(conv_channels[-1], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=classifier_dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        return self.classifier(x)


def build_model(
    model_name: str,
    num_classes: int,
    freeze_backbone: bool = False,
    model_kwargs: Mapping[str, Any] | None = None,
) -> nn.Module:
    model_name = model_name.lower()
    model_kwargs = dict(model_kwargs or {})

    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name in {"custom_cnn", "cnn"}:
        model = CNN(num_classes=num_classes, **model_kwargs)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = False

    return model
