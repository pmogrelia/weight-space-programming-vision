"""The deliberately small classifier used by the MVP."""

from __future__ import annotations

import torch
from torch import nn


class SmallCNN(nn.Module):
    """A 6,164-parameter CNN whose complete weights fit in an editor prompt."""

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(32, num_classes)

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.relu(self.conv1(inputs))
        hidden = self.relu(self.conv2(hidden))
        hidden = self.relu(self.conv3(hidden))
        return self.pool(hidden).flatten(1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(inputs))


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
