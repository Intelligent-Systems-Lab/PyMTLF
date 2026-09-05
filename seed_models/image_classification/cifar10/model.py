import torch
from torch import nn


class SmallCNN(nn.Module):
    def __init__(self, input_channels: int, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.classifier(torch.flatten(self.features(value), 1))


Model = SmallCNN
