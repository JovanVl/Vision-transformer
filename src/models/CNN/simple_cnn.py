import torch
from typing import List


def cnn_sublayer(in_channels: int, out_channels: int):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(2)
    )


class SimpleCNN(torch.nn.Module):
    def __init__(self, in_channels: int, num_of_classes: int, img_size: int = 224, channels: List[int] = None):
        super().__init__()
        if channels is None:
            channels = [32, 64, 128, 256, 256]

        sublayers = []
        prev_channels = in_channels
        for channel in channels:
            sublayers.append(cnn_sublayer(prev_channels, channel))
            prev_channels = channel
        
        self.features = torch.nn.Sequential(*sublayers)
        self.classifier = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(channels[-1], num_of_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)    