import torch
import torch.nn as nn


class DepthwiseSeparableConv2d(nn.Module):
    """
    Convolution séparable en profondeur (Depthwise + Pointwise).
    Utilisée pour réduire drastiquement le nombre de paramètres.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            groups=in_channels,
            bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class ResidualBlock(nn.Module):
    """
    Bloc résiduel composé de deux convolutions séparables et d'un raccourci (shortcut).
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.main_branch = nn.Sequential(
            DepthwiseSeparableConv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=2,
                bias=False
            ),
            nn.BatchNorm2d(out_channels)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.main_branch(x) + self.shortcut(x))


class MiniXception(nn.Module):
    """
    Architecture Mini-Xception complète pour la classification d'émotions sur FER-2013.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 7):
        super().__init__()

        # Stem initial : 1 -> 8 -> 8 canaux (48x48)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )

        # 4 blocs résiduels consécutifs
        self.block1 = ResidualBlock(in_channels=8, out_channels=16)     # 48x48 -> 24x24
        self.block2 = ResidualBlock(in_channels=16, out_channels=32)    # 24x24 -> 12x12
        self.block3 = ResidualBlock(in_channels=32, out_channels=64)    # 12x12 -> 6x6
        self.block4 = ResidualBlock(in_channels=64, out_channels=128)   # 6x6 -> 3x3

        # Tête de classification
        self.head_conv = nn.Sequential(
            DepthwiseSeparableConv2d(128, num_classes, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(inplace=True)
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.head_conv(x)
        x = self.gap(x)
        return torch.flatten(x, start_dim=1)
