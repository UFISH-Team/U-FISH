import torch
from torch import Tensor
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1,
                    stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.shortcut(identity)

        out += identity
        out = self.relu(out)

        return out


class Backbone(nn.Module):
    def __init__(
            self, in_channels=1, base_channels=64,
            out_dim=10000, num_layer=10):
        super(Backbone, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(num_layer):
            if i == 0:
                self.layers.append(ResidualBlock(in_channels, base_channels))
            else:
                self.layers.append(ResidualBlock(base_channels, base_channels))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, out_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class PosRegNet(nn.Module):
    def __init__(
            self, in_channels: int = 1,
            base_channels: int = 64,
            backbone_layers: int = 10,
            backbone_out_dim: int = 10000,
            n_pos: int = 8000, pos_dim: int = 2):
        super().__init__()
        self.n_pos = n_pos
        self.pos_dim = pos_dim
        self.backbone = Backbone(
            in_channels, base_channels,
            backbone_out_dim, backbone_layers)
        self.fc = nn.Linear(
            in_features=backbone_out_dim,
            out_features=n_pos * pos_dim
        )

    def forward(self, x: Tensor):
        x = self.backbone(x)
        x = self.fc(x)
        x = x.view(-1, self.n_pos, self.pos_dim)
        x = torch.sigmoid(x)
        return x
