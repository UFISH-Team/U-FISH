import torch.nn as nn
import torch


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout2d(p=0.2)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        return out


class SpotLearn(nn.Module):
    def __init__(self, input_channel: int = 1) -> None:
        super().__init__()
        self.conv1 = ConvBlock(input_channel, 64)
        self.down1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = ConvBlock(64, 128)
        self.down2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottom = ConvBlock(128, 256)
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = ConvBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = ConvBlock(128, 64)
        self.conv5 = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, x):
        in1 = self.conv1(x)
        x = self.down1(in1)
        in2 = self.conv2(x)
        x = self.down2(in2)
        x = self.bottom(x)
        x = self.up1(x)
        x = torch.cat([x, in2], dim=1)
        x = self.conv3(x)
        x = self.up2(x)
        x = torch.cat([x, in1], dim=1)
        x = self.conv4(x)
        x = self.conv5(x)
        return x
