import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.upconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.upconv(x)


class UNet(nn.Module):
    def __init__(
            self, in_channels=1, out_channels=1,
            num_layers=4, base_channels=64):
        super(UNet, self).__init__()

        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i in range(num_layers):
            input_channels = (
                in_channels if i == 0 else base_channels * 2 ** (i - 1)
            )
            output_channels = base_channels * 2 ** i
            self.encoders.append(ConvBlock(input_channels, output_channels))
            self.downsamples.append(DownConv(output_channels, output_channels))

        self.bottom = ConvBlock(
            base_channels * 2 ** (num_layers - 1),
            base_channels * 2 ** (num_layers)
        )

        self.decoders = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i in range(num_layers, 0, -1):
            input_channels = base_channels * 2 ** i
            output_channels = base_channels * 2 ** (i - 1)
            self.decoders.append(ConvBlock(input_channels, output_channels))
            self.upsamples.append(UpConv(input_channels, output_channels))

        self.final_decoder = ConvBlock(base_channels, out_channels)

    def forward(self, x):
        encodings = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            encodings.append(x)
            x = self.downsamples[i](x)

        x = self.bottom(x)

        for i, decoder in enumerate(self.decoders):
            x = self.upsamples[i](x)
            x = torch.cat([x, encodings[-i - 1]], dim=1)
            x = decoder(x)

        x = self.final_decoder(x)
        return x
