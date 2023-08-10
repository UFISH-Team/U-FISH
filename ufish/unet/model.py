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


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1)
        self.BN = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.BN(out)
        out = self.relu(out)
        out = x + out
        return out


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.resconv = ResidualBlock(out_channels, out_channels)

    def forward(self, x):
        out = self.conv(x)
        out = self.resconv(out)
        return out


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.upconv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        up = F.interpolate(x, scale_factor=2, mode='nearest')
        out = self.upconv(up)
        return out


class ChannelAttention(nn.Module):
    def __init__(self, input_nc, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.f1 = nn.Conv2d(input_nc, input_nc // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.f2 = nn.Conv2d(input_nc // ratio, input_nc, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.f2(self.relu(self.f1(self.avg_pool(x))))
        max_out = self.f2(self.relu(self.f1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1*h*w
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        # 2*h*w
        x = self.conv(x)
        # 1*h*w
        return self.sigmoid(x)


class CBAM(nn.Module):
    # CSP Bottleneck with 3 convolutions
    # ch_in, ch_out, number, shortcut, groups, expansion
    def __init__(self, input_nc, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(input_nc, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out = self.channel_attention(x) * x
        # c*h*w
        # c*h*w * 1*h*w
        out = self.spatial_attention(out) * out
        return out


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
        self.cbams = nn.ModuleList()
        for i in range(num_layers, 0, -1):
            input_channels = base_channels * 2 ** i
            output_channels = base_channels * 2 ** (i - 1)
            self.decoders.append(ConvBlock(input_channels, output_channels))
            self.upsamples.append(UpConv(input_channels, output_channels))
            self.cbams.append(CBAM(output_channels))

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
            x = self.cbams[i](x)
            diffY = encodings[-i - 1].size()[2] - x.size()[2]
            diffX = encodings[-i - 1].size()[3] - x.size()[3]
            x = F.pad(x, (diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2))
            x = torch.cat([x, encodings[-i - 1]], dim=1)
            x = decoder(x)

        x = self.final_decoder(x)
        return x
