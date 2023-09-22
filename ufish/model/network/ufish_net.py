import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        return out


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1)
        self.BN1 = nn.BatchNorm2d(channels)
        self.BN2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.BN1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.BN2(out)
        out = F.relu(out)
        out = out + x
        return out


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownConv, self).__init__()
        self.down_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.conv = ResidualBlock(out_channels)

    def forward(self, x):
        out = self.down_conv(x)
        out = self.conv(out)
        return out


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        up = F.interpolate(x, scale_factor=2, mode='nearest')
        out = self.conv(up)
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
    """Convolutional Block Attention Module"""
    def __init__(self, input_nc, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(input_nc, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x):
        out = self.conv(x)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x):
        out = self.conv(x)
        return out


class BottoleneckBlock(nn.Module):
    def __init__(self, channels):
        super(BottoleneckBlock, self).__init__()
        self.conv1 = ResidualBlock(channels)
        self.cbam = CBAM(channels)
        self.conv2 = ResidualBlock(channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.cbam(out)
        out = self.conv2(out)
        return out


class FinalDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FinalDecoderBlock, self).__init__()
        self.cbam = CBAM(in_channels)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x):
        out = self.cbam(x)
        out = self.conv(out)
        return out


class UFishNet(nn.Module):
    def __init__(
            self, in_channels=1, out_channels=1,
            channel_numbers=[32, 32, 32]):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i, c in enumerate(channel_numbers[:-1]):
            _ei = (in_channels if i == 0 else c)
            self.encoders.append(EncoderBlock(_ei, c))
            self.downsamples.append(DownConv(c, channel_numbers[i+1]))

        self.bottom = BottoleneckBlock(channel_numbers[-1])

        self.decoders = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        rev_nums = channel_numbers[::-1]
        for i, c in enumerate(rev_nums[1:]):
            self.upsamples.append(UpConv(rev_nums[i], c))
            self.decoders.append(DecoderBlock(2*c, c))

        self.final_decoder = FinalDecoderBlock(
            channel_numbers[0], out_channels)

    def forward(self, x):
        encodings = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            encodings.append(x)
            x = self.downsamples[i](x)

        x = self.bottom(x)

        for i, decoder in enumerate(self.decoders):
            x = self.upsamples[i](x)
            diffY = encodings[-i - 1].size()[2] - x.size()[2]
            diffX = encodings[-i - 1].size()[3] - x.size()[3]
            x = F.pad(x, (diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2))
            x = torch.cat([x, encodings[-i - 1]], dim=1)
            x = decoder(x)

        x = self.final_decoder(x)
        return x
