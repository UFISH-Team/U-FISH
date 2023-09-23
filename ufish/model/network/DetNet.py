import torch.nn as nn
import torch


class ConvNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                             kernel_size, padding=padding)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x

class custom_sigmoid(nn.Module):
    def __init__(self, alpha):
        super(custom_sigmoid, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return torch.sigmoid(x - self.alpha)



class DetNet(nn.Module):
    def __init__(self, input_channel: int = 1, alpha=0.5):
        super(DetNet, self).__init__()

        self.conv1 = ConvNorm(1, 16)
        self.skip1 = ConvNorm(16, 16)
        self.skip2 = ConvNorm(16, 16)

        self.conv2 = ConvNorm(16, 32)
        self.skip3 = ConvNorm(32, 32)
        self.skip4 = ConvNorm(32, 32)

        self.conv3 = ConvNorm(32, 64)
        self.skip5 = ConvNorm(64, 64)
        self.skip6 = ConvNorm(64, 64)

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv4 = ConvNorm(64, 32)
        self.skip7 = ConvNorm(32, 32)
        self.skip8 = ConvNorm(32, 32)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv5 = ConvNorm(32, 16)
        self.skip9 = ConvNorm(16, 16)
        self.skip10 = ConvNorm(16, 16)

        self.logit = nn.Conv2d(16, 1, kernel_size=1)
        self.sigmoid = custom_sigmoid(alpha)

    def forward(self, x):
        conv1 = self.conv1(x)
        skip1 = self.skip1(conv1)
        skip2 = self.skip2(skip1)
        add1 = conv1 + skip2
        max1 = nn.functional.max_pool2d(add1, 2)

        conv2 = self.conv2(max1)
        skip3 = self.skip3(conv2)
        skip4 = self.skip4(skip3)
        add2 = conv2 + skip4
        max2 = nn.functional.max_pool2d(add2, 2)

        conv3 = self.conv3(max2)
        skip5 = self.skip5(conv3)
        skip6 = self.skip6(skip5)
        add3 = conv3 + skip6

        up1 = self.up1(add3)
        conv4 = self.conv4(up1)
        skip7 = self.skip7(conv4)
        skip8 = self.skip8(skip7)

        up2 = self.up2(skip8)
        conv5 = self.conv5(up2)
        skip9 = self.skip9(conv5)
        skip10 = self.skip10(skip9)

        logit = self.logit(skip10)
        outputs = self.sigmoid(logit)

        return outputs
        