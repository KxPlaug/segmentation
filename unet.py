import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channels, num_classes, bilinear=True):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        self.factor = 2 if self.bilinear else 1

        self.conv1 = ConvBlock(self.in_channels, 64)
        self.down1 = Downscale(64, 128)
        self.down2 = Downscale(128, 256)
        self.down3 = Downscale(256, 512)
        self.down4 = Downscale(512, 1024//self.factor)

        self.up1 = Upscale(1024, 512//self.factor, bilinear)
        self.up2 = Upscale(512, 256//self.factor, bilinear)
        self.up3 = Upscale(256, 128//self.factor, bilinear)
        self.up4 = Upscale(128, 64, bilinear)

        self.conv2 = nn.Conv2d(64, self.num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        x = self.conv2(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim=None):
        super(ConvBlock, self).__init__()
        if not hidden_dim:
            hidden_dim = out_channels
        self.conv1 = nn.Conv2d(in_channels, hidden_dim,
                               kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden_dim, out_channels,
                               kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class Downscale(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downscale, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv(x)
        return x


class Upscale(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Upscale, self).__init__()
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = ConvBlock(in_channels, out_channels, in_channels//2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels//2, kernel_size=2, stride=2)
            self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
