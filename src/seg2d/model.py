"""U-Net with batch normalization and nearest-neighbor decoder upsampling."""

import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class UNet(nn.Module):
    """Four-level encoder-decoder. Output is N x classes x H x W logits."""

    def __init__(self, n_channels: int = 3, n_classes: int = 4, base_channels: int = 64):
        super().__init__()
        if any(type(n) is not int or n < 1 for n in (n_channels, n_classes, base_channels)):
            raise ValueError("Channel counts must be positive integers.")
        self.n_channels, self.n_classes = n_channels, n_classes
        c = base_channels
        self.inc = DoubleConv(n_channels, c)
        self.down1 = DoubleConv(c, 2*c)
        self.down2 = DoubleConv(2*c, 4*c)
        self.down3 = DoubleConv(4*c, 8*c)
        self.down4 = DoubleConv(8*c, 16*c)
        self.pool = nn.MaxPool2d(2)
        self.up1 = DoubleConv(24*c, 8*c)
        self.up2 = DoubleConv(12*c, 4*c)
        self.up3 = DoubleConv(6*c, 2*c)
        self.up4 = DoubleConv(3*c, c)
        self.outc = nn.Conv2d(c, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.n_channels or min(x.shape[-2:]) < 16:
            raise ValueError("Expected NCHW input with configured channels and H,W >= 16.")
        x1 = self.inc(x)
        x2 = self.down1(self.pool(x1))
        x3 = self.down2(self.pool(x2))
        x4 = self.down3(self.pool(x3))
        x = self.down4(self.pool(x4))
        for skip, block in ((x4, self.up1), (x3, self.up2), (x2, self.up3), (x1, self.up4)):
            # Explicit skip sizes also support odd input dimensions.
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = block(torch.cat((x, skip), dim=1))
        return self.outc(x)
