import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(Conv -> BN -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNetEncoder5(nn.Module):
    """
    U-Net style encoder producing 5 scales:
      f1: 1x     [B, dim,   H,   W]
      f2: 1/2    [B, 2dim,  H/2, W/2]
      f3: 1/4    [B, 4dim,  H/4, W/4]
      f4: 1/8    [B, 8dim,  H/8, W/8]
      f5: 1/16   [B,16dim,  H/16,W/16]

    Notes:


    """

    def __init__(self, in_channels: int, dim: int, bias: bool = False, norm: str = "bn"):
        super().__init__()
        dim = int(dim)
        self.dim = dim

        # 1x
        self.enc1 = DoubleConv(in_channels, dim)

        def down_block(in_ch, out_ch):
            layers = [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=bias),
                nn.BatchNorm2d(out_ch) if norm == "bn" else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=bias),
                nn.BatchNorm2d(out_ch) if norm == "bn" else nn.Identity(),
                nn.ReLU(inplace=True),
            ]
            return nn.Sequential(*layers)

        self.enc2 = down_block(dim, dim * 2)       # 1/2
        self.enc3 = down_block(dim * 2, dim * 4)   # 1/4
        self.enc4 = down_block(dim * 4, dim * 8)   # 1/8
        self.enc5 = down_block(dim * 8, dim * 16)  # 1/16

    def forward(self, x: torch.Tensor):
        f1 = self.enc1(x)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        f4 = self.enc4(f3)
        f5 = self.enc5(f4)
        return f1, f1, f2, f3, f4, f5
