# ultralytics/nn/modules/attention/coordatt.py
import torch
import torch.nn as nn

class CoordAtt(nn.Module):
    """Coordinate Attention (CA)"""
    def __init__(self, c, reduction=32):
        super().__init__()
        mip = max(8, c // reduction)

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.conv1 = nn.Conv2d(c, mip, 1, 1, 0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU(inplace=True)

        self.conv_h = nn.Conv2d(mip, c, 1, 1, 0)
        self.conv_w = nn.Conv2d(mip, c, 1, 1, 0)

    def forward(self, x):
        identity = x
        _, _, h, w = x.shape

        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        return identity * a_w * a_h