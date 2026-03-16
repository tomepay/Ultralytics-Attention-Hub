import torch
from torch import nn

class EMA(nn.Module):
    def __init__(self, channels, factor=32):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(dim=-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g, c//g, h, w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * self.softmax(self.self_attn(x_h, x_w)))
        x2 = self.conv3x3(group_x)
        x11 = self.self_attn(self.agp(x1), x2.reshape(b * self.groups, -1, h * w).permute(0, 2, 1))
        x12 = self.self_attn(self.agp(x2), x1.reshape(b * self.groups, -1, h * w).permute(0, 2, 1))
        v_n = (x11 * x12).reshape(b * self.groups, -1, h, w)
        return v_n.reshape(b, c, h, w)

    def self_attn(self, x, y):
        return (x.reshape(x.size(0), x.size(1), -1) * y.reshape(y.size(0), y.size(1), -1)).reshape(x.size(0), x.size(1), -1)