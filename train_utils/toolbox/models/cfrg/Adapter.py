import torch
import torch.nn as nn
import math
from timm.models.layers import _assert, trunc_normal_
class SepConv(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=3, stride=1, padding=1, affine=True):
        #   depthwise and pointwise convolution, downsample by 2
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            # nn.Upsample(scale_factor=2),
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=stride, padding=padding,
                      groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=1, padding=padding, groups=channel_in,
                      bias=False),
            nn.Conv2d(channel_in, channel_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_out, affine=affine),
            nn.ReLU(inplace=False),
        )


    def forward(self, x):
        return self.op(x)
def init_weights(module):
    for n, m in module.named_modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

class Adapter(nn.Module):
    def __init__(self,channel_in,channel_out):
        super(Adapter, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(channel_in, channel_in, kernel_size=3, stride=1, padding=1,
                      groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_in, affine=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.Conv2d(channel_in, channel_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_out, affine=True),
            nn.ReLU(inplace=False)
        )


    def forward(self, x):

        x = self.op(x)
        return x




if __name__ == '__main__':
    input = torch.randn(1,49,512)
    model = Adapter(c_in=512)
    out = model(input)
    print(out.shape)