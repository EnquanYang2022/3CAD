import torch
import torch.nn as nn
from torch.nn import functional as F


def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )
def upconv2x2(in_channels, out_channels, mode="transpose"):
    if mode == "transpose":
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
    else:
        return nn.Sequential(
            nn.Upsample(mode="bilinear", scale_factor=2),
            conv(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
        )
class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, merge_mode="concat", up_mode="transpose"):
        super(UNetUpBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.upconv = upconv2x2(self.in_channels, self.out_channels, mode=self.up_mode)

        if self.merge_mode == "concat":
            self.conv1 = conv(2 * self.out_channels, self.out_channels)
        else:
            self.conv1 = conv(self.out_channels, self.out_channels)
        self.bn1 = nn.BatchNorm2d(self.out_channels, eps=1e-05)
        self.relu1 = nn.ReLU()
        self.conv2 = conv(self.out_channels, self.out_channels)
        self.bn2 = nn.BatchNorm2d(self.out_channels, eps=1e-05)
        self.relu2 = nn.ReLU()

    def forward(self, from_up, from_down):


        from_up = self.upconv(from_up)

        if self.merge_mode == "concat":
            x = torch.cat((from_up, from_down), 1)
        elif self.merge_mode == "align":
            x = self.align(from_down,from_up)
        else:
            x = from_up
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))

        return x




class UNet(nn.Module):
    def __init__(self, n_channels=3, merge_mode="concat", up_mode="transpose"):
        super(UNet, self).__init__()
        self.n_chnnels = n_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.up1 = UNetUpBlock(1024, 512, merge_mode=self.merge_mode, up_mode=self.up_mode)
        self.up2 = UNetUpBlock(512, 256, merge_mode=self.merge_mode, up_mode=self.up_mode)
        self.up3 = UNetUpBlock(256, 128, merge_mode='None', up_mode=self.up_mode)

        self.up4 = UNetUpBlock(128, 64, merge_mode='None', up_mode=self.up_mode)
        self.conv_final = nn.Sequential(conv(64, 1, 3, 1, 1))

    def forward(self,student_feature,teacher_feature_ano,rec):

        a_mask = 1 - F.cosine_similarity(teacher_feature_ano[0], student_feature[0]).unsqueeze(dim=1)
        b_mask = 1 - F.cosine_similarity(teacher_feature_ano[1],student_feature[1]).unsqueeze(dim=1)
        c_mask = 1 - F.cosine_similarity(teacher_feature_ano[2],student_feature[2]).unsqueeze(dim=1)

        s1,s2,s3 = a_mask*teacher_feature_ano[0],b_mask*teacher_feature_ano[1],c_mask*teacher_feature_ano[2]

        rec1_mask = 1 - F.cosine_similarity(teacher_feature_ano[0], rec[0]).unsqueeze(dim=1)
        rec2_mask = 1 - F.cosine_similarity(teacher_feature_ano[1], rec[1]).unsqueeze(dim=1)
        rec3_mask = 1 - F.cosine_similarity(teacher_feature_ano[2], rec[2]).unsqueeze(dim=1)

        a,b,c = rec1_mask * s1, rec2_mask * s2, rec3_mask * s3
        x4 = self.up1(c, b)
        x3 = self.up2(x4, a)
        x2 = self.up3(x3, a)
        x1 = self.up4(x2, a)
        x0 = self.conv_final(x1)
        x=x0
        # x=torch.softmax(x0, dim=1)
        return x

