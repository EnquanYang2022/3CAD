import timm
import torch
import torch.nn as nn
from .resnet import wide_resnet50_2
from .de_resnet import de_wide_resnet50_2
from .Adapter import Adapter
from .efficientnet.model import EfficientNet
from .Segmentation import UNet

class cfrg_net(nn.Module):
    def __init__(self,model_name = 'ViT-B/14'):  #RN50  ViT-B/32
        super().__init__()
        self.teacher,self.bn = wide_resnet50_2(pretrained=True)
        self.decoder = de_wide_resnet50_2(pretrained=False)
        self.student = EfficientNet.from_name('efficientnet-b0', outblocks=[2, 4, 10], outstrides=[4, 8, 16])
        self.segment = UNet()
        self.adapter1 = Adapter(channel_in=24, channel_out=256)
        self.adapter2 = Adapter(channel_in=40, channel_out=512)
        self.adapter3 = Adapter(channel_in=112, channel_out=1024)
        for name, value in self.teacher.named_parameters():  # 冻结指定层的参数
            value.requires_grad = False

    def forward(self, img,img_aug=None):
        if img_aug!=None:
            teacher_feature_nor = self.teacher(img)
            teacher_feature_ano = self.teacher(img_aug)
            student_feature = self.student(img_aug)['features']
            student_feature[0] = self.adapter1(student_feature[0])
            student_feature[1] = self.adapter2(student_feature[1])
            student_feature[2] = self.adapter3(student_feature[2])
            rec = self.decoder(self.bn(teacher_feature_ano))
            segment_result = self.segment(student_feature,teacher_feature_ano,rec)
            return teacher_feature_nor,teacher_feature_ano,student_feature,rec,segment_result
        else:
            teacher_feature = self.teacher(img)
            student_feature = self.student(img)['features']
            student_feature[0] = self.adapter1(student_feature[0])
            student_feature[1] = self.adapter2(student_feature[1])
            student_feature[2] = self.adapter3(student_feature[2])
            rec = self.decoder(self.bn(teacher_feature))
            segment_result = self.segment(student_feature, teacher_feature, rec)
            return teacher_feature,rec,segment_result



if __name__ == '__main__':
    input_origin = torch.randn(2,3,224,224).cuda()
    input_aug = torch.randn(2,3,640,640)
    model = cfrg_net().cuda()
    teacher_feature,student_feature = model(input_origin)
    print(teacher_feature.shape)

    for it in student_feature:
        print(it.shape)