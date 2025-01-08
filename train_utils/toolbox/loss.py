import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
'''
    对比学习相关损失函数
'''
import torch
import torch.nn.functional as F
import geomloss
#Focal loss
# class FocalLoss(nn.Module):
#     def __init__(self, gamma=0, alpha=None, size_average=True):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         if isinstance(alpha,(float,int,long)): self.alpha = torch.Tensor([alpha,1-alpha])
#         if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
#         self.size_average = size_average
#
#     def forward(self, input, target):
#         if input.dim()>2:
#             input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
#             input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
#             input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
#         target = target.view(-1,1)
#
#         logpt = F.log_softmax(input)
#         logpt = logpt.gather(1,target)
#         logpt = logpt.view(-1)
#         pt = Variable(logpt.data.exp())
#
#         if self.alpha is not None:
#             if self.alpha.type()!=input.data.type():
#                 self.alpha = self.alpha.type_as(input.data)
#             at = self.alpha.gather(0,target.data.view(-1))
#             logpt = logpt * Variable(at)
#
#         loss = -1 * (1-pt)**self.gamma * logpt
#         if self.size_average: return loss.mean()
#         else: return loss.sum()

#这个函数主要是为了使得教师的异常特征和学生的正常特征距离变大
def cal_score(score, target):
    if target == 1:
        return 1 - torch.mean(score)
    else:
        return max(0, torch.mean(score))
def loss_contra_nor_ano(a, b,mask,reduction='mean'):
    current_batchsize = a[0].shape[0]
    target = -torch.ones(current_batchsize).to('cuda')
    loss = 0

    for item in range(len(a)):
        scores = F.cosine_similarity(a[item],b[item])
        now_scores = F.interpolate(scores.unsqueeze(1), size=(mask.shape[-1], mask.shape[-2]), mode='bilinear')
        #首先让教师异常和学生正常拉大,此时教师输入的是正常图片特征，学生只有异常图片输入
        score1 = torch.mul(now_scores,mask)
        for i in range(current_batchsize):
            score1[i] = cal_score(score1[i],target[i])
        if reduction=='mean':
            loss += score1.mean()
        elif reduction == 'sum':
            loss += score1.sum()
    return loss/len(a)
#使得教师和学生的异常特征距离也变大
def loss_contra_ano_ano(a, b,mask,reduction='mean'):
    current_batchsize = a[0].shape[0]
    target = -torch.ones(current_batchsize).to('cuda')
    loss = 0

    for item in range(len(a)):
        scores = F.cosine_similarity(a[item],b[item])
        now_scores = F.interpolate(scores.unsqueeze(1), size=(mask.shape[-1], mask.shape[-2]), mode='bilinear')
        #让教师异常和学生异常拉大,此时教师输入的是异常图片特征，学生只有异常图片输入
        score1 = torch.mul(now_scores,mask)
        for i in range(current_batchsize):
            score1[i] = cal_score(score1[i],target[i])
        if reduction=='mean':
            loss += score1.mean()
        elif reduction == 'sum':
            loss += score1.sum()
    return loss/len(a)

#最优传输，再次拉近教师和学生正常特征
def loss_zuiyou_nor_nor(a, b,mask):
    # cos = torch.nn.CosineSimilarity()
    loss = 0

    for item in range(len(a)):
        #在正常特征进行余弦相似度计算
        t = 1 - F.cosine_similarity(a[item],b[item])
        now_t = F.interpolate(t.unsqueeze(1),size=(mask.shape[-1],mask.shape[-2]),mode='bilinear')
        # loss_wai = torch.mul(1-now_t,mask)#相当于余弦相似度的值与mask做乘法，只保留了GT目标上的余弦相似度的值
        false_mask = (1 - mask) #相当于背景的值为1，目标为0
        loss_nor = torch.mul(now_t,false_mask) #相当于1-余弦相似度的值与背景mask做乘法，相当于背景的地方是1-余弦相似度的值

        #对正常特征和异常特征进行contra损失计算
        all = loss_nor #总的加起来，就得到：目标为余弦相似度的值，背景为1-余弦相似度的值
        loss += torch.mean(all) #让整个loss变小，则loss_wai和loss_each都要变小，则根据公式推理。得出让目标部分余弦相似度变小，背景部分余弦相似度变大，即目标差异变大，背景差异变小
        # if torch.sum(mask) != 0:
        #     loss +=torch.mean(1 - loss_wai)
        # loss += torch.mean(1-cos_loss(a[item].view(a[item].shape[0],-1),
        #                               b[item].view(b[item].shape[0],-1)))
    return loss/3


def loss_cos_nor(a, b,mask):
    # cos = torch.nn.CosineSimilarity()
    loss = 0

    for item in range(len(a)):
        #在正常特征进行余弦相似度计算
        t = 1 - F.cosine_similarity(a[item],b[item])
        now_t = F.interpolate(t.unsqueeze(1),size=(mask.shape[-1],mask.shape[-2]),mode='bilinear')
        # loss_wai = torch.mul(1-now_t,mask)#相当于余弦相似度的值与mask做乘法，只保留了GT目标上的余弦相似度的值
        false_mask = (1 - mask) #相当于背景的值为1，目标为0
        loss_nor = torch.mul(now_t,false_mask) #相当于1-余弦相似度的值与背景mask做乘法，相当于背景的地方是1-余弦相似度的值

        #对正常特征和异常特征进行contra损失计算
        all = loss_nor #总的加起来，就得到：目标为余弦相似度的值，背景为1-余弦相似度的值
        loss += torch.mean(all) #让整个loss变小，则loss_wai和loss_each都要变小，则根据公式推理。得出让目标部分余弦相似度变小，背景部分余弦相似度变大，即目标差异变大，背景差异变小
        # if torch.sum(mask) != 0:
        #     loss +=torch.mean(1 - loss_wai)
        # loss += torch.mean(1-cos_loss(a[item].view(a[item].shape[0],-1),
        #                               b[item].view(b[item].shape[0],-1)))
    return loss/3


class Revisit_RDLoss(nn.Module):
    """
    receive multiple inputs feature
    return multi-task loss:  SSOT loss, Reconstruct Loss, Contrast Loss
    """
    def __init__(self, consistent_shuffle = True):
        super(Revisit_RDLoss, self).__init__()
        self.sinkhorn = geomloss.SamplesLoss(loss='sinkhorn', p=2, blur=0.05, \
                              reach=None, diameter=10000000, scaling=0.95, \
                                truncate=10, cost=None, kernel=None, cluster_scale=None, \
                                  debias=True, potentials=False, verbose=False, backend='auto')
    def forward(self, teacher_nor, rec_feature):
        """
        noised_feature : output of encoder at each_blocks : [noised_feature_block1, noised_feature_block2, noised_feature_block3]
        projected_noised_feature: list of the projection layer's output on noised_features, projected_noised_feature = projection(noised_feature)
        projected_normal_feature: list of the projection layer's output on normal_features, projected_normal_feature = projection(normal_feature)
        """
        loss_ssot =0.0
        for item in range(len(teacher_nor)):
            loss_ssot += self.sinkhorn(torch.softmax(teacher_nor[item].view(teacher_nor[item].shape[0], -1), -1),
                                      torch.softmax(rec_feature[item].view(rec_feature[item].shape[0], -1),-1))

        return loss_ssot/len(teacher_nor)
def loss_ano(a, b,mask):
    # cos = torch.nn.CosineSimilarity()
    loss = 0

    for item in range(len(a)):
        #print(a[item].shape)
        #print(b[item].shape)
        # loss += mse_loss(a[item], b[item])

        # b3=F.interpolate(b2,a[item].shape[-1],mode='bilinear', align_corners=True)
        t = 1 - F.cosine_similarity(a[item],b[item])
        now_t = F.interpolate(t.unsqueeze(1),size=(mask.shape[-1],mask.shape[-2]),mode='bilinear')
        loss_wai = torch.mul(1-now_t,mask)#相当于余弦相似度的值与mask做乘法，只保留了GT目标上的余弦相似度的值

        all = loss_wai
        loss += torch.mean(all) #让整个loss变小，则loss_wai和loss_each都要变小，则根据公式推理。得出让目标部分余弦相似度变小，背景部分余弦相似度变大，即目标差异变大，背景差异变小

    return loss/len(a)
def loss_nor(a, b,mask):
    # cos = torch.nn.CosineSimilarity()
    loss = 0

    for item in range(len(a)):
        #print(a[item].shape)
        #print(b[item].shape)
        # loss += mse_loss(a[item], b[item])

        # b3=F.interpolate(b2,a[item].shape[-1],mode='bilinear', align_corners=True)
        t = 1 - F.cosine_similarity(a[item],b[item])
        now_t = F.interpolate(t.unsqueeze(1),size=(mask.shape[-1],mask.shape[-2]),mode='bilinear')

        false_mask = (1 - mask) #相当于背景的值为1，目标为0
        loss_each = torch.mul(now_t,false_mask) #相当于1-余弦相似度的值与背景mask做乘法，相当于背景的地方是1-余弦相似度的值
        all = loss_each #总的加起来，就得到：目标为余弦相似度的值，背景为1-余弦相似度的值
        loss += torch.mean(all) #让整个loss变小，则loss_wai和loss_each都要变小，则根据公式推理。得出让目标部分余弦相似度变小，背景部分余弦相似度变大，即目标差异变大，背景差异变小

    return loss/len(a)
class FocalLoss(nn.Module):

    def __init__(self, smooth=1e-5, gamma=4, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.smooth = smooth
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        pt = input
        logpt = (pt + 1e-5).log()

        # add label smoothing
        num_class = input.shape[1]
        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != input.device:
            one_hot_key = one_hot_key.to(input.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth, 1.0 - self.smooth)
            logpt = logpt * one_hot_key

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = (-1 * (1 - pt) ** self.gamma * logpt).sum(1)
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


def loss_fucntion(a, b,mask):
    # cos = torch.nn.CosineSimilarity()
    loss = 0

    for item in range(len(a)):
        t = 1 - F.cosine_similarity(a[item],b[item])
        now_t = F.interpolate(t.unsqueeze(1),size=(mask.shape[-1],mask.shape[-2]),mode='bilinear')
        loss_wai = torch.mul(1-now_t,mask)#相当于余弦相似度的值与mask做乘法，只保留了GT目标上的余弦相似度的值
        false_mask = (1 - mask) #相当于背景的值为1，目标为0
        loss_each = torch.mul(now_t,false_mask) #相当于1-余弦相似度的值与背景mask做乘法，相当于背景的地方是1-余弦相似度的值
        all = loss_wai+loss_each #总的加起来，就得到：目标为余弦相似度的值，背景为1-余弦相似度的值
        loss += torch.mean(all) #让整个loss变小
    return loss/3


class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction="mean"):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        assert reduction in ["sum", "mean", "none"]
        self.reduction = reduction

    def forward(self, input, target):
        assert self.weight is None or isinstance(self.weight, torch.Tensor)
        ce = F.cross_entropy(input, target, reduction="none").view(-1)
        pt = torch.exp(-ce)
        if self.weight != None:
            target = target.view(-1)
            weights = self.weight[target]
        else:
            weights = torch.ones_like(target).view(-1)

        focal = weights * ((1 - pt) ** self.gamma)
        if self.reduction == "mean":
            return (focal * ce).sum() / weights.sum()

        elif self.reduction == "sum":
            return (focal * ce).sum()

        else:
            return focal * ce

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self,preds,masks):
        loss = 0.0
        for item in range(len(preds)):
            pred = preds[item]
            mask = masks[item]
            intersection = (pred*mask).sum(axis=(2,3))
            unior = (pred+mask).sum(axis=(2,3))
            dice = (2*intersection+1)/(unior+1)
            dice = torch.mean(1-dice)
            loss += dice
        return loss/len(preds)


class KLDLoss(nn.Module):
    def __init__(self, alpha=1, tau=1, resize_config=None, shuffle_config=None, transform_config=None,\
                 warmup_config=None, earlydecay_config=None):
        super().__init__()
        self.alpha_0 = alpha
        self.alpha = alpha
        self.tau = tau

        self.resize_config = resize_config
        self.shuffle_config = shuffle_config
        self.transform_config = transform_config
        self.warmup_config = warmup_config
        self.earlydecay_config = earlydecay_config

        self.KLD = torch.nn.KLDivLoss(reduction='batchmean')


    def forward(self, x_students, x_teachers, gt=None, n_iter=1):
        loss_all = 0.0
        for item in range(len(x_students)):
            x_student = x_students[item]
            x_teacher = x_teachers
            x_student = F.log_softmax(x_student / self.tau, dim=-1)
            x_teacher = F.softmax(x_teacher / self.tau, dim=-1)

            loss = self.KLD(x_student, x_teacher)
            loss = self.alpha * loss
            loss_all+=loss
        return loss_all


def loss_distill1(a, b):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(b)):
        loss += torch.mean(1-cos_loss(a[item].view(a[item].shape[0],-1),
                                      b[item].view(b[item].shape[0],-1)))
    return loss



def L2(f_):
    return (((f_**2).sum(dim=1))**0.5).reshape(f_.shape[0],1,f_.shape[2],f_.shape[3]) + 1e-8
def similarity(feat):
    feat = feat.float()
    tmp = L2(feat).detach()
    feat = feat/tmp
    feat = feat.reshape(feat.shape[0],feat.shape[1],-1)
    return torch.einsum('icm,icn->imn', [feat, feat])
def sim_dis_compute(f_S, f_T):
    sim_err = ((similarity(f_T) - similarity(f_S))**2)/((f_T.shape[-1]*f_T.shape[-2])**2)/f_T.shape[0]
    sim_dis = sim_err.sum()
    return sim_dis
class CriterionPairWiseforWholeFeatAfterPool(nn.Module):
    def __init__(self, scale=1):
        '''inter pair-wise loss from inter feature maps'''
        super(CriterionPairWiseforWholeFeatAfterPool, self).__init__()
        self.criterion = sim_dis_compute
        self.scale = scale
    def forward(self, preds_Ss, preds_Ts):
        loss_all = 0.0
        for preds_S, preds_T in zip(preds_Ss, preds_Ts):
            feat_S = preds_S
            feat_T = preds_T


            total_w, total_h = feat_T.shape[2], feat_T.shape[3]
            patch_w, patch_h = int(total_w*self.scale), int(total_h*self.scale)
            maxpool = nn.MaxPool2d(kernel_size=(patch_w, patch_h), stride=(patch_w, patch_h), padding=0, ceil_mode=True) # change
            loss = self.criterion(maxpool(feat_S), maxpool(feat_T))
            loss_all+=loss
        return loss_all

def hcl(fstudent, fteacher):
    loss_all = 0.0
    for fs, ft in zip(fstudent, fteacher):
        n,c,h,w = fs.shape
        loss = F.mse_loss(fs, ft, reduction='mean')
        cnt = 1.0
        tot = 1.0
        for l in [4,2,1]:
            if l >=h:
                continue
            tmpfs = F.adaptive_avg_pool2d(fs, (l,l))
            tmpft = F.adaptive_avg_pool2d(ft, (l,l))
            cnt /= 2.0
            loss += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt
            tot += cnt
        loss = loss / tot
        loss_all = loss_all + loss
    return loss_all

def focal_loss(inputs, targets, alpha=-1, gamma=4, reduction="mean"):
    inputs = inputs.float()
    targets = targets.float()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = inputs * targets + (1 - inputs) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

def l1_loss(inputs, targets, reduction="mean"):
    return F.l1_loss(inputs, targets, reduction=reduction)
def MSE_loss(a, b):
    # mse_loss = torch.nn.MSELoss()
    mse_loss = torch.nn.MSELoss()
    loss = 0
    for item in range(len(b)):
        loss += mse_loss(a[item],b[item])
    return loss/len(a)

class SoftLoULoss(nn.Module):
    def __init__(self):
        super(SoftLoULoss, self).__init__()

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        smooth = 1

        intersection = pred * target

        intersection_sum = torch.sum(intersection, dim=(1,2,3))
        pred_sum = torch.sum(pred, dim=(1,2,3))
        target_sum = torch.sum(target, dim=(1,2,3))
        loss = (intersection_sum + smooth) / \
               (pred_sum + target_sum - intersection_sum + smooth)

        loss = 1 - torch.mean(loss)

        return loss


class SoftIoUL1NromLoss(nn.Module):
    def __init__(self, lambda_iou=0.8, lambda_l1=0.2):
        super(SoftIoUL1NromLoss, self).__init__()
        self.softiou = SoftLoULoss()
        self.lambda_iou = lambda_iou
        self.lambda_l1 = lambda_l1

    def forward(self, pred, target):
        iouloss = self.softiou(pred, target)

        batch_size, C, height, width = pred.size()
        pred = (pred > 0).float()
        l1loss = torch.sum(pred) / (batch_size * C * height * width)

        loss = self.lambda_iou * iouloss + self.lambda_l1 * l1loss
        return loss

if __name__ == '__main__':
    in1 = torch.randn(1,3,256,256)
    in2 = torch.randn(1,3,256,256)
    loss = KLDLoss()
    out1,out2 = loss(in1,in2)
    print(out1)
    print(out2)