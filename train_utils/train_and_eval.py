import torch
from train_utils.toolbox.models.model_utils import cal_anomaly_map
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import train_utils.loger_utils as utils
from train_utils.toolbox.metrics import compute_pro
from train_utils.toolbox.loss import loss_distill1,loss_fucntion
def evaluate(model, data_loader, device):

    model.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    aupro_list = []
    with torch.no_grad():
        for i, sample in enumerate(data_loader):
            img_origin, label,targets = sample['img'].to(device), sample['label'].to(device),sample['mask'].to(device)
            #model1
            teacher_feature,rec,segment_result= model(img_origin)
            anomaly_map, _ = cal_anomaly_map(teacher_feature,rec,segment_result,img_origin.shape[-1],
                                             amap_mode='a')

            anomaly_map = gaussian_filter(anomaly_map, sigma=4)


            targets[targets > 0.5] = 1
            targets[targets <= 0.5] = 0
            # print(i)
            if label.item()!=0:
                aupro_list.append(compute_pro(targets.squeeze(0).cpu().numpy().astype(int),
                                          anomaly_map[np.newaxis,:,:]))
            gt_list_px.extend(targets.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(anomaly_map.ravel())
            gt_list_sp.append(np.max(targets.cpu().numpy().astype(int)))
            pr_list_sp.append(np.max(anomaly_map))

        auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 3)
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 3)
        aupro_px = round(np.mean(aupro_list), 3)
        ap_pixel = round(average_precision_score(gt_list_px, pr_list_px), 3)

    return auroc_px,auroc_sp,aupro_px,ap_pixel

def train_one_epoch(model, optimizer, data_loader, device, lr_scheduler, scaler=None):

    model.train()
    model.teacher.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    #定义loss:
    criterion_cos = loss_distill1
    criterion_bce = torch.nn.BCEWithLogitsLoss()
    criterion_function = loss_fucntion
    for i,sample in enumerate(data_loader):
        img_ori,img_aug,aug_mask = sample['img_origin'].to(device),sample['img_aug'].to(device),sample['mask'].to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            teacher_feature_nor,teacher_feature_ano,student_feature,rec,segment_result = model(img_ori,img_aug)

            loss_dis = criterion_function(teacher_feature_ano,student_feature,aug_mask)
            loss_rec = criterion_cos(teacher_feature_nor,rec)
            loss_seg = criterion_bce(segment_result,aug_mask)
            loss = loss_seg +loss_dis+loss_rec
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

    lr_scheduler.step()

    lr = optimizer.param_groups[0]["lr"]
    metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr


