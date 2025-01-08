import os
from typing import Union, List

import torch
from torch.utils import data
import time
from scipy.ndimage import gaussian_filter
import numpy as np
import cv2
import datetime
from train_utils.toolbox.metrics import compute_pro
from sklearn.metrics import roc_auc_score, average_precision_score
from train_utils.toolbox.utils import get_dataset,get_model
from train_utils.toolbox.models.model_utils import cal_anomaly_map
def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def main(args,class_name):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    assert os.path.exists(args.weights), f"weights {args.weights} not found."
    _, testset = get_dataset(data_name=args.data_name, data_path=args.data_path, class_name=class_name)
    num_workers = 4
    test_data_loader = data.DataLoader(testset,
                                      batch_size=1,  # must be 1
                                      num_workers=num_workers,
                                      pin_memory=True,
                                      )

    model = get_model(model_name=args.model_name)
    ckt_path = './save_weights/'+'model_'+class_name+'.pth'
    pretrain_weights = torch.load(ckt_path, map_location='cpu')
    if "model" in pretrain_weights:
        model.load_state_dict(pretrain_weights["model"])
    else:
        model.load_state_dict(pretrain_weights)
    model.to(device)
    t_start = time_synchronized()
    model.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    aupro_list = []
    with torch.no_grad():
        for i, sample in enumerate(test_data_loader):
            img, label,targets = sample['img'].to(device), sample['label'].to(device),sample['mask'].to(device)
            teacher_feature, rec, segment_result = model(img)
            anomaly_map, _ = cal_anomaly_map(teacher_feature, rec, segment_result, img.shape[-1],
                                             amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            targets[targets > 0.5] = 1
            targets[targets <= 0.5] = 0
            if label.item() != 0:
                aupro_list.append(compute_pro(targets.squeeze(0).cpu().numpy().astype(int),
                                              anomaly_map[np.newaxis, :, :]))
            gt_list_px.extend(targets.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(anomaly_map.ravel())
            gt_list_sp.append(np.max(targets.cpu().numpy().astype(int)))
            pr_list_sp.append(np.max(anomaly_map))

        auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 3)
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 3)
        aupro_px = round(np.mean(aupro_list), 3)
        ap_pixel = round(average_precision_score(gt_list_px, pr_list_px), 3)

    t_end = time_synchronized()
    print("inference time: {}".format(t_end - t_start))
    return auroc_px, auroc_sp, aupro_px, ap_pixel

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch model validation")
    parser.add_argument("--data_name", default="mvtec2d", help="data_name")
    parser.add_argument("--model_name", default="cfrg", help="model_name_name")
    parser.add_argument("--data-path", default="/media/Data1/yeq/dataset/MvtecAD/", help="DUTS root")
    parser.add_argument("--weights", default="./save_weights/model_aluminum_ipad.pth")
    parser.add_argument("--device", default="cuda:0", help="training device")
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    s_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_path = "./test_result/{}".format(s_time)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # MVTec AD
    item_list = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                 'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood']
    #3CAD
    # item_list = [
    # "Aluminum_Camera_Cover",
    # "Aluminum_Ipad",
    # "Aluminum_Middle-Frame",
    # "Aluminum_New_Middle_Frame",
    # "Aluminum_New_Ipad",
    # "Aluminum_Pc",
    # "Copper_Stator",
    # "Iron_Stator",
    # ]
    total_roc_auc = []
    total_image_roc_auc = []
    total_aupro_px = []
    total_ap_pixel = []
    for cls in item_list:
        save_path = "./test_result/{}/{}.txt".format(s_time, cls)
        print(cls)
        auroc_px, auroc_sp, aupro_px, ap_pixel =  main(args,class_name=cls)
        total_roc_auc.append(auroc_px)
        total_image_roc_auc.append(auroc_sp)
        total_aupro_px.append(aupro_px)
        total_ap_pixel.append(ap_pixel)

        print('Average pixel ROCAUC: %.3f' % auroc_px)

        print('Average image ROCUAC: %.3f' % auroc_sp)

        print('Average PRO: %.3f' % aupro_px)

        print('Average AP: %.3f' % ap_pixel)
        with open(save_path, "a") as f:
            # 记录每个epoch对应的train_loss、lr以及验证集各指标
            write_info = f"cls_name: {cls} auroc_px: {total_roc_auc:.3f} auroc_sp: {auroc_sp:.3f} aupro_px: {aupro_px:.3f} ap_pixel: {ap_pixel:.3f}\n"
            f.write(write_info)
    print("average_all_classes")
    print('Average pixel ROCAUC: %.3f' % np.mean(total_roc_auc))

    print('Average image ROCUAC: %.3f' % np.mean(total_image_roc_auc))

    print('Average PRO: %.3f' % np.mean(total_aupro_px))

    print('Average AP: %.3f' % np.mean(total_ap_pixel))
    with open(save_path, "a") as f:
        # 记录每个epoch对应的train_loss、lr以及验证集各指标
        write_info = f"cls_name: {cls} auroc_px: {np.mean(total_roc_auc):.3f} auroc_sp: {np.mean(auroc_sp):.3f} aupro_px: {np.mean(aupro_px):.3f} ap_pixel: {np.mean(ap_pixel):.3f}\n"
        f.write(write_info)
