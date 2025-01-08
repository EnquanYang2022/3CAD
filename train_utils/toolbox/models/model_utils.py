import numpy as np
import torch
from torch.nn import functional as F

def cal_anomaly_map(ft_list,fs_list,segmentation_result,out_size=224, amap_mode='mul'):
    if amap_mode == 'mul':
        anomaly_map = np.ones([out_size, out_size])
    else:
        anomaly_map = np.zeros([out_size, out_size])
    a_map_list = []
    for i in range(len(fs_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map


        else:
            anomaly_map += a_map

    segmentation_result = torch.sigmoid(segmentation_result)
    #
    segmentation_result_map = F.interpolate(segmentation_result, size=out_size, mode='bilinear', align_corners=True)
    segmentation_result_map = segmentation_result_map[0, 0, :, :].to('cpu').detach().numpy()
    anomaly_map += segmentation_result_map
    return anomaly_map, a_map_list



