# -*- coding: utf-8 -*-

import os
import cv2
from tqdm import tqdm
from PIL import Image
# pip install pysodmetrics
from py_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure, IoU, CC
import numpy as np
from skimage import io
import scipy.misc
import imageio

_EPS = 1e-16
_TYPE = np.float64

FM = Fmeasure()
# WFM = WeightedFmeasure()
# SM = Smeasure()
EM = Emeasure()
MAE = MAE()
IoU = IoU()
CC = CC()
# str_2=""
#
# str_3=""
# data_root = r"E:\ijcai整理\代码相关\maps\maps\Ours\divide_1\maps"
data_root="/home/lhc/OSAFFDET/net_14/save_models/divide_1_1_b_2_2_k_64_sn_4/"
mask_root = "/home/lhc/dataset/AFF_dataset/divide_1/test/gt/"
map_files=os.listdir(data_root)
for map_file in map_files:
    if map_file[-3:]=="pth":
        continue
    pred_root=os.path.join(data_root,map_file)
    print(pred_root)
    #pred_root = "/home/lhc/OSAFFDET/net_14/save_models/divide_3_PAD_2/epoch_31_iter_1200"
    EM.adaptive_ems=[]
    EM.changeable_ems=[]
    MAE.maes=[]
    IoU.IoUs=[]
    CC.CCs=[]
    FM.changeable_fms=[]

    mask_name_list = sorted(os.listdir(mask_root))
    for mask_name in tqdm(mask_name_list, total=len(mask_name_list)):
        mask_file = os.path.join(mask_root, mask_name)
        mask_images = os.listdir(mask_file)

        for mask_image in mask_images:
            mask_path = os.path.join(mask_root, mask_name, mask_image)
            pred_path = os.path.join(pred_root, mask_name, mask_image)

            if os.path.exists(mask_path) and os.path.exists(pred_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
                if pred.shape != mask.shape:
                    pred = cv2.resize(pred, (mask.shape[1], mask.shape[0]))

                EM.step(pred=pred, gt=mask)
                FM.step(pred=pred, gt=mask)

                MAE.step(pred=pred, gt=mask)
                IoU.step(pred=pred, gt=mask)
                CC.step(pred=pred, gt=mask)

    em = EM.get_results()["em"]
    mae = MAE.get_results()["mae"]
    iou = IoU.get_results()['iou']
    cc = CC.get_results()['cc']
    fm = FM.get_results()["fm"]

    results = {

        # "Smeasure": sm,
        # "wFmeasure": wfm,
        "MAE": mae,
        "adpEm": em["adp"],
        "meanEm": em["curve"].mean(),
        "maxEm": em["curve"].max(),
        # "adpFm": fm["adp"],
        "meanFm": fm["curve"].mean(),
        # "maxFm": fm["curve"].max(),
        'IoU': iou,
        'CC': cc,
    }

    print(results)
    str_1 = ""
    for key in results:
        str_1 += key + " " + str(results[key]) + "\n"
        # print(os.path.join(pred_root, 'results.txt'))
    if os.path.exists(os.path.join(pred_root, 'results_3.txt')):
        os.unlink(os.path.join(pred_root, 'results_3.txt'))
    with open(os.path.join(pred_root, 'results_3.txt'), 'a') as f:
        f.write(str_1)


