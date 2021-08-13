# -*- coding: utf-8 -*-

import os
import cv2
from tqdm import tqdm
from PIL import Image
# pip install pysodmetrics
from py_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure,IoU,CC
import numpy as np
from skimage import io
import scipy.misc
import imageio
_EPS = 1e-16
_TYPE = np.float64

#FM = Fmeasure()
#WFM = WeightedFmeasure()
#SM = Smeasure()
EM = Emeasure()
MAE = MAE()
IoU = IoU()
CC = CC()
#str_2=""
#
#str_3=""
    # data_root = r"E:\ijcai整理\代码相关\maps\maps\Ours\divide_1\maps"
pred_root = "/home/lhc/OSAFFDET/net_14/save_models/divide_3_PAD_2/epoch_18_iter_1200"

mask_root = "/home/lhc/dataset/Co-saliency/co-aff/all_data_4/divide_6/test/gt/"

mask_name_list = sorted(os.listdir(mask_root))
for mask_name in tqdm(mask_name_list, total=len(mask_name_list)):
    mask_file = os.path.join(mask_root, mask_name)
    mask_images = os.listdir(mask_file)
    class_em=[]
    class_fm=[]
    for mask_image in mask_images:
        mask_path = os.path.join(mask_root, mask_name, mask_image)
        pred_path = os.path.join(pred_root, mask_name, mask_image)

        if os.path.exists(mask_path) and os.path.exists(pred_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            if pred.shape != mask.shape:
                pred = cv2.resize(pred, (mask.shape[1], mask.shape[0]))

            EM.step(pred=pred, gt=mask)

            MAE.step(pred=pred, gt=mask)
            IoU.step(pred=pred, gt=mask)
            CC.step(pred=pred, gt=mask)

em = EM.get_results()["em"]
mae = MAE.get_results()["mae"]
iou = IoU.get_results()['iou']
cc = CC.get_results()['cc']

results = {

        #"Smeasure": sm,
        #"wFmeasure": wfm,
        "MAE": mae,
        "adpEm": em["adp"],
        "meanEm": em["curve"].mean(),
        "maxEm": em["curve"].max(),
        #"adpFm": fm["adp"],
        #"meanFm": fm["curve"].mean(),
        #"maxFm": fm["curve"].max(),
        'IoU': iou,
        'CC': cc,
    }

print(results)
str_1 = ""
for key in results:
    str_1 += key + " " + str(results[key]) + "\n"
    #print(os.path.join(pred_root, 'results.txt'))
if os.path.exists(os.path.join(pred_root, 'results_2.txt')):
    os.unlink(os.path.join(pred_root, 'results_2.txt'))
with open(os.path.join(pred_root, 'results_2.txt'), 'a') as f:
    f.write(str_1)

