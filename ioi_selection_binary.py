import json
import numpy as np
from PIL import Image
from utils.utils import rgb2id
from matplotlib import pyplot as plt
import math
import os

def predict(panoptic_model,saliency_model):
    image_dict=json.load(open("results/ioi_"+panoptic_model+".json",'r'))
    prediction_list = []
    gt_list = []
    count=0
    for image_id in image_dict:
        count+=1
        # print(str(count)+"/"+str(len(image_dict)))
        image_name=image_dict[image_id]['image_name']

        saliency_img = Image.open("../" + saliency_model + "/" + image_name.replace("jpg","png"))
        panoptic_img = Image.open("../" + panoptic_model+"/"+ image_name.replace("jpg","png")).convert('RGB')

        sal_img = np.array(saliency_img, dtype=np.uint8)
        sal_vals_ordered = np.sort(sal_img, axis=None)
        threshold = sal_vals_ordered[math.ceil(sal_img.size * 3 / 4) - 1]
        sal_mask = sal_img > threshold

        seg_img = rgb2id(np.array(panoptic_img, dtype=np.uint8))
        instances = image_dict[image_id]['segments_info']
        for instance_id in instances:
            instance_mask = seg_img == int(instance_id)
            intersection = instance_mask * sal_mask

            if np.count_nonzero(intersection) / np.count_nonzero(instance_mask) > 0.5:
                score = 1.0
            else:
                score = 0.0

            prediction_list.append(score)
            if instances[instance_id]['labeled']:
                label = 1
            else:
                label = 0
            gt_list.append(label)
    np.save("results/ciedn_result/"+panoptic_model+"_"+saliency_model+"_binary_gt.npy",np.array(gt_list))
    np.save("results/ciedn_result/"+panoptic_model+"_"+saliency_model+"_binary_pred.npy", np.array(prediction_list))


if __name__=='__main__':
    predict("CIN_panoptic_val","CIN_saliency_val")