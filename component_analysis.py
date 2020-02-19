import os
import json
import skimage.io
import torch
import torch.nn as nn
import numpy as np
import scipy
from torch.autograd import Variable
from torch import FloatTensor


import config
from config import Config
from DatasetLib import OOIDataset,Dataset
from ioi_selection.CIEDN import CIEDN
from compute_metric import compare_mask, write_csv, draw_pictures

import argparse
import yaml

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ins_ext", type=str,
                        default='CIN_panoptic_all_old',
                        help="the path of the image to be detected")
    parser.add_argument("--sem_ext", type=str,
                        default='CIN_semantic_all_old',
                        help="the path of the image to be detected")
    parser.add_argument("--p_intr_ext", type=str,
                        default='CIN_saliency_all_old',
                        help="the path of the image to be detected")
    parser.add_argument("--config", type=str,
                        default="configs/component_analysis_config.yaml",
                        help="the config file path")
    return parser


class CINConfig(Config):
    NAME = "ooi"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1 #10
    NUM_CLASSES = 1+133
    THING_NUM_CLASSES = 1+80
    STUFF_NUM_CLASSES = 1+53


def predict(config, panoptic_model, semantic_model, saliency_model):
    log_file="logs/CIN_ooi_100_selection.pth"
    ciedn = CIEDN().cuda()
    state_dict = torch.load(log_file)
    ciedn.load_state_dict(state_dict, strict=False)

    images = json.load(open("data/middle/ioi_val_images_dict_"+panoptic_model+"_"+saliency_model+".json", 'r'))
    class_dict=json.load(open("data/class_dict.json", 'r'))
    prediction_list = []
    gt_list = []
    count=0
    for image_id in images:
        count+=1
        print(str(count)+"/"+str(len(images)))
        try:
            image=images[image_id]
            # image_id = image['image_id']
            image_name = image['image_name']
            image_width = image['width']
            image_height = image['height']
            scale = 1024 / max(image_height, image_width)
            new_height = round(image_height * scale)
            new_width = round(image_width * scale)
            top_pad = (1024 - new_height) // 2
            bottom_pad = 1024 - new_height - top_pad
            left_pad = (1024 - new_width) // 2
            right_pad = 1024 - new_width - left_pad

            segments_info = image['segments_info']

            labels = []
            boxes = []
            class_ids = []
            for instance_id in segments_info:
                segment_info=segments_info[instance_id]
                category_id = segment_info['category_id']
                class_id = 0
                for idx in class_dict:
                    if class_dict[idx]['category_id'] == category_id:
                        class_id = class_dict[idx]['class_id']
                islabel = segment_info['labeled']
                box = [int(segment_info['bbox'][0]*scale+top_pad),int(segment_info['bbox'][1]*scale+left_pad),int(segment_info['bbox'][2]*scale+top_pad),int(segment_info['bbox'][3]*scale+left_pad)]
                labels.append(islabel)
                class_ids.append(class_id)
                boxes.append(np.array(box))
                class_ids.append(class_id)
            boxes=np.stack(boxes)

            # real
            semantic_img = skimage.io.imread("../"+semantic_model+"/" + image_name.replace("jpg", "png"))
            semantic_img = scipy.misc.imresize(semantic_img, (new_height, new_width), interp='nearest')
            if len(semantic_img.shape)==3:
                semantic_img=semantic_img[:,:,0]
            semantic_label = np.pad(semantic_img, [(top_pad, bottom_pad), (left_pad, right_pad)], mode='constant',constant_values=0)

            saliency_map = skimage.io.imread("../"+saliency_model+"/" + image_name.replace("jpg", "png"))
            saliency_map = scipy.misc.imresize(saliency_map, (new_height, new_width), interp='nearest')
            if len(saliency_map.shape)==3:
                saliency_map=saliency_map[:,:,0]
            saliency_map = np.pad(saliency_map, [(top_pad, bottom_pad), (left_pad, right_pad)], mode='constant',constant_values=0)

            instance_groups = []
            for i in range(boxes.shape[0]):
                y1, x1, y2, x2 = boxes[i][:4]

                instance_group = []

                instance_label = semantic_label[y1:y2, x1:x2]
                instance_label = scipy.misc.imresize(instance_label, (56, 56), interp='nearest') / 134.0
                instance_group.append(instance_label)

                instance_map = saliency_map[y1:y2, x1:x2]
                instance_map = scipy.misc.imresize(instance_map, (56, 56), interp='bilinear') / 255.0
                instance_group.append(instance_map)
                instance_group = np.stack(instance_group)
                instance_groups.append(instance_group)

            instance_groups = np.stack(instance_groups)
            class_ids = np.array(class_ids, dtype=np.float32)
            labels = np.array(labels, dtype=np.float32)

            instance_groups = Variable(FloatTensor(instance_groups)).float().cuda().unsqueeze(0)
            boxes = Variable(FloatTensor(boxes)).float().cuda().unsqueeze(0)
            class_ids = Variable(FloatTensor(class_ids)).float().cuda().unsqueeze(0)
            predictions = ciedn(instance_groups).squeeze(1).data.cpu().numpy()

            num = instance_groups.shape[1]
            for i in range(0, num):
                avg = np.sum(predictions[i * num:(i + 1) * num]) / num
                prediction_list.append(avg)
            gt_list.extend(labels)
        except Exception as e:
            print(str(count)+":", e)

    prediction_list = np.array(prediction_list)
    gt_list = np.array(gt_list)
    np.save("results/ciedn_result/"+panoptic_model+"_"+saliency_model+"_gt.npy", gt_list)
    np.save("results/ciedn_result/"+panoptic_model+"_"+saliency_model+"_pred.npy", prediction_list)

def compute_metric(panoptic_model, saliency_model):
    result = {}
    for a2 in np.arange(0.1, 3, 0.1):
        result[str(round(a2, 2))] = {}
        # gt = np.load("results/ciedn_result/"+panoptic_model+"_"+saliency_model+"_gt.npy")
        # pred = np.load("results/ciedn_result/"+panoptic_model+"_"+saliency_model+"_pred.npy")
        gt = np.load("results/ciedn_result/gt4.npy")
        pred = np.load("results/ciedn_result/pred4.npy")
        precision,recall,f=compare_mask(gt,pred,a2)
        result[str(round(a2,2))]={"precision":precision,"recall":recall,"f":f}
    # json.dump(result,open("results/ciedn_result/"+panoptic_model+"_"+saliency_model+"_result.json",'w'))
    json.dump(result,open("results/ciedn_result/pred4_gt4_result.json",'w'))
    # write_csv(['CIN_panoptic_all'],"results/ciedn_result/"+panoptic_model+"_"+saliency_model+"_result")
    write_csv(['CIN_panoptic_all'], "results/ciedn_result/pred4_gt4_result")
    # draw_pictures(wait_compares,result,saliency_train_model)

if __name__ == '__main__':
    panoptic_model = ''
    semantic_model = ''
    saliency_model = ''
    args = get_parser().parse_args()
    if args.ins_ext:
        panoptic_model = args.ins_ext
    if args.sem_ext:
        semantic_model = args.sem_ext
    if args.p_intr_ext:
        saliency_model = args.p_intr_ext
    if args.config:
        with open(args.config, 'r') as config:
            config_dict = yaml.load(config)
            config = CINConfig()
            for key in config_dict:
                config.key = config_dict[key]
    else:
        config = CINConfig()

    # saliency_model_list=[saliency_train_model,'a-PyTorch-Tutorial-to-Image-Captioning_saiency','DSS-pytorch_saliency','MSRNet_saliency','NLDF_saliency','PiCANet-Implementation_saliency','salgan_saliency']
    # panoptic_model_list=['deeplab_panoptic','maskrcnn_panoptic']

    # predict(config, panoptic_model,semantic_model, saliency_model)
    compute_metric(panoptic_model, saliency_model)