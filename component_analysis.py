import os
import json
import skimage.io
import torch
import torch.nn as nn
import numpy as np
import scipy
from torch.autograd import Variable
from torch import FloatTensor
from matplotlib import pyplot as plt

import config
from config import Config
from DatasetLib import OOIDataset,Dataset
from ioi_selection.CIEDN import CIEDN
from compute_metric import compare_mask, write_csv, draw_pictures
from utils.utils import rgb2id
from middle_process import generate_images_dict

import argparse
import yaml

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

category_dict={}
class_dict=json.load(open("data/class_dict.json",'r'))
for class_id in class_dict:
    category=class_dict[class_id]
    category_dict[str(category['category_id'])]=category

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ins_ext", type=str,
                        default='CIN_panoptic_all',
                        help="the path of the image to be detected")
    parser.add_argument("--sem_ext", type=str,
                        default='CIN_semantic_all',
                        help="the path of the image to be detected")
    parser.add_argument("--p_intr_ext", type=str,
                        default='CIN_saliency_all',
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

class CIN(nn.Module):
    def __init__(self):
        super(CIN, self).__init__()
        self.ciedn = CIEDN()

    def forward(self, instance_groups):
        return self.ciedn(instance_groups)


def predict(config, panoptic_train_model, saliency_train_model, panoptic_model, saliency_model):
    # log_file="logs/CIN_ooi_100_selection.pth"
    ciedn = CIN().cuda()
    state_dict = torch.load(config.WEIGHT_PATH)
    ciedn.load_state_dict(state_dict, strict=False)

    images = json.load(open("results/ioi_"+panoptic_model+".json", 'r'))
    class_dict=json.load(open("data/class_dict.json", 'r'))
    prediction_list = []
    gt_list = []
    count=0
    for image_id in images:
        count+=1
        print(image_id)
        print(str(count)+"/"+str(len(images)))
        # try:
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
        class_ids=[]
        instance_list=[]
        for instance_id in segments_info:
            instance_list.append(instance_id)
            segment_info=segments_info[instance_id]
            category_id = segment_info['category_id']
            class_id = category_dict[str(category_id)]['class_id']
            class_ids.append(class_id)
            islabel = segment_info['labeled']
            box = [int(segment_info['bbox'][0]*scale+top_pad),int(segment_info['bbox'][1]*scale+left_pad),int(segment_info['bbox'][2]*scale+top_pad),int(segment_info['bbox'][3]*scale+left_pad)]
            labels.append(islabel)
            boxes.append(np.array(box))
        if len(segments_info)==0:
            continue
        boxes=np.stack(boxes)
        # real
        semantic_img = skimage.io.imread("../"+panoptic_model.replace("panoptic","semantic")+"/" + image_name.replace("jpg", "png"))
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

            # plt.figure()
            # plt.imshow(instance_label)
            # plt.show()

            instance_map = saliency_map[y1:y2, x1:x2]
            instance_map = scipy.misc.imresize(instance_map, (56, 56), interp='bilinear') / 255.0
            instance_group.append(instance_map)

            # plt.figure()
            # plt.imshow(instance_map)
            # plt.show()

            instance_group = np.stack(instance_group)
            instance_groups.append(instance_group)


        instance_groups = np.stack(instance_groups)
        labels = np.array(labels, dtype=np.float32)

        instance_groups = Variable(FloatTensor(instance_groups)).float().cuda().unsqueeze(0)
        boxes = Variable(FloatTensor(boxes)).float().cuda().unsqueeze(0)
        if config.GPU_COUNT:
            predictions = ciedn(instance_groups).squeeze(1).data.cpu().numpy()
        else:
            predictions = ciedn(instance_groups).squeeze(1).data.numpy()

        num = instance_groups.shape[1]
        for i in range(0, num):
            avg = np.sum(predictions[i * num:(i + 1) * num]) / num
            prediction_list.append(avg)
        gt_list.extend(labels)

    prediction_list = np.array(prediction_list)
    gt_list = np.array(gt_list)
    np.save("results/ciedn_result/"+panoptic_model+"_"+saliency_model+"_gt.npy", gt_list)
    np.save("results/ciedn_result/"+panoptic_model+"_"+saliency_model+"_pred.npy", prediction_list)

def compute_metric(panoptic_train_model, saliency_train_model,compare_list,mode):
    a2 = 0.3
    result = {}
    base=0
    image_dict=json.load(open("results/ioi_"+panoptic_train_model+".json",'r'))
    for image_id in image_dict:
        base+=image_dict[image_id]['base']
    print(base)
    gt = np.load("results/ciedn_result/" + panoptic_train_model + "_" + saliency_train_model + "_gt.npy")
    pred = np.load("results/ciedn_result/" + panoptic_train_model + "_" + saliency_train_model + "_pred.npy")
    precision, recall, f, _recall, _f = compare_mask(gt, pred, a2, base)
    result[saliency_train_model] = {"precision": precision, "recall": recall, "f": f, "_recall": _recall, "_f": _f}
    print("compare")
    if mode == "instance":
        for wait_compare in compare_list:
            compare_base = 0
            compare_image_dict = json.load(open("results/ioi_" + wait_compare + ".json", 'r'))
            for image_id in image_dict:
                compare_base += compare_image_dict[image_id]['base']
            print(compare_base)
            gt = np.load("results/ciedn_result/" + wait_compare + "_" + saliency_train_model + "_gt.npy")
            pred = np.load("results/ciedn_result/" + wait_compare + "_" + saliency_train_model + "_pred.npy")
            precision, recall, f, _recall, _f = compare_mask(gt, pred, a2, compare_base)
            result[wait_compare] = {"precision": precision, "recall": recall, "f": f, "_recall": _recall, "_f": _f}
        json.dump(result,open("results/ciedn_result/result_" + panoptic_train_model + "_" + saliency_train_model + ".json",'w'))
        write_csv([saliency_train_model]+compare_list, "results/ciedn_result/result_" + panoptic_train_model + "_" + saliency_train_model)
        draw_pictures([saliency_train_model]+compare_list, result)
    elif mode=="p_interest":
        for wait_compare in compare_list:
            gt = np.load("results/ciedn_result/" + panoptic_train_model + "_" + wait_compare + "_gt.npy")
            pred = np.load("results/ciedn_result/" + panoptic_train_model + "_" + wait_compare + "_pred.npy")
            precision, recall, f, _recall, _f = compare_mask(gt, pred, a2, base)
            result[wait_compare] = {"precision": precision, "recall": recall, "f": f, "_recall": _recall, "_f": _f}
        json.dump(result,open("results/ciedn_result/result_" + panoptic_train_model + "_" + saliency_train_model + ".json", 'w'))
        write_csv([saliency_train_model]+compare_list, "results/ciedn_result/result_" + panoptic_train_model + "_" + saliency_train_model)
        draw_pictures([saliency_train_model]+compare_list, result)

if __name__ == '__main__':
    panoptic_train_model = 'CIN_panoptic_val'
    semantic_train_model = 'CIN_semantic_val'
    saliency_train_model = 'CIN_saliency_val'

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

    # if panoptic_model!=panoptic_train_model:
    #     predict(config, panoptic_train_model, saliency_train_model, panoptic_model, saliency_train_model)
    #     compare_list=[panoptic_model]
    #     compute_metric(panoptic_train_model, saliency_train_model, compare_list,mode="instance")
    #
    # if saliency_model!=saliency_train_model:
    #     predict(config, panoptic_train_model, saliency_train_model, panoptic_train_model, saliency_model)
    #     compare_list = [saliency_model]
    #     compute_metric(panoptic_train_model, saliency_train_model, compare_list,mode="p_interest")
    #
    # if panoptic_model==panoptic_train_model and saliency_model==saliency_train_model:
    predict(config, panoptic_train_model, saliency_train_model, panoptic_train_model, saliency_train_model)

        # panoptic_model_list = ['thing_panoptic', 'stuff_panoptic']
        # for panoptic_model_inner in panoptic_model_list:
        #     # generate_images_dict(panoptic_model_inner)
        #     predict(config, panoptic_train_model, saliency_train_model, panoptic_model_inner, saliency_train_model)
        # compute_metric(panoptic_train_model, saliency_train_model,panoptic_model_list,mode="instance")
        #
        # saliency_model_list=['a-PyTorch-Tutorial-to-Image-Captioning_saiency','DSS-pytorch_saliency','MSRNet_saliency','NLDF_saliency','PiCANet-Implementation_saliency','salgan_saliency']
        # for saliency_model_inner in saliency_model_list:
        #     predict(config, panoptic_train_model, saliency_train_model, panoptic_train_model, saliency_model_inner)
        # compute_metric(panoptic_train_model, saliency_train_model, saliency_model_list, mode="p_interest")