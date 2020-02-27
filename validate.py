import os
import json
import skimage.io
import torch
import scipy.misc

import config
from config import Config
from DatasetLib import OOIDataset,Dataset
from torch.utils.data.dataloader import DataLoader as TorchDataLoader
from torch.utils.data.dataloader import default_collate
from torch.autograd import Variable
from compute_metric import compare_mask
from CIN import CIN
from utils import utils
import numpy as np

import argparse
import yaml
from compute_metric import compare_mask_wih_a2_and_threshold

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")


class CINConfig(Config):
    NAME = "ooi"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1 #10
    NUM_CLASSES = 1+133
    THING_NUM_CLASSES = 1+80
    STUFF_NUM_CLASSES = 1+53

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default="configs/validate_config.yaml",
                        help="the config file path")
    return parser

def maxminnorm(array):
    max_value=np.max(array)
    min_value=np.min(array)
    newarray=(array-min_value)/(max_value-min_value)
    return newarray

def compute_pixel_iou(bool_mask_pred, bool_mask_gt):
    intersection = bool_mask_pred * bool_mask_gt
    union = bool_mask_pred + bool_mask_gt
    return np.count_nonzero(intersection) / np.count_nonzero(union)

def run(config):
    model = CIN(model_dir=MODEL_DIR, config=config)

    if config.GPU_COUNT:
        model = model.cuda()

    def my_collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        if len(batch) == 0:
            print("No valid data!!!")
            batch = [[torch.from_numpy(np.zeros([1, 1]))]]
        return default_collate(batch)

    state_dict = torch.load(config.WEIGHT_PATH)
    model.load_state_dict(state_dict, strict=False)
    for param in model.named_parameters():
        param[1].requires_grad = False

    gt_images_dict=json.load(open("data/val_images_dict.json"))
    prediction_list=[]
    gt_list=[]
    base=0
    step=0
    for image_id in gt_images_dict:
        step += 1
        print(str(step) + "/" + str(len(gt_images_dict)))

        inner_prediction_list=[]
        inner_gt_list=[]

        image = gt_images_dict[image_id]
        gt_instance_dict=image['instances']
        image_name = image['image_name']
        img = skimage.io.imread(os.path.join(config.IMAGE_PATH, "ioid_images/") + image_name)
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=2)

        pred_dict, ioid_result, instance_dict,panoptic_result_instance_id_map, predictions, instance_list = model.detect([img], limit="selection")
        inner_prediction_list=predictions

        for instance_id in instance_dict:
            mask = panoptic_result_instance_id_map == int(instance_id)
            instance_dict[instance_id]['mask'] = mask

        gt_segmentation_id = utils.rgb2id(scipy.misc.imread("../data/ioid_panoptic/" + image_id.zfill(12) + ".png"))
        for gt_instance_id in gt_instance_dict:
            gt_mask = gt_segmentation_id == int(gt_instance_id)
            gt_instance_dict[gt_instance_id]['mask'] = gt_mask

        instance_pred_gt_dict = {}
        instance_gt_pred_dict = {}
        if len(instance_dict) == 0:
            for gt_instance_id in gt_instance_dict:
                instance_gt_pred_dict[gt_instance_id] = {"labeled": gt_instance_dict[gt_instance_id]['labeled'],"pred": []}
        else:
            for instance_id in instance_dict:
                max_iou = -1
                max_gt_instance_id = ""
                for gt_instance_id in gt_instance_dict:
                    i_iou = compute_pixel_iou(instance_dict[instance_id]['mask'],gt_instance_dict[gt_instance_id]['mask'])
                    if gt_instance_id not in instance_gt_pred_dict:
                        instance_gt_pred_dict[gt_instance_id] = {"labeled": gt_instance_dict[gt_instance_id]['labeled'],
                                                                 "pred": []}
                    if i_iou >= 0.5 and instance_dict[instance_id]['category_id'] == gt_instance_dict[gt_instance_id][
                        'category_id'] and i_iou > max_iou:
                        max_gt_instance_id = gt_instance_id
                        max_iou = i_iou
                        instance_gt_pred_dict[gt_instance_id]['pred'].append(instance_id)
                if max_gt_instance_id != "":
                    instance_pred_gt_dict[instance_id] = {"gt_instance_id": max_gt_instance_id,
                                                          "label": gt_instance_dict[max_gt_instance_id]['labeled']}
                else:
                    instance_pred_gt_dict[instance_id] = {"gt_instance_id": "", "label": False}

        image_base = 0
        for instance_id in instance_gt_pred_dict:
            if instance_gt_pred_dict[instance_id]['labeled'] == True and len(instance_gt_pred_dict[instance_id]['pred']) == 0:
                image_base += 1
        base+=image_base

        for instance_id in instance_dict:
            del instance_dict[instance_id]['mask']

        for gt_instance_id in gt_instance_dict:
            del gt_instance_dict[gt_instance_id]['mask']

        for instance_id in instance_dict:
            instance = instance_dict[instance_id]
            if instance_id in instance_pred_gt_dict:
                instance['labeled'] = instance_pred_gt_dict[instance_id]['label']
            else:
                instance['labeled'] = False

        for instance_id in instance_list:
            inner_gt_list.append(1 if instance_dict[instance_id]['labeled'] else 0)

        prediction_list.extend(inner_prediction_list)
        gt_list.extend(inner_gt_list)

        pl=maxminnorm(np.array(prediction_list))
        gl=np.array(gt_list)
        pl=np.where(pl > 0.4, 1, 0)

        a2=0.3
        TP = np.sum(np.multiply(pl, gl))
        TP_FP = np.sum(pl)
        TP_FN = np.sum(gl)
        precision = TP / TP_FP
        recall = TP / TP_FN
        recall_ = TP / (TP_FN + base)
        f = (a2 + 1) * precision * recall / (a2 * precision + recall)
        f_ = (a2 + 1) * precision * recall_ / (a2 * precision + recall_)
        print("base: "+str(base)+"  precision: "+str(precision)+"  recall: " + str(recall_)+"  f: " + str(f_)+"  recall*: " + str(recall)+"  f*: " + str(f))

    predition_list = np.array(prediction_list)
    gt_list = np.array(gt_list)
    np.save("results/validate/gt.npy", gt_list)
    np.save("results/validate/pred.npy", prediction_list)


if __name__=='__main__':
    args = get_parser().parse_args()
    if args.config:
        with open(args.config, 'r') as config:
            config_dict = yaml.load(config)
            config = CINConfig()
            for key in config_dict:
                config.key = config_dict[key]
    else:
        config = CINConfig()
    run(config)
    # gt=np.load("results/validate/gt.npy")
    # pred=np.nan_to_num(np.load("results/validate/pred.npy"))
    # precision, recall, f, _recall, _f =compare_mask(gt,pred,0.3,16046)
    # print(precision['0.8'])
    # print(recall['0.8'])
    # print(f['0.8'])
    # print(_recall['0.8'])
    # print(_f['0.8'] )
