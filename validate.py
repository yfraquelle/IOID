import os
import json
import skimage.io
import torch


import config
from config import Config
from DatasetLib import OOIDataset,Dataset
from torch.utils.data.dataloader import DataLoader as TorchDataLoader
from torch.utils.data.dataloader import default_collate
from torch.autograd import Variable
from CIN import CIN
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

    state_dict_dir = os.path.join(MODEL_DIR, "CIN_ooi_all.pth")
    state_dict = torch.load(state_dict_dir)
    model.load_state_dict(state_dict, strict=False)

    val_dataset = OOIDataset('val')
    val_set = Dataset(val_dataset, config)
    val_generator = TorchDataLoader(val_set, collate_fn=my_collate_fn, batch_size=1, shuffle=True, num_workers=1)

    step = 0
    gt_list = []
    prediction_list = []
    for inputs in val_generator:
        if len(inputs)!=17:
            print(len(inputs))
            continue
        try:
        # else:
            images = inputs[0]
            image_metas = inputs[1]
            rpn_match = inputs[2]
            rpn_bbox = inputs[3]
            gt_class_ids = inputs[4]
            gt_boxes = inputs[5]
            gt_masks = inputs[6]
            gt_stuff_class_ids = inputs[7]
            gt_stuff_boxes = inputs[8]
            gt_stuff_masks = inputs[9]
            gt_semantic_label = inputs[10]
            gt_influence_map = inputs[11]
            gt_interest_class_ids = inputs[12]
            gt_interest_boxes = inputs[13]
            gt_interest_masks = inputs[14]
            gt_segmentation = inputs[15]
            gt_image_instances = inputs[16]

            # image_metas as numpy array
            image_metas = image_metas.numpy()

            # Wrap in variables
            with torch.no_grad():
                images = Variable(images)
                rpn_match = Variable(rpn_match)
                rpn_bbox = Variable(rpn_bbox)
                gt_class_ids = Variable(gt_class_ids)
                gt_boxes = Variable(gt_boxes)
                gt_masks = Variable(gt_masks)
                gt_stuff_class_ids = Variable(gt_stuff_class_ids)
                gt_stuff_boxes = Variable(gt_stuff_boxes)
                gt_stuff_masks = Variable(gt_stuff_masks)
                gt_semantic_label = Variable(gt_semantic_label)
                gt_influence_map = Variable(gt_influence_map)
                gt_interest_class_ids = Variable(gt_interest_class_ids)
                gt_interest_boxes = Variable(gt_interest_boxes)
                gt_interest_masks = Variable(gt_interest_masks)
                gt_segmentation = Variable(gt_segmentation)

            # To GPU
            if config.GPU_COUNT:
                images = images.cuda()
                rpn_match = rpn_match.cuda()
                rpn_bbox = rpn_bbox.cuda()
                gt_class_ids = gt_class_ids.cuda()
                gt_boxes = gt_boxes.cuda()
                gt_masks = gt_masks.cuda()
                gt_stuff_class_ids = gt_stuff_class_ids.cuda()
                gt_stuff_boxes = gt_stuff_boxes.cuda()
                gt_stuff_masks = gt_stuff_masks.cuda()
                gt_semantic_label = gt_semantic_label.cuda()
                gt_influence_map = gt_influence_map.cuda()
                gt_interest_class_ids = gt_interest_class_ids.cuda()
                gt_interest_boxes = gt_interest_boxes.cuda()
                gt_interest_masks = gt_interest_masks.cuda()
                gt_segmentation = gt_segmentation.cuda()

            predict_input = [images, image_metas, gt_class_ids, gt_boxes, gt_masks, gt_segmentation, gt_image_instances]
            # predictions, segments_info = model.predict_front(predict_input, mode="inference", limit="selection")
            for param in model.named_parameters():
                param[1].requires_grad=False
                # print(param[0] + " " + str(param[1].shape) + " " + str(param[1].requires_grad))
            predictions, pair_label, labels = model.predict_front(predict_input, mode="training", limit="selection")
            predictions = predictions.data.cpu().numpy()
            labels = labels.data.cpu().numpy().squeeze(0)
            num = len(labels)
            gt_list_item = []
            prediction_list_item = []
            for i in range(0, num):
                avg = np.sum(predictions[i*num: (i+1)*num])/num
                prediction_list.append(avg)
                prediction_list_item.append(avg)

            gt_list_item.extend(labels)
            gt_list.extend(labels)

            step += 1
            gt_list_item = np.array(gt_list_item)
            prediction_list_item = np.array(prediction_list_item)
            precision, recall, f = compare_mask_wih_a2_and_threshold(gt_list_item, prediction_list_item, config.SELECTION_THRESHOLD)
            print(step, precision, recall,f)
        except Exception as e:
            print("Error - " + str(step))
            print(e)

    predition_list = np.array(prediction_list)
    gt_list = np.array(gt_list)
    np.save("results/ciedn_result/gt.npy", gt_list)
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