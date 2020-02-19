import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

from config import Config
from CIN import CIN
from utils import visualize
from utils.Dict2Obj import Dict2Obj

import torch

import argparse
import yaml

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# The default image path
IMAGE_DIR = "demo_images"
file_names = os.listdir(IMAGE_DIR)
file_name = file_names[0]
image_path = os.path.join(IMAGE_DIR, file_name)

# The default config path
CONFIG_DIR = "configs"
config_path = os.path.join(CONFIG_DIR, "demo_config.yaml")

class CINConfig(Config):
    NAME = "ooi"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1+133
    THING_NUM_CLASSES = 1+80
    STUFF_NUM_CLASSES = 1+53

def get_parser():
    parser = argparse.ArgumentParser(description="IOID demo for builtin models")
    parser.add_argument("--img", type=str,
                        default=image_path,
                        help="the path of the image to be detected")
    parser.add_argument("--config", type=str,
                        default=config_path,
                        help="the config file path")
    return parser

args = get_parser().parse_args()

if args.img:
    image_path = args.img
if args.config:
    with open(args.config, 'r') as config:
        config_dict = yaml.load(config)
        config = CINConfig()
        for key in config_dict:
            config.key = config_dict[key]

# Create model object.
model = CIN(model_dir=MODEL_DIR, config=config)
if config.GPU_COUNT:
    model = model.cuda()

# Load weights
state_dict_dir = os.path.join(MODEL_DIR, "CIN_ooi_all.pth")
state_dict = torch.load(state_dict_dir)
model.load_state_dict(state_dict, strict=False)


# COCO Class names 134
# Index of the class in the list is its ID. For example, to get ID of;
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['other','person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
               'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
               'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
               'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
               'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
               'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
               'banner', 'blanket', 'bridge', 'cardboard', 'counter', 'curtain', 'door-stuff',
               'floor-wood', 'flower', 'fruit', 'gravel', 'house', 'light', 'mirror-stuff',
               'net', 'pillow', 'platform', 'playingfield', 'railroad', 'river', 'road',
               'roof', 'sand', 'sea', 'shelf', 'snow', 'stairs', 'tent', 'towel',
               'wall-brick', 'wall-stone', 'wall-tile', 'wall-wood', 'water-other',
               'window-blind', 'window-other', 'tree-merged', 'fence-merged',
               'ceiling-merged', 'sky-other-merged', 'cabinet-merged', 'table-merged',
               'floor-other-merged', 'pavement-merged', 'mountain-merged', 'grass-merged',
               'dirt-merged', 'paper-merged', 'food-other-merged', 'building-other-merged',
               'rock-merged', 'wall-other-merged', 'rug-merged']

image = skimage.io.imread(image_path)
if len(image.shape) == 2:
    image = np.stack([image, image, image], axis=2)
# Run detection
results = model.detect([image], limit='selection')
segments_info = results
boxes = []
masks = []
class_ids = []
for key in segments_info:
    boxes.append(segments_info[key]['bbox'])
    masks.append(segments_info[key]['mask'])
    class_name = segments_info[key]['category_name']
    class_id = class_names.index(class_name)
    class_ids.append(class_id)
if len(class_ids) > 0:
    boxes = np.stack(boxes)
    masks = np.stack(masks)
else:
    boxes = np.zeros(shape=(0,0))
    masks = np.zeros(shape=(0,0))

visualize.display_instances(image, boxes, masks, class_ids, class_names)
# print(type(results))
# visualize.display_instances(image, results['thing_boxes'], results['thing_masks'], results['thing_class_ids'], class_names)
