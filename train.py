import os
import json
import skimage.io
import torch


import config
from config import Config
from DatasetLib import OOIDataset,Dataset
from CIN import CIN

import argparse
import yaml

ROOT_DIR = os.getcwd()
IMAGENET_MODEL_PATH = os.path.join(ROOT_DIR, "models/resnet50_imagenet.pth")

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
    parser.add_argument("--setting", type=list,
                        default=[('semantic', 0.01, 30), ('p_interest', 0.01, 10), ('selection', 0.01, 100), ('all', 0.001, 10)],
                        help="the path of the settings in the training process")
    parser.add_argument("--config", type=str,
                        default="configs/train_config.yaml",
                        help="the config file path")
    return parser

def run(settings, config):
    model = CIN(model_dir=MODEL_DIR, config=config)
    if config.GPU_COUNT:
        model = model.cuda()

    #RECENT_MODEL_PATH = model.find_last()[1]
    #print(RECENT_MODEL_PATH)
    # if RECENT_MODEL_PATH is None:
    #    model.load_weights(IMAGENET_MODEL_PATH)
    # else:
    #    model.load_weights(RECENT_MODEL_PATH)
    model.load_part_weights("logs/PFPN_ooi_0034_maskrcnn.pth",mode="segmentation")
    model.load_part_weights("logs/CIN_ooi_0009_saliency.pth", mode="saliency")
    dataset_train = OOIDataset("train")
    dataset_val = OOIDataset("val")

    # for item in settings:
    #     print('Fine tune {} layers'.format(item[0]))
    #     model.train_model(dataset_train, dataset_val,
    #                       learning_rate=item[1],
    #                       epochs=item[2],
    #                       layers=item[0])

    model.train_model(dataset_train, dataset_val,
                      learning_rate=config.LEARNING_RATE*10,
                      epochs=30,
                      layers='selection')

if __name__=='__main__':
    settings = [('semantic', 0.01, 30), ('p_interest', 0.01, 10), ('selection', 0.01, 100), ('all', 0.001, 10)]
    args = get_parser().parse_args()
    if args.setting:
        settings = args.setting
    if args.config:
        with open(args.config, 'r') as config:
            config_dict = yaml.load(config)
            config = CINConfig()
            for key in config_dict:
                config.key = config_dict[key]
    else:
        config = CINConfig()
    run(settings, config)

