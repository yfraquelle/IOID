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
    parser.add_argument("--setting", type=str,
                        default=None,
                        help="the path of the settings in the training process")
    parser.add_argument("--config", type=str,
                        default="configs/train_config.yaml",
                        help="the config file path")
    return parser

def run(settings, config):
    model = CIN(model_dir=MODEL_DIR, config=config)
    if config.GPU_COUNT:
        model = model.cuda()

    model.load_from_maskrcnn()
    # model.load_weights(config.WEIGHT_PATH)
    dataset_train = OOIDataset("train")
    dataset_val = OOIDataset("val")

    for item in settings:
        print('Fine tune {} layers'.format(item[0]))
        model.train_model(dataset_train, dataset_val,
                          learning_rate=item[1],
                          epochs=item[2],
                          layers=item[0])

    # model.train_model(dataset_train, dataset_val,
    #                   learning_rate=config.LEARNING_RATE*10,
    #                   epochs=30,
    #                   layers='selection')

if __name__=='__main__':
    settings = [('semantic', 0.01, 34), ('p_interest', 0.01, 44), ('selection', 0.001, 100)]
    args = get_parser().parse_args()
    if args.setting:
        new_settings=[]
        settings_list = args.setting.split(',')
        for i in range(len(settings_list)//3):
            new_settings.append([])
            new_settings[i].append(settings_list[3*i+0])
            new_settings[i].append(float(settings_list[3*i+1]))
            new_settings[i].append(int(settings_list[3*i+2]))
        settings=new_settings
    if args.config:
        with open(args.config, 'r') as config:
            config_dict = yaml.load(config)
            config = CINConfig()
            for key in config_dict:
                config.key = config_dict[key]
    else:
        config = CINConfig()
    run(settings, config)

