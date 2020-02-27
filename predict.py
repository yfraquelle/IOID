import os
import json
import skimage.io
import numpy as np

from config import Config
from CIN import CIN
import scipy
from utils.utils import IdGenerator
from PIL import Image
from matplotlib import pyplot as plt
from utils import visualize
from middle_process import map_instance_to_gt

import argparse
import yaml

class_names = ['other','person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner', 'blanket', 'bridge', 'cardboard', 'counter', 'curtain', 'door-stuff', 'floor-wood', 'flower', 'fruit', 'gravel', 'house', 'light', 'mirror-stuff', 'net', 'pillow', 'platform', 'playingfield', 'railroad', 'river', 'road', 'roof', 'sand', 'sea', 'shelf', 'snow', 'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone', 'wall-tile', 'wall-wood', 'water-other', 'window-blind', 'window-other', 'tree-merged', 'fence-merged', 'ceiling-merged', 'sky-other-merged', 'cabinet-merged', 'table-merged', 'floor-other-merged', 'pavement-merged', 'mountain-merged', 'grass-merged', 'dirt-merged', 'paper-merged', 'food-other-merged', 'building-other-merged', 'rock-merged', 'wall-other-merged', 'rug-merged']
class_dict=json.load(open("data/class_dict.json",'r'))

ROOT_DIR = os.getcwd()
IMAGENET_MODEL_PATH = os.path.join(ROOT_DIR, "models/resnet50_imagenet.pth") # 没有用

MODEL_DIR = os.path.join(ROOT_DIR, "logs") # 其中的文件是CIN模型训练得到的

id_generator=IdGenerator(class_dict)

class CINConfig(Config):
    NAME = "ooi"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1+133
    THING_NUM_CLASSES = 1+80
    STUFF_NUM_CLASSES = 1+53

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str,
                        default="selection",
                        help = "the mode of the predict")
    parser.add_argument("--config", type=str,
                        default="configs/predict_config.yaml",
                        help="the config file path")
    return parser

def run(mode, config):
    model = CIN(model_dir=MODEL_DIR, config=config)
    if config.GPU_COUNT:
        model = model.cuda()

    if mode=="insttr":
        model.load_weights(config.WEIGHT_PATH)
        val_images=json.load(open(os.path.join(config.JSON_PATH, "val_images_dict.json"),'r'))
        image_collector=dict()
        count=0
        exist=os.listdir("../CIN_saliency_val")
        for image_id in val_images:
            try:
                count+=1
                image = val_images[image_id]
                image_name = image['image_name']
                if image_name.replace(".jpg",".png") in exist:
                    continue

                img=skimage.io.imread(os.path.join(config.IMAGE_PATH, "ioid_images/"+image_name))
                if len(img.shape)==2:
                    img = np.stack([img,img,img],axis=2)
                result=model.detect([img],limit="insttr")[0]

                semantic_labels=result['semantic_segment']
                influence_map=result['influence_map']
                scipy.misc.toimage(influence_map, cmin=0, cmax=1).save("../CIN_saliency_val/" + image_name.replace(".jpg",".png"))
                panoptic_result=np.zeros(img.shape)
                semantic_result = np.zeros(img.shape)

                information_collector={}
                if 'stuff_class_ids' in result:
                    stuff_class_ids, stuff_boxes, stuff_masks = result['stuff_class_ids'], result['stuff_boxes'], result['stuff_masks']
                    for i,class_id in enumerate(stuff_class_ids):
                        category_id=class_dict[str(int(class_id))]['category_id']
                        category_name=class_dict[str(int(class_id))]['name']
                        id,color = id_generator.get_id_and_color(str(category_id))
                        mask=stuff_masks[i]==1
                        panoptic_result[mask]=[int(color[0]),int(color[1]),int(color[2])]
                        semantic_result[mask] = [int(class_id),int(class_id),int(class_id)]
                        information_collector[str(id)]={"id":int(id),"bbox":[int(stuff_boxes[i][0]),int(stuff_boxes[i][1]),int(stuff_boxes[i][2]),int(stuff_boxes[i][3])],"category_id":category_id,"class_id":int(class_id),"category_name":category_name}

                if 'thing_class_ids' in result:
                    thing_class_ids, thing_boxes, thing_masks = result['thing_class_ids'], result['thing_boxes'], result['thing_masks']
                    for i,class_id in enumerate(thing_class_ids):
                        category_id=class_dict[str(int(class_id))]['category_id']
                        category_name=class_dict[str(int(class_id))]['name']
                        id, color = id_generator.get_id_and_color(str(category_id))
                        mask=thing_masks[i]==1
                        panoptic_result[mask]=[int(color[0]),int(color[1]),int(color[2])]
                        semantic_result[mask] = [int(class_id),int(class_id),int(class_id)]
                        information_collector[str(id)]={"id": int(id), "bbox": [int(thing_boxes[i][0]),int(thing_boxes[i][1]),int(thing_boxes[i][2]),int(thing_boxes[i][3])], "category_id": category_id,"class_id":int(class_id),"category_name":category_name}

                Image.fromarray(panoptic_result.astype(np.uint8)).save("../CIN_panoptic_val/" + image_name.replace(".jpg",".png"))
                Image.fromarray(semantic_result.astype(np.uint8)).save("../CIN_semantic_val/" + image_name.replace(".jpg", ".png"))
                image['predictions'] = information_collector
                print(str(count)+"/"+str(len(val_images)))
            except Exception as e:
                print("ERROR: "+image_name)
                print(e)
        map_instance_to_gt(val_images,"CIN_panoptic_val")
    elif mode=="instance":
        model.load_weights(config.WEIGHT_PATH)
        val_images=json.load(open(os.path.join(config.JSON_PATH, "val_images_dict.json"),'r'))
        count=0
        exist=os.listdir("../CIN_panoptic_val")
        for image_id in val_images:
            try:
                count+=1
                image = val_images[image_id]
                image_name=image['image_name']
                if image_name.replace(".jpg",".png") in exist:
                   continue

                img=skimage.io.imread(os.path.join(config.IMAGE_PATH, "ioid_images/"+image_name))
                if len(img.shape)==2:
                    img = np.stack([img,img,img],axis=2)
                result=model.detect([img],limit="instance")[0]

                semantic_labels=result['semantic_segment']
                panoptic_result=np.zeros(img.shape)
                semantic_result=np.zeros(img.shape)
                information_collector={}
                if 'stuff_class_ids' in result:
                    stuff_class_ids,stuff_boxes,stuff_masks=result['stuff_class_ids'],result['stuff_boxes'],result['stuff_masks']
                    for i,class_id in enumerate(stuff_class_ids):
                        category_id=class_dict[str(int(class_id))]['category_id']
                        category_name=class_dict[str(int(class_id))]['name']
                        id,color = id_generator.get_id_and_color(str(category_id))
                        mask=stuff_masks[i]==1
                        panoptic_result[mask]=[int(color[0]),int(color[1]),int(color[2])]
                        semantic_result[mask]=[int(class_id),int(class_id),int(class_id)]
                        information_collector[str(id)]={"id":int(id),"bbox":[int(stuff_boxes[i][0]),int(stuff_boxes[i][1]),int(stuff_boxes[i][2]),int(stuff_boxes[i][3])],"category_id":int(category_id),"class_id":int(class_id),"category_name":category_name}
                if 'thing_class_ids' in result:
                    thing_class_ids, thing_boxes, thing_masks = result['thing_class_ids'], result['thing_boxes'], result['thing_masks']
                    for i,class_id in enumerate(thing_class_ids):
                        category_id=class_dict[str(int(class_id))]['category_id']
                        category_name=class_dict[str(int(class_id))]['name']
                        id, color = id_generator.get_id_and_color(str(category_id))
                        mask=thing_masks[i]==1
                        panoptic_result[mask]=[int(color[0]),int(color[1]),int(color[2])]
                        semantic_result[mask]=[int(class_id),int(class_id),int(class_id)]
                        information_collector[str(id)]={"id": int(id), "bbox": [int(thing_boxes[i][0]),int(thing_boxes[i][1]),int(thing_boxes[i][2]),int(thing_boxes[i][3])], "category_id": int(category_id),"class_id":int(class_id),"category_name":category_name}
                Image.fromarray(panoptic_result.astype(np.uint8)).save("../CIN_panoptic_val/" + image_name.replace(".jpg",".png"))
                Image.fromarray(semantic_result.astype(np.uint8)).save("../CIN_semantic_val/" + image_name.replace(".jpg",".png"))
                image['predictions']=information_collector
                print(str(count)+"/"+str(len(val_images)))
            except Exception as e:
                print("ERROR: "+image_name)
                print(e)
        map_instance_to_gt(val_images,"CIN_panoptic_val")
    elif mode=="p_interest":
        print("generate p_interest")
        model.load_weights(config.WEIGHT_PATH)
        val_images = json.load(open(os.path.join(config.JSON_PATH, "val_images_dict.json"), 'r'))
        count = 0
        exist = os.listdir("../CIN_saliency_val")
        for image_id in val_images:
            try:
                count += 1
                image = val_images[image_id]
                image_name = image['image_name']
                if image_name.replace(".jpg", ".png") in exist:
                    continue

                img = skimage.io.imread(os.path.join(config.IMAGE_PATH, "ioid_images/"+image_name))
                if len(img.shape)==2:
                    img = np.stack([img,img,img],axis=2)
                influence_map = model.detect([img], limit="p_interest")[0]["influence_map"]
                scipy.misc.toimage(influence_map, cmin=0, cmax=1).save("../CIN_saliency_val/" + image_name.replace(".jpg", ".png"))
                print(str(count) + "/" + str(len(val_images)))
            except Exception as e:
                print("ERROR: " + image_name)
                print(e)
    elif mode=="selection":
        model.load_weights(config.WEIGHT_PATH)
        val_images=json.load(open(os.path.join(config.JSON_PATH, "val_images_dict.json"),'r'))

        CIEDN_pred_dict = {}
        count = 0
        for image_id in val_images:
            try:
                print(image_id)
                count+=1
                print(str(count)+"/"+str(len(val_images)))
                image = val_images[image_id]
                image_name=image['image_name']
                img=skimage.io.imread(os.path.join(config.IMAGE_PATH, "ioid_images/")+image_name)
                if len(img.shape)==2:
                    img = np.stack([img,img,img],axis=2)

                pred_dict,ioid_result,segments_info,panoptic_result_instance_id_map,prediction_list,instance_list=model.detect([img], limit="selection")
                scipy.misc.imsave("results/CIEDN_pred/" + image_name.replace(".jpg", ".png"), ioid_result)
                CIEDN_pred_dict[str(image_id)] = pred_dict
                print("{}/{}".format(count,len(val_images)))
            except Exception as e:
                print("ERROR: "+image_name)
                print(e)
        json.dump(CIEDN_pred_dict, open("results/CIEDN_pred_dict.json", 'w'))

    else:
        print("mode not exists")

import sys
if __name__=='__main__':
    args = get_parser().parse_args()
    if args.mode:
        mode = args.mode
    if args.config:
        with open(args.config, 'r') as config:
            config_dict = yaml.load(config)
            config = CINConfig()
            for key in config_dict:
                config.key = config_dict[key]

    run(mode, config)