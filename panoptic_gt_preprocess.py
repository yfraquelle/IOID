from __future__ import print_function

from utils.cocoapi import CocoPanoptic
import numpy as np
import scipy.misc
import random
import os
import json
import scipy.ndimage

from utils import utils
from collections import defaultdict
# -*- coding: utf-8 -*-

#############################
# global variables #
#############################
# total: 118287
root_dir = "data"
data_dir = '/home/magus/datasets/coco/train2017'  # train data
label_dir = '/home/magus/datasets/coco/annotations/panoptic_train2017'  # train label
label_colors_file = os.path.join(root_dir, "panoptic_coco_categories.json")  # coco categories
panoptic_json = '/home/magus/datasets/coco/annotations/panoptic_train2017.json'

# create dir for label index
label_idx_dir = os.path.join("", "../panoptic_ioi")
if not os.path.exists(label_idx_dir):
    os.makedirs(label_idx_dir)

color2index = {}

def divide_train_val(val_rate=0.2, shuffle=True, random_seed=None):
    val_label_file = os.path.join(root_dir, "val.csv")  # validation file
    train_label_file = os.path.join(root_dir, "train.csv")  # train file
    data_list = os.listdir(data_dir)
    data_len = len(data_list)
    val_len = int(data_len * val_rate)

    if random_seed:
        random.seed(random_seed)

    if shuffle:
        data_idx = random.sample(range(data_len), data_len)
    else:
        data_idx = list(range(data_len))

    val_idx = [data_list[i] for i in data_idx[:val_len]]
    train_idx = [data_list[i] for i in data_idx[val_len:]]

    # create val.csv
    v = open(val_label_file, "w")
    v.write("img,label\n")
    for name in val_idx:
        if 'jpg' not in name:
            continue
        img_name = os.path.join(data_dir, name)
        lab_name = os.path.join(label_idx_dir, name.split(".")[0] + ".png.npy")
        v.write("{},{}\n".format(img_name, lab_name))

    # create train.csv
    t = open(train_label_file, "w")
    t.write("img,label\n")
    for name in train_idx:
        if 'jpg' not in name:
            continue
        img_name = os.path.join(data_dir, name)
        lab_name = os.path.join(label_idx_dir, name.split(".")[0] + ".png.npy")
        t.write("{},{}\n".format(img_name, lab_name))

def rgb2id(color):
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        if color.dtype == np.uint8:
            color = color.astype(np.uint8)
        return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    return color[0] + 256 * color[1] + 256 * 256 * color[2]

def generate_new_color(segmentation, segments_info, category_dict):
    segmentation_id = rgb2id(segmentation)

    segmentation[:, :, :] = 0
    for segment_info in segments_info:
        segment_id = segment_info['id']
        category_id = segment_info['category_id']
        color = category_dict[str(category_id)]['color']
        mask = segmentation_id == segment_id
        segmentation[mask] = color

    return segmentation

def generate_category(segmentation, segments_info, category_dict):
    segmentation_id = rgb2id(segmentation)
    semantic=np.zeros((segmentation.shape[0],segmentation.shape[1]))
    for segment_info in segments_info:
        segment_id = segment_info['id']
        category_id = segment_info['category_id']
        class_id = category_dict[str(category_id)]['idx']
        mask = segmentation_id == segment_id
        semantic[mask] = class_id
    return semantic.astype(np.uint8)

def parse_label(all_img_name):
    cocoPanoptic = CocoPanoptic(panoptic_json)
    # change label to class index
    class_dict = json.load(open("data/class_dict.json",'r'))
    category_dict = json.load(open("data/category_dict.json",'r'))

    for class_id in class_dict:
        category = class_dict[class_id]
        class_id=category['idx']
        category_id=category["id"]
        color = tuple(category['color'])
        color2index[color] = class_id

    count=0
    for name in os.listdir(label_dir):
        if name.replace(".png","") in all_img_name:
            count += 1
            filename = os.path.join(label_idx_dir, name)
            print('%d in 45024' % count)
            if os.path.exists(filename):
                print("Skip %s" % (name))
                continue
            img = os.path.join(label_dir, name)
            img = scipy.misc.imread(img, mode='RGB')
            img_h = img.shape[0]
            img_w = img.shape[1]
            scale = min(512/img_h,512/img_w)

            # img = scipy.ndimage.zoom(img, [scale, scale,1],mode="nearest")
            # print(img.shape)
            img = scipy.misc.imresize(img,(round(img_h*scale),round(img_w*scale)),interp="nearest")
            h, w = img.shape[:2]
            top_pad = (512 - h) // 2
            bottom_pad = 512 - h - top_pad
            left_pad = (512 - w) // 2
            right_pad = 512 - w - left_pad
            padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]

            img_id = int(name.split('.')[0])
            segments_info = cocoPanoptic.loadAnns(img_id)[0]['segments_info']
            img = np.pad(img, padding, mode='constant', constant_values=0)

            idx_mat = generate_category(img, segments_info, category_dict)

            print(idx_mat.shape)
            scipy.misc.imsave(filename, idx_mat)
            print("Finish %s" % (name))
    print('Done')

class IdGenerator():
    def __init__(self, categories):
        self.taken_colors = set([0, 0, 0])
        self.categories = categories
        for category in self.categories.values():
            if category['isthing'] == 0:
                self.taken_colors.add(tuple(category['color']))

    def get_color(self, cat_id):
        def random_color(base, max_dist=30):
            new_color = base + np.random.randint(low=-max_dist,
                                                 high=max_dist+1,
                                                 size=3)
            return tuple(np.maximum(0, np.minimum(255, new_color)))

        # category = self.categories[cat_id]
        # find the category whose id is equal to cat_id
        category = {}
        for class_id in self.categories:
            item = self.categories[class_id]
            if str(item['id']) == cat_id:
                category = item
        if category['isthing'] == 0:
            return category['color']
        base_color_array = category['color']
        base_color = tuple(base_color_array)
        if base_color not in self.taken_colors:
            self.taken_colors.add(base_color)
            return base_color
        else:
            while True:
                color = random_color(base_color_array)
                if color not in self.taken_colors:
                     self.taken_colors.add(color)
                     return color

    def get_id(self, cat_id):
        color = self.get_color(cat_id)
        return rgb2id(color)

    def get_id_and_color(self, cat_id):
        color = self.get_color(cat_id)
        return rgb2id(color), color

if __name__ == '__main__':
    train_images_dict = json.load(open("data/train_images_dict.json", 'r'))
    train_images_name = [train_images_dict[train_image_id]['image_name'] for train_image_id in train_images_dict]
    val_images_dict = json.load(open("data/val_images_dict.json", 'r'))
    val_images_name = [val_images_dict[val_image_id]['image_name'] for val_image_id in val_images_dict]
    all_img_name = train_images_name + val_images_name
    parse_label(all_img_name)