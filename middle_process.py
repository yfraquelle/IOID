import numpy as np
import json
import os
import skimage
import utils.utils as utils
import scipy.misc
from collections import defaultdict
from matplotlib import pyplot as plt
import multiprocessing
from PIL import Image

def extract_json_from_panoptic_semantic(segmentation_model,semantic_model,only_val=False):
    if only_val:
        images_png = []
        val_images_dict = json.load(open("data/val_images_dict.json"))
        for val_image_id in val_images_dict:
            images_png.append(val_images_dict[val_image_id]['image_name'].replace('jpg', 'png'))
    else:
        images_png=os.listdir("../"+segmentation_model)
    class_id_dict=json.load(open("data/class_dict.json",'r'))
    instance_panoptic_all={}
    count=0
    for image_png in images_png:
        count+=1
        print("extract_json_from_panoptic_semantic:"+str(count)+"/"+str(len(images_png)))
        try:
            semantic = skimage.io.imread("../"+semantic_model+"/"+image_png)
            panoptic = skimage.io.imread("../"+segmentation_model+"/"+image_png)
            segmentation_id = utils.rgb2id(panoptic)
            segmentation_id_list=np.unique(segmentation_id)
            instances={}
            for instance_id in segmentation_id_list:
                mask = np.where(segmentation_id == instance_id, 1, 0)
                if np.sum(mask)==0:
                    continue
                box=utils.extract_bbox(mask)
                class_id = np.unique(semantic[segmentation_id == instance_id])[0]
                category = class_id_dict[str(class_id)]
                instances[str(instance_id)]={'id':int(instance_id),'class_id':int(class_id),'category_id':category['id'],'category_name':category['name'],'bbox':[int(box[0]),int(box[1]),int(box[2]),int(box[3])]}
            instance_panoptic_all[image_png[:-4].lstrip("0")]=instances
        except Exception as e:
            print(e)
    json.dump(instance_panoptic_all,open("data/middle/ioi_"+segmentation_model+".json",'w'))

def map_instance_to_gt(segmentation_model):
    def compute_pixel_iou(bool_mask_pred, bool_mask_gt):
        intersection = bool_mask_pred * bool_mask_gt
        union = bool_mask_pred + bool_mask_gt
        return np.count_nonzero(intersection) / np.count_nonzero(union)

    gt_train_image_instances = json.load(open("data/train_images_dict.json", 'r'))
    gt_val_image_instances = json.load(open('data/val_images_dict.json', 'r'))
    gt_image_instances = dict(gt_train_image_instances, **gt_val_image_instances)
    image_instances = json.load(open("data/middle/ioi_"+segmentation_model+".json", 'r'))
    instance_pred_gt_dict = {}
    instance_gt_pred_dict = {}
    count = 0
    for image_id in image_instances:
        count += 1
        print("map_instance_to_gt:"+str(count) + "/" + str(len(image_instances)))
        instance_pred_gt_dict[image_id] = {}
        instance_gt_pred_dict[image_id] = defaultdict(list)

        segmentation = scipy.misc.imread("../"+segmentation_model+"/" + image_id.zfill(12) + ".png")
        gt_segmentation = scipy.misc.imread(
            "/home/magus/datasets/coco/annotations/panoptic_train2017/" + image_id.zfill(12) + ".png")
        instance_dict = image_instances[image_id]
        gt_instance_dict = gt_image_instances[image_id]['instances']

        segmentation_id = utils.rgb2id(segmentation)
        for instance_id in instance_dict:
            mask = segmentation_id == int(instance_id)
            instance_dict[instance_id]['mask'] = mask

        gt_segmentation_id = utils.rgb2id(gt_segmentation)
        for gt_instance_id in gt_instance_dict:
            gt_mask = gt_segmentation_id == int(gt_instance_id)
            gt_instance_dict[gt_instance_id]['mask'] = gt_mask

        for instance_id in instance_dict:
            max_iou = -1
            max_gt_instance_id = ""
            for gt_instance_id in gt_instance_dict:
                i_iou = compute_pixel_iou(instance_dict[instance_id]['mask'], gt_instance_dict[gt_instance_id]['mask'])
                if gt_instance_id not in instance_gt_pred_dict[image_id]:
                    instance_gt_pred_dict[image_id][gt_instance_id] = {
                        "labeled": gt_instance_dict[gt_instance_id]['labeled'], "pred": []}
                if i_iou >= 0.5 and instance_dict[instance_id]['category_id'] == gt_instance_dict[gt_instance_id][
                    'category_id'] and i_iou > max_iou:
                    max_gt_instance_id = gt_instance_id
                    max_iou = i_iou
                    instance_gt_pred_dict[image_id][gt_instance_id]['pred'].append(instance_id)
            if max_gt_instance_id != "":
                instance_pred_gt_dict[image_id][instance_id] = {"gt_instance_id": max_gt_instance_id,
                                                                "label": gt_instance_dict[max_gt_instance_id][
                                                                    'labeled']}
            else:
                instance_pred_gt_dict[image_id][instance_id] = {"gt_instance_id": "", "label": False}

        for instance_id in instance_dict:
            instance_dict[instance_id]['mask'] = ""

        for gt_instance_id in gt_instance_dict:
            gt_instance_dict[gt_instance_id]['mask'] = ""

    json.dump(instance_pred_gt_dict, open("data/middle/ioi_instance_pred_gt_"+segmentation_model+".json", 'w'))
    json.dump(instance_gt_pred_dict, open("data/middle/ioi_instance_gt_pred_"+segmentation_model+".json", 'w'))

def generate_image_dict(segmentation_model):
    pred_to_gt = json.load(open("data/middle/ioi_instance_pred_gt_" + segmentation_model + ".json", 'r'))
    all_dict = json.load(open("data/middle/ioi_" + segmentation_model + ".json", 'r'))
    train_gt_images = json.load(open("data/train_images_dict.json", 'r'))
    val_gt_images = json.load(open("data/val_images_dict.json", 'r'))
    all_gt_images = dict(train_gt_images, **val_gt_images)
    ioi_val_images_id = list(val_gt_images.keys())
    ioi_val_images_dict = {}
    for image_id in all_dict:
        image_info = all_gt_images[image_id]
        segments_info = all_dict[image_id]
        for instance_id in segments_info:
            if instance_id == "0":
                pass
            instance = segments_info[instance_id]
            if instance_id in pred_to_gt[image_id]:
                instance['labeled'] = pred_to_gt[image_id][instance_id]['label']
        if image_id in ioi_val_images_id:
            ioi_val_images_dict[image_id] = {"image_id": image_info['image_id'],
                                             "image_name": image_info['image_name'],
                                             "height": image_info['height'], "width": image_info['width'],
                                             'segments_info': segments_info}
    json.dump(ioi_val_images_dict, open("data/middle/ioi_val_images_dict_"+segmentation_model+".json", 'w'))

def split_dataset(segmentation_model):
    pred_to_gt=json.load(open("data/middle/ioi_instance_pred_gt_"+segmentation_model+".json",'r'))
    all_dict=json.load(open("data/middle/ioi_"+segmentation_model+".json",'r'))
    train_gt_images = json.load(open("data/train_images_dict.json", 'r'))
    val_gt_images = json.load(open("data/val_images_dict.json", 'r'))
    all_gt_images = dict(train_gt_images, **val_gt_images)
    ioi_val_images_png=[]
    for val_image_id in val_gt_images:
        ioi_val_images_png.append(val_gt_images[val_image_id]['image_name'].replace('jpg','png'))
    ioi_train_images_png=[]
    ioi_val_images_id=[int(name[:-4].lstrip()) for name in ioi_val_images_png]
    ioi_train_images_id=[]
    ioi_val_images_dict={}
    ioi_train_images_dict={}
    for image_id in all_dict:
        image_info=all_gt_images[image_id]
        segments_info=all_dict[image_id]
        for instance_id in segments_info:
            if instance_id=="0":
                pass
            instance=segments_info[instance_id]
            if instance_id in pred_to_gt[image_id]:
                instance['labeled'] = pred_to_gt[image_id][instance_id]['label']
        if int(image_id) in ioi_val_images_id:
            ioi_val_images_dict[image_id] = {"image_id": image_info['image_id'], "image_name": image_info['image_name'],
                                             "height": image_info['height'], "width": image_info['width'],
                                             'segments_info': segments_info}
        else:
            ioi_train_images_png.append(image_info['image_name'].replace(".jpg",".png"))
            ioi_train_images_id.append(image_id)
            ioi_train_images_dict[image_id] = {"image_id": image_info['image_id'], "image_name": image_info['image_name'],
                                               "height": image_info['height'],"width": image_info['width'],
                                               'segments_info': segments_info}
    json.dump(ioi_val_images_id,open("data/middle/ioi_val_images_id_"+segmentation_model+".json",'w'))
    json.dump(ioi_train_images_id,open("data/middle/ioi_train_images_id_"+segmentation_model+".json",'w'))
    json.dump(ioi_val_images_png, open("data/middle/ioi_val_images_png_" + segmentation_model + ".json", 'w'))
    json.dump(ioi_train_images_png, open("data/middle/ioi_train_images_png_"+segmentation_model+".json", 'w'))
    json.dump(ioi_val_images_dict,open("data/middle/ioi_val_images_dict_"+segmentation_model+".json",'w'))
    json.dump(ioi_train_images_dict,open("data/middle/ioi_train_images_dict_"+segmentation_model+".json",'w'))
    json.dump(dict(ioi_train_images_dict, **ioi_val_images_dict ),open("data/middle/ioi_all_images_dict_"+segmentation_model+".json",'w'))

def filter_by_saliency(segmentation_model,saliency_model):
    saliency_images=os.listdir("../"+saliency_model)
    try:
        train=json.load(open("data/middle/ioi_train_images_dict_"+segmentation_model+".json",'r'))
        exist_train=dict()
        for image_id in train:
            if train[image_id]['image_name'].replace(".jpg",".png") in saliency_images:
                exist_train[image_id]=train[image_id]
        json.dump(exist_train,open("data/middle/ioi_train_images_dict_"+segmentation_model+"_"+saliency_model+".json",'w'))
        print("filter_by_saliency:"+str(len(exist_train))+"/"+str(len(train)))
    except:
        print("no train data")
    val=json.load(open("data/middle/ioi_val_images_dict_"+segmentation_model+".json",'r'))
    exist_val=dict()
    for image_id in val:
        if val[image_id]['image_name'].replace(".jpg",".png") in saliency_images:
            exist_val[image_id]=val[image_id]
    json.dump(exist_val,open("data/middle/ioi_val_images_dict_"+segmentation_model+"_"+saliency_model+".json",'w'))
    # print("data/middle/ioi_val_images_dict_"+segmentation_model+".json")
    print("filter_by_saliency:"+str(len(exist_val))+"/"+str(len(val)))

def compute_instance_saliency(segmentation_model,saliency_model):
    try:
        images_dict=json.load(open("data/middle/ioi_train_images_dict_"+segmentation_model+"_"+saliency_model+".json",'r'))
        results_len=len(images_dict)
        results_file = [images_dict[img_id]['image_name'].replace(".jpg",".png") for img_id in images_dict]
        cpu_cnt = multiprocessing.cpu_count()
        step = max(int(results_len / cpu_cnt), 1)
        pool = multiprocessing.Pool()
        procs = []
        for begin in range(0, results_len, step):
            end = begin + step
            if end > results_len:
                end = results_len
            procs.append(pool.apply_async(run_proc, (begin, end,results_file,images_dict,segmentation_model,saliency_model)))

        instance_saliency = {}
        for proc in procs:
            instance_saliency = {**proc.get(), **instance_saliency}
        json.dump(instance_saliency, open('data/middle/ioi_train_images_dict_with_diff_saliency_'+segmentation_model+"_"+saliency_model+'.json', 'w'))
    except Exception as e:
        print(e)
    images_dict = json.load(open("data/middle/ioi_val_images_dict_" + segmentation_model + "_" + saliency_model + ".json", 'r'))
    results_len = len(images_dict)
    results_file = [images_dict[img_id]['image_name'].replace(".jpg", ".png") for img_id in images_dict]
    cpu_cnt = multiprocessing.cpu_count()
    step = max(int(results_len / cpu_cnt), 1)
    pool = multiprocessing.Pool()
    procs = []
    for begin in range(0, results_len, step):
        end = begin + step
        if end > results_len:
            end = results_len
        procs.append(pool.apply_async(run_proc, (begin, end, results_file, images_dict, segmentation_model,saliency_model)))

    instance_saliency = {}
    for proc in procs:
        instance_saliency = {**proc.get(), **instance_saliency}
    json.dump(instance_saliency, open('data/middle/ioi_val_images_dict_with_diff_saliency_' + segmentation_model+"_"+saliency_model + '.json', 'w'))

def compute_instance_saliency_list(segmentation_model,saliency_train_model,saliency_model_list):
    images_dict = json.load(open("data/middle/ioi_val_images_dict_" + segmentation_model + "_"+saliency_train_model+".json", 'r'))
    for saliency_model in saliency_model_list:
        print(saliency_model)
        results_len = len(images_dict)
        results_file = [images_dict[img_id]['image_name'].replace(".jpg", ".png") for img_id in images_dict]
        cpu_cnt = multiprocessing.cpu_count()
        step = max(int(results_len / cpu_cnt), 1)
        pool = multiprocessing.Pool()
        procs = []
        for begin in range(0, results_len, step):
            end = begin + step
            if end > results_len:
                end = results_len
            procs.append(pool.apply_async(run_proc, (begin, end, results_file, images_dict, segmentation_model,saliency_model)))

        instance_saliency = {}
        for proc in procs:
            instance_saliency = {**proc.get(), **instance_saliency}
        images_dict=instance_saliency
    print('data/middle/ioi_val_images_dict_with_diff_saliency_' + segmentation_model + '.json')
    print(len(images_dict))
    json.dump(images_dict,open('data/middle/ioi_val_images_dict_with_diff_saliency_' + segmentation_model + '.json','w'))

def run_proc(begin, end, results_file, images_dict, segmentation_model,saliency_model):
    print('Computing [%d, %d)...' % (begin, end))

    instance_saliency = {}
    for result_file in results_file[begin:end]:
        img_id = result_file[:-4].lstrip('0')
        if not img_id in images_dict:
            print('[Warning] Images dict file does not contain image_id: "%s", this image will be skipped.' % img_id)
            continue

        segmentation_file = os.path.join("../"+segmentation_model, result_file)
        if not os.path.exists(segmentation_file):
            print('[Warning] the segmentation file "%s" does not exist, this image will be skipped.' % segmentation_file)
            continue

        result_file = os.path.join("../"+saliency_model, result_file)
        if not os.path.exists(result_file):
            print('[Warning] the result file "%s" does not exist, this image will be skipped.' % result_file)
            continue

        result_img = Image.open(result_file).convert('L')
        seg_img = Image.open(segmentation_file).convert('RGB')

        if result_img.size != seg_img.size:
            result_img = result_img.resize(seg_img.size)

        saliency_mask = np.array(result_img, dtype=np.uint8)
        seg_mask = utils.rgb2id(np.array(seg_img, dtype=np.uint8))

        instances = images_dict[img_id]["segments_info"]
        for instance_id in instances:
            instance = instances[instance_id]
            instance_mask=seg_mask == int(instance_id)
            saliencys=sorted(saliency_mask[instance_mask],reverse=True)
            if saliency_mask[instance_mask].shape[0]==0:
                instance[saliency_model+'_max'] = 0
                instance[saliency_model+'_mean'] = 0
                instance[saliency_model+'_q3'] = 0
            else:
                instance_mask = (instance_mask).astype(np.uint8)
                instance[saliency_model+'_max'] = int(saliencys[0])#int(np.amax(instance_mask * saliency_mask))
                instance[saliency_model+'_mean'] = int(np.sum(saliencys)/np.count_nonzero(instance_mask))#int(np.sum(instance_mask * saliency_mask) / np.count_nonzero(instance_mask))
                instance[saliency_model+'_q3'] = int(saliencys[len(saliencys) // 4])
        instance_saliency[img_id] = images_dict[img_id]

    print('Complete [%d, %d)...' % (begin, end))
    return instance_saliency

import sys

if __name__=='__main__':
    segmentation_model = "CIN_panoptic_all"
    semantic_model = "CIN_semantic_all"
    saliency_model = "CIN_saliency_all"
    # extract_json_from_panoptic_semantic(segmentation_model,semantic_model)
    # map_instance_to_gt(segmentation_model)
    # split_dataset(segmentation_model)
    # filter_by_saliency(segmentation_model, saliency_model)
    # compute_instance_saliency(segmentation_model, saliency_model)
    # for segmentation in ['maskrcnn_panoptic','deeplab_panoptic']:
    #     extract_json_from_panoptic_semantic(segmentation, segmentation.replace("panoptic","semantic"),only_val=True)
    #     map_instance_to_gt(segmentation)
    #     generate_image_dict(segmentation)
    #     filter_by_saliency(segmentation, saliency_model)
    compute_instance_saliency_list(segmentation_model,saliency_model, \
                                   ['a-PyTorch-Tutorial-to-Image-Captioning_saiency', \
                                    'DSS-pytorch_saliency', 'MSRNet_saliency', 'NLDF_saliency', \
                                    'PiCANet-Implementation_saliency', 'salgan_saliency'])
