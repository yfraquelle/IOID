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

def generate_images_dict(panoptic_model):
    class_id_dict=json.load(open("data/class_dict.json",'r'))
    val_dict=json.load(open("data/val_images_dict.json",'r'))
    count=0
    for image_id in val_dict:
        count+=1
        image_info=val_dict[image_id]
        print(str(count)+"/"+str(len(val_dict)))
        semantic = skimage.io.imread("../"+panoptic_model.replace("panoptic","semantic")+"/"+image_info['image_name'].replace("jpg","png"))
        panoptic = skimage.io.imread("../"+panoptic_model+"/"+image_info['image_name'].replace("jpg","png"))
        segmentation_id = utils.rgb2id(panoptic)
        segmentation_id_list=np.unique(segmentation_id)
        instances={}
        for instance_id in segmentation_id_list:
            if instance_id==0:
                continue
            mask = np.where(segmentation_id == instance_id, 1, 0)
            if np.sum(mask)==0:
                print(image_png+" "+str(instance_id))
                continue
            box=utils.extract_bbox(mask)
            class_id = np.unique(semantic[segmentation_id == instance_id])[0]
            category = class_id_dict[str(class_id)]
            instances[str(instance_id)]={'id':int(instance_id),'class_id':int(class_id),'category_id':category['category_id'],'category_name':category['name'],'bbox':[int(box[0]),int(box[1]),int(box[2]),int(box[3])]}
        image_info['predictions']=instances
    map_instance_to_gt(val_dict,panoptic_model)

def map_instance_to_gt(val_images,save_name):
    def compute_pixel_iou(bool_mask_pred, bool_mask_gt):
        intersection = bool_mask_pred * bool_mask_gt
        union = bool_mask_pred + bool_mask_gt
        return np.count_nonzero(intersection) / np.count_nonzero(union)

    ioi_val_images_dict = {}
    count = 0
    base = 0
    for image_id in val_images:
        count += 1
        print("map_instance_to_gt:"+str(count) + "/" + str(len(val_images)))

        segmentation = scipy.misc.imread("../"+save_name+"/" + image_id.zfill(12) + ".png")
        gt_segmentation = scipy.misc.imread("../data/ioid_panoptic/" + image_id.zfill(12) + ".png")

        image_info=val_images[image_id]
        instance_dict = image_info['predictions']
        gt_instance_dict = image_info['instances']

        segmentation_id = utils.rgb2id(segmentation)
        for instance_id in instance_dict:
            mask = segmentation_id == int(instance_id)
            instance_dict[instance_id]['mask'] = mask

        gt_segmentation_id = utils.rgb2id(gt_segmentation)
        for gt_instance_id in gt_instance_dict:
            gt_mask = gt_segmentation_id == int(gt_instance_id)
            gt_instance_dict[gt_instance_id]['mask'] = gt_mask

        instance_pred_gt_dict = {}
        instance_gt_pred_dict = {}
        if len(instance_dict)==0:
            for gt_instance_id in gt_instance_dict:
                instance_gt_pred_dict[gt_instance_id] = {"labeled": gt_instance_dict[gt_instance_id]['labeled'],"pred": []}
        else:
            for instance_id in instance_dict:
                max_iou = -1
                max_gt_instance_id = ""
                for gt_instance_id in gt_instance_dict:
                    i_iou = compute_pixel_iou(instance_dict[instance_id]['mask'], gt_instance_dict[gt_instance_id]['mask'])
                    if gt_instance_id not in instance_gt_pred_dict:
                        instance_gt_pred_dict[gt_instance_id] = {"labeled": gt_instance_dict[gt_instance_id]['labeled'], "pred": []}
                    if i_iou >= 0.5 and instance_dict[instance_id]['category_id'] == gt_instance_dict[gt_instance_id]['category_id'] and i_iou > max_iou:
                        max_gt_instance_id = gt_instance_id
                        max_iou = i_iou
                        instance_gt_pred_dict[gt_instance_id]['pred'].append(instance_id)
                if max_gt_instance_id != "":
                    instance_pred_gt_dict[instance_id] = {"gt_instance_id": max_gt_instance_id,"label": gt_instance_dict[max_gt_instance_id]['labeled']}
                else:
                    instance_pred_gt_dict[instance_id] = {"gt_instance_id": "", "label": False}

        base = 0
        for instance_id in instance_gt_pred_dict:
            if instance_gt_pred_dict[instance_id]['labeled'] == True and len(instance_gt_pred_dict[instance_id]['pred']) == 0:
                base += 1

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

        ioi_val_images_dict[image_id] = {"image_id": image_info['image_id'], "image_name": image_info['image_name'],
                                         "base":base, "height": image_info['height'], "width": image_info['width'],
                                         'segments_info': instance_dict}

    json.dump(ioi_val_images_dict, open("results/ioi_"+save_name+".json", 'w'))

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
    generate_images_dict("thing_panoptic")
    generate_images_dict("stuff_panoptic")

