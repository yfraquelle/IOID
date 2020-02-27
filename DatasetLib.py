import os
import random
import json

import torch
from torch.utils.data.dataset import Dataset as TorchDataset
from torch.utils.data.dataloader import DataLoader as TorchDataLoader
from torch.utils.data.dataloader import default_collate

import numpy as np
import skimage.color
import skimage.io
import scipy.misc
from PIL import Image
from matplotlib import pyplot as plt

from utils.utils import generate_pyramid_anchors, rgb2id, resize_image, resize_mask, resize_map, minimize_mask, \
    compute_overlaps, extract_bboxes
from utils.formatting_utils import compose_image_meta, mold_image
from config import Config


############################################################
#  Data Generator
############################################################


def load_image_gt(dataset, config, image_id, augment=False, use_mini_mask=False):
    # Load image and mask
    image_name = dataset.image_info[str(image_id)]['image_name']
    # print(image_name)
    image = dataset.load_image(image_id)
    shape = image.shape
    image, window, scale, padding = resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        max_dim=config.IMAGE_MAX_DIM,
        padding=config.IMAGE_PADDING)
    image_meta = compose_image_meta(image_id, shape, window)

    thing_mask, thing_class_ids, stuff_mask, stuff_class_ids, influence_mask, influence_class_ids = dataset.load_mask(image_id)
    thing_mask = resize_mask(thing_mask, scale, padding)  # 1024
    stuff_mask = resize_mask(stuff_mask, scale, padding)  # 1024
    influence_mask = resize_mask(influence_mask, scale, padding)  # 1024
    influence_mask = resize_map(influence_mask, 1 / 8)  # 1024 -> 128
    # Resize masks to smaller size to reduce memory usage
    thing_bbox = extract_bboxes(thing_mask)
    stuff_bbox = extract_bboxes(stuff_mask)
    influence_bbox = extract_bboxes(influence_mask)

    if use_mini_mask:
        thing_mask = minimize_mask(
            thing_bbox, thing_mask, config.MINI_MASK_SHAPE)
        stuff_mask = minimize_mask(
            stuff_bbox, stuff_mask, config.MINI_MASK_SHAPE)

    segmentation = skimage.io.imread(os.path.join(dataset.annotation_dir, image_name.replace("jpg", "png")))

    semantic_label = np.zeros_like(segmentation)
    segmentation_instance_id_map=rgb2id(segmentation)
    instance_id_list=list(dataset.image_info[str(image_id)]['instances'].keys())
    for instance_id in instance_id_list:
        instance=dataset.image_info[str(image_id)]['instances'][instance_id]
        instance_mask=segmentation_instance_id_map==int(instance_id)
        semantic_label[instance_mask]=dataset.category_info[str(instance['category_id'])]['class_id']
    semantic_label=semantic_label[:,:,0]

    semantic_label_h = semantic_label.shape[0]
    semantic_label_w = semantic_label.shape[1]
    semantic_label_scale = min(500 / semantic_label_h, 500 / semantic_label_w)
    semantic_label = scipy.misc.imresize(semantic_label, (round(semantic_label_h * semantic_label_scale), round(semantic_label_w * semantic_label_scale)), interp="nearest")


    h, w = semantic_label.shape[:2]
    top_pad = (500 - h) // 2
    bottom_pad = 500 - h - top_pad
    left_pad = (500 - w) // 2
    right_pad = 500 - w - left_pad
    padding = [(top_pad, bottom_pad), (left_pad, right_pad)]
    semantic_label = np.pad(semantic_label, padding, mode='constant', constant_values=0)

    image_info = dataset.image_info[str(image_id)]

    # Random horizontal flips.
    if augment:
        if random.randint(0, 1):
            image = np.fliplr(image)
            thing_mask = np.fliplr(thing_mask)
            semantic_label = np.fliplr(semantic_label)
            segmentation = np.fliplr(segmentation)
    return image, image_meta, thing_class_ids, thing_bbox, thing_mask, stuff_class_ids, stuff_bbox, stuff_mask, \
           semantic_label, segmentation, image_info, influence_class_ids, influence_bbox, influence_mask


def build_rpn_targets(image_shape, anchors, gt_class_ids, gt_boxes, config):
    """Given the anchors and GT boxes, compute overlaps and identify positive
    anchors and deltas to refine them to match their corresponding GT boxes.

    anchors: [num_anchors, (y1, x1, y2, x2)]
    gt_class_ids: [num_gt_boxes] Integer class IDs.
    gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]

    Returns:
    rpn_match: [N] (int32) matches between anchors and GT boxes.
               1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_bbox: [N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    """
    # RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
    # RPN bounding boxes: [max anchors per image, (dy, dx, log(dh), log(dw))]
    rpn_bbox = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4))

    # Handle COCO crowds
    # A crowd box in COCO is a bounding box around several instances. Exclude
    # them from training. A crowd box is given a negative class ID.
    crowd_ix = np.where(gt_class_ids < 0)[0]
    if crowd_ix.shape[0] > 0:
        # Filter out crowds from ground truth class IDs and boxes
        non_crowd_ix = np.where(gt_class_ids > 0)[0]
        crowd_boxes = gt_boxes[crowd_ix]
        gt_class_ids = gt_class_ids[non_crowd_ix]
        gt_boxes = gt_boxes[non_crowd_ix]
        # Compute overlaps with crowd boxes [anchors, crowds]
        crowd_overlaps = compute_overlaps(anchors, crowd_boxes)
        crowd_iou_max = np.amax(crowd_overlaps, axis=1)
        no_crowd_bool = (crowd_iou_max < 0.001)
    else:
        # All anchors don't intersect a crowd
        no_crowd_bool = np.ones([anchors.shape[0]], dtype=bool)

    # Compute overlaps [num_anchors, num_gt_boxes]
    overlaps = compute_overlaps(anchors, gt_boxes)

    # Match anchors to GT Boxes
    # If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
    # If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
    # Neutral anchors are those that don't match the conditions above,
    # and they don't influence the loss function.
    # However, don't keep any GT box unmatched (rare, but happens). Instead,
    # match it to the closest anchor (even if its max IoU is < 0.3).
    #
    # 1. Set negative anchors first. They get overwritten below if a GT box is
    # matched to them. Skip boxes in crowd areas.
    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
    rpn_match[(anchor_iou_max < 0.3) & (no_crowd_bool)] = -1
    # 2. Set an anchor for each GT box (regardless of IoU value).
    # TODO: If multiple anchors have the same IoU match all of them
    gt_iou_argmax = np.argmax(overlaps, axis=0)
    rpn_match[gt_iou_argmax] = 1
    # 3. Set anchors with high overlap as positive.
    rpn_match[anchor_iou_max >= 0.7] = 1

    # Subsample to balance positive and negative anchors
    # Don't let positives be more than half the anchors
    ids = np.where(rpn_match == 1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
    if extra > 0:
        # Reset the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0
    # Same for negative proposals
    ids = np.where(rpn_match == -1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE -
                        np.sum(rpn_match == 1))
    if extra > 0:
        # Rest the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0

    # For positive anchors, compute shift and scale needed to transform them
    # to match the corresponding GT boxes.
    ids = np.where(rpn_match == 1)[0]
    ix = 0  # index into rpn_bbox
    # TODO: use box_refinment() rather than duplicating the code here
    for i, a in zip(ids, anchors[ids]):
        # Closest gt box (it might have IoU < 0.7)
        gt = gt_boxes[anchor_iou_argmax[i]]

        # Convert coordinates to center plus width/height.
        # GT Box
        gt_h = gt[2] - gt[0]
        gt_w = gt[3] - gt[1]
        gt_center_y = gt[0] + 0.5 * gt_h
        gt_center_x = gt[1] + 0.5 * gt_w
        # Anchor
        a_h = a[2] - a[0]
        a_w = a[3] - a[1]
        a_center_y = a[0] + 0.5 * a_h
        a_center_x = a[1] + 0.5 * a_w

        # Compute the bbox refinement that the RPN should predict.
        rpn_bbox[ix] = [
            (gt_center_y - a_center_y) / a_h,
            (gt_center_x - a_center_x) / a_w,
            np.log(gt_h / a_h),
            np.log(gt_w / a_w),
        ]
        # Normalize
        rpn_bbox[ix] /= config.RPN_BBOX_STD_DEV
        ix += 1

    return rpn_match, rpn_bbox


class Dataset(TorchDataset):
    def __init__(self, dataset, config, augment=True):
        self.b = 0  # batch item index
        self.image_index = -1
        self.image_ids = np.copy(dataset.image_ids)
        self.error_count = 0

        self.dataset = dataset
        self.config = config
        self.augment = augment

        # Anchors
        # [anchor_count, (y1, x1, y2, x2)]
        self.anchors = generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                                config.RPN_ANCHOR_RATIOS,
                                                config.BACKBONE_SHAPES,
                                                config.BACKBONE_STRIDES,
                                                config.RPN_ANCHOR_STRIDE)

    def __getitem__(self, image_index):
        # try:
            image_id = self.image_ids[image_index]
            image, image_metas, gt_thing_class_ids, gt_thing_boxes, gt_thing_masks, \
            gt_stuff_class_ids, gt_stuff_boxes, gt_stuff_masks, gt_semantic_label, gt_segmentation, \
            gt_image_instances, gt_influence_class_ids, gt_influence_boxes, gt_influence_masks = \
                load_image_gt(self.dataset, self.config, image_id,
                              augment=self.augment, use_mini_mask=self.config.USE_MINI_MASK)

            if gt_thing_class_ids.shape[0] == 0 or gt_stuff_class_ids.shape[0] == 0 or gt_influence_class_ids.shape[0] == 0:
                print(str(image_id)+" error")
                return None

            # RPN Targets
            rpn_match, rpn_bbox = build_rpn_targets(image.shape, self.anchors,
                                                    gt_thing_class_ids, gt_thing_boxes, self.config)

            # If more instances than fits in the array, sub-sample from them.
            if gt_thing_boxes.shape[0] > self.config.MAX_GT_INSTANCES:
                ids = np.random.choice(
                    np.arange(gt_thing_boxes.shape[0]), self.config.MAX_GT_INSTANCES, replace=False)
                gt_thing_class_ids = gt_thing_class_ids[ids]
                gt_thing_boxes = gt_thing_boxes[ids]
                gt_thing_masks = gt_thing_masks[:, :, ids]

            # Add to batch
            rpn_match = rpn_match[:, np.newaxis]
            images = mold_image(image.astype(np.float32), self.config)

            # Convert
            images = torch.from_numpy(images.transpose(2, 0, 1)).float()
            image_metas = torch.from_numpy(image_metas.astype(np.float32))
            rpn_match = torch.from_numpy(rpn_match)
            rpn_bbox = torch.from_numpy(rpn_bbox).float()

            if gt_thing_class_ids.shape[0] > 0:
                gt_thing_class_ids = torch.from_numpy(gt_thing_class_ids)
                gt_thing_boxes = torch.from_numpy(gt_thing_boxes).float()
                gt_thing_masks = gt_thing_masks.astype(int).transpose(2, 0, 1)
                # for i in range(gt_thing_masks.shape[0]):
                #     plt.figure()
                #     plt.imshow(gt_thing_masks[i])
                #     plt.show()
                gt_thing_masks = torch.from_numpy(gt_thing_masks).float()
            else:
                gt_thing_class_ids = torch.IntTensor()
                gt_thing_boxes = torch.FloatTensor()
                gt_thing_masks = torch.FloatTensor()

            if gt_stuff_class_ids.shape[0] > 0:
                gt_stuff_class_ids = torch.from_numpy(gt_stuff_class_ids)
                gt_stuff_boxes = torch.from_numpy(gt_stuff_boxes).float()
                gt_stuff_masks = gt_stuff_masks.astype(int).transpose(2, 0, 1)
                # for i in range(gt_stuff_masks.shape[0]):
                #     plt.figure()
                #     plt.imshow(gt_stuff_masks[i])
                #     plt.show()
                gt_stuff_masks = torch.from_numpy(gt_stuff_masks).float()
            else:
                gt_stuff_class_ids = torch.IntTensor()
                gt_stuff_boxes = torch.FloatTensor()
                gt_stuff_masks = torch.FloatTensor()

            gt_semantic_label = torch.from_numpy(
                np.ascontiguousarray(gt_semantic_label, dtype=np.uint8)).long()

            gt_segmentation = torch.from_numpy(
                np.ascontiguousarray(gt_segmentation, dtype=np.uint8)).long()

            if gt_influence_class_ids.shape[0] > 0:
                gt_influence_class_ids = torch.from_numpy(
                    gt_influence_class_ids)
                gt_influence_boxes = torch.from_numpy(
                    gt_influence_boxes).float()
                gt_influence_masks = gt_influence_masks.astype(
                    int).transpose(2, 0, 1)
                # for i in range(gt_stuff_masks.shape[0]):
                #     plt.figure()
                #     plt.imshow(gt_stuff_masks[i])
                #     plt.show()
                gt_influence_map = torch.from_numpy(
                    np.max(gt_influence_masks, axis=0)).float()
                gt_influence_masks = torch.from_numpy(
                    gt_influence_masks).float()
            else:
                gt_influence_class_ids = torch.IntTensor()
                gt_influence_boxes = torch.FloatTensor()
                gt_influence_masks = torch.FloatTensor()
                gt_influence_map = torch.FloatTensor()

            # plt.figure()
            # plt.imshow(gt_segmentation)
            # plt.show()
            # print(gt_image_instances)
            # # print(gt_thing_masks.shape)
            # # print(gt_stuff_masks.shape)
            # print(gt_influence_map.shape)

            return images, image_metas, rpn_match, rpn_bbox, \
                   gt_thing_class_ids, gt_thing_boxes, gt_thing_masks, \
                   gt_stuff_class_ids, gt_stuff_boxes, gt_stuff_masks, \
                   gt_semantic_label, gt_influence_map, \
                   gt_influence_class_ids, gt_influence_boxes, gt_influence_masks, \
                   gt_segmentation, gt_image_instances

        # except:
        #     return None

    def __len__(self):
        return self.image_ids.shape[0]


############################################################
#  Dataset
############################################################


class BaseDataset(object):
    def __init__(self):
        self.image_dir = "../data/ioid_images"
        self.annotation_dir = "../data/ioid_panoptic"
        # self.interest_label_file = "data/interest_objects_by_image_all.json"
        self.interest_info = dict()
        self.image_ids = []
        self.image_info = dict()
        self.num_images = 0

        self.class_ids = []
        self.class_names = []
        self.class_info = dict()
        self.num_classes = 0

        self.thing_class_ids = []
        self.thing_class_names = []
        self.thing_class_info = dict()
        self.thing_num_classes = 0

        self.stuff_class_ids = []
        self.stuff_class_names = []
        self.stuff_class_info = dict()
        self.stuff_num_classes = 0

        self.anns = dict()

    def prepare(self):
        self.image_ids = np.array(
            list(map(eval, list(self.image_info.keys()))))
        self.num_images = len(self.image_info)
        self.num_classes = len(self.class_info)
        self.class_ids = np.array(
            list(map(eval, list(self.class_info.keys()))))
        self.class_names = [
            self.class_info[str(class_id)]["name"] for class_id in self.class_ids]

        thing_class_info_list = [self.class_info[str(
            class_id)] for class_id in self.class_ids if self.class_info[str(class_id)]["isthing"]]
        for thing_class_info in thing_class_info_list:
            self.thing_class_info[str(
                thing_class_info['class_id'])] = thing_class_info
        self.thing_class_ids = np.array(
            list(map(eval, list(self.thing_class_info.keys()))))
        self.thing_num_classes = len(self.thing_class_info)
        self.thing_class_names = [self.thing_class_info[str(
            thing_class_id)]["name"] for thing_class_id in self.thing_class_ids]

        stuff_class_info_list = [self.class_info[str(
            class_id)] for class_id in self.class_ids if not self.class_info[str(class_id)]["isthing"]]
        for stuff_class_info in stuff_class_info_list:
            self.stuff_class_info[str(
                stuff_class_info['class_id'])] = stuff_class_info
        self.stuff_class_ids = np.array(
            list(map(eval, list(self.stuff_class_info.keys()))))
        self.stuff_num_classes = len(self.stuff_class_info)
        self.stuff_class_names = [self.stuff_class_info[str(
            stuff_class_id)]["name"] for stuff_class_id in self.stuff_class_ids]

        for image_id in self.image_info:
            image = self.image_info[image_id]
            for instance_id in image['instances']:
                # labeled = self.interest_info[image_id]['instances'][str(
                #     segment['id'])]['labeled']
                # if labeled:
                #     segment['labeled'] = 1
                # else:
                #     segment['labeled'] = 0
                self.anns[image_id + "_" + instance_id] = image['instances'][instance_id]

    def load_image(self, image_id):
        image_path = os.path.join(
            self.image_dir, self.image_info[str(image_id)]['image_name'])
        image = skimage.io.imread(image_path)
        if len(image.shape)==2:
            image = np.stack([image,image,image],axis=2)
            print(image.shape)
        # image = Image.open(image_path)
        return image

    def load_mask(self, image_id):
        mask = np.empty([0, 0, 0])
        class_ids = np.empty([0], np.int32)
        return mask, class_ids


class OOIDataset(BaseDataset):
    def __init__(self, mode="train"):
        super(OOIDataset, self).__init__()
        self.load(mode)

    def load(self, mode):
        self.class_info = json.load(open("data/class_dict.json", 'r'))
        self.category_info={}
        for class_id in self.class_info:
            self.category_info[str(self.class_info[class_id]['category_id'])]=self.class_info[class_id]
        self.image_info = json.load(
            open("data/" + mode + "_images_dict.json", 'r'))
        self.prepare()

    def load_mask(self, image_id, type="thing"):
        image_info = self.image_info[str(image_id)]
        # image_interest_info = self.interest_info[str(image_id)]
        annotations = image_info['instances']
        masks_file = image_info['image_name'].replace("jpg", "png")
        masks_path = os.path.join(self.annotation_dir, masks_file)

        segmentation = np.array(Image.open(masks_path), dtype=np.uint8)

        segmentation_id = rgb2id(segmentation)

        thing_masks = []
        thing_class_ids = []
        stuff_masks = []
        stuff_class_ids = []
        influence_masks = []
        influence_class_ids = []
        for segment_info_id in annotations:
            segment_info = annotations[segment_info_id]
            category_id = segment_info['category_id']
            # find the class_id if we know the category_id
            class_id = 0
            for key in self.class_info:
                if self.class_info[key]['category_id'] == category_id:
                    class_id = self.class_info[key]['class_id']
            # class_id = self.category_info[str(category_id)]['idx']
            id = segment_info['id']
            islabel = segment_info['labeled']
            iscrowd = segment_info['iscrowd']
            isthing = self.class_info[str(class_id)]['isthing']
            mask = (segmentation_id == segment_info['id'])
            if isthing:
                if iscrowd:
                    thing_class_ids.append(-1 * class_id)
                else:
                    thing_class_ids.append(class_id)
                thing_masks.append(mask)
            else:
                stuff_class_ids.append(class_id)
                stuff_masks.append(mask)
            if islabel:
                influence_class_ids.append(class_id)
                influence_masks.append(mask)

        if thing_class_ids:
            thing_masks = np.stack(thing_masks, axis=2)
            thing_class_ids = np.array(thing_class_ids, dtype=np.int32)
        else:
            thing_masks = np.empty([0, 0, 0])
            thing_class_ids = np.empty([0], np.int32)
        if stuff_class_ids:
            stuff_masks = np.stack(stuff_masks, axis=2)
            stuff_class_ids = np.array(stuff_class_ids, dtype=np.int32)
        else:
            stuff_masks = np.empty([0, 0, 0])
            stuff_class_ids = np.empty([0], np.int32)
        if influence_class_ids:
            influence_masks = np.stack(influence_masks, axis=2)
            influence_class_ids = np.array(influence_class_ids, dtype=np.int32)
        else:
            influence_masks = np.empty([0, 0, 0])
            influence_class_ids = np.empty([0], np.int32)

        # print(thing_masks.shape)
        # print(stuff_masks.shape)
        # print(influence_masks.shape)

        return thing_masks, thing_class_ids, stuff_masks, stuff_class_ids, influence_masks, influence_class_ids


if __name__ == '__main__':
    class CINConfig(Config):
        NAME = "ooi"
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        NUM_CLASSES = 1 + 133
        THING_NUM_CLASSES = 1 + 80
        STUFF_NUM_CLASSES = 1 + 53


    config = CINConfig()
    dataset_val = OOIDataset("val")
    val_set = Dataset(dataset_val, config)
    # train_set.__getitem__(5)
    # train_set.__getitem__(20)
    # print("dataset_train", dataset_train.num_images)
    # print("train_set", train_set.__len__())
    #
    def my_collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        if len(batch) == 0:
            print("No valid data!!!")
            batch = [[torch.from_numpy(np.zeros([1, 1]))]]
        return default_collate(batch)

    val_generator = TorchDataLoader(val_set, collate_fn=my_collate_fn, batch_size=1, shuffle=True, num_workers=1)
    step = 0
    for inputs in val_generator:
        if len(inputs) != 17:
            print("length of inputs", len(inputs))
            print("inputs", inputs)
            continue
        print(str(step)+"/9000")
        step+=1

    '''
    result=load_image_gt(ooiDataset,OISMPSConfig(),9,True,True)
    thing_masks=result[4]
    stuff_masks=result[7]
    semantic_label=result[8]
    influence_mask=result[11]
    # print(thing_masks.shape)
    # print(stuff_masks.shape)
    # print(semantic_label.shape)
    # print(influence_mask.shape)
    for i in range(thing_masks.shape[2]):
        plt.figure()
        plt.imshow(thing_masks[:,:,i])
        plt.show()
    for i in range(stuff_masks.shape[2]):
        plt.figure()
        plt.imshow(stuff_masks[:,:,i])
        plt.show()
    plt.figure()
    plt.imshow(semantic_label)
    plt.show()
    gt_influence_masks = influence_mask.astype(int).transpose(2, 0, 1)
    gt_influence_map = np.max(gt_influence_masks, axis=0)
    plt.figure()
    plt.imshow(gt_influence_map)
    plt.show()
    '''
