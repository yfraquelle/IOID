import numpy as np
import math
from utils import utils
import scipy.misc
import scipy.ndimage
from matplotlib import pyplot as plt
from config import Config

class CINConfig(Config):
    NAME = "ooi"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1 #10
    NUM_CLASSES = 1+133
    THING_NUM_CLASSES = 1+80
    STUFF_NUM_CLASSES = 1+53

config = CINConfig()


def resize_things(detections, mrcnn_mask, new_shape):
    zero_ix = np.where(detections[:, 4] == 0)[0]
    N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

    boxes = np.multiply(detections[:N,:4],np.array([new_shape[0]/1024,new_shape[1]/1024,new_shape[0]/1024,new_shape[1]/1024])).astype(np.int32)#move_box_specific_shape_thing(detections[:N,:4],image_metas,new_shape)

    class_ids = detections[:N, 4].astype(np.int32)
    scores = detections[:N, 5]
    masks = mrcnn_mask[np.arange(N), :, :, class_ids]

    # Filter out detections with zero area. Often only happens in early
    # stages of training when the network weights are still a bit random.
    exclude_ix = np.where(
        (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
    if exclude_ix.shape[0] > 0:
        boxes = np.delete(boxes, exclude_ix, axis=0)
        class_ids = np.delete(class_ids, exclude_ix, axis=0)
        scores = np.delete(scores, exclude_ix, axis=0)
        masks = np.delete(masks, exclude_ix, axis=0)

    class_ids=class_ids.reshape([-1, 1])

    # Resize masks to original image size and set boundary threshold.
    if masks.shape[0]>0:
        temp_masks=masks.transpose(1,2,0)
        masks= utils.expand_mask(boxes, temp_masks, new_shape).transpose(2, 0, 1)
    else:
        masks=np.empty((0,new_shape[0],new_shape[1]))

    return class_ids,boxes,masks,scores

def resize_stuffs(detections,masks,new_shape):
    boxes = np.multiply(detections[:,:4], np.array([new_shape[0]/500, new_shape[1]/500, new_shape[0]/500, new_shape[1]/500])).astype(np.int32)
    class_ids = detections[:, 4].reshape([-1, 1])
    exclude_ix = np.where(
        (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]

    if exclude_ix.shape[0] > 0:
        boxes = np.delete(boxes, exclude_ix, axis=0)
        class_ids = np.delete(class_ids, exclude_ix, axis=0)
        masks = np.delete(masks, exclude_ix, axis=0)
    stuff_masks = []
    if masks.shape[0]>0:
        for i in range(masks.shape[0]):
            m = np.where(scipy.misc.imresize(masks[i], new_shape, interp='bilinear') >= 128, 1, 0)
            stuff_masks.append(m)
        masks=np.stack(stuff_masks)
    else:
        masks = np.empty((0, new_shape[0], new_shape[1]))

    return class_ids,boxes,masks

def influence_from_full_box_mask(class_ids,boxes,masks,influence_map):
    threshold = 0.5
    box_influence_all = []
    mask_influence_all = []
    mask_bi = []
    for i in range(class_ids.shape[0]):
        box = boxes[i]
        mask = np.where(masks[i] >= threshold, 1, 0).astype(np.uint8)
        mask_bi.append(mask)
        box_influence = np.sum(influence_map[box[0]:box[2],box[1]-box[3]])
        mask_influence = np.sum(np.multiply(mask,influence_map))
        box_influence_all.append(box_influence)
        mask_influence_all.append(mask_influence)
    box_influence_all = np.concatenate(box_influence_all,axis=0) # x,1
    mask_influence_all = np.concatenate(mask_influence,axis=0) # x,1
    mask_bi=np.concatenate(mask_bi,axis=0) # x,128,128
    return box_influence_all, mask_influence_all,mask_bi

def influence_from_hightest_mask(class_ids,boxes,masks,influence_map):
    threshold = 0.5
    mask_influence_all=[]
    class_ids_bi = []
    boxes_bi = []
    mask_bi = []
    for i in range(masks.shape[0]):
        mask = np.where(masks[i] >= threshold, 1, 0).astype(np.uint8)
        if np.sum(np.where(mask==1))>0:
            class_ids_bi.append(class_ids[i])
            boxes_bi.append(boxes[i])
            mask_bi.append(mask)
            mask_influence = np.max(np.multiply(mask, influence_map))
            mask_influence_all.append(mask_influence)
    class_ids_bi = np.stack(class_ids_bi,axis=0)
    boxes_bi = np.stack(boxes_bi,axis=0)
    mask_bi = np.stack(mask_bi, axis=0)  # x,128,128
    mask_influence_all = np.array(mask_influence_all).reshape((-1,1)) # x,1
    return class_ids_bi,boxes_bi,mask_bi,mask_influence_all

def compute_box_overlap():
    pass

def compute_distance_points(point1,point2):
    return math.sqrt(math.pow(point1[0]-point2[0],2)+math.pow(point1[1]-point2[1],2))
def compute_distance_lists(list1,list2):
    min_d=compute_distance_points(list1[0],list2[0])
    for point1 in list1:
        for point2 in list2:
            d=compute_distance_points(point1,point2)
            min_d=min(min_d,d)
            if min_d<=3:
                return min_d
    return min_d

def compute_mask_distances(class_ids_sort,masks_sort):
    distances_sort=[]
    id_position_group=dict()
    for i in range(class_ids_sort.shape[0]):
        id=int(class_ids_sort[i][0])
        mask = masks_sort[i]
        localtion = np.where(mask==1)
        points=[]
        for j in range(localtion[0].shape[0]):
            points.append((int(localtion[0][j]),int(localtion[1][j])))
        id_position_group[id]=points

    for id_index in range(class_ids_sort.shape[0]):
        id = int(class_ids_sort[id_index][0])
        if id not in id_position_group:
            continue
        distance = []
        for wait_id_index in range(class_ids_sort.shape[0]):
            wait_id = int(class_ids_sort[wait_id_index][0])
            if wait_id not in id_position_group:
                continue
            # print(" " + str(wait_id))
            if wait_id_index < id_index:
                distance.append(distances_sort[wait_id_index][id_index])
            elif wait_id_index == id_index:
                distance.append(1)
            else:
                min_distance_id_waitid = compute_distance_lists(id_position_group[id], id_position_group[wait_id])
                if min_distance_id_waitid==0:
                    min_distance_id_waitid=1
                distance.append(1/min_distance_id_waitid)
        distances_sort.append(distance)
    distances_sort = np.array(distances_sort, dtype=np.float32)
    return distances_sort

def pack_influential_elements(thing_detections,thing_masks,stuff_detections,stuff_masks,influence_map):
    # (x,6) num_detections*(y1,x1,y2,x2,class_id,score)  (x,28,28,81)  (x,5)bbox,class  (x,500,500)  (128,128)
    thing_class_ids,thing_boxes,thing_masks,thing_scores = resize_things(thing_detections,thing_masks,(config.INSTANCE_SIZE,config.INSTANCE_SIZE))
    stuff_class_ids,stuff_boxes,stuff_masks = resize_stuffs(stuff_detections,stuff_masks,(config.INSTANCE_SIZE,config.INSTANCE_SIZE))

    thing_isthing_class_ids = np.concatenate([thing_class_ids,np.ones((thing_class_ids.shape))],axis=1) # x,2
    stuff_isthing_class_ids = np.concatenate([stuff_class_ids, np.zeros((stuff_class_ids.shape))], axis=1)  # y,2
    class_isthing_ids = np.concatenate([thing_isthing_class_ids,stuff_isthing_class_ids],axis=0) # x+y,2
    boxes = np.concatenate([thing_boxes,stuff_boxes],axis=0) # x+y,4
    masks = np.concatenate([thing_masks,stuff_masks],axis=0) # x+y,50,50 consist_mask

    influence_map_50=scipy.misc.imresize(influence_map, size=(1024,1024),interp='bilinear')

    class_isthing_ids, boxes, masks, mask_influence = influence_from_hightest_mask(class_isthing_ids,boxes,masks,influence_map_50) # binary_mask
    id_array=np.arange(0,class_isthing_ids.shape[0]).reshape(-1,1)
    influence_input = np.concatenate([id_array,class_isthing_ids, mask_influence], axis=1)  # [x+y,4] id class_id is_thing influence

    influence_input_sort_with_idx=sorted(influence_input,key=lambda x:x[3],reverse=True)

    influence_input_sort=[]
    class_ids_sort=[]
    boxes_sort=[]
    masks_sort = []
    for influence_input_with_idx in influence_input_sort_with_idx:
        idx=int(influence_input_with_idx[0])
        influence_input_single=influence_input[idx][1:]
        influence_input_sort.append(influence_input_single)
        class_id=class_isthing_ids[idx][0]
        class_ids_sort.append(class_id)
        box=boxes[idx]
        boxes_sort.append(box)
        mask=masks[idx]
        masks_sort.append(mask)
    influence_input_sort=np.array(influence_input_sort,dtype=np.float32)
    class_ids_sort = np.array(class_ids_sort).reshape(-1,1)
    boxes_sort = np.stack(boxes_sort, axis=0)
    masks_sort = np.stack(masks_sort, axis=0)

    return influence_input_sort, class_ids_sort, boxes_sort, masks_sort

def compute_pixel_iou(bool_mask_pred, bool_mask_gt):
    intersection = bool_mask_pred * bool_mask_gt
    union = bool_mask_pred + bool_mask_gt
    return np.count_nonzero(intersection) / np.count_nonzero(union)

def map_pred_with_gt_mask(gt_class_ids,gt_masks,instance_class_ids,instance_boxes,instance_masks,threshold):

    gt_interest_label = []
    for pred_idx,pred_mask in enumerate(instance_masks):
        max_gt_idx=-1
        max_gt_iou=-1
        for gt_idx,gt_mask in enumerate(gt_masks):
            pixel_iou = compute_pixel_iou(pred_mask, gt_mask)
            if pixel_iou >= threshold and gt_class_ids[gt_idx]==instance_class_ids[pred_idx] and pixel_iou>max_gt_iou:
                max_gt_idx = gt_idx
                max_gt_iou = pixel_iou
        if max_gt_iou>-1:
            gt_interest_label.append(1)
        else:
            gt_interest_label.append(0)

    return np.array(gt_interest_label,dtype=np.float32)


def resize_thing_masks(thing_detections,mrcnn_mask):
    scale=config.INSTANCE_SIZE/28

    zero_ix = np.where(thing_detections[:, 4] == 0)[0]
    N = zero_ix[0] if zero_ix.shape[0] > 0 else thing_detections.shape[0]

    thing_class_ids=thing_detections[:N,4].astype(np.int32)
    thing_boxes=thing_detections[:N,:4]
    thing_scores = thing_detections[:N, 5]
    thing_masks = mrcnn_mask[np.arange(N), :, :, thing_class_ids]

    # Filter out detections with zero area. Often only happens in early
    # stages of training when the network weights are still a bit random.
    exclude_ix = np.where(
        (thing_boxes[:, 2] - thing_boxes[:, 0]) * (thing_boxes[:, 3] - thing_boxes[:, 1]) <= 0)[0]
    if exclude_ix.shape[0] > 0:
        thing_boxes = np.delete(thing_boxes, exclude_ix, axis=0)
        thing_class_ids = np.delete(thing_class_ids, exclude_ix, axis=0)
        thing_scores = np.delete(thing_scores, exclude_ix, axis=0)
        thing_masks = np.delete(thing_masks, exclude_ix, axis=0)

    thing_class_ids = thing_class_ids.reshape([-1, 1])
    thing_masks = scipy.ndimage.zoom(thing_masks, zoom=[1, scale, scale], order=0)
    return thing_class_ids,thing_boxes,thing_masks,thing_scores

def resize_stuff_masks(stuff_detections,stuff_masks):
    scale=1024/500
    stuff_class_ids=stuff_detections[:,4:5]
    stuff_boxes=stuff_detections[:,:4]
    stuff_mask_mini=[]

    for i in range(stuff_masks.shape[0]):
        y1, x1, y2, x2 = (stuff_boxes[i][:4]*scale).astype(np.int32)
        mask = scipy.ndimage.zoom(stuff_masks[i], zoom=[scale, scale], order=0)
        instance_mask = mask[y1:y2, x1:x2]
        instance_mask = scipy.ndimage.zoom(instance_mask, [config.INSTANCE_SIZE/500, config.INSTANCE_SIZE/500], mode='nearest',order=0)
        stuff_mask_mini.append(instance_mask)
    stuff_mask_mini=np.stack(stuff_mask_mini)
    stuff_boxes=stuff_boxes*scale
    return stuff_class_ids,stuff_boxes,stuff_mask_mini

def filter_thing_masks(thing_detections,mrcnn_mask,image_shape,window):
    if (thing_detections.shape[0] == 0):
        thing_class_ids = []
        thing_boxes = []
        thing_masks_umold = []
        thing_scores = []
        return thing_class_ids, thing_boxes, thing_masks_umold, thing_scores
    zero_ix = np.where(thing_detections[:, 4] == 0)[0]
    N = zero_ix[0] if zero_ix.shape[0] > 0 else thing_detections.shape[0]

    thing_class_ids=thing_detections[:N,4].astype(np.int32)
    thing_boxes=thing_detections[:N,:4]
    thing_scores = thing_detections[:N, 5]
    thing_masks = mrcnn_mask[np.arange(N), :, :, thing_class_ids]

    # Filter out detections with zero area. Often only happens in early
    # stages of training when the network weights are still a bit random.
    exclude_ix = np.where(
        (thing_boxes[:, 2] - thing_boxes[:, 0]) * (thing_boxes[:, 3] - thing_boxes[:, 1]) <= 0)[0]
    if exclude_ix.shape[0] > 0:
        thing_boxes = np.delete(thing_boxes, exclude_ix, axis=0)
        thing_class_ids = np.delete(thing_class_ids, exclude_ix, axis=0)
        thing_scores = np.delete(thing_scores, exclude_ix, axis=0)
        thing_masks = np.delete(thing_masks, exclude_ix, axis=0)

    h_scale = image_shape[0] / (window[2] - window[0]) # h_ori_image/h_box_image
    w_scale = image_shape[1] / (window[3] - window[1]) # w_ori_image/w_box_image
    scale = min(h_scale, w_scale)
    shifts = np.array([window[0], window[1], window[0], window[1]])
    scales = np.array([scale, scale, scale, scale])
    thing_boxes = np.multiply(thing_boxes-shifts,scales)
    thing_class_ids = thing_class_ids.reshape([-1, 1])

    #resize
    thing_masks_unmold = []
    final_thing_class_ids = []
    final_thing_boxes = []
    final_thing_scores = []
    for i in range(thing_class_ids.shape[0]):
        threshold = 0.5
        y1, x1, y2, x2 = thing_boxes[i].astype(np.int32)

        mask = scipy.misc.imresize(thing_masks[i], (y2 - y1, x2 - x1), interp='bilinear').astype(np.float32) / 255.0
        mask = np.where(mask >= threshold, 1, 0).astype(np.uint8)
        full_image = np.zeros(image_shape[:2], dtype=np.uint8)
        full_image[y1:y2,x1:x2]=mask
        thing_box=utils.extract_bbox(full_image)
        if (thing_box[2]-thing_box[0])*(thing_box[3]-thing_box[1])>0:
            final_thing_class_ids.append(thing_class_ids[i])
            final_thing_boxes.append(thing_box)
            thing_masks_unmold.append(full_image)
            final_thing_scores=thing_scores[i]
    thing_masks_unmold=np.stack(thing_masks_unmold)
    final_thing_class_ids=np.array(final_thing_class_ids)
    final_thing_boxes=np.stack(final_thing_boxes)
    final_thing_scores=np.array(final_thing_scores)
    return final_thing_class_ids,final_thing_boxes,thing_masks_unmold,final_thing_scores

def filter_stuff_masks(stuff_detections,stuff_masks,image_shape,window):
    if(stuff_detections.shape[0] == 0 and stuff_masks.shape[0] == 0):
        stuff_class_ids = []
        stuff_boxes = []
        stuff_masks_umold = []
        return stuff_class_ids, stuff_boxes, stuff_masks_umold
    stuff_class_ids=stuff_detections[:,4:5]
    stuff_boxes=stuff_detections[:,:4]

    h, w = image_shape[:2]
    mask_scale = max(h,w)/500.0
    top_pad = (max(h,w) - h) // 2
    left_pad = (max(h,w) - w) // 2
    shifts = np.array([top_pad, left_pad, top_pad, left_pad])
    stuff_boxes = stuff_boxes * mask_scale - shifts#np.multiply(, scales)
    stuff_masks=scipy.ndimage.zoom(stuff_masks, zoom=[1, mask_scale, mask_scale], order=0)
    stuff_masks_umold=[]
    final_stuff_class_ids = []
    final_stuff_boxes = []
    for i in range(stuff_class_ids.shape[0]):
        stuff_mask=stuff_masks[i][top_pad:h+top_pad,left_pad:w+left_pad]
        stuff_box=utils.extract_bbox(stuff_mask)
        if (stuff_box[2]-stuff_box[0])*(stuff_box[3]-stuff_box[1])>0:
            final_stuff_class_ids.append(stuff_class_ids[i])
            final_stuff_boxes.append(stuff_box)
            stuff_masks_umold.append(stuff_mask)
    stuff_masks_umold=np.stack(stuff_masks_umold)
    final_stuff_class_ids=np.array(final_stuff_class_ids)
    final_stuff_boxes=np.stack(final_stuff_boxes)
    return final_stuff_class_ids,final_stuff_boxes,stuff_masks_umold

def resize_influence_map(influence_map,new_shape):
    return scipy.misc.imresize(influence_map, new_shape, interp='bilinear')
def resize_semantic_label(semantic_label,new_shape):
    return scipy.ndimage.zoom(semantic_label, [new_shape[0]/semantic_label.shape[0],new_shape[1]/semantic_label.shape[1]], mode='nearest',order=0)

def extract_piece_group(thing_detections,thing_masks,stuff_detections,stuff_masks,influence_map,semantic_label):
    thing_class_ids,thing_boxes,thing_masks,thing_scores=resize_thing_masks(thing_detections,thing_masks)
    stuff_class_ids,stuff_boxes,stuff_masks=resize_stuff_masks(stuff_detections,stuff_masks)
    # [n,1] [n,4]:1024 [n,56,56]
    instance_class_ids=np.concatenate([thing_class_ids,stuff_class_ids],axis=0)
    instance_boxes=np.concatenate([thing_boxes,stuff_boxes],axis=0)
    instance_masks=np.concatenate([thing_masks,stuff_masks],axis=0)

    influence_map=resize_influence_map(influence_map,(1024,1024))
    semantic_label=resize_semantic_label(semantic_label,(1024,1024))

    plt.figure()
    plt.imshow(semantic_label)
    plt.show()

    plt.figure()
    plt.imshow(influence_map,cmap="gray")
    plt.show()


    instance_piece_groups=[]
    for i in range(instance_class_ids.shape[0]):
        small_box=instance_boxes[i][:4]
        y1, x1, y2, x2 = small_box.astype(np.int32)
        mask = instance_masks[i]
        instance_group = []

        instance_group.append(mask)

        instance_label = semantic_label[y1:y2, x1:x2]
        instance_label = scipy.ndimage.zoom(instance_label, [config.INSTANCE_SIZE/instance_label.shape[0], config.INSTANCE_SIZE//instance_label.shape[1]], mode='nearest',order=0)
        instance_group.append(instance_label)

        instance_map = influence_map[y1:y2, x1:x2]
        instance_map = scipy.misc.imresize(instance_map, (config.INSTANCE_SIZE, config.INSTANCE_SIZE), interp='bilinear')
        instance_group.append(instance_map)
        instance_group = np.stack(instance_group)
        instance_piece_groups.append(instance_group)

        # plt.figure()
        # plt.imshow(mask,cmap="gray")
        # plt.show()
        #
        # plt.figure()
        # plt.imshow(instance_label)
        # plt.show()
        #
        # plt.figure()
        # plt.imshow(instance_map,cmap="gray")
        # plt.show()
    instance_piece_groups = np.stack(instance_piece_groups)

    return instance_piece_groups, instance_class_ids, instance_boxes,instance_masks
if __name__=='__main__':
    class_ids_sort=np.array([1,2,3,4])
    masks_sort=np.array([[[0,1,1,0],
                          [1,1,1,0],
                          [0,0,0,0],
                          [0,0,0,0]],
                         [[0,0,0,0],
                          [0,0,0,0],
                          [1,0,0,0],
                          [1,1,0,0]],
                         [[0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,1],
                          [0,0,0,1]],
                         [[0,0,0,0],
                          [0,0,1,0],
                          [0,0,1,0],
                          [0,0,0,0]]])
    distances_sort=compute_mask_distances(class_ids_sort,masks_sort)
    for distance in distances_sort:
        print(distance)
