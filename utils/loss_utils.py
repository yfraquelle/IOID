import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

############################################################
#  Loss Functions
############################################################

def compute_rpn_class_loss(rpn_match, rpn_class_logits):
    """RPN anchor classifier loss.

    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for FG/BG.
    """

    # Squeeze last dim to simplify
    rpn_match = rpn_match.squeeze(2)

    # Get anchor classes. Convert the -1/+1 match to 0/1 values.
    anchor_class = (rpn_match == 1).long()

    # Positive and Negative anchors contribute to the loss,
    # but neutral anchors (match value = 0) don't.
    indices = torch.nonzero(rpn_match != 0)

    # Pick rows that contribute to the loss and filter out the rest.
    rpn_class_logits = rpn_class_logits[indices.data[:,0],indices.data[:,1],:]
    anchor_class = anchor_class[indices.data[:,0],indices.data[:,1]]

    # Crossentropy loss
    loss = F.cross_entropy(rpn_class_logits, anchor_class)

    return loss

def compute_rpn_bbox_loss(target_bbox, rpn_match, rpn_bbox):
    """Return the RPN bounding box loss graph.

    target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
        Uses 0 padding to fill in unsed bbox deltas.
    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
    """

    # Squeeze last dim to simplify
    rpn_match = rpn_match.squeeze(2)

    # Positive anchors contribute to the loss, but negative and
    # neutral anchors (match value of 0 or -1) don't.
    indices = torch.nonzero(rpn_match==1)

    # Pick bbox deltas that contribute to the loss
    rpn_bbox = rpn_bbox[indices.data[:,0],indices.data[:,1]]

    # Trim target bounding box deltas to the same length as rpn_bbox.
    target_bbox = target_bbox[0,:rpn_bbox.size()[0],:]

    # Smooth L1 loss
    loss = F.smooth_l1_loss(rpn_bbox, target_bbox)

    return loss


def compute_mrcnn_class_loss(target_class_ids, pred_class_logits):
    """Loss for the classifier head of Mask RCNN.

    target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero
        padding to fill in the array.
    pred_class_logits: [batch, num_rois, num_classes]
    """

    # Loss
    if target_class_ids.size():
        loss = F.cross_entropy(pred_class_logits,target_class_ids.long())
    else:
        loss = Variable(torch.FloatTensor([0]), requires_grad=False)
        if target_class_ids.is_cuda:
            loss = loss.cuda()

    return loss


def compute_mrcnn_bbox_loss(target_bbox, target_class_ids, pred_bbox):
    """Loss for Mask R-CNN bounding box refinement.

    target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
    target_class_ids: [batch, num_rois]. Integer class IDs.
    pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
    """
    if target_class_ids.size():
        # Only positive ROIs contribute to the loss. And only
        # the right class_id of each ROI. Get their indicies.
        positive_roi_ix = torch.nonzero(target_class_ids > 0)[:, 0]
        positive_roi_class_ids = target_class_ids[positive_roi_ix.data].long()
        indices = torch.stack((positive_roi_ix,positive_roi_class_ids), dim=1)

        # Gather the deltas (predicted and true) that contribute to loss
        target_bbox = target_bbox[indices[:,0].data,:]
        pred_bbox = pred_bbox[indices[:,0].data,indices[:,1].data,:]

        # Smooth L1 loss
        loss = F.smooth_l1_loss(pred_bbox, target_bbox)
    else:
        loss = Variable(torch.FloatTensor([0]), requires_grad=False)
        if target_class_ids.is_cuda:
            loss = loss.cuda()

    return loss


def compute_mrcnn_mask_loss(target_masks, target_class_ids, pred_masks):
    """Mask binary cross-entropy loss for the masks head.

    target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_masks: [batch, proposals, height, width, num_classes] float32 tensor
                with values from 0 to 1.
    """
    if target_class_ids.size():
        # Only positive ROIs contribute to the loss. And only
        # the class specific mask of each ROI.
        positive_ix = torch.nonzero(target_class_ids > 0)[:, 0]
        positive_class_ids = target_class_ids[positive_ix.data].long()
        indices = torch.stack((positive_ix, positive_class_ids), dim=1)

        # Gather the masks (predicted and true) that contribute to loss
        y_true = target_masks[indices[:,0].data,:,:]
        y_pred = pred_masks[indices[:,0].data,indices[:,1].data,:,:]

        # Binary cross entropy
        loss = F.binary_cross_entropy(y_pred, y_true)
    else:
        loss = Variable(torch.FloatTensor([0]), requires_grad=False)
        if target_class_ids.is_cuda:
            loss = loss.cuda()

    return loss
def compute_saliency_loss(saliency_pred,saliency_tar,loss_ratio=[0.5,0.5, 0.5, 0.8, 1]):
    saliency_tar = saliency_tar.unsqueeze(0)
    loss = 0
    for i in range(5):
        # print(saliency_pred[4-i].shape)
        # print(saliency_tar.shape)
        loss += F.binary_cross_entropy(saliency_pred[4 - i], saliency_tar) * loss_ratio[4 - i]
        # print(float(loss))
        if i==1:
            saliency_tar = F.max_pool2d(saliency_tar, 2, 2)
    return loss

def compute_saliency_loss_finetune(saliency_pred,saliency_tar,loss_ratio=[0.5,0.5, 0.5, 0.8,0.8, 1]):
    saliency_tar = saliency_tar.unsqueeze(0)
    loss = 0
    for i in range(6):
        # print(saliency_pred[5-i].shape)
        # print(saliency_tar.shape)
        loss += F.binary_cross_entropy(saliency_pred[5 - i], saliency_tar) * loss_ratio[5 - i]
        # print(float(loss))
        if saliency_tar.size()[2] > 32:
            saliency_tar = F.max_pool2d(saliency_tar, 2, 2)
    return loss

def compute_semantic_loss(semantic_segment,semantic_target):
    semantic_thing=Variable(torch.zeros((semantic_segment.shape[0],81,semantic_segment.shape[2],semantic_segment.shape[3]))).cuda()
    semantic_segment=semantic_segment[:,81:,:,:]
    semantic_segment=torch.cat([semantic_thing,semantic_segment],dim=1)
    loss = nn.CrossEntropyLoss(weight=None, ignore_index=0, size_average=True)(semantic_segment,semantic_target)
    return loss

def compute_psp_semantic_loss(semantic_segment,semantic_classes,semantic_target,class_weights=None,alpha=1.0):
    seg_criterion = nn.CrossEntropyLoss(weight=None, ignore_index=0, size_average=True)
    #seg_criterion = nn.NLLLoss2d(weight=class_weights)
    #cls_criterion = nn.BCEWithLogitsLoss(weight=class_weights)
    #y_cls=[]
    #exist=np.unique(semantic_target.data.cpu().numpy())
    #for class_id in range(0,semantic_classes.shape[1]):
    #    if class_id in exist:
    #        y_cls.append(1.5)
    #    else:
    #        y_cls.append(0)
    #y_cls=Variable(torch.from_numpy(np.array(y_cls))).cuda().float().unsqueeze(0)
    seg_loss=seg_criterion(semantic_segment, semantic_target)
    #cls_loss=cls_criterion(semantic_classes.double(), y_cls.double())
    return seg_loss#+alpha*cls_loss.float()

def compute_interest_loss(interest_label,gt_interest_label):
    return nn.MSELoss()(interest_label,gt_interest_label)

def compute_losses_CIN(rpn_match, rpn_bbox, rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask,target_semantic_label,semantic_label, interest_target, interest_preds):

    rpn_class_loss = compute_rpn_class_loss(rpn_match, rpn_class_logits)
    rpn_bbox_loss = compute_rpn_bbox_loss(rpn_bbox, rpn_match, rpn_pred_bbox)
    mrcnn_class_loss = compute_mrcnn_class_loss(target_class_ids, mrcnn_class_logits)
    mrcnn_bbox_loss = compute_mrcnn_bbox_loss(target_deltas, target_class_ids, mrcnn_bbox)
    mrcnn_mask_loss = compute_mrcnn_mask_loss(target_mask, target_class_ids, mrcnn_mask)
    semantic_loss = compute_semantic_loss(semantic_label,target_semantic_label)
    saliency_loss = compute_saliency_loss(interest_preds, interest_target)

    return [rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss,semantic_loss,saliency_loss]
def compute_losses_CIN_newsem(rpn_match, rpn_bbox, rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask,target_semantic_label,semantic_label,semantic_cls, interest_target, interest_preds):

    rpn_class_loss = compute_rpn_class_loss(rpn_match, rpn_class_logits)
    rpn_bbox_loss = compute_rpn_bbox_loss(rpn_bbox, rpn_match, rpn_pred_bbox)
    mrcnn_class_loss = compute_mrcnn_class_loss(target_class_ids, mrcnn_class_logits)
    mrcnn_bbox_loss = compute_mrcnn_bbox_loss(target_deltas, target_class_ids, mrcnn_bbox)
    mrcnn_mask_loss = compute_mrcnn_mask_loss(target_mask, target_class_ids, mrcnn_mask)
    semantic_loss = compute_psp_semantic_loss(semantic_label,semantic_cls,target_semantic_label)
    saliency_loss = compute_saliency_loss(interest_preds, interest_target)

    return [rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss,semantic_loss,saliency_loss]

def compute_losses_PFPN(rpn_match, rpn_bbox, rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask,target_semantic_label,semantic_label):

    rpn_class_loss = compute_rpn_class_loss(rpn_match, rpn_class_logits)
    rpn_bbox_loss = compute_rpn_bbox_loss(rpn_bbox, rpn_match, rpn_pred_bbox)
    mrcnn_class_loss = compute_mrcnn_class_loss(target_class_ids, mrcnn_class_logits)
    mrcnn_bbox_loss = compute_mrcnn_bbox_loss(target_deltas, target_class_ids, mrcnn_bbox)
    mrcnn_mask_loss = compute_mrcnn_mask_loss(target_mask, target_class_ids, mrcnn_mask)
    semantic_loss = compute_semantic_loss(semantic_label,target_semantic_label)

    return [rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss,semantic_loss]
def compute_losses_PFPN_newsem(rpn_match, rpn_bbox, rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask,target_semantic_label,semantic_label,semantic_cls):

    rpn_class_loss = compute_rpn_class_loss(rpn_match, rpn_class_logits)
    rpn_bbox_loss = compute_rpn_bbox_loss(rpn_bbox, rpn_match, rpn_pred_bbox)
    mrcnn_class_loss = compute_mrcnn_class_loss(target_class_ids, mrcnn_class_logits)
    mrcnn_bbox_loss = compute_mrcnn_bbox_loss(target_deltas, target_class_ids, mrcnn_bbox)
    mrcnn_mask_loss = compute_mrcnn_mask_loss(target_mask, target_class_ids, mrcnn_mask)
    semantic_loss = compute_psp_semantic_loss(semantic_label,semantic_cls,target_semantic_label)

    return [rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss,semantic_loss]
