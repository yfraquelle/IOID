import os
import re
import datetime
import json

import scipy
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict

import torch
from torch import FloatTensor
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader as TorchDataLoader
from torch.utils.data.dataloader import default_collate

from utils import utils, visualize


from backbone.ResNet import ResNet
from instance_extraction.FPN import FPN
from instance_extraction.RPN import RPN
from instance_extraction.FPN_heads import Classifier,Mask,Semantic
from interest_estimation.Saliency import SaNet
from instance_extraction.Proposal import proposal_layer
from instance_extraction.Detection import detection_layer, generate_stuff
from instance_extraction.DetectionTarget import detection_target_layer
from utils.formatting_utils import mold_image, compose_image_meta
from DatasetLib import Dataset
from utils.log_utils import log, printProgressBar
from utils.loss_utils import compute_losses_CIN, compute_losses_PFPN, compute_saliency_loss, compute_interest_loss, compute_semantic_loss
from ioi_selection.CIEDN import CIEDN
from utils.Selection import extract_piece_group, map_pred_with_gt_mask, resize_influence_map,resize_semantic_label,filter_stuff_masks,filter_thing_masks
from utils.utils import IdGenerator
from compute_metric import maxminnorm

############################################################
#  MaskRCNN
############################################################

class CIN(nn.Module):
    """Encapsulates the Mask RCNN model functionality.
    """

    def __init__(self, config, model_dir):
        """
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        super(CIN, self).__init__()
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self.build(config=config)
        self.initialize_weights()
        self.loss_history = []
        self.val_loss_history = []

        self.class_dict = json.load(open("data/class_dict.json",'r'))
        self.category_dict={}
        for class_id in self.class_dict:
            self.category_dict[class_id]=self.class_dict[class_id]

        self.id_generator = IdGenerator(json.load(open("data/class_dict.json",'r')))

    def build(self, config):
        """Build Mask R-CNN architecture.
        """

        # Image size must be dividable by 2 multiple times
        h, w = config.IMAGE_SHAPE[:2]
        if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")

        # Build the shared convolutional layers.
        # Bottom-up Layers
        # Returns a list of the last layers of each stage, 5 in total.
        # Don't create the thead (stage 5), so we pick the 4th item in the list.
        self.resnet = ResNet("resnet101", stage5=True)

        # Top-down Layers
        # TODO: add assert to varify feature map sizes match what's in config
        self.fpn = FPN(out_channels=256)

        # Generate Anchors
        self.anchors = Variable(torch.from_numpy(utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                                                                config.RPN_ANCHOR_RATIOS,
                                                                                config.BACKBONE_SHAPES,
                                                                                config.BACKBONE_STRIDES,
                                                                                config.RPN_ANCHOR_STRIDE)).float(), requires_grad=False)
        if self.config.GPU_COUNT:
            self.anchors = self.anchors.cuda()


        # Salient
        self.saliency=SaNet()

        # RPN
        self.rpn = RPN(len(config.RPN_ANCHOR_RATIOS), config.RPN_ANCHOR_STRIDE, 256)

        # FPN Classifier
        self.classifier = Classifier(256, config.POOL_SIZE, config.IMAGE_SHAPE, config.THING_NUM_CLASSES)

        # FPN Mask
        self.mask = Mask(256, config.MASK_POOL_SIZE, config.IMAGE_SHAPE, config.THING_NUM_CLASSES)

        # FPN Semantic
        self.semantic = Semantic(self.config.NUM_CLASSES)

        # OOI Selection
        self.ciedn = CIEDN()

        # Fix batch norm layers
        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters(): p.requires_grad = False

        self.apply(set_bn_fix)

    def initialize_weights(self):
        """Initialize model weights.
        """

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def set_trainable(self, layer_regex, model=None, indent=0, verbose=1):
        """Sets model layers as trainable if their names match
        the given regular expression.
        """

        for param in self.named_parameters():
            layer_name = param[0]
            trainable = bool(re.fullmatch(layer_regex, layer_name))
            if not trainable:
                param[1].requires_grad = False
            else:
                param[1].requires_grad = True

    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.

        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """

        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:
            # /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5
            regex = r".*/\w+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/CIN\_\w+(\d{4})\.pth"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                self.epoch = int(m.group(6))

        # Directory for training logs
        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
            self.config.NAME.lower(), now))
        # self.log_dir = os.path.join(self.model_dir, 'ooi20200108T1851')

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "CIN_{}_*epoch*.pth".format(
            self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{:04d}")

    def find_last(self):
        """Finds the last checkpoint file of the last trained model in the
        model directory.
        Returns:
            log_dir: The directory where events and weights are saved
            checkpoint_path: the path to the last checkpoint file
        """
        # Get directory names. Each directory corresponds to a model
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.config.NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            return None, None
        # Pick last directory
        dir_name = os.path.join(self.model_dir, dir_names[-1])
        # Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("CIN"), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            return dir_name, None
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return dir_name, checkpoint

    def load_weights(self, filepath):
        """Modified version of the correspoding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exlude: list of layer names to excluce
        """
        if os.path.exists(filepath):
            state_dict = torch.load(filepath)
            self.load_state_dict(state_dict, strict=False)
        else:
            print("Weight file not found ...")

        # Update the log directory
        self.set_log_dir(filepath)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def load_part_weights(self,filepath,mode="instance"):
        if os.path.exists(filepath):
            state_dict = torch.load(filepath)
            if mode == "p_interest":
                state_dict_to_load = dict()
                for name in state_dict:
                    if name.split(".")[0] == "saliency":
                        state_dict_to_load[name] = state_dict[name]
                self.load_state_dict(state_dict_to_load, strict=False)
            elif mode == "semantic":
                state_dict_to_load = dict()
                for name in state_dict:
                    if name.split(".")[0] == "semantic":
                        state_dict_to_load[name] = state_dict[name]
                self.load_state_dict(state_dict_to_load, strict=False)
            elif mode == "insttr":
                state_dict_to_load = dict()
                for name in state_dict:
                    if name.split(".")[0] == ["resnet", "fpn", "rpn", "classifier", "mask", "semantic", "saliency"]:
                        state_dict_to_load[name] = state_dict[name]
                self.load_state_dict(state_dict_to_load, strict=False)
            elif mode == "selection":
                state_dict_to_load = dict()
                for name in state_dict:
                    if name.split(".")[0] == "cirnn":
                        new_name = name.replace('cirnn', 'ciedn')
                        state_dict_to_load[new_name] = state_dict[name]
                self.load_state_dict(state_dict_to_load, strict=False)
            elif mode == "all":
                self.load_state_dict(state_dict, strict=False)
            else:
                self.load_state_dict(state_dict, strict=False)
        else:
            print("Weight file not found ...")
            exit()
        self.set_log_dir()
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        print(self.log_dir)

    def load_from_maskrcnn(self):
        state_dict = torch.load("models/mask_rcnn_coco.pth")
        resnet_dict=dict()
        other_dict=dict()
        for name in state_dict:
            if name[:5]=="fpn.C":
                resnet_dict["resnet.C"+name[5:]]=state_dict[name]
            else:
                other_dict[name]=state_dict[name]
        self.load_state_dict(resnet_dict, strict=False)
        self.load_state_dict(other_dict, strict=False)
        self.set_log_dir()
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def train_model(self,train_dataset, val_dataset, learning_rate, epochs, layers):
        self.training_layers=layers
        layer_regex = {
            "semantic": r"(semantic.*)",
            "p_interest": r"(saliency.*)",
            "selection": r"(ciedn.*)"
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        def my_collate_fn(batch):
            batch = list(filter(lambda x: x is not None, batch))
            if len(batch) == 0:
                print("No valid data!!!")
                batch = [[torch.from_numpy(np.zeros([1,1]))]]
            return default_collate(batch)

        train_set = Dataset(train_dataset, self.config)
        train_generator = TorchDataLoader(train_set, collate_fn=my_collate_fn, batch_size=1, shuffle=True, num_workers=0)

        val_set = Dataset(val_dataset, self.config)
        val_generator = TorchDataLoader(val_set, collate_fn=my_collate_fn, batch_size=1, shuffle=True, num_workers=0)

        self.set_trainable(layers)

        optimizers=[]
        if self.training_layers == "semantic":
            trainables_wo_bn = [param for name, param in self.named_parameters() if param.requires_grad and not 'bn' in name]
            trainables_only_bn = [param for name, param in self.named_parameters() if param.requires_grad and 'bn' in name]

            optimizer = optim.SGD([
                {'params': trainables_wo_bn, 'weight_decay': self.config.WEIGHT_DECAY},
                {'params': trainables_only_bn}
            ], lr=learning_rate, momentum=self.config.LEARNING_MOMENTUM)
            optimizers.append(optimizer)
        elif self.training_layers == 'p_interest':
            optimizer_encoder = torch.optim.SGD([{'params':[param for name, param in self.saliency.conv1.named_parameters()]},
                                                 {'params':[param for name, param in self.saliency.conv2.named_parameters()]},
                                                 {'params':[param for name, param in self.saliency.conv3.named_parameters()]}], lr=learning_rate, momentum=0.9, weight_decay=0.0005)
            trainables_decoder_no_bn = [param for name, param in self.saliency.decoder.named_parameters() if param.requires_grad and not 'bn' in name]
            trainables_decoder_only_bn = [param for name, param in self.saliency.decoder.named_parameters() if param.requires_grad and 'bn' in name]
            optimizer_decoder = torch.optim.SGD([{'params': trainables_decoder_no_bn,},{'params': trainables_decoder_only_bn}], lr=learning_rate * 10, momentum=0.9,weight_decay=0.0005)
            optimizers.append(optimizer_encoder)
            optimizers.append(optimizer_decoder)
        elif self.training_layers == "selection":
            optimizer= torch.optim.Adam(self.ciedn.parameters(), lr=learning_rate)
            optimizers.append(optimizer)
        else:
            print("training mode not exists")
            exit()

        for epoch in range(self.epoch + 1, epochs + 1):
            log("Epoch {}/{}.".format(epoch, epochs))

            if self.training_layers == "semantic":
                loss, loss_semantic = self.train_epoch(train_generator, optimizers, self.config.STEPS_PER_EPOCH)
                val_loss, val_loss_semantic = self.valid_epoch(val_generator, self.config.VALIDATION_STEPS)
                
                self.loss_history.append([loss, loss_semantic])
                self.val_loss_history.append([val_loss, val_loss_semantic])
                
                visualize.plot_loss("semantic_loss", 1, self.loss_history, self.val_loss_history, save=True, log_dir=self.log_dir)
            elif self.training_layers=='p_interest':
                loss,loss_influence = self.train_epoch(train_generator, optimizers, self.config.STEPS_PER_EPOCH)
                val_loss, val_loss_influence = self.valid_epoch(val_generator, self.config.VALIDATION_STEPS)

                self.loss_history.append([loss, loss_influence])
                self.val_loss_history.append([val_loss,val_loss_influence])

                visualize.plot_loss("p_interest", 1, self.loss_history, self.val_loss_history, save=True, log_dir=self.log_dir)
            elif self.training_layers=='selection':
                loss, loss_interest = self.train_epoch(train_generator, optimizers, self.config.STEPS_PER_EPOCH)
                val_loss, val_loss_interest = self.valid_epoch(val_generator, self.config.VALIDATION_STEPS)

                self.loss_history.append([loss, loss_interest])
                self.val_loss_history.append([val_loss, val_loss_interest])

                visualize.plot_loss("loss_interest", 1, self.loss_history, self.val_loss_history, save=True, log_dir=self.log_dir)
            else:
                print("training mode not exists")
                exit()

            # Save model
            torch.save(self.state_dict(), self.checkpoint_path.format(epoch))

        self.epoch = epochs

    def train_epoch(self, datagenerator, optimizers, steps):
        batch_count = 0
        loss_sum = 0
        loss_rpn_class_sum = 0
        loss_rpn_bbox_sum = 0
        loss_mrcnn_class_sum = 0
        loss_mrcnn_bbox_sum = 0
        loss_mrcnn_mask_sum = 0
        loss_semantic_sum = 0
        loss_influence_sum = 0
        loss_interest_sum = 0
        step = 0

        for optimizer in optimizers:
            optimizer.zero_grad()
        for inputs in datagenerator:
            if len(inputs)!=17:
                continue
            try:
                batch_count += 1

                images = inputs[0]
                image_metas = inputs[1].int().data.numpy()
                rpn_match = inputs[2]
                rpn_bbox = inputs[3]
                gt_class_ids = inputs[4]
                gt_boxes = inputs[5]
                gt_masks = inputs[6]
                gt_stuff_class_ids = inputs[7]
                gt_stuff_boxes = inputs[8]
                gt_stuff_masks = inputs[9]
                gt_semantic_label = inputs[10]
                gt_influence_map = inputs[11]
                gt_interest_class_ids = inputs[12]
                gt_interest_boxes = inputs[13]
                gt_interest_masks = inputs[14]
                gt_segmentation = inputs[15]
                gt_image_instances = inputs[16]

                images = Variable(images)
                rpn_match = Variable(rpn_match)
                rpn_bbox = Variable(rpn_bbox)
                gt_class_ids = Variable(gt_class_ids)
                gt_boxes = Variable(gt_boxes)
                gt_masks = Variable(gt_masks)
                gt_stuff_class_ids = Variable(gt_stuff_class_ids)
                gt_stuff_boxes = Variable(gt_stuff_boxes)
                gt_stuff_masks = Variable(gt_stuff_masks)
                gt_semantic_label = Variable(gt_semantic_label)
                gt_influence_map = Variable(gt_influence_map)
                # gt_interest_class_ids = Variable(gt_interest_class_ids)
                gt_interest_boxes = Variable(gt_interest_boxes)
                gt_interest_masks = Variable(gt_interest_masks)
                gt_segmentation = Variable(gt_segmentation)

                # To GPU
                if self.config.GPU_COUNT:
                    images = images.cuda()
                    rpn_match = rpn_match.cuda()
                    rpn_bbox = rpn_bbox.cuda()
                    gt_class_ids = gt_class_ids.cuda()
                    gt_boxes = gt_boxes.cuda()
                    gt_masks = gt_masks.cuda()
                    gt_stuff_class_ids = gt_stuff_class_ids.cuda()
                    gt_stuff_boxes = gt_stuff_boxes.cuda()
                    gt_stuff_masks = gt_stuff_masks.cuda()
                    gt_semantic_label = gt_semantic_label.cuda()
                    gt_influence_map = gt_influence_map.cuda()
                    # gt_interest_class_ids = gt_interest_class_ids.cuda()
                    gt_interest_boxes = gt_interest_boxes.cuda()
                    gt_interest_masks = gt_interest_masks.cuda()
                    gt_segmentation = gt_segmentation.cuda()

                if self.training_layers == "semantic":
                    # Run object detection
                    predict_input = [images, image_metas]
                    semantic_label = self.predict_front(predict_input, mode='training', limit="semantic")
                    semantic_loss = compute_semantic_loss(semantic_label, gt_semantic_label)
                    loss = semantic_loss

                    # Backpropagation
                    loss.backward()
                    torch.nn.utils.clip_grad_norm(self.parameters(), 5.0)
                    if (batch_count % self.config.BATCH_SIZE) == 0:
                        for optimizer in optimizers:
                            optimizer.step()
                            optimizer.zero_grad()
                        batch_count = 0

                    if self.config.GPU_COUNT:
                        printProgressBar(step + 1, steps, prefix="\t{}/{}".format(step + 1, steps),
                                         suffix="Complete - loss: {:.5f} - semantic_loss: {:.5f} ".format(
                                             loss.data.cpu()[0], semantic_loss.data.cpu()[0]), length=10)
                        loss_sum += loss.data.cpu()[0] / steps
                        loss_semantic_sum += semantic_loss.data.cpu()[0] / steps
                    else:
                        printProgressBar(step + 1, steps, prefix="\t{}/{}".format(step + 1, steps),
                                         suffix="Complete - loss: {:.5f} - semantic_loss: {:.5f} ".format(
                                             loss.data[0], semantic_loss.data[0]), length=10)
                        loss_sum += loss.data[0] / steps
                        loss_semantic_sum += semantic_loss.data[0] / steps

                    # Break after 'steps' steps
                    if step == steps - 1:
                        return loss_sum, loss_semantic_sum
                    else:
                        step += 1
                elif self.training_layers=='p_interest':
                    # Run object detection
                    predict_input = [images, image_metas]
                    influence_preds = self.predict_front(predict_input, mode='training', limit="p_interest")
                    influence_loss = compute_saliency_loss(influence_preds,gt_influence_map)
                    loss=influence_loss

                    # Backpropagation
                    loss.backward()
                    torch.nn.utils.clip_grad_norm(self.parameters(), 5.0)
                    if (batch_count % self.config.BATCH_SIZE) == 0:
                        for optimizer in optimizers:
                            optimizer.step()
                            optimizer.zero_grad()
                        batch_count = 0

                    if self.config.GPU_COUNT:
                        printProgressBar(step + 1, steps, prefix="\t{}/{}".format(step + 1, steps),suffix="Complete - influence_loss: {:.5f}".format(influence_loss.data.cpu()[0]), length=10)
                        loss_sum += loss.data.cpu()[0] / steps
                        loss_influence_sum += influence_loss.data.cpu()[0] / steps
                    else:
                        printProgressBar(step + 1, steps, prefix="\t{}/{}".format(step + 1, steps),suffix="Complete - influence_loss: {:.5f}".format(influence_loss.data[0]), length=10)
                        loss_sum += loss.data[0] / steps
                        loss_influence_sum += influence_loss.data[0] / steps

                    # Break after 'steps' steps
                    if step == steps - 1:
                        return loss_sum, loss_influence_sum
                    else:
                        step += 1
                elif self.training_layers=='selection':
                    loss_func = nn.MSELoss()
                    detection_result = self.predict_front([images, image_metas],mode='inference', limit="insttr")
                    predict_input = [images, image_metas, detection_result, gt_segmentation, gt_image_instances]
                    predictions, pair_labels, labels = self.predict_front(predict_input, mode="training", limit="selection")

                    interest_loss = loss_func(predictions, pair_labels)
                    loss = interest_loss

                    # Backpropagation
                    loss.backward()
                    torch.nn.utils.clip_grad_norm(self.parameters(), 5.0)
                    if (batch_count % self.config.BATCH_SIZE) == 0:
                        for optimizer in optimizers:
                            optimizer.step()
                            optimizer.zero_grad()
                        batch_count = 0

                    if self.config.GPU_COUNT:
                        printProgressBar(step + 1, steps, prefix="\t{}/{}".format(step + 1, steps),
                                         suffix="Complete - interest_loss: {:.5f}".format(interest_loss.data.cpu()[0]),
                                         length=10)

                        loss_sum += loss.data.cpu()[0] / steps
                        loss_interest_sum += interest_loss.data.cpu()[0] / steps
                    else:
                        printProgressBar(step + 1, steps, prefix="\t{}/{}".format(step + 1, steps),
                                         suffix="Complete - interest_loss: {:.5f}".format(interest_loss.data[0]),
                                         length=10)

                        loss_sum += loss.data[0] / steps
                        loss_interest_sum += interest_loss.data[0] / steps

                    # Break after 'steps' steps
                    if step == steps - 1:
                        return loss_sum, loss_interest_sum
                    else:
                        step += 1
                else:
                    print("training mode not exists")
                    exit()
            except Exception as e:
                print("Error - "+str(step))
                print(e)

    def valid_epoch(self, datagenerator, steps):
        step = 0
        loss_sum = 0
        loss_rpn_class_sum = 0
        loss_rpn_bbox_sum = 0
        loss_mrcnn_class_sum = 0
        loss_mrcnn_bbox_sum = 0
        loss_mrcnn_mask_sum = 0
        loss_semantic_sum = 0
        loss_influence_sum = 0
        loss_interest_sum = 0

        for inputs in datagenerator:
            if len(inputs)!=17:
                print(len(inputs))
                continue
            try:
                images = inputs[0]
                image_metas = inputs[1].int().data.numpy()
                rpn_match = inputs[2]
                rpn_bbox = inputs[3]
                gt_class_ids = inputs[4]
                gt_boxes = inputs[5]
                gt_masks = inputs[6]
                gt_stuff_class_ids = inputs[7]
                gt_stuff_boxes = inputs[8]
                gt_stuff_masks = inputs[9]
                gt_semantic_label = inputs[10]
                gt_influence_map = inputs[11]
                gt_interest_class_ids = inputs[12]
                gt_interest_boxes = inputs[13]
                gt_interest_masks = inputs[14]
                gt_segmentation = inputs[15]
                gt_image_instances = inputs[16]

                # Wrap in variables
                images = Variable(images, volatile=True)
                rpn_match = Variable(rpn_match, volatile=True)
                rpn_bbox = Variable(rpn_bbox, volatile=True)
                gt_class_ids = Variable(gt_class_ids, volatile=True)
                gt_boxes = Variable(gt_boxes, volatile=True)
                gt_masks = Variable(gt_masks, volatile=True)
                gt_stuff_class_ids = Variable(gt_stuff_class_ids, volatile=True)
                gt_stuff_boxes = Variable(gt_stuff_boxes, volatile=True)
                gt_stuff_masks = Variable(gt_stuff_masks, volatile=True)
                gt_semantic_label = Variable(gt_semantic_label, volatile=True)
                gt_influence_map = Variable(gt_influence_map, volatile=True)
                gt_interest_class_ids = Variable(gt_interest_class_ids, volatile=True)
                gt_interest_boxes = Variable(gt_interest_boxes, volatile=True)
                gt_interest_masks = Variable(gt_interest_masks, volatile=True)
                gt_segmentation = Variable(gt_segmentation, volatile=True)

                # To GPU
                if self.config.GPU_COUNT:
                    images = images.cuda()
                    rpn_match = rpn_match.cuda()
                    rpn_bbox = rpn_bbox.cuda()
                    gt_class_ids = gt_class_ids.cuda()
                    gt_boxes = gt_boxes.cuda()
                    gt_masks = gt_masks.cuda()
                    gt_stuff_class_ids = gt_stuff_class_ids.cuda()
                    gt_stuff_boxes = gt_stuff_boxes.cuda()
                    gt_stuff_masks = gt_stuff_masks.cuda()
                    gt_semantic_label = gt_semantic_label.cuda()
                    gt_influence_map = gt_influence_map.cuda()
                    gt_interest_class_ids = gt_interest_class_ids.cuda()
                    gt_interest_boxes = gt_interest_boxes.cuda()
                    gt_interest_masks = gt_interest_masks.cuda()
                    gt_segmentation = gt_segmentation.cuda()

                if self.training_layers == "semantic":
                    # Run object detection
                    predict_input = [images, image_metas]
                    semantic_label = self.predict_front(predict_input, mode='training', limit="semantic")
                    semantic_loss = compute_semantic_loss(semantic_label, gt_semantic_label)
                    loss = semantic_loss

                    if self.config.GPU_COUNT:
                        printProgressBar(step + 1, steps, prefix="\t{}/{}".format(step + 1, steps),
                                         suffix="Complete - loss: {:.5f} - semantic_loss: {:.5f} ".format(
                                             loss.data.cpu()[0], semantic_loss.data.cpu()[0]), length=10)

                        loss_sum += loss.data.cpu()[0] / steps
                        loss_semantic_sum += semantic_loss.data.cpu()[0] / steps
                    else:
                        printProgressBar(step + 1, steps, prefix="\t{}/{}".format(step + 1, steps),
                                         suffix="Complete - loss: {:.5f} - semantic_loss: {:.5f} ".format(
                                             loss.data[0], semantic_loss.data[0]), length=10)

                        loss_sum += loss.data[0] / steps
                        loss_semantic_sum += semantic_loss.data[0] / steps

                    # Break after 'steps' steps
                    if step == steps - 1:
                        return loss_sum, loss_semantic_sum
                    else:
                        step += 1
                elif self.training_layers == 'p_interest':
                    # Run object detection
                    predict_input = [images, image_metas]
                    influence_preds = self.predict_front(predict_input, mode='training', limit="p_interest")

                    influence_loss = compute_saliency_loss(influence_preds, gt_influence_map)
                    loss = influence_loss

                    if self.config.GPU_COUNT:
                        printProgressBar(step + 1, steps, prefix="\t{}/{}".format(step + 1, steps),
                                         suffix="Complete - influence_loss: {:.5f}".format(influence_loss.data.cpu()[0]),
                                         length=10)

                        loss_sum += loss.data.cpu()[0] / steps
                        loss_influence_sum += influence_loss.data.cpu()[0] / steps
                    else:
                        printProgressBar(step + 1, steps, prefix="\t{}/{}".format(step + 1, steps),
                                         suffix="Complete - influence_loss: {:.5f}".format(
                                             influence_loss.data[0]),
                                         length=10)

                        loss_sum += loss.data[0] / steps
                        loss_influence_sum += influence_loss.data[0] / steps

                    # Break after 'steps' steps
                    if step == steps - 1:
                        return loss_sum, loss_influence_sum
                    else:
                        step += 1
                elif self.training_layers=='selection':
                    # Run object detection
                    detection_result=self.predict_front([images, image_metas], mode="inference", limit="insttr")
                    predictions, pair_labels, labels = self.predict_front([images, image_metas, detection_result, gt_segmentation, gt_image_instances], mode="training", limit="selection")
                    loss_func = nn.MSELoss()
                    interest_loss = loss_func(predictions, pair_labels)
                    loss = interest_loss

                    if self.config.GPU_COUNT:
                        printProgressBar(step + 1, steps, prefix="\t{}/{}".format(step + 1, steps),
                                         suffix="Complete - interest_loss: {:.5f}".format(interest_loss.data.cpu()[0]),
                                         length=10)

                        loss_sum += loss.data.cpu()[0] / steps
                        loss_interest_sum += interest_loss.data.cpu()[0] / steps
                    else:
                        printProgressBar(step + 1, steps, prefix="\t{}/{}".format(step + 1, steps),
                                         suffix="Complete - interest_loss: {:.5f}".format(interest_loss.data[0]),
                                         length=10)

                        loss_sum += loss.data[0] / steps
                        loss_interest_sum += interest_loss.data[0] / steps
                    # Break after 'steps' steps
                    if step == steps - 1:
                        return loss_sum, loss_interest_sum
                    else:
                        step += 1
                else:
                    print("training mode not exists")
                    exit()
            except Exception as e:
                print("Error - "+str(step))
                print(e)

    def detect(self, images, limit="instance"):
        # Mold inputs to format expected by the neural network
        print(images[0].shape)
        molded_images, image_metas = self.mold_inputs(images)

        image_metas=image_metas.int().data.numpy()

        if self.config.GPU_COUNT:
            molded_images=Variable(molded_images, volatile=True).cuda()
        else:
            molded_images = Variable(molded_images, volatile=True)

        image_id, image_shape, window = image_metas[0][0], image_metas[0][1:4], image_metas[0][4:8]
        top_pad, left_pad, top_pad_h, left_pad_w = window[0], window[1], window[2], window[3]

        if limit in ["instance","p_interest","insttr"]:
            result = self.predict_front([molded_images, image_metas], mode='inference', limit=limit)
            return [result]
        elif limit=="selection":
            result = self.predict_front([molded_images, image_metas], mode='inference', limit="insttr")  # [x,5],[x,28,28,81]
            predictions, segments_info, panoptic_result, instance_list = self.predict_front([molded_images, image_metas, result], mode='inference', limit=limit)
            num = len(segments_info)  # the num of the instances
            prediction_list = []
            for i in range(0, num):
                avg = np.sum(predictions[i * num:(i + 1) * num]) / num
                prediction_list.append(avg)
            idx = 0
            CIRNN_pred_dict = {}
            ioid_result=np.zeros_like(panoptic_result)
            panoptic_result_instance_id_map=utils.rgb2id(panoptic_result)
            for segment_info_id in segments_info:
                if prediction_list[idx] > self.config.SELECTION_THRESHOLD:
                    CIRNN_pred_dict[segment_info_id] = segments_info[segment_info_id]
                    ioid_result[panoptic_result_instance_id_map==int(segment_info_id)]=utils.id2rgb(int(segment_info_id))
                idx += 1
            return CIRNN_pred_dict, ioid_result, segments_info,panoptic_result_instance_id_map, prediction_list, instance_list

    def predict_front(self, input, mode, limit=""): #image_metas is a int numpy array
        molded_images = input[0]
        image_metas = input[1]
        image_id = image_metas[0][0]

        if mode == 'training':
            self.train()
        elif mode == 'inference':
            self.eval()

            # Set batchnorm always in eval mode during training
            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            self.apply(set_bn_eval)


        if limit == 'selection':
            if mode=="training":
                detection_result = input[2]
                gt_segmentation = input[3]
                image_info = input[4]
                gt_instance_dict = image_info['instances']

                semantic_labels, panoptic_result, segments_info = self.predict_segment(detection_result, image_metas)
                ioi_image_dict = self.map_instance_to_gt(gt_instance_dict, segments_info,gt_segmentation, panoptic_result, image_metas)

                ioi_segments_info = ioi_image_dict['segments_info']

                image_shape = image_metas[0][1:4].astype('int32')  # 420,640,3
                instance_groups, boxes, class_ids, labels, pair_label, instance_list = self.construct_dataset(semantic_labels,
                                                                                               detection_result['influence_map'],
                                                                                               panoptic_result,
                                                                                               ioi_segments_info,
                                                                                               image_shape,
                                                                                               mode)
                instance_groups = Variable(torch.unsqueeze(torch.from_numpy(instance_groups), 0)).float()
                boxes = Variable(torch.unsqueeze(torch.from_numpy(boxes), 0)).float()
                class_ids = Variable(torch.unsqueeze(torch.from_numpy(class_ids), 0)).float()
                gt = Variable(torch.unsqueeze(torch.from_numpy(labels), 0)).float()
                labels = Variable(torch.unsqueeze(torch.from_numpy(labels), 0)).float()
                pair_label = Variable(torch.from_numpy(pair_label)).float().squeeze(0)
                if self.config.GPU_COUNT:
                    instance_groups=instance_groups.cuda()
                    boxes=boxes.cuda()
                    class_ids=class_ids.cuda()
                    gt=gt.cuda()
                    labels=labels.cuda()
                    pair_label=pair_label.cuda()

                predictions = self.ciedn(instance_groups).squeeze(1)
                return predictions, pair_label, labels
            elif mode=="inference":
                detection_result = input[2]

                semantic_labels, panoptic_result, segments_info = self.predict_segment(detection_result, image_metas)

                image_shape = image_metas[0][1:4].astype('int32')  # 420,640,3
                instance_groups, boxes, class_ids, labels, pair_label, instance_list = self.construct_dataset(semantic_labels,
                                                         detection_result['influence_map'],
                                                         panoptic_result,
                                                         segments_info,
                                                         image_shape,
                                                         mode)

                if self.config.GPU_COUNT:
                    instance_groups = Variable(FloatTensor(instance_groups)).float().cuda().unsqueeze(0)  # 1,19,2,56,56
                    predictions = self.ciedn(instance_groups).squeeze(1).data.cpu().numpy()
                else:
                    instance_groups = Variable(FloatTensor(instance_groups)).float().unsqueeze(0)  # 1,19,2,56,56
                    predictions = self.ciedn(instance_groups).squeeze(1).data.numpy()
                return predictions, segments_info, panoptic_result, instance_list
        else: # training - semantic/p_interest ; inference - instance/p_interest/insttr
            [c1_out, c2_out, c3_out, c4_out, c5_out] = self.resnet(molded_images)

            if limit == "p_interest":
                influence_preds = self.saliency(c1_out, c2_out, c3_out, c4_out, c5_out)  # (1,4,128,128)
                if mode == "training":
                    return influence_preds
                elif mode == "inference":
                    influence_map=self.unmold_p_interest(influence_preds[4], image_metas)
                    return {"influence_map":influence_map}
            else: # training - semantic ; inference - instance/insttr
                [p2_out, p3_out, p4_out, p5_out, p6_out] = self.fpn(c1_out, c2_out, c3_out, c4_out, c5_out)

                # Note that P6 is used in RPN, but not in the classifier heads.
                rpn_feature_maps = [p2_out, p3_out, p4_out, p5_out, p6_out]
                mrcnn_feature_maps = [p2_out, p3_out, p4_out, p5_out]

                # Loop through pyramid layers
                layer_outputs = []  # list of lists
                for p in rpn_feature_maps:
                    layer_outputs.append(self.rpn(p))

                # Concatenate layer outputs
                # Convert from list of lists of level outputs to list of lists
                # of outputs across levels.
                # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
                outputs = list(zip(*layer_outputs))
                outputs = [torch.cat(list(o), dim=1) for o in outputs]
                rpn_class_logits, rpn_class, rpn_bbox = outputs

                # Generate proposals
                # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
                # and zero padded.
                proposal_count = self.config.POST_NMS_ROIS_TRAINING if mode == "training" \
                    else self.config.POST_NMS_ROIS_INFERENCE
                rpn_rois = proposal_layer([rpn_class, rpn_bbox],
                                          proposal_count=proposal_count,
                                          nms_threshold=self.config.RPN_NMS_THRESHOLD,
                                          anchors=self.anchors,
                                          config=self.config)

                semantic_segment = self.semantic(mrcnn_feature_maps)

                if limit == "semantic":
                    if mode == "training":
                        return semantic_segment
                    else:
                        print("inference semantic not exists")
                        exit()
                else: # inference - instance/insttr
                    if limit == "instance":
                        mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.classifier(mrcnn_feature_maps, rpn_rois)
                        detections = detection_layer(self.config, rpn_rois, mrcnn_class, mrcnn_bbox, image_metas)  # 34,6
                        h, w = self.config.IMAGE_SHAPE[:2]
                        scale = Variable(torch.from_numpy(np.array([h, w, h, w])).float(), requires_grad=False)
                        if self.config.GPU_COUNT:
                            scale=scale.cuda()
                        if len(detections.shape)>1:
                            detection_boxes = detections[:, :4] / scale

                            # Add back batch dimension
                            detection_boxes = detection_boxes.unsqueeze(0)

                            # Create masks for detections
                            mrcnn_mask = self.mask(mrcnn_feature_maps, detection_boxes)  # x, 134, 28, 28

                            # Add back batch dimension
                            detections = detections.unsqueeze(0)  # [1, x, 6]
                            mrcnn_mask = mrcnn_mask.unsqueeze(0)  # [1, x, 81, 28, 28]
                        # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！ THING
                        else:
                            detections=torch.Tensor()
                            mrcnn_mask=torch.Tensor()
                            if self.config.GPU_COUNT:
                                detections=detections.cuda()
                                mrcnn_mask=mrcnn_mask.cuda()

                        # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！ THING
                        result=self.detect_objects(image_metas, detections, mrcnn_mask, semantic_segment)
                        return result
                    elif limit == "insttr":
                        influence_map = self.saliency(c1_out, c2_out, c3_out, c4_out, c5_out)[4]  # (1,1,128,128)
                        mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.classifier(mrcnn_feature_maps, rpn_rois)
                        detections = detection_layer(self.config, rpn_rois, mrcnn_class, mrcnn_bbox, image_metas)  # 34,6

                        if len(detections.shape)>1:
                            h, w = self.config.IMAGE_SHAPE[:2]
                            scale = Variable(torch.from_numpy(np.array([h, w, h, w])).float(), requires_grad=False)
                            if self.config.GPU_COUNT:
                                scale = scale.cuda()

                            detection_boxes = detections[:, :4] / scale

                            # Add back batch dimension
                            detection_boxes = detection_boxes.unsqueeze(0)

                            # Create masks for detections
                            mrcnn_mask = self.mask(mrcnn_feature_maps, detection_boxes)  # x, 134, 28, 28

                            # Add back batch dimension
                            detections = detections.unsqueeze(0)  # [1, x, 6]
                            mrcnn_mask = mrcnn_mask.unsqueeze(0)  # [1, x, 81, 28, 28]
                        else:
                            detections=torch.Tensor()
                            mrcnn_mask=torch.Tensor()
                            if self.config.GPU_COUNT:
                                detections=detections.cuda()
                                mrcnn_mask=mrcnn_mask.cuda()

                        result = self.detect_objects(image_metas, detections, mrcnn_mask, semantic_segment)
                        influence_map = self.unmold_p_interest(influence_map,image_metas)
                        result['influence_map']=influence_map
                        return result
                    else:
                        print("mode not exists")
                        exit()

    def mold_inputs(self, images):
        """Takes a list of images and modifies them to the format expected
        as an input to the neural network.
        images: List of image matricies [height,width,depth]. Images can have
            different sizes.

        Returns 3 Numpy matricies:
        molded_images: [N, h, w, 3]. Images resized and normalized.
        image_metas: [N, length of meta data]. Details about each image.
        """
        molded_images = []
        image_metas = []
        for image in images:
            # Resize image to fit the model expected size
            # TODO: move resizing to mold_image()
            molded_image, window, scale, padding = utils.resize_image(
                image,
                min_dim=self.config.IMAGE_MIN_DIM,
                max_dim=self.config.IMAGE_MAX_DIM,
                padding=self.config.IMAGE_PADDING)
            molded_image = mold_image(molded_image, self.config)
            # Build image_meta
            image_meta = compose_image_meta(0, image.shape, window)
            # Append
            molded_images.append(molded_image)
            image_metas.append(image_meta)
        # Pack into arrays
        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)

        molded_images=torch.from_numpy(molded_images.transpose(0, 3, 1, 2)).float()
        image_metas = torch.from_numpy(image_metas).float()

        return molded_images, image_metas

    def detect_objects(self,image_metas, thing_detections, thing_masks, semantic_segment):
        image_id, image_shape, window = image_metas[0][0], image_metas[0][1:4], image_metas[0][4:8]
        top_pad, left_pad, top_pad_h, left_pad_w = window[0], window[1], window[2], window[3]

        result = {}
        
        semantic_segment, stuff_detections, stuff_masks = generate_stuff(self.config, semantic_segment)  # [y, 5],[y, 500, 500] [1, 134, 512, 512]

        if len(thing_detections.shape) > 1 and len(stuff_detections.shape) > 1 and self.config.GPU_COUNT:
            if self.config.GPU_COUNT:
                thing_detections = thing_detections.data.cpu().numpy()
                thing_masks = thing_masks.permute(0, 1, 3, 4, 2).data.cpu().numpy()
                stuff_detections = stuff_detections.data.cpu().numpy()  # [y,5]
                stuff_masks = stuff_masks.data.cpu().numpy()  # [y,500,500]
            else:
                thing_detections = thing_detections.data.numpy()
                thing_masks = thing_masks.permute(0, 1, 3, 4, 2).data.numpy()
                stuff_detections = stuff_detections.data.numpy()  # [y,5]
                stuff_masks = stuff_masks.data.numpy()  # [y,500,500]
            thing_detections = thing_detections.squeeze(0)  # [x,6]
            thing_masks = thing_masks.squeeze(0)  # [x,28,28,81]

            semantic_segment = resize_semantic_label(semantic_segment, (self.config.IMAGE_SIZE, self.config.IMAGE_SIZE))
            semantic_segment = semantic_segment[top_pad:top_pad_h, left_pad:left_pad_w]
            semantic_segment = scipy.ndimage.zoom(semantic_segment, [image_shape[0] / (top_pad_h - top_pad),
                                                                     image_shape[1] / (left_pad_w - left_pad)],
                                                  mode='nearest', order=0)

            thing_class_ids, thing_boxes, thing_masks_unmold, thing_scores = filter_thing_masks(thing_detections,
                                                                                                thing_masks,
                                                                                                image_shape, window)
            stuff_class_ids, stuff_boxes, stuff_masks_unmold = filter_stuff_masks(stuff_detections, stuff_masks,
                                                                                  image_shape, window)
            result={
                "thing_boxes": thing_boxes,
                "thing_class_ids": thing_class_ids,
                "thing_scores": thing_scores,
                "thing_masks": thing_masks_unmold,
                "stuff_boxes": stuff_boxes,
                "stuff_class_ids": stuff_class_ids,
                "stuff_masks": stuff_masks_unmold,
                "semantic_segment": semantic_segment
            }
        elif len(thing_detections.shape) == 1 and len(stuff_detections.shape) > 1 and self.config.GPU_COUNT:
            if self.config.GPU_COUNT:
                stuff_detections = stuff_detections.data.cpu().numpy()  # [y,5]
                stuff_masks = stuff_masks.data.cpu().numpy()  # [y,500,500]
            else:
                stuff_detections = stuff_detections.data.numpy()  # [y,5]
                stuff_masks = stuff_masks.data.numpy()  # [y,500,500]

            semantic_segment = resize_semantic_label(semantic_segment, (self.config.IMAGE_SIZE, self.config.IMAGE_SIZE))
            semantic_segment = semantic_segment[top_pad:top_pad_h, left_pad:left_pad_w]
            semantic_segment = scipy.ndimage.zoom(semantic_segment, [image_shape[0] / (top_pad_h - top_pad),
                                                                     image_shape[1] / (left_pad_w - left_pad)],
                                                  mode='nearest', order=0)

            stuff_class_ids, stuff_boxes, stuff_masks_unmold = filter_stuff_masks(stuff_detections, stuff_masks,
                                                                                  image_shape, window)

            result={
                "stuff_boxes": stuff_boxes,
                "stuff_class_ids": stuff_class_ids,
                "stuff_masks": stuff_masks_unmold,
                "semantic_segment": semantic_segment
            }
        elif len(thing_detections.shape) > 1 and len(stuff_detections.shape) == 1 and self.config.GPU_COUNT:
            if self.config.GPU_COUNT:
                thing_detections = thing_detections.data.cpu().numpy()
                thing_masks = thing_masks.permute(0, 1, 3, 4, 2).data.cpu().numpy()
            else:
                thing_detections = thing_detections.data.numpy()
                thing_masks = thing_masks.permute(0, 1, 3, 4, 2).data.numpy()
            thing_detections = thing_detections.squeeze(0)  # [x,6]
            thing_masks = thing_masks.squeeze(0)  # [x,28,28,81]

            semantic_segment = resize_semantic_label(semantic_segment, (self.config.IMAGE_SIZE, self.config.IMAGE_SIZE))
            semantic_segment = semantic_segment[top_pad:top_pad_h, left_pad:left_pad_w]
            semantic_segment = scipy.ndimage.zoom(semantic_segment, [image_shape[0] / (top_pad_h - top_pad),
                                                                     image_shape[1] / (left_pad_w - left_pad)],
                                                  mode='nearest', order=0)

            thing_class_ids, thing_boxes, thing_masks_unmold, thing_scores = filter_thing_masks(thing_detections,
                                                                                                thing_masks,
                                                                                                image_shape, window)

            result={
                "thing_boxes": thing_boxes,
                "thing_class_ids": thing_class_ids,
                "thing_scores": thing_scores,
                "thing_masks": thing_masks_unmold,
                "semantic_segment": semantic_segment
            }
        else:
            result={
                "semantic_segment": semantic_segment
            }
        return result
    
    def unmold_p_interest(self,influence_map, image_metas):
        if self.config.GPU_COUNT:
            influence_map = influence_map.squeeze(0).squeeze(0).data.cpu().numpy()
        else:
            influence_map = influence_map.squeeze(0).squeeze(0).data.numpy()

        image_id, image_shape, window = image_metas[0][0], image_metas[0][1:4], image_metas[0][4:8]
        top_pad, left_pad, top_pad_h, left_pad_w = window[0], window[1], window[2], window[3]

        # influence_map=influence_map.squeeze(0).squeeze(0).data.cpu().numpy()
        influence_map = resize_influence_map(influence_map, (self.config.IMAGE_SIZE, self.config.IMAGE_SIZE))
        influence_map = influence_map[top_pad:top_pad_h, left_pad:left_pad_w]
        influence_map = scipy.misc.imresize(influence_map, image_shape[:2], interp='bilinear')
        
        return influence_map
    def predict_segment(self, result, image_metas):
        image_shape = image_metas[0][1:4]
        panoptic_result=np.zeros(image_shape)
        semantic_result = np.zeros(image_shape)

        information_collector={}
        class_dict = self.class_dict
        id_generator = self.id_generator
        if 'stuff_class_ids' in result:
            stuff_class_ids, stuff_boxes, stuff_masks = result['stuff_class_ids'], result['stuff_boxes'], result['stuff_masks']
            for i,class_id in enumerate(stuff_class_ids):
                category_id=class_dict[str(int(class_id))]['category_id']
                category_name=class_dict[str(int(class_id))]['name']
                id,color = id_generator.get_id_and_color(str(category_id))
                mask=stuff_masks[i]==1
                panoptic_result[mask]=color
                semantic_result[mask] = [int(class_id),int(class_id),int(class_id)]
                information_collector[str(id)]={"id":int(id),"bbox":[int(stuff_boxes[i][0]),int(stuff_boxes[i][1]),int(stuff_boxes[i][2]),int(stuff_boxes[i][3])], \
                                                "class_id": int(class_id), "category_id":int(category_id),"category_name":category_name, \
                                                'mask': mask}
        if 'thing_class_ids' in result:
            thing_class_ids, thing_boxes, thing_masks = result['thing_class_ids'], result['thing_boxes'], result['thing_masks']
            for i,class_id in enumerate(thing_class_ids):
                category_id=class_dict[str(int(class_id))]['category_id']
                category_name=class_dict[str(int(class_id))]['name']
                id, color = id_generator.get_id_and_color(str(category_id))
                mask=thing_masks[i]==1          # 426,640
                panoptic_result[mask]=color     # 426,640,3
                semantic_result[mask] = [int(class_id),int(class_id),int(class_id)]
                information_collector[str(id)]={"id": int(id), "bbox": [int(thing_boxes[i][0]),int(thing_boxes[i][1]),int(thing_boxes[i][2]),int(thing_boxes[i][3])], \
                                                "class_id": int(class_id), "category_id": int(category_id),"category_name":category_name, \
                                                'mask': mask}
        return semantic_result, panoptic_result, information_collector

    def construct_dataset(self, semantic_label, saliency_map, panoptic_result, ioi_segments_info, image_shape, mode): # input numpy, output numpy
        image_width = image_shape[1]
        image_height = image_shape[0]
        scale = self.config.IMAGE_SIZE / max(image_height, image_width)
        new_height = int(round(image_height * scale))
        new_width = int(round(image_width * scale))
        top_pad = (self.config.IMAGE_SIZE - new_height) // 2
        bottom_pad = self.config.IMAGE_SIZE - new_height - top_pad
        left_pad = (self.config.IMAGE_SIZE - new_width) // 2
        right_pad = self.config.IMAGE_SIZE - new_width - left_pad

        semantic_label = scipy.misc.imresize(semantic_label.astype(np.uint8), (new_height, new_width), interp='nearest')
        if len(semantic_label.shape) == 3:
            semantic_label = semantic_label[:, :, 0]
        semantic_label = np.pad(semantic_label, [(top_pad, bottom_pad), (left_pad, right_pad)], mode='constant',
                                constant_values=0)

        saliency_map = scipy.misc.imresize(saliency_map, (new_height, new_width), interp='nearest')
        if len(saliency_map.shape) == 3:
            saliency_map = saliency_map[:, :, 0]
        saliency_map = np.pad(saliency_map, [(top_pad, bottom_pad), (left_pad, right_pad)], mode='constant',constant_values=0)

        labels = []
        class_ids = []
        boxes = []
        instance_list=[]
        for instance_id in ioi_segments_info:
            instance_list.append(instance_id)
            segment_info = ioi_segments_info[instance_id]
            category_id = segment_info['category_id']
            # find the class_id from the class_dict whose id is equal to the category_id
            class_id = segment_info["class_id"]
            box = [int(segment_info['bbox'][0] * scale + top_pad), int(segment_info['bbox'][1] * scale + left_pad),
                   int(segment_info['bbox'][2] * scale + top_pad), int(segment_info['bbox'][3] * scale + left_pad)]
            # plt.figure()
            # plt.imshow(semantic_label)
            # plt.gca().add_patch(plt.Rectangle((segment_info['bbox'][1] * scale + left_pad,segment_info['bbox'][0] * scale + top_pad),
            #                                   (segment_info['bbox'][3] * scale + left_pad)-(segment_info['bbox'][1] * scale + left_pad),
            #                                   (segment_info['bbox'][2] * scale + left_pad)-(segment_info['bbox'][0] * scale + left_pad),
            #                                   color='green',fill=False,linewidth=1))
            # plt.show()
            # plt.figure()
            # plt.imshow(panoptic_result)
            # plt.gca().add_patch(plt.Rectangle((segment_info['bbox'][1], segment_info['bbox'][0]),
            #                                   segment_info['bbox'][3] - segment_info['bbox'][1],
            #                                   segment_info['bbox'][2] - segment_info['bbox'][0],
            #                                   color='green', fill=False, linewidth=1))
            # plt.show()
            if mode == 'training':
                islabel = 1 if segment_info['labeled'] else 0
            elif mode == 'inference':
                islabel = 0
            labels.append(islabel)
            boxes.append(np.array(box))
            class_ids.append(class_id)
        boxes = np.stack(boxes)

        instance_groups = []
        for i in range(boxes.shape[0]):
            y1, x1, y2, x2 = boxes[i][:4]
            instance_group = []

            instance_label = semantic_label[y1:y2, x1:x2]
            instance_label = scipy.misc.imresize(instance_label, (self.config.INSTANCE_SIZE, self.config.INSTANCE_SIZE), interp='nearest') / 134.0
            instance_group.append(instance_label)

            # plt.figure()
            # plt.imshow(instance_label)
            # plt.show()

            instance_map = saliency_map[y1:y2, x1:x2]
            instance_map = scipy.misc.imresize(instance_map, (self.config.INSTANCE_SIZE, self.config.INSTANCE_SIZE), interp='bilinear') / 255.0
            instance_group.append(instance_map)

            # plt.figure()
            # plt.imshow(instance_map)
            # plt.show()

            instance_group = np.stack(instance_group)
            instance_groups.append(instance_group)

        pair_label = []
        for i, a_label_value in enumerate(labels):
            for j, b_label_value in enumerate(labels):
                pair_label.append((a_label_value + b_label_value) / 2.0)
        pair_label = np.array(pair_label)
        instance_groups = np.stack(instance_groups)
        class_ids = np.array(class_ids, dtype=np.float32)
        labels = np.array(labels, dtype=np.float32)
        return instance_groups, boxes, class_ids, labels, pair_label, instance_list

    def map_instance_to_gt(self, gt_instance_dict, instance_dict, gt_segmentation, segmentation, image_metas):
        def compute_pixel_iou(bool_mask_pred, bool_mask_gt):
            intersection = bool_mask_pred * bool_mask_gt
            union = bool_mask_pred + bool_mask_gt
            return np.count_nonzero(intersection) / np.count_nonzero(union)  # np.count_nonzero: 计算数组中非零元素的个数

        instance_pred_gt_dict = {}
        instance_gt_pred_dict = {}

        if self.config.GPU_COUNT:
            gt_segmentation = gt_segmentation.squeeze(0).data.cpu().numpy()
        else:
            gt_segmentation = gt_segmentation.squeeze(0).data.numpy()

        gt_segmentation_id = utils.rgb2id(gt_segmentation)
        segmentation_id = utils.rgb2id(segmentation)

        for instance_id in instance_dict:
            mask = segmentation_id == int(instance_id)
            instance_dict[instance_id]['mask'] = mask

        for gt_instance_id in gt_instance_dict:
            gt_mask = gt_segmentation_id == int(gt_instance_id)
            gt_instance_dict[gt_instance_id]['mask'] = gt_mask

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
                        instance_gt_pred_dict[gt_instance_id] = {"labeled": gt_instance_dict[gt_instance_id]['labeled'].data.numpy()[0]==1, "pred": []}
                    if i_iou >= self.config.MAP_IOU and instance_dict[instance_id]['category_id'] == gt_instance_dict[gt_instance_id]['category_id'].data.numpy()[0] and i_iou > max_iou:
                        max_gt_instance_id = gt_instance_id
                        max_iou = i_iou
                        instance_gt_pred_dict[gt_instance_id]['pred'].append(instance_id)
                if max_gt_instance_id != "":
                    instance_pred_gt_dict[instance_id] = {"gt_instance_id": max_gt_instance_id,
                                                          "label": gt_instance_dict[max_gt_instance_id]['labeled'].data.numpy()[0]==1}
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
            if instance_id in instance_pred_gt_dict:
                instance_dict[instance_id]['labeled'] = instance_pred_gt_dict[instance_id]['label']
            else:
                instance_dict[instance_id]['labeled'] = False
        image_id, image_shape, window = image_metas[0][0], image_metas[0][1:4], image_metas[0][4:8]

        ioi_images_dict = {"image_id": int(image_id), "image_name": str(image_id).zfill(12)+".jpg",
                           "height": int(image_shape[0]), "width": int(image_shape[1]),
                           'segments_info': instance_dict,"base":base}
        return ioi_images_dict