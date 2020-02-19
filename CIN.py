import os
import re
import datetime
import json

import scipy
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict

import torch
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
from panoptic_gt_preprocess import IdGenerator
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

    def load_part_weights(self,filepath,mode="segmentation"):
        if os.path.exists(filepath):
            state_dict = torch.load(filepath)
            if mode in ['heads', '3+', '4+', '5+', 'segmentation']:
                state_dict_to_load = dict()
                for name in state_dict:
                    if name.split(".")[0] in ["resnet", "fpn", "rpn", "classifier", "mask", "semantic"]:
                        state_dict_to_load[name] = state_dict[name]
                self.load_state_dict(state_dict_to_load, strict=False)
            elif mode == "saliency":
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
            elif mode == "sesa":
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
        # classifier_dict = dict()
        # mask_dict = dict()
        # rpn_dict = dict()
        for name in state_dict:
            # if name.find("classifier")>=0:
            #     classifier_dict[name]=state_dict[name]
            # if name.find("mask")>=0:
            #     mask_dict[name]=state_dict[name]
            # if name.find("rpn")>=0:
            #     rpn_dict[name]=state_dict[name]
            if name[:5]=="fpn.C":
                resnet_dict["resnet.C"+name[5:]]=state_dict[name]
            else:
                other_dict[name]=state_dict[name]
        self.load_state_dict(resnet_dict, strict=False)
        self.load_state_dict(other_dict, strict=False)
        # self.load_state_dict(classifier_dict, strict=False)
        # self.load_state_dict(mask_dict, strict=False)
        # self.load_state_dict(rpn_dict, strict=False)
        self.set_log_dir()
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def train_model(self,train_dataset, val_dataset, learning_rate, epochs, layers):
        self.training_layers=layers
        layer_regex = {
            # all layers but the backbone
            "heads": r"(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)|(semantic.*)",
            # From a specific Resnet stage and up
            "3+": r"(resnet.C3.*)|(resnet.C4.*)|(resnet.C5.*)|(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)|(semantic.*)",
            "4+": r"(resnet.C4.*)|(resnet.C5.*)|(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)|(semantic.*)",
            "5+": r"(resnet.C5.*)|(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)|(semantic.*)",
            "segmentation": r"(resnet.*)|(fpn.*)|(rpn.*)|(classifier.*)|(mask.*)|(semantic.*)",
            "saliency": r"(saliency.*)",
            "semantic": r"(semantic.*)",
            "new_heads":r"(saliency.*)|(semantic.*)",
            "sesa": r"(resnet.*)|(fpn.*)|(rpn.*)|(classifier.*)|(mask.*)|(semantic.*)|(saliency.*)",
            "selection": r"(ciedn.*)",
            # All layers
            "all": r"(.*)",
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
        train_generator = TorchDataLoader(train_set, collate_fn=my_collate_fn, batch_size=1, shuffle=True, num_workers=1)

        val_set = Dataset(val_dataset, self.config)
        val_generator = TorchDataLoader(val_set, collate_fn=my_collate_fn, batch_size=1, shuffle=True, num_workers=1)

        self.set_trainable(layers)

        optimizers=[]
        if self.training_layers in ['heads', '3+', '4+', '5+', 'segmentation',"semantic"]:
            trainables_wo_bn = [param for name, param in self.named_parameters() if param.requires_grad and not 'bn' in name]
            trainables_only_bn = [param for name, param in self.named_parameters() if param.requires_grad and 'bn' in name]

            optimizer = optim.SGD([
                {'params': trainables_wo_bn, 'weight_decay': self.config.WEIGHT_DECAY},
                {'params': trainables_only_bn}
            ], lr=learning_rate, momentum=self.config.LEARNING_MOMENTUM)
            optimizers.append(optimizer)
        elif self.training_layers == 'new_heads':
            trainables_wo_bn = [param for name, param in self.named_parameters() if
                                param.requires_grad and not 'bn' in name and not 'saliency' in name]
            trainables_only_bn = [param for name, param in self.named_parameters() if
                                  param.requires_grad and 'bn' in name and not 'saliency' in name]
            optimizer_seg = optim.SGD([
                {'params': trainables_wo_bn, 'weight_decay': self.config.WEIGHT_DECAY},
                {'params': trainables_only_bn}
            ], lr=learning_rate, momentum=self.config.LEARNING_MOMENTUM)
            optimizers.append(optimizer_seg)

            optimizer_encoder = torch.optim.SGD(
                [{'params': [param for name, param in self.saliency.conv1.named_parameters() if param.requires_grad]},
                 {'params': [param for name, param in self.saliency.conv2.named_parameters() if param.requires_grad]},
                 {'params': [param for name, param in self.saliency.conv3.named_parameters() if param.requires_grad]}], lr=learning_rate,
                momentum=0.9, weight_decay=0.0005)
            trainables_decoder_no_bn = [param for name, param in self.saliency.decoder.named_parameters() if
                                        param.requires_grad and not 'bn' in name]
            trainables_decoder_only_bn = [param for name, param in self.saliency.decoder.named_parameters() if
                                          param.requires_grad and 'bn' in name]
            optimizer_decoder = torch.optim.SGD(
                [{'params': trainables_decoder_no_bn, }, {'params': trainables_decoder_only_bn}], lr=learning_rate * 10,
                momentum=0.9, weight_decay=0.0005)
            optimizers.append(optimizer_encoder)
            optimizers.append(optimizer_decoder)
        elif self.training_layers == 'saliency':
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
        elif self.training_layers == "all":
            trainables_wo_bn = [param for name, param in self.named_parameters() if
                                param.requires_grad and not 'bn' in name]
            trainables_only_bn = [param for name, param in self.named_parameters() if
                                  param.requires_grad and 'bn' in name]

            optimizer_seg = optim.SGD([
                {'params': trainables_wo_bn, 'weight_decay': self.config.WEIGHT_DECAY},
                {'params': trainables_only_bn}
            ], lr=learning_rate, momentum=self.config.LEARNING_MOMENTUM)
            optimizers.append(optimizer_seg)

            optimizer_encoder = torch.optim.SGD(
                [{'params': [param for name, param in self.saliency.conv1.named_parameters()]},
                 {'params': [param for name, param in self.saliency.conv2.named_parameters()]},
                 {'params': [param for name, param in self.saliency.conv3.named_parameters()]}], lr=learning_rate,
                momentum=0.9, weight_decay=0.0005)
            trainables_decoder_no_bn = [param for name, param in self.saliency.decoder.named_parameters() if
                                        param.requires_grad and not 'bn' in name]
            trainables_decoder_only_bn = [param for name, param in self.saliency.decoder.named_parameters() if
                                          param.requires_grad and 'bn' in name]
            optimizer_decoder = torch.optim.SGD(
                [{'params': trainables_decoder_no_bn, }, {'params': trainables_decoder_only_bn}], lr=learning_rate * 10,
                momentum=0.9, weight_decay=0.0005)
            optimizers.append(optimizer_encoder)
            optimizers.append(optimizer_decoder)

            optimizer_selec = torch.optim.Adam(self.ciedn.parameters(), lr=learning_rate*10)
            optimizers.append(optimizer_selec)
        else:
            pass

        for epoch in range(self.epoch + 1, epochs + 1):
            log("Epoch {}/{}.".format(epoch, epochs))

            if self.training_layers in ['heads','3+','4+','5+','segmentation']:
                print("segmentation")
                # Training
                loss, loss_rpn_class, loss_rpn_bbox, loss_mrcnn_class, loss_mrcnn_bbox, loss_mrcnn_mask, loss_semantic = self.train_epoch(
                    train_generator, optimizers, self.config.STEPS_PER_EPOCH)

                # Validation
                val_loss, val_loss_rpn_class, val_loss_rpn_bbox, val_loss_mrcnn_class, val_loss_mrcnn_bbox, val_loss_mrcnn_mask, val_loss_semantic = self.valid_epoch(
                    val_generator, self.config.VALIDATION_STEPS)

                self.loss_history.append([loss, loss_rpn_class, loss_rpn_bbox, loss_mrcnn_class, loss_mrcnn_bbox, loss_mrcnn_mask,loss_semantic])
                self.val_loss_history.append([val_loss, val_loss_rpn_class, val_loss_rpn_bbox, val_loss_mrcnn_class, val_loss_mrcnn_bbox,val_loss_mrcnn_mask,val_loss_semantic])

                # Statistics
                visualize.plot_loss("loss", 0, self.loss_history, self.val_loss_history, save=True, log_dir=self.log_dir)
                visualize.plot_loss("rpn_class_loss", 1, self.loss_history, self.val_loss_history, save=True, log_dir=self.log_dir)
                visualize.plot_loss("rpn_bbox_loss", 2, self.loss_history, self.val_loss_history, save=True, log_dir=self.log_dir)
                visualize.plot_loss("mrcnn_class_loss", 3, self.loss_history, self.val_loss_history, save=True, log_dir=self.log_dir)
                visualize.plot_loss("mrcnn_bbox_loss", 4, self.loss_history, self.val_loss_history, save=True, log_dir=self.log_dir)
                visualize.plot_loss("mrcnn_mask_loss", 5, self.loss_history, self.val_loss_history, save=True, log_dir=self.log_dir)
                visualize.plot_loss("semantic_loss", 6, self.loss_history, self.val_loss_history, save=True, log_dir=self.log_dir)

            elif self.training_layers=='saliency':
                loss,loss_influence = self.train_epoch(train_generator, optimizers, self.config.STEPS_PER_EPOCH)
                val_loss, val_loss_influence = self.valid_epoch(val_generator, self.config.VALIDATION_STEPS)

                self.loss_history.append([loss, loss_influence])
                self.val_loss_history.append([val_loss,val_loss_influence])

                visualize.plot_loss("saliency", 1, self.loss_history, self.val_loss_history, save=True, log_dir=self.log_dir)
            elif self.training_layers=="semantic":
                loss, loss_semantic = self.train_epoch(train_generator, optimizers, self.config.STEPS_PER_EPOCH)
                val_loss, val_loss_semantic = self.valid_epoch(val_generator, self.config.VALIDATION_STEPS)

                self.loss_history.append([loss, loss_semantic])
                self.val_loss_history.append([val_loss, val_loss_semantic])
                visualize.plot_loss("loss", 0, self.loss_history, self.val_loss_history, save=True,
                                    log_dir=self.log_dir)
                visualize.plot_loss("semantic_loss", 1, self.loss_history, self.val_loss_history, save=True,
                                    log_dir=self.log_dir)
            elif self.training_layers=="new_heads":
                loss, loss_semantic,loss_influence = self.train_epoch(train_generator, optimizers, self.config.STEPS_PER_EPOCH)
                val_loss, val_loss_semantic, val_loss_influence = self.valid_epoch(val_generator, self.config.VALIDATION_STEPS)

                self.loss_history.append([loss, loss_semantic,loss_influence])
                self.val_loss_history.append([val_loss, val_loss_semantic,val_loss_influence])
                visualize.plot_loss("loss", 0, self.loss_history, self.val_loss_history, save=True,
                                    log_dir=self.log_dir)
                visualize.plot_loss("semantic_loss", 1, self.loss_history, self.val_loss_history, save=True,
                                    log_dir=self.log_dir)
                visualize.plot_loss("saliency", 2, self.loss_history, self.val_loss_history, save=True,
                                    log_dir=self.log_dir)
            elif self.training_layers=="sesa":
                loss, loss_rpn_class, loss_rpn_bbox, loss_mrcnn_class, loss_mrcnn_bbox, loss_mrcnn_mask, loss_semantic, loss_influence = self.train_epoch(
                    train_generator, optimizers, self.config.STEPS_PER_EPOCH)

                # Validation
                val_loss, val_loss_rpn_class, val_loss_rpn_bbox, val_loss_mrcnn_class, val_loss_mrcnn_bbox, val_loss_mrcnn_mask, val_loss_semantic, val_loss_influence = self.valid_epoch(
                    val_generator, self.config.VALIDATION_STEPS)

                self.loss_history.append(
                    [loss, loss_rpn_class, loss_rpn_bbox, loss_mrcnn_class, loss_mrcnn_bbox, loss_mrcnn_mask,
                     loss_semantic,loss_influence])
                self.val_loss_history.append(
                    [val_loss, val_loss_rpn_class, val_loss_rpn_bbox, val_loss_mrcnn_class, val_loss_mrcnn_bbox,
                     val_loss_mrcnn_mask, val_loss_semantic,val_loss_influence])

                # Statistics
                visualize.plot_loss("loss", 0, self.loss_history, self.val_loss_history, save=True,
                                    log_dir=self.log_dir)
                visualize.plot_loss("rpn_class_loss", 1, self.loss_history, self.val_loss_history, save=True,
                                    log_dir=self.log_dir)
                visualize.plot_loss("rpn_bbox_loss", 2, self.loss_history, self.val_loss_history, save=True,
                                    log_dir=self.log_dir)
                visualize.plot_loss("mrcnn_class_loss", 3, self.loss_history, self.val_loss_history, save=True,
                                    log_dir=self.log_dir)
                visualize.plot_loss("mrcnn_bbox_loss", 4, self.loss_history, self.val_loss_history, save=True,
                                    log_dir=self.log_dir)
                visualize.plot_loss("mrcnn_mask_loss", 5, self.loss_history, self.val_loss_history, save=True,
                                    log_dir=self.log_dir)
                visualize.plot_loss("semantic_loss", 6, self.loss_history, self.val_loss_history, save=True,
                                    log_dir=self.log_dir)
                visualize.plot_loss("saliency_loss", 7, self.loss_history, self.val_loss_history, save=True,
                                    log_dir=self.log_dir)
            elif self.training_layers=='selection':
                loss, loss_interest = self.train_epoch(train_generator, optimizers, self.config.STEPS_PER_EPOCH)
                val_loss, val_loss_interest = self.valid_epoch(val_generator, self.config.VALIDATION_STEPS)

                self.loss_history.append([loss, loss_interest])
                self.val_loss_history.append([val_loss, val_loss_interest])

                visualize.plot_loss("loss_interest", 1, self.loss_history, self.val_loss_history, save=True, log_dir=self.log_dir)

            elif self.training_layers=='all':
                loss, loss_rpn_class, loss_rpn_bbox, loss_mrcnn_class, loss_mrcnn_bbox, loss_mrcnn_mask, loss_semantic,loss_influence, loss_interest = self.train_epoch(train_generator, optimizers, self.config.STEPS_PER_EPOCH)
                val_loss, val_loss_rpn_class, val_loss_rpn_bbox, val_loss_mrcnn_class, val_loss_mrcnn_bbox, val_loss_mrcnn_mask, val_loss_semantic, val_loss_influence, val_loss_interest = self.valid_epoch(val_generator, self.config.VALIDATION_STEPS)

                self.loss_history.append([loss, loss_rpn_class, loss_rpn_bbox, loss_mrcnn_class, loss_mrcnn_bbox, loss_mrcnn_mask, loss_semantic,loss_influence, loss_interest])
                self.val_loss_history.append([val_loss, val_loss_rpn_class, val_loss_rpn_bbox, val_loss_mrcnn_class, val_loss_mrcnn_bbox, val_loss_mrcnn_mask, val_loss_semantic, val_loss_influence, val_loss_interest])

                visualize.plot_loss("loss", 0, self.loss_history, self.val_loss_history, save=True, log_dir=self.log_dir)
                visualize.plot_loss("loss_rpn_class", 1, self.loss_history, self.val_loss_history, save=True, log_dir=self.log_dir)
                visualize.plot_loss("loss_rpn_bbox", 2, self.loss_history, self.val_loss_history, save=True, log_dir=self.log_dir)
                visualize.plot_loss("loss_mrcnn_class", 3, self.loss_history, self.val_loss_history, save=True, log_dir=self.log_dir)
                visualize.plot_loss("loss_mrcnn_bbox", 4, self.loss_history, self.val_loss_history, save=True, log_dir=self.log_dir)
                visualize.plot_loss("loss_mrcnn_mask", 5, self.loss_history, self.val_loss_history, save=True, log_dir=self.log_dir)
                visualize.plot_loss("loss_semantic", 6, self.loss_history, self.val_loss_history, save=True, log_dir=self.log_dir)
                visualize.plot_loss("loss_influence", 7, self.loss_history, self.val_loss_history, save=True, log_dir=self.log_dir)
                visualize.plot_loss("loss_interest", 8, self.loss_history, self.val_loss_history, save=True, log_dir=self.log_dir)
            else:
                pass

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
            #else:
            try:
                batch_count += 1

                images = inputs[0]
                image_metas = inputs[1]
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

                if self.training_layers in ['heads','3+','4+','5+','segmentation']:
                    # Run object detection
                    predict_input = [images, image_metas, gt_class_ids, gt_boxes, gt_masks]
                    rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask,\
                        semantic_label = self.predict_front(predict_input, mode='training',limit="segmentation")

                    # Compute losses
                    rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss, semantic_loss = compute_losses_PFPN(
                        rpn_match, rpn_bbox, rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits,target_deltas, mrcnn_bbox, target_mask, mrcnn_mask,
                        gt_semantic_label, semantic_label)
                    loss = rpn_class_loss + rpn_bbox_loss + mrcnn_class_loss + mrcnn_bbox_loss + mrcnn_mask_loss + semantic_loss

                    # Backpropagation
                    loss.backward()
                    torch.nn.utils.clip_grad_norm(self.parameters(), 5.0)
                    if (batch_count % self.config.BATCH_SIZE) == 0:
                        for optimizer in optimizers:
                            optimizer.step()
                            optimizer.zero_grad()
                        batch_count = 0

                    # Progress
                    printProgressBar(step + 1, steps, prefix="\t{}/{}".format(step + 1, steps),
                                     suffix="Complete - loss: {:.5f} - rpn_class_loss: {:.5f} - rpn_bbox_loss: {:.5f} - mrcnn_class_loss: {:.5f} - mrcnn_bbox_loss: {:.5f} - mrcnn_mask_loss: {:.5f} - semantic_loss: {:.5f}".format(
                                         loss.data.cpu()[0], rpn_class_loss.data.cpu()[0], rpn_bbox_loss.data.cpu()[0],
                                         mrcnn_class_loss.data.cpu()[0], mrcnn_bbox_loss.data.cpu()[0],
                                         mrcnn_mask_loss.data.cpu()[0],semantic_loss.data.cpu()[0]), length=10)

                    # Statistics
                    loss_sum += loss.data.cpu()[0]/steps
                    loss_rpn_class_sum += rpn_class_loss.data.cpu()[0]/steps
                    loss_rpn_bbox_sum += rpn_bbox_loss.data.cpu()[0]/steps
                    loss_mrcnn_class_sum += mrcnn_class_loss.data.cpu()[0]/steps
                    loss_mrcnn_bbox_sum += mrcnn_bbox_loss.data.cpu()[0]/steps
                    loss_mrcnn_mask_sum += mrcnn_mask_loss.data.cpu()[0]/steps
                    loss_semantic_sum += semantic_loss.data.cpu()[0] / steps

                    # Break after 'steps' steps
                    if step==steps-1:
                        return loss_sum, loss_rpn_class_sum, loss_rpn_bbox_sum, loss_mrcnn_class_sum, loss_mrcnn_bbox_sum, loss_mrcnn_mask_sum, loss_semantic_sum
                    else:
                        step += 1

                elif self.training_layers=='saliency':
                    # Run object detection
                    predict_input = [images, image_metas]
                    influence_preds = self.predict_front(predict_input, mode='training', limit="saliency")
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

                    printProgressBar(step + 1, steps, prefix="\t{}/{}".format(step + 1, steps),suffix="Complete - influence_loss: {:.5f}".format(influence_loss.data.cpu()[0]), length=10)

                    loss_sum += loss.data.cpu()[0] / steps
                    loss_influence_sum += influence_loss.data.cpu()[0] / steps

                    # Break after 'steps' steps
                    if step == steps - 1:
                        return loss_sum, loss_influence_sum
                    else:
                        step += 1
                elif self.training_layers=="semantic":
                    # Run object detection
                    predict_input = [images, image_metas,gt_class_ids, gt_boxes, gt_masks]
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

                    printProgressBar(step + 1, steps, prefix="\t{}/{}".format(step + 1, steps),
                                     suffix="Complete - loss: {:.5f} - semantic_loss: {:.5f} ".format(
                                         loss.data.cpu()[0], semantic_loss.data.cpu()[0]), length=10)

                    loss_sum += loss.data.cpu()[0] / steps
                    loss_semantic_sum += semantic_loss.data.cpu()[0] / steps

                    # Break after 'steps' steps
                    if step == steps - 1:
                        return loss_sum, loss_semantic_sum
                    else:
                        step += 1
                elif self.training_layers=="new_heads":
                    # Run object detection
                    predict_input = [images, image_metas]
                    semantic_label,influence_preds = self.predict_front(predict_input, mode='training', limit="new_heads")
                    influence_loss = compute_saliency_loss(influence_preds, gt_influence_map)
                    semantic_loss = compute_semantic_loss(semantic_label,gt_semantic_label)
                    loss = influence_loss+semantic_loss

                    # Backpropagation
                    loss.backward()
                    torch.nn.utils.clip_grad_norm(self.parameters(), 5.0)
                    if (batch_count % self.config.BATCH_SIZE) == 0:
                        for optimizer in optimizers:
                            optimizer.step()
                            optimizer.zero_grad()
                        batch_count = 0

                    printProgressBar(step + 1, steps, prefix="\t{}/{}".format(step + 1, steps),
                                     suffix="Complete - loss: {:.5f} - semantic_loss: {:.5f} - influence_loss: {:.5f}".format(loss.data.cpu()[0],semantic_loss.data.cpu()[0],influence_loss.data.cpu()[0]),
                                     length=10)

                    loss_sum += loss.data.cpu()[0] / steps
                    loss_semantic_sum += semantic_loss.data.cpu()[0] / steps
                    loss_influence_sum += influence_loss.data.cpu()[0] / steps

                    # Break after 'steps' steps
                    if step == steps - 1:
                        return loss_sum, loss_semantic_sum, loss_influence_sum
                    else:
                        step += 1
                elif self.training_layers=="sesa":
                    # Run object detection
                    predict_input = [images, image_metas, gt_class_ids, gt_boxes, gt_masks]
                    rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask, \
                    semantic_label,influence_preds = self.predict_front(predict_input, mode='training', limit="sesa")

                    # Compute losses
                    rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss, semantic_loss = compute_losses_PFPN(
                        rpn_match, rpn_bbox, rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits,
                        target_deltas, mrcnn_bbox, target_mask, mrcnn_mask,
                        gt_semantic_label, semantic_label)
                    influence_loss = compute_saliency_loss(influence_preds, gt_influence_map)
                    loss = rpn_class_loss + rpn_bbox_loss + mrcnn_class_loss + mrcnn_bbox_loss + mrcnn_mask_loss + semantic_loss + influence_loss

                    # Backpropagation
                    loss.backward()
                    torch.nn.utils.clip_grad_norm(self.parameters(), 5.0)
                    if (batch_count % self.config.BATCH_SIZE) == 0:
                        for optimizer in optimizers:
                            optimizer.step()
                            optimizer.zero_grad()
                        batch_count = 0

                    # Progress
                    printProgressBar(step + 1, steps, prefix="\t{}/{}".format(step + 1, steps),
                                     suffix="Complete - loss: {:.5f} - rpn_class_loss: {:.5f} - rpn_bbox_loss: {:.5f} - mrcnn_class_loss: {:.5f} - mrcnn_bbox_loss: {:.5f} - mrcnn_mask_loss: {:.5f} - semantic_loss: {:.5f} - influence_loss: {:.5f}".format(
                                         loss.data.cpu()[0], rpn_class_loss.data.cpu()[0], rpn_bbox_loss.data.cpu()[0],
                                         mrcnn_class_loss.data.cpu()[0], mrcnn_bbox_loss.data.cpu()[0],
                                         mrcnn_mask_loss.data.cpu()[0], semantic_loss.data.cpu()[0],influence_loss.data.cpu()[0]), length=10)

                    # Statistics
                    loss_sum += loss.data.cpu()[0] / steps
                    loss_rpn_class_sum += rpn_class_loss.data.cpu()[0] / steps
                    loss_rpn_bbox_sum += rpn_bbox_loss.data.cpu()[0] / steps
                    loss_mrcnn_class_sum += mrcnn_class_loss.data.cpu()[0] / steps
                    loss_mrcnn_bbox_sum += mrcnn_bbox_loss.data.cpu()[0] / steps
                    loss_mrcnn_mask_sum += mrcnn_mask_loss.data.cpu()[0] / steps
                    loss_semantic_sum += semantic_loss.data.cpu()[0] / steps
                    loss_influence_sum += influence_loss.data.cpu()[0] / steps

                    # Break after 'steps' steps
                    if step == steps - 1:
                        return loss_sum, loss_rpn_class_sum, loss_rpn_bbox_sum, loss_mrcnn_class_sum, loss_mrcnn_bbox_sum, loss_mrcnn_mask_sum, loss_semantic_sum, loss_influence_sum
                    else:
                        step += 1
                elif self.training_layers=='selection':

                    loss_func = nn.MSELoss()
                    predict_input = [images, image_metas, gt_class_ids, gt_boxes, gt_masks, gt_segmentation, gt_image_instances]
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

                    printProgressBar(step + 1, steps, prefix="\t{}/{}".format(step + 1, steps),
                                     suffix="Complete - interest_loss: {:.5f}".format(interest_loss.data.cpu()[0]),
                                     length=10)

                    loss_sum += loss.data.cpu()[0] / steps
                    loss_interest_sum += interest_loss.data.cpu()[0] / steps

                    # Break after 'steps' steps
                    if step == steps - 1:
                        return loss_sum, loss_interest_sum
                    else:
                        step += 1
                elif self.training_layers=='all':
                    # Run object detection
                    predict_input = [images, image_metas, gt_class_ids, gt_boxes, gt_masks, gt_segmentation, gt_image_instances]
                    rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask, thing_detections, thing_masks, \
                    semantic_labels, influence_preds, predictions, pair_labels = self.predict_front(predict_input, mode='training',limit="all")
                    influence_map=influence_preds[4]

                    instance_class_ids, instance_boxes, instance_masks, interest_label,semantic_segment = self.predict_back(image_metas,thing_detections,thing_masks,semantic_labels,influence_map)

                    gt_interest_label = map_pred_with_gt_mask(gt_interest_class_ids.squeeze(0),
                                                              gt_interest_masks.squeeze(0),
                                                              instance_class_ids.squeeze(0), instance_masks.squeeze(0),
                                                              0.5)

                    loss_func = nn.MSELoss()

                    # Compute losses
                    rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss, semantic_loss, influence_loss = compute_losses_CIN(
                        rpn_match, rpn_bbox, rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits,
                        target_deltas, mrcnn_bbox, target_mask, mrcnn_mask,
                        gt_semantic_label, semantic_labels, gt_influence_map, influence_preds)
                    interest_loss = loss_func(predictions, pair_labels)

                    loss = rpn_class_loss + rpn_bbox_loss + mrcnn_class_loss + mrcnn_bbox_loss + mrcnn_mask_loss + semantic_loss + influence_loss + interest_loss

                    # Backpropagation
                    loss.backward()
                    torch.nn.utils.clip_grad_norm(self.parameters(), 5.0)
                    if (batch_count % self.config.BATCH_SIZE) == 0:
                        for optimizer in optimizers:
                            optimizer.step()
                            optimizer.zero_grad()
                        batch_count = 0

                    # Progress
                    printProgressBar(step + 1, steps, prefix="\t{}/{}".format(step + 1, steps),
                                     suffix="Complete - loss: {:.5f} - rpn_class_loss: {:.5f} - rpn_bbox_loss: {:.5f} - mrcnn_class_loss: {:.5f} - mrcnn_bbox_loss: {:.5f} - mrcnn_mask_loss: {:.5f} - semantic_loss: {:.5f} - influence_loss: {:.5f} - interest_loss: {:.5f}".format(
                                         loss.data.cpu()[0], rpn_class_loss.data.cpu()[0], rpn_bbox_loss.data.cpu()[0],
                                         mrcnn_class_loss.data.cpu()[0], mrcnn_bbox_loss.data.cpu()[0],
                                         mrcnn_mask_loss.data.cpu()[0], semantic_loss.data.cpu()[0],
                                         influence_loss.data.cpu()[0],interest_loss.data.cpu()[0]), length=10)

                    # Statistics
                    loss_sum += loss.data.cpu()[0] / steps
                    loss_rpn_class_sum += rpn_class_loss.data.cpu()[0] / steps
                    loss_rpn_bbox_sum += rpn_bbox_loss.data.cpu()[0] / steps
                    loss_mrcnn_class_sum += mrcnn_class_loss.data.cpu()[0] / steps
                    loss_mrcnn_bbox_sum += mrcnn_bbox_loss.data.cpu()[0] / steps
                    loss_mrcnn_mask_sum += mrcnn_mask_loss.data.cpu()[0] / steps
                    loss_semantic_sum += semantic_loss.data.cpu()[0] / steps
                    loss_influence_sum += influence_loss.data.cpu()[0] / steps
                    loss_interest_sum += interest_loss.data.cpu()[0] / steps

                    # Break after 'steps' steps
                    if step == steps - 1:
                        return loss_sum, loss_rpn_class_sum, loss_rpn_bbox_sum, loss_mrcnn_class_sum, loss_mrcnn_bbox_sum, loss_mrcnn_mask_sum, loss_semantic_sum, loss_influence_sum, loss_interest_sum
                    else:
                        step += 1

                else:
                    pass
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
            #else:
            try:
                images = inputs[0]
                image_metas = inputs[1]
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

                # image_metas as numpy array
                image_metas = image_metas.numpy()

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

                if self.training_layers in ['heads', '3+', '4+', '5+', 'segmentation']:
                    # Run object detection
                    predict_input = [images, image_metas, gt_class_ids, gt_boxes, gt_masks]
                    rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask, semantic_label = self.predict_front(predict_input, mode='training',limit="segmentation")

                    # Compute losses
                    rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss, semantic_loss = compute_losses_PFPN(
                        rpn_match, rpn_bbox, rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits,target_deltas, mrcnn_bbox, target_mask, mrcnn_mask,
                        gt_semantic_label, semantic_label)
                    loss = rpn_class_loss + rpn_bbox_loss + mrcnn_class_loss + mrcnn_bbox_loss + mrcnn_mask_loss + semantic_loss

                    # Progress
                    printProgressBar(step + 1, steps, prefix="\t{}/{}".format(step + 1, steps),
                                     suffix="Complete - loss: {:.5f} - rpn_class_loss: {:.5f} - rpn_bbox_loss: {:.5f} - mrcnn_class_loss: {:.5f} - mrcnn_bbox_loss: {:.5f} - mrcnn_mask_loss: {:.5f} - semantic_loss: {:.5f}".format(
                                         loss.data.cpu()[0], rpn_class_loss.data.cpu()[0], rpn_bbox_loss.data.cpu()[0],
                                         mrcnn_class_loss.data.cpu()[0], mrcnn_bbox_loss.data.cpu()[0],
                                         mrcnn_mask_loss.data.cpu()[0], semantic_loss.data.cpu()[0], length=10))

                    # Statistics
                    loss_sum += loss.data.cpu()[0]/steps
                    loss_rpn_class_sum += rpn_class_loss.data.cpu()[0]/steps
                    loss_rpn_bbox_sum += rpn_bbox_loss.data.cpu()[0]/steps
                    loss_mrcnn_class_sum += mrcnn_class_loss.data.cpu()[0]/steps
                    loss_mrcnn_bbox_sum += mrcnn_bbox_loss.data.cpu()[0]/steps
                    loss_mrcnn_mask_sum += mrcnn_mask_loss.data.cpu()[0]/steps
                    loss_semantic_sum += semantic_loss.data.cpu()[0] / steps

                    # Break after 'steps' steps
                    if step==steps-1:
                        return loss_sum, loss_rpn_class_sum, loss_rpn_bbox_sum, loss_mrcnn_class_sum, loss_mrcnn_bbox_sum, loss_mrcnn_mask_sum, loss_semantic_sum
                    else:
                        step += 1

                elif self.training_layers == 'saliency':
                    # Run object detection
                    predict_input = [images, image_metas]
                    influence_preds = self.predict_front(predict_input, mode='training', limit="saliency")

                    influence_loss = compute_saliency_loss(influence_preds, gt_influence_map)
                    loss = influence_loss

                    printProgressBar(step + 1, steps, prefix="\t{}/{}".format(step + 1, steps),
                                     suffix="Complete - influence_loss: {:.5f}".format(influence_loss.data.cpu()[0]),
                                     length=10)

                    loss_sum += loss.data.cpu()[0] / steps
                    loss_influence_sum += influence_loss.data.cpu()[0] / steps

                    # Break after 'steps' steps
                    if step == steps - 1:
                        return loss_sum, loss_influence_sum
                    else:
                        step += 1
                elif self.training_layers == "semantic":
                    # Run object detection
                    predict_input = [images, image_metas]
                    semantic_label = self.predict_front(predict_input, mode='training',limit="semantic")
                    semantic_loss = compute_semantic_loss(semantic_label, gt_semantic_label)
                    loss = semantic_loss

                    printProgressBar(step + 1, steps, prefix="\t{}/{}".format(step + 1, steps),
                                     suffix="Complete - loss: {:.5f} - semantic_loss: {:.5f} ".format(loss.data.cpu()[0], semantic_loss.data.cpu()[0]),length=10)

                    loss_sum += loss.data.cpu()[0] / steps
                    loss_semantic_sum += semantic_loss.data.cpu()[0] / steps

                    # Break after 'steps' steps
                    if step == steps - 1:
                        return loss_sum, loss_semantic_sum
                    else:
                        step += 1
                elif self.training_layers == "new_heads":
                    # Run object detection
                    predict_input = [images, image_metas]
                    semantic_label, influence_preds = self.predict_front(predict_input, mode='training',
                                                                         limit="new_heads")
                    influence_loss = compute_saliency_loss(influence_preds, gt_influence_map)
                    semantic_loss = compute_semantic_loss(semantic_label, gt_semantic_label)
                    loss = influence_loss + semantic_loss

                    printProgressBar(step + 1, steps, prefix="\t{}/{}".format(step + 1, steps),
                                     suffix="Complete - loss: {:.5f} - semantic_loss: {:.5f} - influence_loss: {:.5f}".format(
                                         loss.data.cpu()[0], semantic_loss.data.cpu()[0], influence_loss.data.cpu()[0]),
                                     length=10)

                    loss_sum += loss.data.cpu()[0] / steps
                    loss_semantic_sum += semantic_loss.data.cpu()[0] / steps
                    loss_influence_sum += influence_loss.data.cpu()[0] / steps

                    # Break after 'steps' steps
                    if step == steps - 1:
                        return loss_sum, loss_semantic_sum, loss_influence_sum
                    else:
                        step += 1
                elif self.training_layers == "sesa":
                    # Run object detection
                    predict_input = [images, image_metas, gt_class_ids, gt_boxes, gt_masks]
                    rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask, semantic_label = self.predict_front(
                        predict_input, mode='training', limit="segmentation")
                    predict_input = [images, image_metas]
                    influence_preds = self.predict_front(predict_input, mode='training', limit="saliency")

                    # Compute losses
                    rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss, semantic_loss = compute_losses_PFPN(
                        rpn_match, rpn_bbox, rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits,
                        target_deltas, mrcnn_bbox, target_mask, mrcnn_mask,
                        gt_semantic_label, semantic_label)
                    influence_loss = compute_saliency_loss(influence_preds, gt_influence_map)
                    loss = rpn_class_loss + rpn_bbox_loss + mrcnn_class_loss + mrcnn_bbox_loss + mrcnn_mask_loss + semantic_loss + influence_loss

                    # Progress
                    printProgressBar(step + 1, steps, prefix="\t{}/{}".format(step + 1, steps),
                                     suffix="Complete - loss: {:.5f} - rpn_class_loss: {:.5f} - rpn_bbox_loss: {:.5f} - mrcnn_class_loss: {:.5f} - mrcnn_bbox_loss: {:.5f} - mrcnn_mask_loss: {:.5f} - semantic_loss: {:.5f} - influence_loss: {:.5f}".format(
                                         loss.data.cpu()[0], rpn_class_loss.data.cpu()[0], rpn_bbox_loss.data.cpu()[0],
                                         mrcnn_class_loss.data.cpu()[0], mrcnn_bbox_loss.data.cpu()[0],
                                         mrcnn_mask_loss.data.cpu()[0], semantic_loss.data.cpu()[0],influence_loss.data.cpu()[0], length=10))

                    # Statistics
                    loss_sum += loss.data.cpu()[0] / steps
                    loss_rpn_class_sum += rpn_class_loss.data.cpu()[0] / steps
                    loss_rpn_bbox_sum += rpn_bbox_loss.data.cpu()[0] / steps
                    loss_mrcnn_class_sum += mrcnn_class_loss.data.cpu()[0] / steps
                    loss_mrcnn_bbox_sum += mrcnn_bbox_loss.data.cpu()[0] / steps
                    loss_mrcnn_mask_sum += mrcnn_mask_loss.data.cpu()[0] / steps
                    loss_semantic_sum += semantic_loss.data.cpu()[0] / steps
                    loss_influence_sum += influence_loss.data.cpu()[0] / steps

                    # Break after 'steps' steps
                    if step == steps - 1:
                        return loss_sum, loss_rpn_class_sum, loss_rpn_bbox_sum, loss_mrcnn_class_sum, loss_mrcnn_bbox_sum, loss_mrcnn_mask_sum, loss_semantic_sum, loss_influence_sum
                    else:
                        step += 1
                elif self.training_layers=='selection':
                    # Run object detection
                    predict_input = [images, image_metas, gt_class_ids, gt_boxes, gt_masks, gt_segmentation, gt_image_instances]
                    predictions, pair_labels, labels = self.predict_front(predict_input, mode="training", limit="selection")
                    loss_func = nn.MSELoss()
                    interest_loss = loss_func(predictions, pair_labels)
                    loss = interest_loss

                    printProgressBar(step + 1, steps, prefix="\t{}/{}".format(step + 1, steps),
                                     suffix="Complete - interest_loss: {:.5f}".format(interest_loss.data.cpu()[0]),
                                     length=10)

                    loss_sum += loss.data.cpu()[0] / steps
                    loss_interest_sum += interest_loss.data.cpu()[0] / steps

                    # Break after 'steps' steps
                    if step == steps - 1:
                        return loss_sum, loss_interest_sum
                    else:
                        step += 1

                elif self.training_layers == 'all':
                    # Run object detection
                    predict_input = [images, image_metas, gt_class_ids, gt_boxes, gt_masks, gt_segmentation, gt_image_instances]
                    rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask, thing_detections, thing_masks, \
                    semantic_labels,influence_preds, predictions, pair_labels = self.predict_front(predict_input, mode='training', limit="all")
                    influence_map = influence_preds[4]

                    instance_class_ids, instance_boxes, instance_masks, interest_label, semantic_segment = self.predict_back(
                        image_metas, thing_detections, thing_masks, semantic_labels, influence_map)

                    loss_func = nn.MSELoss()
                    interest_loss = loss_func(predictions, pair_labels)

                    # Compute losses
                    rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss, semantic_loss, influence_loss = compute_losses_CIN(
                        rpn_match, rpn_bbox, rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits,
                        target_deltas, mrcnn_bbox, target_mask, mrcnn_mask,
                        gt_semantic_label, semantic_labels, gt_influence_map, influence_preds)

                    loss = rpn_class_loss + rpn_bbox_loss + mrcnn_class_loss + mrcnn_bbox_loss + mrcnn_mask_loss + semantic_loss + influence_loss + interest_loss

                    # Statistics
                    loss_sum += loss.data.cpu()[0] / steps
                    loss_rpn_class_sum += rpn_class_loss.data.cpu()[0] / steps
                    loss_rpn_bbox_sum += rpn_bbox_loss.data.cpu()[0] / steps
                    loss_mrcnn_class_sum += mrcnn_class_loss.data.cpu()[0] / steps
                    loss_mrcnn_bbox_sum += mrcnn_bbox_loss.data.cpu()[0] / steps
                    loss_mrcnn_mask_sum += mrcnn_mask_loss.data.cpu()[0] / steps
                    loss_semantic_sum += semantic_loss.data.cpu()[0] / steps
                    loss_influence_sum += influence_loss.data.cpu()[0] / steps
                    loss_interest_sum += interest_loss.data.cpu()[0] / steps

                    # Break after 'steps' steps
                    if step == steps - 1:
                        return loss_sum, loss_rpn_class_sum, loss_rpn_bbox_sum, loss_mrcnn_class_sum, loss_mrcnn_bbox_sum, loss_mrcnn_mask_sum, loss_semantic_sum, loss_influence_sum, loss_interest_sum
                    else:
                        step += 1
                else:
                    pass
            except Exception as e:
                print("Error - "+str(step))
                print(e)

    def detect(self, images, limit="segmentation"):
        """Runs the detection pipeline.

        images: List of images, potentially of different sizes.

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        """

        # Mold inputs to format expected by the neural network
        molded_images, image_metas, windows = self.mold_inputs(images)

        # Convert images to torch tensor
        molded_images = torch.from_numpy(molded_images.transpose(0, 3, 1, 2)).float()

        # To GPU
        if self.config.GPU_COUNT:
            molded_images = molded_images.cuda()

        # Wrap in variable
        molded_images = Variable(molded_images, volatile=True)

        if limit=="saliency":
            influence_map = self.predict_front([molded_images, image_metas], mode='inference', limit=limit)
            influence_map=influence_map.squeeze(0).squeeze(0).data.cpu().numpy()
            # influence_map=np.where(influence_map*255>100,1,0)
            #plt.figure()
            #plt.imshow(influence_map)
            #plt.show()
            image_id, image_shape, window = image_metas[0][0],image_metas[0][1:4],image_metas[0][4:8]
            top_pad, left_pad, top_pad_h, left_pad_w = window[0], window[1], window[2], window[3]
            influence_map = resize_influence_map(influence_map,(self.config.IMAGE_SIZE,self.config.IMAGE_SIZE))
            influence_map = influence_map[top_pad:top_pad_h, left_pad:left_pad_w]
            influence_map = scipy.misc.imresize(influence_map, image_shape[:2], interp='bilinear')
            return influence_map
        elif limit == "segmentation":
            thing_detections, thing_masks, semantic_labels = self.predict_front(
                [molded_images, image_metas], mode='inference',limit="segmentation") # [x,5],[x,28,28,81]
            semantic_segment, stuff_detections, stuff_masks = generate_stuff(semantic_labels)  # [y, 5],[y, 500, 500]

            results = []
            # image=images[0]
            if len(thing_detections.shape) > 0 and len(stuff_detections.shape)>0:
                thing_detections = thing_detections.data.cpu().numpy()
                thing_masks = thing_masks.permute(0, 1, 3, 4, 2).data.cpu().numpy()
                thing_detections = thing_detections.squeeze(0)  # [x,6]
                thing_masks = thing_masks.squeeze(0)  # [x,28,28,81]
                stuff_detections = stuff_detections.data.cpu().numpy()  # [y,5]
                stuff_masks = stuff_masks.data.cpu().numpy()  # [y,500,500]

                image_id, image_shape, window = image_metas[0][0],image_metas[0][1:4],image_metas[0][4:8]
                top_pad, left_pad, top_pad_h, left_pad_w = window[0], window[1], window[2], window[3]

                semantic_segment = resize_semantic_label(semantic_segment,(self.config.IMAGE_SIZE,self.config.IMAGE_SIZE))
                semantic_segment = semantic_segment[top_pad:top_pad_h, left_pad:left_pad_w]
                semantic_segment = scipy.ndimage.zoom(semantic_segment, [image_shape[0]/(top_pad_h-top_pad),image_shape[1]/(left_pad_w-left_pad)] , mode='nearest', order=0)

                thing_class_ids, thing_boxes, thing_masks_unmold, thing_scores = filter_thing_masks(thing_detections,thing_masks,image_shape,window)
                stuff_class_ids, stuff_boxes, stuff_masks_unmold = filter_stuff_masks(stuff_detections, stuff_masks,image_shape,window)

                results.append({
                    "thing_boxes": thing_boxes,
                    "thing_class_ids": thing_class_ids,
                    "thing_scores": thing_scores,
                    "thing_masks": thing_masks_unmold,
                    "stuff_boxes": stuff_boxes,
                    "stuff_class_ids": stuff_class_ids,
                    "stuff_masks": stuff_masks_unmold,
                    "semantic_segment": semantic_segment
                })
            elif len(thing_detections.shape) == 0 and len(stuff_detections.shape)>0:
                stuff_detections = stuff_detections.data.numpy()  # [y,5]
                stuff_masks = stuff_masks.data.numpy()  # [y,500,500]

                results.append({
                    "stuff_rois": stuff_detections,
                    "stuff_masks": stuff_masks,
                    "segment": semantic_segment
                })
            elif len(thing_detections.shape) > 0 and len(stuff_detections.shape)==0:
                thing_detections = thing_detections.data.cpu().numpy()
                thing_masks = thing_masks.permute(0, 1, 3, 4, 2).data.cpu().numpy()
                thing_detections = thing_detections.squeeze(0)  # [x,6]
                thing_masks = thing_masks.squeeze(0)  # [x,28,28,81]

                image_id, image_shape, window = image_metas[0][0],image_metas[0][1:4],image_metas[0][4:8]
                top_pad, left_pad, top_pad_h, left_pad_w = window[0], window[1], window[2], window[3]

                semantic_segment = resize_semantic_label(semantic_segment,(self.config.IMAGE_SIZE,self.config.IMAGE_SIZE))
                semantic_segment = semantic_segment[top_pad:top_pad_h, left_pad:left_pad_w]
                semantic_segment = scipy.ndimage.zoom(semantic_segment, [image_shape[0]/(top_pad_h-top_pad),image_shape[1]/(left_pad_w-left_pad)] , mode='nearest', order=0)

                thing_class_ids, thing_boxes, thing_masks_unmold, thing_scores = filter_thing_masks(thing_detections,thing_masks,image_shape,window)

                results.append({
                    "thing_boxes": thing_boxes,
                    "thing_class_ids": thing_class_ids,
                    "thing_scores": thing_scores,
                    "thing_masks": thing_masks_unmold,
                    "semantic_segment": semantic_segment
                })
            else:
                results.append({
                    "semantic_segment": semantic_segment
                })
            return results
        elif limit == "sesa":
            thing_detections, thing_masks, semantic_labels,influence_map = self.predict_front([molded_images, image_metas], mode='inference',limit=limit) # [x,5],[x,28,28,81]
            semantic_segment, stuff_detections, stuff_masks = generate_stuff(semantic_labels)  # [y, 5],[y, 500, 500] [1, 134, 512, 512]

            results = []
            # image=images[0]
            if len(thing_detections.shape) > 0:
                thing_detections = thing_detections.data.cpu().numpy()
                thing_masks = thing_masks.permute(0, 1, 3, 4, 2).data.cpu().numpy()
                thing_detections = thing_detections.squeeze(0)  # [x,6]
                thing_masks = thing_masks.squeeze(0)  # [x,28,28,81]
                stuff_detections = stuff_detections.data.cpu().numpy()  # [y,5]
                stuff_masks = stuff_masks.data.cpu().numpy()  # [y,500,500]
                influence_map = influence_map.squeeze(0).squeeze(0).data.cpu().numpy()  # [128,128]'

                image_id, image_shape, window = image_metas[0][0],image_metas[0][1:4],image_metas[0][4:8]
                top_pad, left_pad, top_pad_h, left_pad_w = window[0], window[1], window[2], window[3]
                influence_map = resize_influence_map(influence_map,(self.config.IMAGE_SIZE,self.config.IMAGE_SIZE))
                semantic_segment = resize_semantic_label(semantic_segment,(self.config.IMAGE_SIZE,self.config.IMAGE_SIZE))
                influence_map = influence_map[top_pad:top_pad_h, left_pad:left_pad_w]
                semantic_segment = semantic_segment[top_pad:top_pad_h, left_pad:left_pad_w]
                semantic_segment = scipy.ndimage.zoom(semantic_segment, [image_shape[0]/(top_pad_h-top_pad),image_shape[1]/(left_pad_w-left_pad)] , mode='nearest', order=0)
                influence_map = scipy.misc.imresize(influence_map, image_shape[:2], interp='bilinear')

                thing_class_ids, thing_boxes, thing_masks_unmold, thing_scores = filter_thing_masks(thing_detections,thing_masks,image_shape,window)
                stuff_class_ids, stuff_boxes, stuff_masks_unmold = filter_stuff_masks(stuff_detections, stuff_masks,image_shape,window)

                results.append({
                    "thing_boxes": thing_boxes,
                    "thing_class_ids": thing_class_ids,
                    "thing_scores": thing_scores,
                    "thing_masks": thing_masks_unmold,
                    "stuff_boxes": stuff_boxes,
                    "stuff_class_ids": stuff_class_ids,
                    "stuff_masks": stuff_masks_unmold,
                    "semantic_segment": semantic_segment,
                    'influence_map': influence_map
                })
            else:
                influence_map = influence_map.squeeze(0).squeeze(0).data.cpu().numpy()  # [128,128]
                stuff_detections = stuff_detections.data.numpy()  # [y,5]
                stuff_masks = stuff_masks.data.numpy()  # [y,500,500]

                results.append({
                    "stuff_rois": stuff_detections,
                    "stuff_masks": stuff_masks,
                    "segment": semantic_segment,
                    'influence': influence_map
                })
            return results
        elif limit=="selection":
            # Run object detection [b,x,c,h,w]
            predictions, segments_info = self.predict_front([molded_images, image_metas], mode='inference', limit=limit)
            num = len(segments_info)  # the num of the instances
            prediction_list = []
            # predictions = predictions.cpu().detach().numpy()
            for i in range(0, num):
                avg = np.sum(predictions[i * num:(i + 1) * num]) / num
                prediction_list.append(avg)
            idx = 0
            CIRNN_pred_dict = {}
            prediction_list = maxminnorm(prediction_list)
            for segment_info_id in segments_info:
                if prediction_list[idx] > self.config.SELECTION_THRESHOLD:
                    CIRNN_pred_dict[segment_info_id] = segments_info[segment_info_id]
                idx += 1
            return CIRNN_pred_dict

    def predict_front(self, input, mode, limit=""):
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

        # ！！！！！！！！！！！！！！！！！！！ START ！！！！！！！！！！！！！！！！！！！
        # print("Image:")
        # print(molded_images.shape)
        # Feature extraction
        # for param in self.named_parameters():
        #     print(param[0]+" "+str(param[1].shape)+" "+str(param[1].requires_grad))
        [c1_out, c2_out, c3_out, c4_out, c5_out] = self.resnet(molded_images)

        # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！ ResNet
        if limit == "saliency":
            influence_preds = self.saliency(c1_out, c2_out, c3_out, c4_out, c5_out)  # (1,4,128,128)
            if mode == "training":
                return influence_preds
            elif mode == "inference":
                return influence_preds[4]
        # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！ Saliency

        [p2_out, p3_out, p4_out, p5_out, p6_out] = self.fpn(c1_out, c2_out, c3_out, c4_out, c5_out)

        # Note that P6 is used in RPN, but not in the classifier heads.
        rpn_feature_maps = [p2_out, p3_out, p4_out, p5_out, p6_out]
        mrcnn_feature_maps = [p2_out, p3_out, p4_out, p5_out]

        # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！ FPN

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

        # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！ RPN & ROIAlign

        semantic_segment = self.semantic(mrcnn_feature_maps)
        # print("Semantic:")
        # print(semantic_segment.shape) # (1,134,500,500)
        # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！ SEMANTIC
        if mode == "training" and limit == "semantic":
            return semantic_segment
        if mode == "training" and limit == "new_heads":
            influence_preds = self.saliency(c1_out, c2_out, c3_out, c4_out, c5_out)
            return [semantic_segment, influence_preds]

        if mode == 'training':
            gt_class_ids = input[2]
            gt_boxes = input[3]
            gt_masks = input[4]

            # Normalize coordinates
            h, w = self.config.IMAGE_SHAPE[:2]
            scale = Variable(torch.from_numpy(np.array([h, w, h, w])).float(), requires_grad=False)
            if self.config.GPU_COUNT:
                scale = scale.cuda()

            gt_boxes = gt_boxes / scale

            # Generate detection targets
            # Subsamples proposals and generates target outputs for training
            # Note that proposal class IDs, gt_boxes, and gt_masks are zero
            # padded. Equally, returned rois and targets are zero padded.
            rois, target_class_ids, target_deltas, target_mask = \
                detection_target_layer(rpn_rois, gt_class_ids, gt_boxes, gt_masks, self.config)

            if not rois.size():
                mrcnn_class_logits = Variable(torch.FloatTensor())
                mrcnn_class = Variable(torch.IntTensor())
                mrcnn_bbox = Variable(torch.FloatTensor())
                mrcnn_mask = Variable(torch.FloatTensor())
                if self.config.GPU_COUNT:
                    mrcnn_class_logits = mrcnn_class_logits.cuda()
                    mrcnn_class = mrcnn_class.cuda()
                    mrcnn_bbox = mrcnn_bbox.cuda()
                    mrcnn_mask = mrcnn_mask.cuda()
            else:
                mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.classifier(mrcnn_feature_maps, rois)
                mrcnn_mask = self.mask(mrcnn_feature_maps, rois)
            # print(mrcnn_class_logits.shape) # x,91
            # print(mrcnn_class.shape) # x,91
            # print(mrcnn_bbox.shape)  # x,91,4 [y1,x1,y2,x2]
            # print(mrcnn_mask.shape)  # x,91,28,28
            # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！ THING

            if limit == "segmentation":
                return [rpn_class_logits, rpn_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox,
                        target_mask, mrcnn_mask, semantic_segment]
            elif limit == "sesa":
                influence_preds = self.saliency(c1_out, c2_out, c3_out, c4_out, c5_out)
                return [rpn_class_logits, rpn_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox,
                        target_mask, mrcnn_mask, semantic_segment, influence_preds]
            elif limit == 'selection':
                # TODO_wdd: add the ciedn model
                gt_segmentation = input[5]
                image_info = input[6]
                gt_instance_dict = image_info['instances']
                influence_map = self.saliency(c1_out, c2_out, c3_out, c4_out, c5_out)[4]  # (1,1,128,128)
                mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.classifier(mrcnn_feature_maps, rpn_rois)
                detections = detection_layer(self.config, rpn_rois, mrcnn_class, mrcnn_bbox, image_metas)  # 34,6

                # Add back batch dimension
                if len(detections.shape)>1:
                    detections = detections.unsqueeze(0)  # [1, x, 6]
                mrcnn_mask = mrcnn_mask.unsqueeze(0)  # [1, x, 81, 28, 28]

                # detect 中 predict_front 的结果
                # detections, mrcnn_mask, semantic_segment, influence_map # [1, x, 6],[1, x, 81, 28, 28],[1,134,500,500],[1,1,128,128]
                # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！ predict_front mode = inference

                result = self.detect_objects(detections, mrcnn_mask, semantic_segment, influence_map, image_metas)[0]
                # result detect 中的result
                # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！ detect

                semantic_labels, influence_map, panoptic_result, segments_info = self.predict_segment(result, image_metas)
                # semantic_labels, influence_map, panoptic_result, segments_info
                # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！ CIN_predict_alls

                instance_pred_gt_dict, instance_gt_pred_dict = self.map_instance_to_gt(gt_instance_dict, segments_info,
                                                                                  gt_segmentation, panoptic_result)

                ioi_image_dict = self.generate_image_dict(image_info, instance_pred_gt_dict, segments_info)
                # add the labeled attribute in the segments_info
                # ioi_image_dict
                # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！ middle_process

                ioi_segments_info = ioi_image_dict['segments_info']
                if torch.is_tensor(image_metas[0][1:4]):
                    image_shape = image_metas[0][1:4].numpy().astype('int32')
                else:
                    image_shape = image_metas[0][1:4].astype('int32')

                instance_groups, boxes, class_ids, labels, pair_label = self.construct_dataset(semantic_labels,
                                                                                               influence_map,
                                                                                               panoptic_result,
                                                                                               ioi_segments_info,
                                                                                               image_shape,
                                                                                               mode)
                # instance_groups, boxes, class_ids, labels,pair_label
                # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！ dataset
                if self.config.GPU_COUNT:
                    instance_groups = Variable(torch.unsqueeze(torch.from_numpy(instance_groups), 0)).float().cuda()
                    boxes = Variable(torch.unsqueeze(torch.from_numpy(boxes), 0)).float().cuda()
                    class_ids = Variable(torch.unsqueeze(torch.from_numpy(class_ids), 0)).float().cuda()
                    gt = Variable(torch.unsqueeze(torch.from_numpy(labels), 0)).float().cuda()
                    labels = Variable(torch.unsqueeze(torch.from_numpy(labels), 0)).float().cuda()
                    pair_label = Variable(torch.from_numpy(pair_label)).float().cuda().squeeze(0)

                predictions = self.ciedn(instance_groups).squeeze(1)
                # print("instance_groups", instance_groups.shape)
                # print("labels", labels.shape)
                # print("pair_label", pair_label.shape)
                # print("predictions", len(predictions))
                return predictions, pair_label, labels
            elif limit == "all":
                gt_segmentation = input[5]
                image_info = input[6]
                gt_instance_dict = image_info['instances']
                influence_map = self.saliency(c1_out, c2_out, c3_out, c4_out, c5_out)[4]  # (1,1,128,128)
                mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.classifier(mrcnn_feature_maps, rpn_rois)
                detections = detection_layer(self.config, rpn_rois, mrcnn_class, mrcnn_bbox, image_metas)  # 34,6

                # Add back batch dimension
                detections = detections.unsqueeze(0)  # [1, x, 6]
                mrcnn_mask = mrcnn_mask.unsqueeze(0)  # [1, x, 81, 28, 28]

                # detect 中 predict_front 的结果
                # detections, mrcnn_mask, semantic_segment, influence_map # [1, x, 6],[1, x, 81, 28, 28],[1,134,500,500],[1,1,128,128]
                # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！ predict_front mode = inference

                result = self.detect_objects(detections, mrcnn_mask, semantic_segment, influence_map, image_metas)[0]
                # result detect 中的result
                # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！ detect

                semantic_labels, influence_map, panoptic_result, segments_info = self.predict_segment(result,
                                                                                                      image_metas)
                # semantic_labels, influence_map, panoptic_result, segments_info
                # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！ CIN_predict_alls

                instance_pred_gt_dict, instance_gt_pred_dict = self.map_instance_to_gt(gt_instance_dict, segments_info,
                                                                                       gt_segmentation, panoptic_result)

                ioi_image_dict = self.generate_image_dict(image_info, instance_pred_gt_dict, segments_info)
                # add the labeled attribute in the segments_info
                # ioi_image_dict
                # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！ middle_process

                ioi_segments_info = ioi_image_dict['segments_info']
                if torch.is_tensor(image_metas[0][1:4]):
                    image_shape = image_metas[0][1:4].numpy().astype('int32')
                else:
                    image_shape = image_metas[0][1:4].astype('int32')

                instance_groups, boxes, class_ids, labels, pair_label = self.construct_dataset(semantic_labels,
                                                                                               influence_map,
                                                                                               panoptic_result,
                                                                                               ioi_segments_info,
                                                                                               image_shape,
                                                                                               mode)
                # instance_groups, boxes, class_ids, labels,pair_label
                # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！ dataset
                if self.config.GPU_COUNT:
                    instance_groups = Variable(torch.unsqueeze(torch.from_numpy(instance_groups), 0)).float().cuda()
                    boxes = Variable(torch.unsqueeze(torch.from_numpy(boxes), 0)).float().cuda()
                    class_ids = Variable(torch.unsqueeze(torch.from_numpy(class_ids), 0)).float().cuda()
                    gt = Variable(torch.unsqueeze(torch.from_numpy(labels), 0)).float().cuda()
                    labels = Variable(torch.unsqueeze(torch.from_numpy(labels), 0)).float().cuda()
                    pair_label = Variable(torch.from_numpy(pair_label)).float().cuda().squeeze(0)

                predictions = self.ciedn(instance_groups).squeeze(1)
                return [rpn_class_logits, rpn_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox,
                            target_mask, mrcnn_mask, thing_detections, thing_masks, semantic_segment, influence_preds,\
                        predictions, pair_label]

        elif mode == 'inference':
            if limit == "segmentation":
                mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.classifier(mrcnn_feature_maps, rpn_rois)
                detections = detection_layer(self.config, rpn_rois, mrcnn_class, mrcnn_bbox, image_metas)  # 34,6

                h, w = self.config.IMAGE_SHAPE[:2]
                scale = Variable(torch.from_numpy(np.array([h, w, h, w])).float(), requires_grad=False)
                if self.config.GPU_COUNT:
                    scale = scale.cuda()
                # print(detections.shape)
                detection_boxes = detections[:, :4] / scale

                # Add back batch dimension
                detection_boxes = detection_boxes.unsqueeze(0)

                # Create masks for detections
                mrcnn_mask = self.mask(mrcnn_feature_maps, detection_boxes)  # x, 134, 28, 28

                # Add back batch dimension
                detections = detections.unsqueeze(0)  # [1, x, 6]
                mrcnn_mask = mrcnn_mask.unsqueeze(0)  # [1, x, 81, 28, 28]

                # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！ THING

                return [detections, mrcnn_mask,
                        semantic_segment]  # , influence_map] # [1, x, 6],[1, x, 81, 28, 28],[1,134,500,500],[1,1,128,128]
            elif limit == "sesa":
                influence_map = self.saliency(c1_out, c2_out, c3_out, c4_out, c5_out)[4]  # (1,1,128,128)
                # print("Influence:")
                # print(influence_map.shape)  # (1,1,128,128)

                # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！ Saliency

                # Network Heads
                # Proposal classifier and BBox regressor heads
                mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.classifier(mrcnn_feature_maps, rpn_rois)

                detections = detection_layer(self.config, rpn_rois, mrcnn_class, mrcnn_bbox, image_metas)  # 34,6

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

                # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！ THING

                return [detections, mrcnn_mask, semantic_segment,
                        influence_map]  # [1, x, 6],[1, x, 81, 28, 28],[1,134,500,500],[1,1,128,128]

            elif limit == 'selection':
                # TODO_wdd: 添加selection分支代码
                influence_map = self.saliency(c1_out, c2_out, c3_out, c4_out, c5_out)[4]  # (1,1,128,128)
                mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.classifier(mrcnn_feature_maps, rpn_rois)

                detections = detection_layer(self.config, rpn_rois, mrcnn_class, mrcnn_bbox, image_metas)  # 34,6

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
                # detections, mrcnn_mask, semantic_segment, influence_map

                result = self.detect_objects(detections, mrcnn_mask, semantic_segment, influence_map, image_metas)[0]
                # visualize.display_instances(img, result['thing_boxes'], result['thing_masks'], result['thing_class_ids'], class_names)
                semantic_labels, influence_map, panoptic_result, segments_info = self.predict_segment(result, image_metas)

                image_shape = image_metas[0][1:4].astype('int32') # 420,640,3
                instance_groups = self.construct_dataset(semantic_labels,
                                                         influence_map,
                                                         panoptic_result,
                                                         segments_info,
                                                         image_shape,
                                                         mode)
                if self.config.GPU_COUNT:
                    instance_groups = Variable(torch.unsqueeze(torch.from_numpy(instance_groups), 0)).float().cuda() # 1,19,2,56,56

                predictions = self.ciedn(instance_groups).squeeze(1).data.cpu().numpy()
                return predictions, segments_info

    def predict_back(self,image_metas,thing_detections,thing_masks,semantic_segment,influence_map):
        img=scipy.misc.imread("/media/yuf/Data/magus/database/COCO/train2017/"+str(int(image_metas[0][0])).zfill(12)+".jpg")
        plt.figure()
        plt.imshow(img)
        plt.show()

        semantic_segment, stuff_detections, stuff_masks = generate_stuff(semantic_segment)  # [y, 5],[y, 500, 500]
        # semantic_segment = self.expand_semantic_predict(semantic_segment, image_metas)

        if self.config.GPU_COUNT:
            stuff_detections = stuff_detections.cuda()
            stuff_masks = stuff_masks.cuda()

        thing_detections = thing_detections.data.cpu().numpy()
        thing_masks = thing_masks.permute(0, 1, 3, 4, 2).data.cpu().numpy()
        thing_detections = thing_detections.squeeze(0)  # [x,6]
        thing_masks = thing_masks.squeeze(0)  # [x,28,28,81]
        stuff_detections = stuff_detections.data.cpu().numpy()  # [y,5]
        stuff_masks = stuff_masks.data.cpu().numpy()  # [y,500,500]
        influence_map = influence_map.squeeze(0).squeeze(0).data.cpu().numpy()  # [128,128]

        instance_piece_groups, instance_class_ids, instance_boxes, instance_masks = extract_piece_group(thing_detections,
                                                                                                 thing_masks,
                                                                                                 stuff_detections,
                                                                                                 stuff_masks,
                                                                                                 influence_map,
                                                                                                 semantic_segment)
        # print(instance_piece_groups.shape)
        # print(instance_class_ids.shape)
        # print(instance_boxes.shape)
        # print(instance_masks.shape)

        instance_piece_groups=Variable(torch.from_numpy(instance_piece_groups)).unsqueeze(0)
        instance_class_ids=Variable(torch.from_numpy(instance_class_ids)).unsqueeze(0)
        instance_boxes=Variable(torch.from_numpy(instance_boxes)).unsqueeze(0)
        if self.config.GPU_COUNT:
            instance_piece_groups = instance_piece_groups.cuda()
            instance_class_ids = instance_class_ids.cuda()
            instance_boxes = instance_boxes.cuda()

        # interest_label = self.ciedn(instance_piece_groups,instance_boxes,instance_class_ids).squeeze(2)
        # TODO
        interest_label = None
        return instance_class_ids, instance_boxes, instance_masks,interest_label,semantic_segment

    def mold_inputs(self, images):
        """Takes a list of images and modifies them to the format expected
        as an input to the neural network.
        images: List of image matricies [height,width,depth]. Images can have
            different sizes.

        Returns 3 Numpy matricies:
        molded_images: [N, h, w, 3]. Images resized and normalized.
        image_metas: [N, length of meta data]. Details about each image.
        windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
            original image (padding excluded).
        """
        molded_images = []
        image_metas = []
        windows = []
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
            windows.append(window)
            image_metas.append(image_meta)
        # Pack into arrays
        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)
        windows = np.stack(windows)
        return molded_images, image_metas, windows

    def unmold_detections(self, detections, mrcnn_mask, image_shape, window):
        """Reformats the detections of one image from the format of the neural
        network output to a format suitable for use in the rest of the
        application.

        detections: [N, (y1, x1, y2, x2, class_id, score)]
        mrcnn_mask: [N, height, width, num_classes]
        image_shape: [height, width, depth] Original size of the image before resizing
        window: [y1, x1, y2, x2] Box in the image where the real image is
                excluding the padding.

        Returns:
        boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
        class_ids: [N] Integer class IDs for each bounding box
        scores: [N] Float probability scores of the class_id
        masks: [height, width, num_instances] Instance masks
        """

        # How many detections do we have?
        # Detections array is padded with zeros. Find the first class_id == 0.
        zero_ix = np.where(detections[:, 4] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

        # Extract boxes, class_ids, scores, and class-specific masks
        boxes = detections[:N, :4]
        class_ids = detections[:N, 4].astype(np.int32)
        scores = detections[:N, 5]
        masks = mrcnn_mask[np.arange(N), :, :, class_ids]

        # Compute scale and shift to translate coordinates to image domain.
        h_scale = image_shape[0] / (window[2] - window[0]) # h_ori_image/h_box_image
        w_scale = image_shape[1] / (window[3] - window[1]) # w_ori_image/w_box_image
        scale = min(h_scale, w_scale)
        shift = window[:2]  # y, x
        scales = np.array([scale, scale, scale, scale])
        shifts = np.array([shift[0], shift[1], shift[0], shift[1]])

        # Translate bounding boxes to image domain
        boxes = np.multiply(boxes - shifts, scales).astype(np.int32)

        # Filter out detections with zero area. Often only happens in early
        # stages of training when the network weights are still a bit random.
        exclude_ix = np.where(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            masks = np.delete(masks, exclude_ix, axis=0)
            N = class_ids.shape[0]

        # Resize masks to original image size and set boundary threshold.
        full_masks = []
        for i in range(N):
            # Convert neural network mask to full size mask
            full_mask = utils.unmold_mask(masks[i], boxes[i], image_shape)
            full_masks.append(full_mask)
        full_masks = np.stack(full_masks, axis=-1) \
            if full_masks else np.empty((0,) + masks.shape[1:3])

        return boxes, class_ids, scores, full_masks

    def expand_semantic_predict(self,semantic_segment, image_metas):
        window = image_metas[0][4:]
        window = (window * 500 / self.config.IMAGE_SIZE).astype(np.int32)
        semantic_segment = semantic_segment[window[0]:window[2], window[1]:window[3]]
        semantic_segment = scipy.n(semantic_segment, (500, 500), interp="nearest")###########!!!!!!wrong -> zoom
        return semantic_segment

    def detect_objects(self, detections, mrcnn_mask, semantic_segment, influence_map, image_metas):
        semantic_segment, stuff_detections, stuff_masks = generate_stuff(semantic_segment)  # [y, 5],[y, 500, 500] [1, 134, 512, 512]
        result = []
        if len(detections.shape) > 0:
            detections = detections.data.cpu().numpy()
            mrcnn_masks = mrcnn_mask.permute(0, 1, 3, 4, 2).data.cpu().numpy()
            if len(detections.shape) > 1:
                detections = detections.squeeze(0)  # [x,6]
            mrcnn_masks = mrcnn_masks.squeeze(0)  # [x,28,28,81]
            stuff_detections = stuff_detections.data.cpu().numpy()  # [y,5]
            stuff_masks = stuff_masks.data.cpu().numpy()  # [y,500,500]
            influence_map = influence_map.squeeze(0).squeeze(0).data.cpu().numpy()  # [128,128]

            image_id, image_shape, window = image_metas[0][0],image_metas[0][1:4],image_metas[0][4:8]
            # image_shape = [ int(i) for i in image_shape.data.cpu().numpy()]
            if type(image_shape) is np.ndarray:
                image_shape = image_shape.astype('int32')
                window = window.astype('int32')
            else:
                image_shape = image_shape.data.cpu().numpy().astype('int32')
                window = window.data.cpu().numpy().astype('int32')
            top_pad, left_pad, top_pad_h, left_pad_w = window[0], window[1], window[2], window[3]
            influence_map = resize_influence_map(influence_map,(self.config.IMAGE_SIZE,self.config.IMAGE_SIZE))
            semantic_segment = resize_semantic_label(semantic_segment,(self.config.IMAGE_SIZE,self.config.IMAGE_SIZE))

            influence_map = influence_map[top_pad:top_pad_h, left_pad:left_pad_w]
            semantic_segment = semantic_segment[top_pad:top_pad_h, left_pad:left_pad_w]
            semantic_segment = scipy.ndimage.zoom(semantic_segment, [image_shape[0]/(top_pad_h-top_pad),image_shape[1]/(left_pad_w-left_pad)] , mode='nearest', order=0)
            influence_map = scipy.misc.imresize(influence_map, image_shape[:2], interp='bilinear')

            thing_class_ids, thing_boxes, thing_masks_unmold, thing_scores = filter_thing_masks(detections,mrcnn_masks,image_shape,window)
            stuff_class_ids, stuff_boxes, stuff_masks_unmold = filter_stuff_masks(stuff_detections, stuff_masks,image_shape,window)

            result.append({
                "thing_boxes": thing_boxes,
                "thing_class_ids": thing_class_ids,
                "thing_scores": thing_scores,
                "thing_masks": thing_masks_unmold,
                "stuff_boxes": stuff_boxes,
                "stuff_class_ids": stuff_class_ids,
                "stuff_masks": stuff_masks_unmold,
                "semantic_segment": semantic_segment,
                'influence_map': influence_map
            })
        else:
            influence_map = influence_map.squeeze(0).squeeze(0).data.cpu().numpy()  # [128,128]
            stuff_detections = stuff_detections.data.numpy()  # [y,5]
            stuff_masks = stuff_masks.data.numpy()  # [y,500,500]

            result.append({
                "stuff_rois": stuff_detections,
                "stuff_masks": stuff_masks,
                "semantic_segment": semantic_segment,
                'influence_map': influence_map
            })
        return result

    def predict_segment(self, result, image_metas):
        semantic_labels=result['semantic_segment'] # CIN_semantic_all中的一张图片

        influence_map=result['influence_map'] # CIN_saliency_all中的一张图片

        stuff_class_ids,stuff_boxes,stuff_masks=result['stuff_class_ids'],result['stuff_boxes'],result['stuff_masks']
        thing_class_ids, thing_boxes, thing_masks = result['thing_class_ids'], result['thing_boxes'], result['thing_masks']

        if type(image_metas) is np.ndarray:
            image_shape = image_metas[0][1:4].astype('int32')
        else:
            image_shape = image_metas[0][1:4].cpu().numpy().astype('int32')
        panoptic_result=np.zeros(image_shape)

        information_collector={}
        class_dict = self.class_dict
        id_generator = self.id_generator
        for i,class_id in enumerate(stuff_class_ids):
            category_id=class_dict[str(int(class_id))]['id']
            category_name=class_dict[str(int(class_id))]['name']
            id,color = id_generator.get_id_and_color(str(category_id))
            mask=stuff_masks[i]==1
            panoptic_result[mask]=color
            information_collector[str(id)]={"id":int(id),"bbox":[int(stuff_boxes[i][0]),int(stuff_boxes[i][1]),int(stuff_boxes[i][2]),int(stuff_boxes[i][3])], \
                                            "category_id":category_id,"category_name":category_name, \
                                            'mask': mask}

        for i,class_id in enumerate(thing_class_ids):
            category_id=class_dict[str(int(class_id))]['id']
            category_name=class_dict[str(int(class_id))]['name']
            id, color = id_generator.get_id_and_color(str(category_id))
            mask=thing_masks[i]==1          # 426,640
            panoptic_result[mask]=color     # 426,640,3
            information_collector[str(id)]={"id": int(id), "bbox": [int(thing_boxes[i][0]),int(thing_boxes[i][1]),int(thing_boxes[i][2]),int(thing_boxes[i][3])], \
                                            "category_id": category_id,"category_name":category_name, \
                                            'mask': mask}
        return semantic_labels, influence_map, panoptic_result, information_collector

    def construct_dataset(self, semantic_label, saliency_map, panoptic_result, ioi_segments_info, image_shape, mode):
        # if the mode is training, we keep the labeled attribute
        # if the mode is inference, we ignore the labeled attribute
        # image_id = ioi_image_dict['image_id']
        # image_name = ioi_image_dict['image_name']
        image_width = image_shape[1]
        image_height = image_shape[0]
        scale = self.config.IMAGE_SIZE / max(image_height, image_width)
        new_height = int(round(image_height * scale))
        new_width = int(round(image_width * scale))
        top_pad = (self.config.IMAGE_SIZE - new_height) // 2
        bottom_pad = self.config.IMAGE_SIZE - new_height - top_pad
        left_pad = (self.config.IMAGE_SIZE - new_width) // 2
        right_pad = self.config.IMAGE_SIZE - new_width - left_pad

        semantic_label = scipy.misc.imresize(semantic_label, (new_height, new_width), interp='nearest')
        if len(semantic_label.shape) == 3:
            semantic_label = semantic_label[:, :, 0]
        semantic_label = np.pad(semantic_label, [(top_pad, bottom_pad), (left_pad, right_pad)], mode='constant',
                                constant_values=0)

        saliency_map = scipy.misc.imresize(saliency_map, (new_height, new_width), interp='nearest')
        if len(saliency_map.shape) == 3:
            saliency_map = saliency_map[:, :, 0]
        saliency_map = np.pad(saliency_map, [(top_pad, bottom_pad), (left_pad, right_pad)], mode='constant',
                              constant_values=0)

        labels = []
        class_ids = []
        boxes = []

        for instance_id in ioi_segments_info:
            segment_info = ioi_segments_info[instance_id]
            category_id = segment_info['category_id']
            # find the class_id from the class_dict whose id is equal to the category_id
            class_id = 0
            for key in self.class_dict:
                class_info = self.class_dict[key]
                if class_info['id'] == category_id:
                    class_id = class_info['idx']
                    break
            box = [int(segment_info['bbox'][0] * scale + top_pad), int(segment_info['bbox'][1] * scale + left_pad),
                   int(segment_info['bbox'][2] * scale + top_pad), int(segment_info['bbox'][3] * scale + left_pad)]
            if mode == 'training':
                islabel = segment_info['labeled']
            elif mode == 'inference':
                islabel = ''
            labels.append(islabel)
            boxes.append(np.array(box))
            class_ids.append(class_id)
        boxes = np.stack(boxes)

        instance_groups = []
        for i in range(boxes.shape[0]):
            y1, x1, y2, x2 = boxes[i][:4]
            if y1 < 0:
                y1 = 0
            if x1 < 0:
                x1 = 0
            instance_group = []

            instance_label = semantic_label[y1:y2, x1:x2]
            # print("semantic_label shape============================", semantic_label.shape)
            # print("instance_label shape============================",instance_label.shape)
            # print("y1,x1,y2,x2", y1, x1, y2, x2)
            instance_label = scipy.misc.imresize(instance_label, (self.config.INSTANCE_SIZE, self.config.INSTANCE_SIZE), interp='nearest') / 134.0
            instance_group.append(instance_label)

            instance_map = saliency_map[y1:y2, x1:x2]
            instance_map = scipy.misc.imresize(instance_map, (self.config.INSTANCE_SIZE, self.config.INSTANCE_SIZE), interp='bilinear') / 255.0
            instance_group.append(instance_map)
            instance_group = np.stack(instance_group)
            instance_groups.append(instance_group)

        if mode == 'inference':
            instance_groups = np.stack(instance_groups) #13, 2, 56, 56
            return instance_groups

        pair_label = []
        for i, a_label in enumerate(labels):
            for j, b_label in enumerate(labels):
                pair_label.append((a_label + b_label) / 2.0)
        pair_label = np.array(pair_label)
        instance_groups = np.stack(instance_groups)
        class_ids = np.array(class_ids, dtype=np.float32)
        labels = np.array(labels, dtype=np.float32)
        return instance_groups, boxes, class_ids, labels, pair_label

    def map_instance_to_gt(self, gt_instance_dict, instance_dict, gt_segmentation, segmentation):
        def compute_pixel_iou(bool_mask_pred, bool_mask_gt):
            intersection = bool_mask_pred * bool_mask_gt
            union = bool_mask_pred + bool_mask_gt
            return np.count_nonzero(intersection) / np.count_nonzero(union)  # np.count_nonzero: 计算数组中非零元素的个数

        count = 0
        instance_pred_gt_dict = {}
        instance_gt_pred_dict = defaultdict(list)

        segmentation_id = utils.rgb2id(segmentation)
        for instance_id in instance_dict:
            mask = segmentation_id == int(instance_id)
            instance_dict[instance_id]['mask'] = mask

        gt_segmentation = gt_segmentation.squeeze(0).data.cpu().numpy()
        gt_segmentation_id = utils.rgb2id(gt_segmentation)

        for gt_instance_id in gt_instance_dict:
            gt_mask = gt_segmentation_id == int(gt_instance_id)
            gt_instance_dict[gt_instance_id]['mask'] = gt_mask

        for instance_id in instance_dict:
            max_iou = -1
            max_gt_instance_id = ""
            for gt_instance_id in gt_instance_dict:
                i_iou = compute_pixel_iou(instance_dict[instance_id]['mask'], gt_instance_dict[gt_instance_id]['mask'])
                if gt_instance_id not in instance_gt_pred_dict:
                    instance_gt_pred_dict[gt_instance_id] = {
                        "labeled": gt_instance_dict[gt_instance_id]['labeled'], "pred": []}
                if i_iou >= self.config.MAP_IOU and instance_dict[instance_id]['category_id'] == gt_instance_dict[gt_instance_id][
                    'category_id'] and i_iou > max_iou:
                    max_gt_instance_id = gt_instance_id
                    max_iou = i_iou
                    instance_gt_pred_dict[gt_instance_id]['pred'].append(instance_id)
            if max_gt_instance_id != "":
                instance_pred_gt_dict[instance_id] = {"gt_instance_id": max_gt_instance_id,
                                                      "label": gt_instance_dict[max_gt_instance_id]['labeled']}
            else:
                instance_pred_gt_dict[instance_id] = {"gt_instance_id": "", "label": False}

        return instance_pred_gt_dict, instance_gt_pred_dict

    def generate_image_dict(self, image_info, pred_to_gt, segments_info):
        for instance_id in segments_info:
            if instance_id == "0":
                pass
            if instance_id in pred_to_gt:
                segments_info[instance_id]['labeled'] = pred_to_gt[instance_id]['label']

        ioi_images_dict = {"image_id": image_info['image_id'], "image_name": image_info['image_name'],
                           "height": image_info['height'], "width": image_info['width'],
                           'segments_info': segments_info}

        return ioi_images_dict