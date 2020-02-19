import torch.nn as nn
import torch.nn.functional as F

from utils.pytorch_utils import SamePad2d, GroupNorm
# from utils.utils import SamePad2d, GroupNorm
from instance_extraction.ROIAlign import pyramid_roi_align

############################################################
#  Feature Pyramid Network Heads
############################################################

class Classifier(nn.Module):
    def __init__(self, depth, pool_size, image_shape, num_classes):
        super(Classifier, self).__init__()
        self.depth = depth
        self.pool_size = pool_size
        self.image_shape = image_shape
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(self.depth, 1024, kernel_size=self.pool_size, stride=1)
        self.bn1 = nn.BatchNorm2d(1024, eps=0.001, momentum=0.01)
        self.conv2 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(1024, eps=0.001, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)

        self.linear_class = nn.Linear(1024, num_classes)
        self.softmax = nn.Softmax(dim=1)

        self.linear_bbox = nn.Linear(1024, num_classes * 4)

    def forward(self, x, rois):
        x = pyramid_roi_align([rois]+x, self.pool_size, self.image_shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = x.view(-1,1024)
        mrcnn_class_logits = self.linear_class(x)
        mrcnn_probs = self.softmax(mrcnn_class_logits)

        mrcnn_bbox = self.linear_bbox(x)
        mrcnn_bbox = mrcnn_bbox.view(mrcnn_bbox.size()[0], -1, 4)

        return [mrcnn_class_logits, mrcnn_probs, mrcnn_bbox]

class Mask(nn.Module):
    def __init__(self, depth, pool_size, image_shape, num_classes):
        super(Mask, self).__init__()
        self.depth = depth
        self.pool_size = pool_size
        self.image_shape = image_shape
        self.num_classes = num_classes
        self.padding = SamePad2d(kernel_size=3, stride=1)
        self.conv1 = nn.Conv2d(self.depth, 256, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(256, eps=0.001)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(256, eps=0.001)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(256, eps=0.001)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(256, eps=0.001)
        self.deconv = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, rois):
        x = pyramid_roi_align([rois] + x, self.pool_size, self.image_shape)
        x = self.conv1(self.padding(x))
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(self.padding(x))
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(self.padding(x))
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(self.padding(x))
        x = self.bn4(x)
        x = self.relu(x)
        x = self.deconv(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.sigmoid(x)

        return x

class Semantic(nn.Module):
    def __init__(self,num_classes):
        super(Semantic, self).__init__()

        self.num_classes = num_classes

        # Semantic branch
        self.semantic_branch = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, self.num_classes, kernel_size=1, stride=1, padding=0)
        # num_groups, num_channels
        self.gn1 = GroupNorm(128,128)
        self.gn2 = GroupNorm(256,256)


    def forward(self, mrcnn_feature_maps):
        p2_out = mrcnn_feature_maps[0] #256
        p3_out = mrcnn_feature_maps[1] #128
        p4_out = mrcnn_feature_maps[2] #64
        p5_out = mrcnn_feature_maps[3] #32

        # Semantic
        _, _, h, w = p2_out.size()
        # 32, 256->256, 256
        s5 = F.upsample(F.relu(self.gn2(self.conv2(p5_out))), size=(h,w),mode='bilinear')
        # 256, 256->256, 256
        s5 = F.upsample(F.relu(self.gn2(self.conv2(s5))), size=(h,w),mode='bilinear')
        # 256, 256->256, 128
        s5 = F.upsample(F.relu(self.gn1(self.semantic_branch(s5))), size=(h,w),mode='bilinear')

        # 64, 256->256, 256
        s4 = F.upsample(F.relu(self.gn2(self.conv2(p4_out))), size=(h,w),mode='bilinear')
        # 256, 256->256, 128
        s4 = F.upsample(F.relu(self.gn1(self.semantic_branch(s4))), size=(h,w),mode='bilinear')

        # 128, 256->256, 128
        s3 = F.upsample(F.relu(self.gn1(self.semantic_branch(p3_out))), size=(h,w),mode='bilinear')

        # 256, 256->256, 128
        s2 = F.relu(self.gn1(self.semantic_branch(p2_out)))
        return F.upsample(self.conv3(s2 + s3 + s4 + s5), size=(500, 500),mode='bilinear') # 500

