import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

# from utils.utils import unfold
from utils.pytorch_utils import unfold
############################################################
#  PiCANet
############################################################

def make_layers(cfg, in_channels):
    layers = []
    dilation_flag = False
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'm':
            layers += [nn.MaxPool2d(kernel_size=1, stride=1)]
            dilation_flag = True
        else:
            if not dilation_flag:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=2, dilation=2)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class DecoderCell(nn.Module):
    def __init__(self, size, in_channel, out_channel, mode):
        super(DecoderCell, self).__init__()
        self.bn_en = nn.BatchNorm2d(in_channel)
        self.conv1 = nn.Conv2d(2 * in_channel, in_channel, kernel_size=1, padding=0)
        self.mode = mode
        if mode == 'G':
            self.picanet = PicanetG(size, in_channel)
        elif mode == 'L':
            self.picanet = PicanetL(in_channel)
        elif mode == 'C':
            self.picanet = None
        else:
            assert 0
        if not mode == 'C':
            self.conv2 = nn.Conv2d(2 * in_channel, out_channel, kernel_size=1, padding=0)
            self.bn_feature = nn.BatchNorm2d(out_channel)
            self.conv3 = nn.Conv2d(out_channel, 1, kernel_size=1, padding=0)
        else:
            self.conv2 = nn.Conv2d(in_channel, 1, kernel_size=1, padding=0)

    def forward(self, *input):
        assert len(input) <= 2
        if input[1] is None:
            en = input[0]
            dec = input[0]
        else:
            en = input[0]
            dec = input[1]
        if dec.size()[2] * 2 == en.size()[2]:
            dec = F.upsample(dec, scale_factor=2, mode='bilinear')
        elif dec.size()[2] != en.size()[2]:
            assert 0
        en = self.bn_en(en)
        en = F.relu(en)
        fmap = torch.cat((en, dec), dim=1)  # F
        fmap = self.conv1(fmap)
        fmap = F.relu(fmap)
        if not self.mode == 'C':
            # print(fmap.size())
            fmap_att = self.picanet(fmap)  # F_att
            x = torch.cat([fmap, fmap_att], 1)
            x = self.conv2(x)
            x = self.bn_feature(x)
            dec_out = F.relu(x)
            _y = self.conv3(dec_out)
            _y = F.sigmoid(_y)
        else:
            dec_out = self.conv2(fmap)
            _y = F.sigmoid(dec_out)

        return dec_out, _y

class PicanetG(nn.Module):
    def __init__(self, size, in_channel):
        super(PicanetG, self).__init__()
        self.renet = Renet(size, in_channel, 22*22)
        self.in_channel = in_channel

    def forward(self, *input):
        x = input[0]
        size = x.size()
        kernel = self.renet(x)
        kernel = F.softmax(kernel, 1)
        x = unfold(x, kernel_size=[22, 22], dilation=[3, 3])
        x = x.view(size[0], size[1], 22 * 22)
        kernel = kernel.view(size[0], 22*22, -1)
        x = torch.bmm(x, kernel)
        x = x.view(size[0], size[1], size[2], size[3])
        return x

class PicanetL(nn.Module):
    def __init__(self, in_channel):
        super(PicanetL, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 256, kernel_size=5, dilation=2, padding=4)
        self.conv2 = nn.Conv2d(256, 5*5, kernel_size=1)

    def forward(self, *input):
        x = input[0]
        size = x.size()
        kernel = self.conv1(x)
        kernel = self.conv2(kernel)
        kernel = F.softmax(kernel, 1)
        kernel = kernel.view(size[0], 1, size[2] * size[3], 5 * 5)

        x = unfold(x, kernel_size=[5, 5], dilation=[2, 2], padding=[4,4])
        x = x.view(size[0], size[1], size[2] * size[3], -1)
        x = torch.mul(x, kernel)
        x = torch.sum(x, dim=3)
        x = x.view(size[0], size[1], size[2], size[3])
        return x

class Renet(nn.Module):
    def __init__(self, size, in_channel, out_channel):
        super(Renet, self).__init__()
        self.size = size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.vertical = nn.LSTM(input_size=in_channel, hidden_size=256, batch_first=True,
                                bidirectional=True)  # each row
        self.horizontal = nn.LSTM(input_size=512, hidden_size=256, batch_first=True,
                                  bidirectional=True)  # each column
        self.conv = nn.Conv2d(512, out_channel, 1)
        # self.fc = nn.Linear(512 * size * size, 10)

    def forward(self, *input):
        x = input[0]
        temp = []
        # size = x.size()  # batch, in_channel, height, width
        x = torch.transpose(x, 1, 3)  # batch, width, height, in_channel
        for i in range(self.size):
            h, _ = self.vertical(x[:, :, i, :])
            temp.append(h)  # batch, width, 512
        x = torch.stack(temp, dim=2)  # batch, width, height, 512
        temp = []
        for i in range(self.size):
            h, _ = self.horizontal(x[:, i, :, :])
            temp.append(h)  # batch, width, 512
        x = torch.stack(temp, dim=3)  # batch, height, 512, width
        x = torch.transpose(x, 1, 2)  # batch, 512, height, width
        x = self.conv(x)
        return x

class SaNet(nn.Module):
    def __init__(self):
        super(SaNet,self).__init__()
        cfg = {'Mode': "GGLL",
                'Size': [64, 64, 64, 128, 128],
               'Channel': [2048, 1024, 512, 256, 64],
               'loss_ratio': [0.5, 0.5, 0.5, 0.8, 1]}

        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)

        self.decoder = nn.ModuleList()
        for i in range(4):
            self.decoder.append(DecoderCell(size=cfg['Size'][i],
                        in_channel=cfg['Channel'][i],
                        out_channel=cfg['Channel'][i + 1],
                        mode=cfg['Mode'][i]))
        self.decoder.append(DecoderCell(size=cfg['Size'][4],
                                        in_channel=cfg['Channel'][4],
                                        out_channel=1,
                                        mode='C'))
    def forward(self, c1_out,c2_out,c3_out,c4_out,c5_out):
        c1_out = self.conv1(c1_out) # 64 256, 256 -> 128, 128
        c2_out = self.conv2(c2_out) # 256 256, 256 -> 128, 128
        c3_out = self.conv3(c3_out) # 512 128, 128 -> 64, 64
        # c4_out # 1024, 64, 64
        c5_out = F.upsample(c5_out, size=(64, 64)) # 2048 32, 32 -> 64, 64
        en_out = [c1_out,c2_out,c3_out,c4_out,c5_out]
        pred = []
        dec = None
        # Bottom-up
        for i in range(5):
            dec, _pred = self.decoder[i](en_out[4 - i], dec)
            # print(_pred.shape)
            pred.append(_pred)
        return pred
