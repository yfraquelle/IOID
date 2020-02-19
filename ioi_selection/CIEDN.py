import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import math

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.sa_conv1=nn.Conv2d(1,8,kernel_size=3,stride=2)
        self.sa_ac1=nn.ReLU()
        self.sa_conv2=nn.Conv2d(8,32,kernel_size=3,stride=2)
        self.sa_ac2=nn.ReLU()
        self.sa_mp1=nn.MaxPool2d(kernel_size=1, stride=1)
        self.sa_conv3=nn.Conv2d(32,64,kernel_size=3,stride=2)
        self.sa_ac3=nn.ReLU()

        self.ca_conv1=nn.Conv2d(1,8,kernel_size=3,stride=2)
        self.ca_ac1=nn.ReLU()
        self.ca_conv2=nn.Conv2d(8,32,kernel_size=3,stride=2)
        self.ca_ac2=nn.ReLU()
        self.ca_mp1=nn.MaxPool2d(kernel_size=1, stride=1)
        self.ca_conv3=nn.Conv2d(32,64,kernel_size=3,stride=2)
        self.ca_ac3=nn.ReLU()


    def forward(self, embeddings):
        # print(embeddings.shape)
        sa_c1_out=self.sa_conv1(embeddings[0,:,:1,:,:])
        sa_c1_out=self.sa_ac1(sa_c1_out)
        # print(a_c1_out.data.cpu().numpy())
        sa_c2_out=self.sa_conv2(sa_c1_out)
        sa_c2_out = self.sa_ac2(sa_c2_out)
        # print(a_c2_out.data.cpu().numpy())
        # print(a_c2_out.shape)
        sa_c2_out=self.sa_mp1(sa_c2_out)
        # print(a_c2_out.shape)
        sa_c3_out=self.sa_conv3(sa_c2_out)
        sa_c3_out = self.sa_ac3(sa_c3_out)
        # print(a_c3_out.shape)
        sa_c4_out=sa_c3_out.view(-1,64*6*6)
        # print(sa_c4_out)

        ca_c1_out=self.ca_conv1(embeddings[0,:,1:,:,:])
        ca_c1_out=self.ca_ac1(ca_c1_out)
        # print(a_c1_out.data.cpu().numpy())
        ca_c2_out=self.ca_conv2(ca_c1_out)
        ca_c2_out = self.ca_ac2(ca_c2_out)
        # print(a_c2_out.data.cpu().numpy())
        ca_c2_out=self.ca_mp1(ca_c2_out)
        # print(a_c2_out.shape)
        ca_c3_out=self.ca_conv3(ca_c2_out)
        ca_c3_out = self.ca_ac3(ca_c3_out)
        # print(a_c3_out.shape)
        ca_c4_out=ca_c3_out.view(-1,64*6*6)

        return torch.cat((sa_c4_out,ca_c4_out),dim=1)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(2304*2*2,1152)
        self.fc2 = nn.Linear(1152,576)
        self.ac1 = nn.ReLU()
        self.fc3 = nn.Linear(576, 288)
        self.fc4 = nn.Linear(288, 72)
        self.ac2 = nn.ReLU()
        self.fc5 = nn.Linear(72, 18)
        self.fc6 = nn.Linear(18, 9)
        self.ac3 = nn.ReLU()
        self.fc7 = nn.Linear(9,1)

    def forward(self, pairs):
        c1_out=self.fc1(pairs)
        c2_out = self.fc2(c1_out)
        c2_put = self.ac1(c2_out)
        # print(c2_out.shape)
        c3_out = self.fc3(c2_out)
        c4_out = self.fc4(c3_out)
        c4_put = self.ac2(c4_out)
        # print(c4_out.shape)
        c5_out = self.fc5(c4_out)
        c6_out = self.fc6(c5_out)
        c6_put = self.ac3(c6_out)
        # print(c6_out.shape)
        c7_out = self.fc7(c6_out)
        # print(c7_out.shape)
        # print(output)
        return c7_out


class CIEDN(nn.Module):
    def __init__(self,):
        super(CIEDN, self).__init__()
        self.encoder = Encoder()
        self.decoder =Decoder()
        self.sig=nn.Sigmoid()

    def forward(self, instance_groups):
        encoder_output = self.encoder(instance_groups) #[t, 2304]
        pair_groups=[]
        for i,a_group in enumerate(encoder_output):
            for j,b_group in enumerate(encoder_output):
                b_group=encoder_output[j]
                pair_groups.append(torch.cat([a_group,b_group]))
        pair_groups=torch.stack(pair_groups)
        output = self.decoder(pair_groups)
        return output
