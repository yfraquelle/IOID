import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable

############################################################
#  Pytorch Utility Functions
############################################################

def unique1d(tensor):
    if tensor.size()[0] == 0 or tensor.size()[0] == 1:
        return tensor
    tensor = tensor.sort()[0]
    unique_bool = tensor[1:] != tensor [:-1]
    first_element = Variable(torch.ByteTensor([True]), requires_grad=False)
    if tensor.is_cuda:
        first_element = first_element.cuda()
    unique_bool = torch.cat((first_element, unique_bool),dim=0)
    return tensor[unique_bool.data]

def intersect1d(tensor1, tensor2):
    aux = torch.cat((tensor1, tensor2),dim=0)
    aux = aux.sort()[0]
    return aux[:-1][(aux[1:] == aux[:-1]).data]

def log2(x):
    """Implementatin of Log2. Pytorch doesn't have a native implemenation."""
    ln2 = Variable(torch.log(torch.FloatTensor([2.0])), requires_grad=False)
    if x.is_cuda:
        ln2 = ln2.cuda()
    return torch.log(x) / ln2

class SamePad2d(nn.Module):
    """Mimics tensorflow's 'SAME' padding.
    """

    def __init__(self, kernel_size, stride):
        super(SamePad2d, self).__init__()
        self.kernel_size = torch.nn.modules.utils._pair(kernel_size)
        self.stride = torch.nn.modules.utils._pair(stride)

    def forward(self, input):
        in_width = input.size()[2]
        in_height = input.size()[3]
        out_width = math.ceil(float(in_width) / float(self.stride[0]))
        out_height = math.ceil(float(in_height) / float(self.stride[1]))
        pad_along_width = ((out_width - 1) * self.stride[0] +
                           self.kernel_size[0] - in_width)
        pad_along_height = ((out_height - 1) * self.stride[1] +
                            self.kernel_size[1] - in_height)
        pad_left = math.floor(pad_along_width / 2)
        pad_top = math.floor(pad_along_height / 2)
        pad_right = pad_along_width - pad_left
        pad_bottom = pad_along_height - pad_top
        return F.pad(input, (pad_left, pad_right, pad_top, pad_bottom), 'constant', 0)

    def __repr__(self):
        return self.__class__.__name__

class GroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels, eps=1e-5):
        super(GroupNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(1,num_channels,1,1))
        self.bias = nn.Parameter(torch.zeros(1,num_channels,1,1))
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x):
        N,C,H,W = x.size()
        G = self.num_groups
        assert C % G == 0

        x = x.view(N,G,-1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x-mean) / (var+self.eps).sqrt()
        x = x.view(N,C,H,W)
        return x * self.weight + self.bias

def unfold(input,kernel_size,dilation,padding=[0,0],stride=[1,1]):
    input=input.data
    window_size=[int((input.shape[2+0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0]+1),
                 int((input.shape[2+1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1]+1)]
    final_result=[]
    for j in range(len(input)): # batch
        single_input=input[j]
        single_result=[]
        for i in range(len(single_input)): # channel
            x=single_input[i]
            # print(x.shape)
            x = x.unfold(0, window_size[0], dilation[0])
            # print(x.shape)
            x = x.unfold(1, window_size[1], dilation[1])
            # print(x.shape)
            xs=[]
            for k in range(padding[0]):
                x_zero=Variable(torch.zeros((x.shape[0],x.shape[1],window_size[0]+2*padding[1]))).cuda()
                xs.append(x_zero)
            for k in range(x.shape[2]):
                x_element = x[:, :, k, :]
                x_zero = Variable(torch.zeros((x.shape[0], x.shape[1], window_size[0] + 2 * padding[1]))).cuda()
                x_zero[:,:,padding[1]:padding[1]+x.shape[3]]=x_element
                xs.append(x_zero)
            for k in range(padding[0]):
                x_zero=Variable(torch.zeros((x.shape[0],x.shape[1],window_size[0]+2*padding[1]))).cuda()
                xs.append(x_zero)
            x = torch.cat(xs,dim=2)
            xs=[]
            for k in range(x.shape[0]):
                xs.append(x[k,:,:])
            x = torch.cat(xs, dim=0)
            single_result.append(x)
        single_result=torch.cat(single_result,dim=0)
        final_result.append(single_result)
    final_result=torch.stack(final_result,dim=0)
    return final_result
