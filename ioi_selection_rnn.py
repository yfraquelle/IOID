#nomask seems not to be impacted by epoch
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import random
import json
import math
import torch.nn.functional as F

torch.manual_seed(1)  # reproducible

# Hyper Parameters
INPUT_SIZE = 2  # category_id saliency distance
HIDDEN_SIZE = 2 # hidden
BATCH_SIZE=1000
LR = 0.01  # learning rate

class LSTM_V(nn.Module):
    def __init__(self, input_size, hidden_size,num_layers=1,bias=True, batch_first=True, dropout=0,bidirectional=False):
        super(LSTM_V, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional
        )
        # self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

        self.acti = nn.ReLU()
        self.o2o = nn.Linear(hidden_size, 1)

    def forward(self, inputs,batch_num,max_len):
        out_pack, _ = self.lstm(inputs, None)

        out = self.acti(out_pack[0])
        out = self.o2o(out)
        return out


def load_labeled_data(mode,panoptic_model,saliency_model,max_len):
    images_json=json.load(open("data/ioi_"+mode+"_images_dict_with_diff_saliency_"+panoptic_model+".json",'r'))
    image_id_list=list()
    for image_id in images_json:
        image_id_list.append((image_id,len(images_json[image_id]['segments_info'])))
    image_id_list=sorted(image_id_list,key=lambda x:x[1],reverse=True)

    input_data=[] #(image_num,instance_num,2)
    gt_data=[] #(image_num.instance_num,1)
    length_list=list()
    for image_id,instance_num in image_id_list:
        image=images_json[image_id]
        instances=image['segments_info']
        if instance_num>=max_len:
            print(instance_num)
            continue
        instances_sortlist=[(instance_id,instances[instance_id][saliency_model+'_max']) for instance_id in instances]
        instances_sortlist=sorted(instances_sortlist,key=lambda x:x[1],reverse=True)

        for instance_id,saliency in instances_sortlist:
            instance=instances[instance_id]
            category_id=instance['class_id']
            saliency = instance[saliency_model+'_max']
            if instance['labeled']==True:
                labeled = 1
            else:
                labeled = 0
            input_data.extend([category_id,saliency])
            gt_data.extend([labeled])
        length_list.append(instance_num)
        for i in range(0,max_len-instance_num):
            input_data.append(0)
            input_data.append(0)
            gt_data.append(0)
    input_data = np.array(input_data,dtype=np.float32).reshape((len(length_list), max_len, 2))
    gt_data = np.array(gt_data,dtype=np.float32).reshape((len(length_list), max_len, 1))

    return input_data,gt_data,length_list

def train_by_batch(panoptic_model,saliency_model,max_len):
    input_data,gt_data,length_list=load_labeled_data("train",panoptic_model,saliency_model,max_len)
    count=len(length_list)

    rnn = LSTM_V(INPUT_SIZE, HIDDEN_SIZE).cuda()
    rnn.zero_grad()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  # optimize all rnn parameters
    loss_func = nn.MSELoss()

    for epoch in range(1,51):
        print("Epoch:" + str(epoch))
        input_loader = list()
        gt_loader = list()
        length_loader = list()
        box_num=int(count/BATCH_SIZE+1)
        for i in range(0,box_num):
            input_loader.append(list())
            gt_loader.append(list())
            length_loader.append(list())

        for i in range(0, count):
            box=random.randint(0,box_num-1)
            input_loader[box].append(input_data[i])
            gt_loader[box].append(gt_data[i])
            length_loader[box].append(length_list[i])
        for i in range(0,box_num):
            input_loader[i]=np.array(input_loader[i])
            gt_loader[i]=np.array(gt_loader[i])

        for i in range(0,box_num):
            input_data_in=input_loader[i]
            gt_data_in=gt_loader[i]
            length_list_in=length_loader[i]
            max_length_curr=length_list_in[0]
            batch_num=len(length_list_in)

            input_tensor=Variable(torch.from_numpy(input_data_in), requires_grad=True).cuda()
            input_tensor = torch.nn.utils.rnn.pack_padded_sequence(input_tensor,length_list_in, batch_first=True)
            gt_tensor=Variable(torch.from_numpy(gt_data_in)).cuda()
            # mask
            gt_tensor = torch.nn.utils.rnn.pack_padded_sequence(gt_tensor,length_list_in, batch_first=True)[0]

            predictions = rnn(input_tensor,batch_num,max_length_curr)
            loss = loss_func(predictions, gt_tensor)  # cross entropy loss
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
        print("loss:"+str(loss.data.cpu().numpy()))
            # exit()
        if epoch%10==0:
            torch.save(rnn, 'logs/lstm/'+str(epoch)+'.pkl')

def predict(panoptic_model,saliency_model,max_len):
    rnn = torch.load('logs/lstm/50.pkl')
    input_data, gt_data, length_list = load_labeled_data("val",panoptic_model,saliency_model,max_len)
    input_tensor = Variable(torch.from_numpy(input_data), requires_grad=True).cuda()
    input_tensor = torch.nn.utils.rnn.pack_padded_sequence(input_tensor, length_list, batch_first=True)
    gt_tensor = Variable(torch.from_numpy(gt_data)).cuda()
    gt_tensor = torch.nn.utils.rnn.pack_padded_sequence(gt_tensor, length_list, batch_first=True)[0].data.cpu().numpy()
    prediction=rnn(input_tensor,2000,max_len).data.cpu().numpy()
    np.save("results/lstm/"+panoptic_model+"/"+saliency_moel+"gt.npy",gt_tensor)
    np.save("results/lstm/"+panoptic_model+"/"+saliency_moel+"pred.npy",prediction)

import sys
if __name__=='__main__':
    if sys.argv[1]=="train":
        train_by_batch("CIN_panoptic_all","CIN_saliency_all",116)
    elif sys.argv[1]=="pred":
        predict("CIN_panoptic_all", "CIN_saliency_all",116)
