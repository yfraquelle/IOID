import numpy as np
import math
import json
import csv

def maxminnorm(array):
    max_value=np.max(array)
    min_value=np.min(array)
    array=(array-min_value)/(max_value-min_value)
    return array

def compare_mask(gt_data,predictions,a2):
    predictions=maxminnorm(predictions)
    TP=dict()
    TP_FP=dict()
    TP_FN=dict()
    precision_dict=dict()
    recall_dict=dict()
    f_dict=dict()
    for threshold in np.arange(0.3,0.91,0.05):
        prediction = np.where(predictions>threshold,1,0)
        TP[threshold] = np.sum(np.multiply(prediction, gt_data))
        TP_FP[threshold] = np.sum(prediction)
        TP_FN[threshold] = np.sum(gt_data)
        precision=TP[threshold]/TP_FP[threshold]
        recall=TP[threshold]/TP_FN[threshold]
        f=(a2 + 1) * precision * recall / (a2 * precision + recall)
        precision_dict[str(round(threshold,2))]=round(precision,3)
        recall_dict[str(round(threshold,2))]=round(recall,3)
        f_dict[str(round(threshold,2))]=round(f,3)
        # print(str(round(threshold,2))+":"+str(round(precision,3))+" "+str(round(recall,3)))
        # print(str(round(f,3)))
    return precision_dict,recall_dict,f_dict

def compare_mask_wih_a2_and_threshold(gt_data, predictions,threshold):
    a2 = 0.3
    predictions = maxminnorm(predictions)
    TP = dict()
    TP_FP = dict()
    TP_FN = dict()
    precision_dict = dict()
    recall_dict = dict()
    f_dict = dict()
    prediction = np.where(predictions > threshold, 1, 0)
    TP[threshold] = np.sum(np.multiply(prediction, gt_data))
    TP_FP[threshold] = np.sum(prediction)
    TP_FN[threshold] = np.sum(gt_data)
    precision = TP[threshold] / TP_FP[threshold]
    recall = TP[threshold] / TP_FN[threshold]
    f = (a2 + 1) * precision * recall / (a2 * precision + recall)
    return precision, recall, f

from matplotlib import pyplot as plt

def draw_pictures(methods,result,saliency_model):
    result_a2 = result["0.3"]
    threshold_list = sorted(list(result_a2[saliency_model]['precision'].keys()))
    x_list=list(threshold_list)

    plt.figure(figsize=(8, 8))
    # plt.subplot(1, 2, 1)
    # plt.title('precision')
    for index,method in enumerate(methods):
        method = method.split("/")[-1]
        precision_list = result_a2[method]['precision']
        y_precision = []
        for threshold in threshold_list:
            y_precision.append(precision_list[threshold])
        if index==0:
            plt.plot(x_list, y_precision,'r', label=method+"-p")
        elif index<=6:
            plt.plot(x_list, y_precision, 'y', label=method + "-p")
        elif index<=8:
            plt.plot(x_list, y_precision, 'g', label=method + "-p")
        elif index<=10:
            plt.plot(x_list, y_precision, 'b', label=method + "-p")

    # plt.subplot(1, 2, 2)
    # plt.title('recall')
    for index,method in enumerate(methods):
        method = method.split("/")[-1]
        recall_list = result_a2[method]['recall']
        y_recall = []
        for threshold in threshold_list:
            y_recall.append(recall_list[threshold])
        if index==0:
            plt.plot(x_list, y_recall,'r', label=method+"-r")
        elif index<=6:
            plt.plot(x_list, y_recall, 'y', label=method + "-r")
        elif index<=8:
            plt.plot(x_list, y_recall, 'g', label=method + "-r")
        elif index<=10:
            plt.plot(x_list, y_recall, 'b', label=method + "-r")

    plt.show()

def write_csv(methods,result_file):
    result=json.load(open(result_file+".json",'r'))
    with open(result_file+'.csv', 'w+', newline='') as csv_file:
        writer = csv.writer(csv_file)
        a_list=sorted(list(result.keys()))
        for a2 in a_list:
            result_a2=result[a2]
            writer.writerow([a2])
            # threshold_list = sorted(list(result_a2["CIN_saliency_all"]['precision'].keys()))
            threshold_list = sorted(list(result_a2['precision'].keys()))
            header = [" "]
            for threshold in threshold_list:
                header.append(threshold + "-p")
                header.append(threshold + "-r")
                header.append(threshold + "-f")
            writer.writerow(header)
            precision_list = result_a2['precision']
            recall_list = result_a2['recall']
            f_list = result_a2['f']
            result_row = ['CIN']
            for threshold in threshold_list:
                result_row.append(precision_list[threshold])
                result_row.append(recall_list[threshold])
                result_row.append(f_list[threshold])
            writer.writerow(result_row)
            # for method in methods:
            #     method=metholistd.split("/")[-1]
            #     precision_list=result_a2[method]['precision']
            #     recall_list=result_a2[method]['recall']
            #     f_list=result_a2[method]['f']
            #     result_row=[method]
            #     for threshold in threshold_list:
            #         result_row.append(precision_list[threshold])
            #         result_row.append(recall_list[threshold])
            #         result_row.append((f_list[threshold]))
            #     writer.writerow(result_row)

if __name__=="__main__":
    saliency_train_model="CIN_saliency_all_old"
    panoptic_train_model="CIN_panoptic_all_old"
    wait_compares=[saliency_train_model,'a-PyTorch-Tutorial-to-Image-Captioning_saiency','DSS-pytorch_saliency','MSRNet_saliency','NLDF_saliency','PiCANet-Implementation_saliency','salgan_saliency',
                   'maskrcnn_panoptic','deeplab_panoptic','../../../lstm/'+panoptic_train_model+'/'+saliency_train_model,'../../../only/'+panoptic_train_model+'/'+saliency_train_model]
    result={}
    for a2 in np.arange(0.1, 3, 0.1):
        result[str(round(a2, 2))] = {}
        for wait_compare in wait_compares:
            gt = np.load("results/cirnn_result_pair_influence/"+panoptic_train_model+"/"+saliency_train_model+"/"+wait_compare+"/gt.npy")
            pred = maxminnorm(np.load("results/cirnn_result_pair_influence/"+panoptic_train_model+"/"+saliency_train_model+"/"+wait_compare+"/pred.npy"))
            precision,recall,f=compare_mask(gt,pred,a2)
            result[str(round(a2,2))][wait_compare]={"precision":precision,"recall":recall,"f":f}
    result_file="results/result_old"
    json.dump(result,open(result_file+".json",'w'))
    write_csv(wait_compares,result_file)
    draw_pictures(wait_compares,result,saliency_train_model)