import numpy as np
import math
import json
import csv

def maxminnorm(array):
    max_value=np.max(array)
    min_value=np.min(array)
    array=(array-min_value)/(max_value-min_value)
    return array

def compare_mask(gt_data,predictions,a2, base):
    predictions=maxminnorm(predictions)
    TP=dict()
    TP_FP=dict()
    TP_FN=dict()
    precision_dict=dict()
    recall_dict=dict()
    f_dict=dict()
    _recall_dict=dict()
    _f_dict=dict()
    for threshold in np.arange(0.3,0.91,0.01):
        prediction = np.where(predictions>threshold,1,0)
        TP[threshold] = np.sum(np.multiply(prediction, gt_data))
        TP_FP[threshold] = np.sum(prediction)
        TP_FN[threshold] = np.sum(gt_data)
        precision=TP[threshold]/TP_FP[threshold]
        recall=TP[threshold]/TP_FN[threshold]
        recall_=TP[threshold]/(TP_FN[threshold]+base)
        if a2 * precision + recall == 0:
            f = 0
        else:
            f=(a2 + 1) * precision * recall / (a2 * precision + recall)
            f_ = (a2 + 1) * precision * recall_ / (a2 * precision + recall_)
        precision_dict[str(round(threshold,2))]=round(precision,3)
        recall_dict[str(round(threshold,2))]=round(recall,3)
        f_dict[str(round(threshold,2))]=round(f,3)
        _recall_dict[str(round(threshold,2))]=round(recall_,3)
        _f_dict[str(round(threshold, 2))] = round(f_, 3)
        # print(str(round(threshold,2))+":"+str(round(precision,3))+" "+str(round(recall,3)))
        # print(str(round(f,3)))
    return precision_dict,recall_dict,f_dict, _recall_dict, _f_dict

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

def draw_pictures(methods,result):
    threshold_list = sorted(list(result[methods[0]]['precision'].keys()))
    x_list = list(threshold_list)

    plt.figure(figsize=(8, 8))
    for index, method in enumerate(methods):
        method = method.split("/")[-1]
        precision_list = result[method]['precision']
        y_precision = []
        for threshold in threshold_list:
            y_precision.append(precision_list[threshold])
        # print(x_list)
        # print(y_precision)
        if index == 0:
            plt.plot(x_list, y_precision, 'r', label=method + "-p")
        elif index <= 6:
            plt.plot(x_list, y_precision, 'y', label=method + "-p")
        elif index <= 8:
            plt.plot(x_list, y_precision, 'g', label=method + "-p")
        elif index <= 10:
            plt.plot(x_list, y_precision, 'b', label=method + "-p")

    for index, method in enumerate(methods):
        method = method.split("/")[-1]
        recall_list = result[method]['recall']
        y_recall = []
        for threshold in threshold_list:
            y_recall.append(recall_list[threshold])
        # print(y_recall)
        if index == 0:
            plt.plot(x_list, y_recall, 'r', label=method + "-r")
        elif index <= 6:
            plt.plot(x_list, y_recall, 'y', label=method + "-r")
        elif index <= 8:
            plt.plot(x_list, y_recall, 'g', label=method + "-r")
        elif index <= 10:
            plt.plot(x_list, y_recall, 'b', label=method + "-r")

    for index, method in enumerate(methods):
        method = method.split("/")[-1]
        recall_list = result[method]['f']
        y_recall = []
        for threshold in threshold_list:
            y_recall.append(recall_list[threshold])
        # print(y_recall)
        if index == 0:
            plt.plot(x_list, y_recall, 'r', label=method + "-f")
        elif index <= 6:
            plt.plot(x_list, y_recall, 'y', label=method + "-f")
        elif index <= 8:
            plt.plot(x_list, y_recall, 'g', label=method + "-f")
        elif index <= 10:
            plt.plot(x_list, y_recall, 'b', label=method + "-f")

    # plt.legend()
    plt.show()

    plt.figure(figsize=(8, 8))

    for index, method in enumerate(methods):
        method = method.split("/")[-1]
        precision_list = result[method]['precision']
        y_precision = []
        for threshold in threshold_list:
            y_precision.append(precision_list[threshold])
        # print(x_list)
        # print(y_precision)
        if index == 0:
            plt.plot(x_list, y_precision, 'r', label=method + "-p")
        elif index <= 6:
            plt.plot(x_list, y_precision, 'y', label=method + "-p")
        elif index <= 8:
            plt.plot(x_list, y_precision, 'g', label=method + "-p")
        elif index <= 10:
            plt.plot(x_list, y_precision, 'b', label=method + "-p")

    for index, method in enumerate(methods):
        method = method.split("/")[-1]
        recall_list = result[method]['_recall']
        y_recall = []
        for threshold in threshold_list:
            y_recall.append(recall_list[threshold])
        # print(y_recall)
        if index == 0:
            plt.plot(x_list, y_recall, 'r', label=method + "-r")
        elif index <= 6:
            plt.plot(x_list, y_recall, 'y', label=method + "-r")
        elif index <= 8:
            plt.plot(x_list, y_recall, 'g', label=method + "-r")
        elif index <= 10:
            plt.plot(x_list, y_recall, 'b', label=method + "-r")

    for index, method in enumerate(methods):
        method = method.split("/")[-1]
        recall_list = result[method]['_f']
        y_recall = []
        for threshold in threshold_list:
            y_recall.append(recall_list[threshold])
        # print(y_recall)
        if index == 0:
            plt.plot(x_list, y_recall, 'r', label=method + "-f")
        elif index <= 6:
            plt.plot(x_list, y_recall, 'y', label=method + "-f")
        elif index <= 8:
            plt.plot(x_list, y_recall, 'g', label=method + "-f")
        elif index <= 10:
            plt.plot(x_list, y_recall, 'b', label=method + "-f")

    # plt.legend()
    plt.show()

def write_csv(methods,result_file):
    result=json.load(open(result_file+".json",'r'))
    with open(result_file+'.csv', 'w+', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # a_list=sorted(list(result.keys()))
        # for a2 in a_list:
        #     writer.writerow([a2])
        for method in methods:
            result_method=result[method]
            # threshold_list = sorted(list(result_a2["CIN_saliency_all"]['precision'].keys()))
            threshold_list = sorted(list(result[method]['precision'].keys()))
            header = [" "]
            for threshold in threshold_list:
                header.append(threshold + "-p")
                header.append(threshold + "-r")
                header.append(threshold + "-f")
                header.append(threshold + '-r*')
                header.append(threshold + '-f*')
            writer.writerow(header)
            precision_list = result_method['precision']
            recall_list = result_method['recall']
            f_list = result_method['f']
            _recall_list = result_method['_recall']
            _f_list = result_method['_f']
            result_row = [method]
            for threshold in threshold_list:
                result_row.append(precision_list[threshold])
                result_row.append(_recall_list[threshold])
                result_row.append(_f_list[threshold])
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
    saliency_train_model="CIN_saliency_val"
    panoptic_train_model="CIN_panoptic_val"
