import numpy as np
import math
import json
import csv

def maxminnorm(array):
    max_value=np.max(array)
    min_value=np.min(array)
    array=(array-min_value)/(max_value-min_value)
    return array

def compare_mask(gt_data,predictions,a2, base,threshold):
    predictions=maxminnorm(predictions)
    TP=dict()
    TP_FP=dict()
    TP_FN=dict()
    
    prediction = np.where(predictions>threshold,1,0)
    TP[threshold] = np.count_nonzero(gt_data*prediction)
    TP_FP[threshold] = np.count_nonzero(prediction)
    TP_FN[threshold] = np.count_nonzero(gt_data)
    precision=TP[threshold]/TP_FP[threshold]
    recall=TP[threshold]/TP_FN[threshold]
    recall_=TP[threshold]/(TP_FN[threshold]+base)
    if a2 * precision + recall == 0:
        f = 0
    else:
        f=(a2 + 1) * precision * recall / (a2 * precision + recall)
        f_ = (a2 + 1) * precision * recall_ / (a2 * precision + recall_)
    
    return precision,recall,f, recall_, f_
