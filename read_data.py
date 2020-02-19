import json
import numpy as np
import torch

result_pred = np.load("results/ciedn_result/CIN_panoptic_all_CIN_saliency_all_pred.npy")
result_gt   = np.load("results/ciedn_result/CIN_panoptic_all_CIN_saliency_all_gt.npy")

result_pred_old = np.load("results/ciedn_result/pred.npy")
result_gt_old = np.load("results/ciedn_result/gt.npy")

print(result_pred.shape)
print(result_gt.shape)

print(result_pred_old.shape)
print(result_gt_old.shape)