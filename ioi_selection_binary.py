import json
import numpy as np

def predict(panoptic_model,saliency_model):
    image_dict=json.load(open("results/val_images_dict_with_saliency.json",'r'))
    gt_list=[]
    prediction=[]
    for image_id in image_dict:
        instances=image_dict[image_id]['instances']
        for instance_id in instances:
            if instances[instance_id]['labeled']:
                gt_list.append(1)
            else:
                gt_list.append(0)
            prediction.append(instances[instance_id][saliency_model+'_max']/255.0)
    np.save("results/ciedn_result/"+panoptic_model+"_"+saliency_model+"_binary_gt.npy",np.array(gt_list))
    np.save("results/ciedn_result/"+panoptic_model+"_"+saliency_model+"_binary_pred.npy", np.array(prediction))

if __name__=='__main__':
    predict("CIN_panoptic_val","CIN_saliency_val")