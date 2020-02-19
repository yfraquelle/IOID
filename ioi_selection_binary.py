import json
import numpy as np

def predict(panoptic_model,saliency_model):
    image_dict=json.load(open("data/ioi_val_images_dict_with_diff_saliency_"+panoptic_model+".json",'r'))
    gt_list=[]
    prediction=[]
    for image_id in image_dict:
        instances=image_dict[image_id]['segments_info']
        for instance_id in instances:
            if instances[instance_id]['labeled']:
                gt_list.append(1)
            else:
                gt_list.append(0)
            prediction.append(instances[instance_id][saliency_model+'_max']/255.0)
    np.save("results/only/"+panoptic_model+"/"+saliency_model+"/gt.npy",np.array(gt_list))
    np.save("results/only/"+panoptic_model+"/"+saliency_model+"/pred.npy", np.array(prediction))

if __name__=='__main__':
    predict("CIN_panoptic_all_old","CIN_saliency_all_old")