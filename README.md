## Instance of Interest Detection

## Requirements
opencv_python==3.4.3.18  
numpy==1.16.2  
scikit_image==0.14.2  
torchvision==0.2.1  
torch==0.4.1  
scipy==1.1.0  
matplotlib==3.0.0  
Pillow==7.0.0    
tensorboardX==2.0  
PyYAML==3.13  
cffi==1.12.2  

## Folder Creation
```
mkdir logs
mkdir models
mkdir data
mkdir results
mkdir ../data
mkdir ../CIN_semantic_val
mkdir ../CIN_panoptic_val
mkdir ../CIN_saliency_val
mkdir ../CIN_semantic_train
mkdir ../CIN_panoptic_train
mkdir ../CIN_saliency_train
mkdir results/CIEDN_pred
mkdir results/ciedn_result
mkdir results/validate
```

## Dataset
We construct the first dataset for IOID based on the training set of MSCOCO 2017, which contains 123,287 images with manually labelled captions and panoptic segmentation. We construct our dataset by representing each IOI with its category and its region in the panoptic segmentation. There are 133 categories of IOIs in total, the same as those provided by the panoptic segmentation in MSCOCO 2017, containing 80 thing categories, such as person, ball and cow, and 53 stuff categories, such as wall, tree and mountain. We further divide the datasets into the training set and the test set, which contain 36,000 images and 9,000 images, respectively.  
The dataset consists of the following sections:
1. The original images which can be downloaded at https://drive.google.com/open?id=1yRyduTD58_lL1GI4oGoUdhpi3gnjzvgO. After downloading the dataset, you should put the image folder at the "data" folder which is paralell with the root.
2. The panoptic segmentation images which can be downloaded at https://drive.google.com/open?id=1nxvSLhNkk7Vc2HEEXquG51tESwEHK07T. After downloading the dataset, you should put the image folder at the "data" folder which is paralell with the root.
3. Dataset annotations in json format which can be downloaded at https://drive.google.com/open?id=1dWZZf5PPokmWvAgT0_Cel5QTnocc1T2t. After downloading the json dataset, you should put the files at the "data" folder which is the child folder of the root.

In order to verify the effectiveness of the method, we compare the results of our method with some of the other methods. The data of the instance extraction experiment can be downloaded at https://drive.google.com/open?id=1wFupKzYt0sabBw_UjMc6AWLDwnbidLsj. And the data of the interest estimation experiment can be downloaded at <https://drive.google.com/open?id=1dU_YF6p3cwXR2Kq51O2y-Gft5WiRvKb_> .



## Quick Start
To visualize the result of the instance of interest detection, we provide a demo and it can be performed in the following script:
```python
python demo.py −−img <image path> −−config <configuration file path>
```
You can train you own model by running the following script:  
TIPS:setting sample: semantic,0.01,34,p_interest,0.01,44,selection,0.01,100  
```python
python train.py −−setting <setting> −−config <configuration file path>
```
Based on the pretrained model, you can predict all the images in the dataset by running the following script:
```python
python predict.py −−mode <mode> --train_val_mode <train_val_mode> −−config <configuration file path>
```
To validate the performance of the CIN model, the validate.py file can be performed in the following script:
```python
python validate.py −−config < configuration file path>
```
In order to verify the effectiveness of the method, the component_analysis.py file can be performed in the following script:
```python
python component_analysis.py −−ins_ext <panoptic segmentation path> −−sem_ext <semantic segmentation path> −−p_intr <interest estimation path> --selection_model <selection model> −−config <configuration file path>
```
We provide two interest selection variants in "ioi_selection_binary.py" and "ioi_selection_rnn.py". Before using these two variants, you have to generate medium results:
```python
python predict.py --mode insttr --train_val_mode val −−config <configuration file path>
python middle_process.py --mode val
```  
If you want to train the model in "ioi_selection_rnn.py" by yourself, you need to generate medium results for training set:
```python
python predict.py --mode insttr --train_val_mode train −−config <configuration file path>
python middle_process.py --mode train
python ioi_selection_rnn.py train
```

## Docker environment
We provide docker image with all software dependencies: https://drive.google.com/file/d/1IQneKJpYU34tyREmuDC9G81nI1ekIiCX/view?usp=sharing .
```
sudo docker pull xxxx
sudo nvidia-docker run --rm -ti -v /path/to/parent/floder/of/IOID:/workspace --ipc=host xxxx /bin/bash
```


## Links
You can download some pretrained models at https://drive.google.com/open?id=167nT9zWvmWN2YQ_SKoMO7faqHE2LMcX2 and https://drive.google.com/open?id=1COzdQtxtA0v4bkb6MOuUX7QpbWJPsibm. After downloading the pretrained model, you should put the files at the "models" folder which is the child folder of the root.

This project refers to https://github.com/Ugness/PiCANet-Implementation and https://github.com/multimodallearning/pytorch-mask-rcnn .  
Our baselines refers to https://github.com/RuochenFan/S4Net and https://github.com/multimodallearning/pytorch-mask-rcnn .  
Our components analysis refers to https://github.com/dmechea/PanopticSegmentation, https://github.com/AceCoooool/DSS-pytorch.git, https://github.com/zhimingluo/NLDF, https://github.com/Xyuan13/MSRNet, https://github.com/Ugness/PiCANet-Implementation, https://github.com/batsa003/salgan and https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning .