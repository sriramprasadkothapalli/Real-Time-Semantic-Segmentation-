# Real-Time-Semantic-Segmentation
The main objective of the project is to perform realtime semantic segmentation on CamVid dataset with different approaches of deep learning and then testing it on the realtime data.

## Libraries 
Install these libraries to run  the code

```
import os
import numpy as np
import cv2 
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import re
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from tqdm import tqdm as tqdm

```

## Pipeline 


- Import required libraries, including segmentation_models_pytorch and torch.

- Set encoder parameters (e.g., vgg16 for encoder, imagenet for encoder weights).
Specify activation function as softmax.


- Initialize a segmentation model (e.g., PSPNet) with the predefined encoder and weights. Define a preprocessing function using the encoder’s preprocessing method.

- Set up the Adam optimizer with a learning rate of 0.0001, define the loss function as DiceLoss. Load and Prepare Data:

- Create datasets for training and validation using a custom Dataset class. Apply augmentations to the training dataset.

- Initialize data loaders for training and validation datasets using DataLoader.
- Train the Model:

- Define the training loop to iterate over epochs and batches, updating model weights using the optimizer and computing the loss.

- Define metrics for model evaluation, including IoU, accuracy, F-score, recall, and precision. Evaluate the model on the validation dataset using these metrics.

- Visualize predicted masks alongside ground truth masks. Save frames from a video as individual image files using OpenCV. Generate a video from the predicted masks and save it as an MP4 file.

## Results


The following masks have been generated 
![image](https://github.com/sriramprasadkothapalli/Real-Time-Semantic-Segmentation-/assets/143056659/226437ec-e459-4980-a933-7e191b6dcb11)
![image](https://github.com/sriramprasadkothapalli/Real-Time-Semantic-Segmentation-/assets/143056659/2edd24b9-5e22-434d-9455-3bebb5b613eb)
![image](https://github.com/sriramprasadkothapalli/Real-Time-Semantic-Segmentation-/assets/143056659/9c4364fe-71e2-490e-9847-cb3ff20f09af)
![image](https://github.com/sriramprasadkothapalli/Real-Time-Semantic-Segmentation-/assets/143056659/4efd2196-9b19-453a-9937-d4d132030748)


### Videos

- With RESNET and UNET https://drive.google.com/file/d/1r29oBECa-1zw425y2t8SzA91jlzfoJfg/view?usp=sharing
- With Custom Model https://drive.google.com/file/d/1ludgU0OSnIXkzqrkePuzxHh1qYYUal5z/view?usp=sharing




