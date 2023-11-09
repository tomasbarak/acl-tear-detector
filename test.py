import os
import sys
sys.path.append('../../mrnet')

import torch
import model
from dataloader import MRDataset
from tqdm import tqdm_notebook

import cv2

import numpy as np

task = 'acl'
plane = 'sagittal'
prefix = 'v3'

model_name = [name  for name in os.listdir('./models/') 
              if (task in name) and 
                 (plane in name) and 
                 (prefix in name)][0]

is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")

mrnet = torch.load(f'./models/{model_name}')
mrnet = mrnet.to(device)

_ = mrnet.eval()

dataset = MRDataset('./data/', 
                    task, 
                    plane, 
                    transform=None, 
                    train=False)
loader = torch.utils.data.DataLoader(dataset, 
                                     batch_size=1, 
                                     shuffle=False, 
                                     num_workers=0, 
                                     drop_last=False)

def returnCAM(feature_conv, weight_softmax, class_idx):
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    slice_cams = []
    for s in range(bz):
        for idx in class_idx:
            cam = weight_softmax[idx].dot(feature_conv[s].reshape((nc, h*w)))
            cam = cam.reshape(h, w)
            cam = cam - np.min(cam)
            cam_img = cam / np.max(cam)
            cam_img = np.uint8(255 * cam_img)
            slice_cams.append(cv2.resize(cam_img, size_upsample))
    return slice_cams