import shutil
import sys
sys.path.append('../../mrnet')

import torch
import model as model
from dataloader import MRDataset
from tqdm import notebook as tqdm_notebook

tqdm_notebook = tqdm_notebook.tqdm_notebook

import os
import numpy as np
import cv2

import torchvision

from PIL import Image

task = 'acl'
plane = 'sagittal'
prefix = 'test'

model_name = [name  for name in os.listdir('../models/') 
              if (task in name) and 
                 (plane in name) and 
                 (prefix in name)][0]

is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")

mrnet: model.MRNet = torch.load(f'../models/{model_name}')
mrnet = mrnet.to(device)

_ = mrnet.eval()

dataset = MRDataset('../data/', 
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
            
            # Convert feature_conv[s] to a NumPy array
            feature_conv_np = torch.as_tensor(feature_conv[s]).cpu().numpy()

            # Reshape weight_softmax[idx] to have the same number of columns as feature_conv[s].reshape((nc, h*w))
            weight_softmax_np = torch.as_tensor(weight_softmax[idx]).cpu().numpy().reshape(1, 1)

            print(type(feature_conv_np))
            print(type(weight_softmax_np))

            cam = weight_softmax_np.dot(feature_conv_np.reshape((nc, h*w)))
            cam = cam.reshape(h, w)
            cam = cam - np.min(cam)
            cam_img = cam / np.max(cam)
            cam_img = np.uint8(255 * cam_img)
            slice_cams.append(cv2.resize(cam_img, size_upsample))
    return slice_cams

patients = []

for i, (image, label, _) in tqdm_notebook(enumerate(loader), total=len(loader)):
    patient_data = {}
    patient_data['mri'] = image
    patient_data['label'] = label[0][0].item()
    patient_data['id'] = '0' * (4 - len(str(i))) + str(i)
    patients.append(patient_data)

acl = list(filter(lambda d: d['label'] == 1, patients))

print(f'Number of patients with ACL: {len(acl)}')

for name, module in mrnet.named_children():
    if not name.startswith('params'):
        print(name)
        print(module)
        print('------')

def create_patiens_cam(case, plane):
    patient_id = case['id']
    mri = case['mri']

    print(f'Patient ID: {patient_id}')

    folder_path = f'./CAMS/{plane}/{patient_id}/'
    if os.path.isdir(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)
    os.makedirs(folder_path + 'slices/')
    os.makedirs(folder_path + 'cams/')
    
    params = list(mrnet.parameters())
    weight_softmax = np.squeeze(params[-2].cpu().data.numpy())
    
    num_slices = mri.shape[1]
    print(f'Number of slices: {num_slices}')
    global feature_blobs

    #Set the features blob as the output of the last convolutional layer

    feature_blobs = []
    def hook_feature(module, input, output):
        feature_blobs.append(output.data.cpu().numpy())

    finalconv_name = 'pooling_layer'

    mrnet._modules.get(finalconv_name).register_forward_hook(hook_feature)

    mri = mri.to(device)

    logit = mrnet(mri)

    h_x = torch.softmax(logit, dim=1).data.squeeze(0)

    probs, idx = h_x.sort(0, True)
    probs = probs.cpu().numpy()

    idx = idx.cpu().numpy()

    weight_softmax = torch.as_tensor(weight_softmax).cpu().detach().numpy()

    print(type(weight_softmax))

    print(f"IDX: {idx[:1]} WEIGHT_SOFTMAX_SHAPE: {weight_softmax.shape} IDX_W: {weight_softmax[idx[:1]].shape}")

    slice_cams = returnCAM(feature_blobs[-1], weight_softmax, idx[:1])
    
    for s in tqdm_notebook(range(num_slices), leave=False):
        print(s)
        slice_pil = (torchvision.transforms.ToPILImage()(mri.cpu()[0][s] / 255))
        slice_pil.save(folder_path + f'slices/{s}.png', 
                       dpi=(300, 300))
         
        img = mri[0][s].cpu().numpy()
        img = img.transpose(1, 2, 0)
        heatmap = (cv2
                    .cvtColor(cv2.applyColorMap(
                        cv2.resize(slice_cams[s], (256, 256)),
                        cv2.COLORMAP_JET), 
                               cv2.COLOR_BGR2RGB)
                  )
        result = heatmap * 0.3 + img * 0.5  
        
        pil_img_cam = Image.fromarray(np.uint8(result))
        pil_img_cam.save(folder_path + f'cams/{s}.png', dpi=(300, 300))

for case in tqdm_notebook(acl):
    print(f'Patient ID: {case["id"]}')
    create_patiens_cam(case, plane)