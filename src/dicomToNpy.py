import pydicom
import os
import numpy as np
import cv2
from skimage import exposure

dicom_folder = "mri/29581351 BARAK TOMAS/532299 RM DE RODILLA DERECHA SIN CONTRASTE/MR PD FINO CORONAL" # Set the folder of your dicom files that inclued images 
jpg_folder = 'converted/jpg' # Set the folder of your output folder for jpg files 
npy_folder = 'converted/npy' # Set the folder of your output folder for npy files

if not os.path.exists(dicom_folder):
    os.makedirs(dicom_folder)
if not os.path.exists(jpg_folder):
    os.makedirs(jpg_folder)
if not os.path.exists(npy_folder):
    os.makedirs(npy_folder)


# Step 1. prepare your input(.dcm) and output(.jpg) filepath 
dcm_jpg_map = {}
for dicom_f in os.listdir(dicom_folder):
    dicom_filepath = os.path.join(dicom_folder, dicom_f)
    jpg_f = dicom_f.replace('.dcm', '.jpg') 
    jpg_filepath = os.path.join(jpg_folder,jpg_f)
    dcm_jpg_map[dicom_filepath] = jpg_filepath

unstacked_list = []
for dicom_filepath, jpg_filepath in dcm_jpg_map.items():
    # convert dicom file into jpg file
    dicom = pydicom.read_file(dicom_filepath)
    if (not "PixelData" in dicom):
        continue
    np_pixel_array = dicom.pixel_array

    np_pixel_array=exposure.equalize_adapthist(np_pixel_array)

    #Resize from 560x560 to 256x256
    np_pixel_array = cv2.resize(np_pixel_array, (256, 256))

    unstacked_list.append(np_pixel_array)

    image_to_save = cv2.convertScaleAbs(np_pixel_array, alpha=(255.0))

    #Check if the folder exists
    cv2.imwrite(jpg_filepath, image_to_save)

final_array = np.array(unstacked_list)

print(final_array.shape)

np.save(os.path.join(npy_folder, "1.npy"), final_array)