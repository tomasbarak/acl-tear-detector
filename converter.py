import pydicom
#IMPORT ALL FROM pydicom.pixel_data_handlers.util as util
import pydicom.pixel_data_handlers.util as util
import os
import numpy as np
import cv2
from skimage import exposure

dicom_folder = './testing/sagittal01/' # Set the folder of your dicom files that inclued images 
jpg_folder = './testing/sagittal01_JPG' # Set the folder of your output folder for jpg files 
# Step 1. prepare your input(.dcm) and output(.jpg) filepath 
dcm_jpg_map = {}
for dicom_f in os.listdir(dicom_folder):
    print(dicom_f)
    dicom_filepath = os.path.join(dicom_folder, dicom_f)
    jpg_f = dicom_f.replace('.dcm', '.jpg') 
    jpg_filepath = os.path.join(jpg_folder,jpg_f)
    dcm_jpg_map[dicom_filepath] = jpg_filepath

# Now, dcm_jpg_map is key,value pair of input dcm filepath and output jpg filepath

for dicom_filepath, jpg_filepath in dcm_jpg_map.items():
    # convert dicom file into jpg file
    dicom = pydicom.read_file(dicom_filepath)
    if (dicom.__len__() != 367):
        continue
    np_pixel_array = dicom.pixel_array

    np_pixel_array=exposure.equalize_adapthist(np_pixel_array)

    image_to_save = cv2.convertScaleAbs(np_pixel_array, alpha=(255.0))

    cv2.imwrite(jpg_filepath, image_to_save)

    cv2.imshow('image', image_to_save)

# import numpy as np

# npy_path = './data/train/sagittal/0110.npy'

# npy = np.load(npy_path)

# #Remove the first dimension
# npy = npy[0]

# #Show the shape of the numpy array
# print(npy.shape)

# #Convert npy array to image using opencv
# cv2.imwrite('0000.jpg', npy)

# #Show 
# cv2.imshow('image', npy)

# cv2.waitKey(0)