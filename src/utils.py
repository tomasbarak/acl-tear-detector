#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pickle
import platform

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from ipywidgets import interact
from scipy import ndimage
from sklearn.metrics import (ConfusionMatrixDisplay, auc,
                             balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.preprocessing import label_binarize
from sklearn.utils import class_weight, shuffle
import keras
from keras.models import load_model


# In[2]:


def explore_3D_volume(vol: np.ndarray, cmap: str = 'gray'):
    """
    Given a 3D volumteric array with shape (Z,X,Y). This function will create an interactive
    widget to check out all the 2D arrays with shape (X,Y) inside the 3D array.
    The purpose of this function to visually inspect the 2D arrays in the image.

    Args:
        vol (np.ndarray): 3D array with shape (Z,X,Y) that represents the volume of a MRI image
        cmap (str, optional): color map use to plot the slices in matplotlib.pyplot
    """
    def fn(SLICE):
        """
        This function plots MRI slice.

        Args:
            SLICE (NumPy array): MRI Slice
        """
        plt.figure(figsize=(5, 5))
        plt.imshow(vol[SLICE, :, :], cmap=cmap)

    interact(fn, SLICE=(0, vol.shape[0] - 1))


# In[3]:


def compare_3D_volume(vol_before: np.ndarray, vol_after: np.ndarray, cmap: str = 'gray'):
    """
    Given two 3D volumetric arrays with shape (Z,X,Y). This function will create an interactive
    widget to check out all the 2D arrays with shape (X,Y) inside the 3D arrays.
    The purpose of this function to visual compare the 2D arrays after some transformations.

    Args:
        vol_before (np.ndarray): 3D array with shape (Z,X,Y) that represents the volume of a MRI image, before any transform
        vol_after (np.ndarray): 3D array with shape (Z,X,Y) that represents the volume of a MRI image, after some transform
        cmap (str, optional): Which color map use to plot the slices in matplotlib.pyplot
    """
    assert vol_after.shape == vol_before.shape

    def fn(SLICE):
        """
        This function plots before and after MRI slices.

        Args:
            SLICE (NumPy array): MRI Slice
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, sharex='col', sharey='row', figsize=(10, 10))

        ax1.set_title('Before', fontsize=15)
        ax1.imshow(vol_before[SLICE, :, :], cmap=cmap)

        ax2.set_title('After', fontsize=15)
        ax2.imshow(vol_after[SLICE, :, :], cmap=cmap)

        plt.tight_layout()

    interact(fn, SLICE=(0, vol_before.shape[0] - 1))


# In[4]:


def resize_3D_volume(vol, target_size=(30, 256, 256)):
    """
    Given a 3D volumteric array with shape (Z,X,Y). This function will resize
    the image across z-axis.
    The purpose of this function to standardise the depth of MRI image.

    Args:
        vol: 3D array with shape (Z,X,Y) that represents the volume of a MRI image
        target_size: target size to shape into the volumetric data

    Returns:
        np.ndarray: Returns the resized MRI volume
    """
    # Set the desired depth
    desired_depth, desired_width, desired_height = target_size
    # Get current depth
    current_depth = vol.shape[0]
    current_width = vol.shape[1]
    current_height = vol.shape[2]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Resize across z-axis
    resized_vol = ndimage.zoom(vol, (depth_factor, width_factor, height_factor), order=1)
    return resized_vol


# In[5]:


def denoise_3D_volume(vol):
    """Summary

    Args:
        vol (np.ndarray): MRI volume to denoise

    Returns:
        np.ndarray: Returns denoised MRI volume
    """
    vol_sitk = sitk.GetImageFromArray(vol)
    denoised_vol_sitk = sitk.CurvatureFlow(vol_sitk, timeStep=0.01, numberOfIterations=7)
    denoised_vol = sitk.GetArrayFromImage(denoised_vol_sitk)
    return denoised_vol


# In[6]:


def bias_field_correction_volume(vol):
    """Summary

    Args:
        vol (np.ndarray): MRI volume to perform bias field correction

    Returns:
        np.ndarray: Returns bias field corrected MRI volume
    """
    # Convert the NumPy array to SimpleITK image
    vol_sitk = sitk.GetImageFromArray(vol)

    # Perform bias field correction using N4BiasFieldCorrection
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    bias_corrected_vol_sitk = corrector.Execute(vol_sitk)

    # Get the NumPy array representation of the bias-corrected volume
    bias_corrected_vol = sitk.GetArrayFromImage(bias_corrected_vol_sitk)

    return bias_corrected_vol


# In[7]:


def efficient_bias_field_correction_volume(vol):
    """Summary

    Args:
        vol (np.ndarray): MRI volume to perform efficient bias field correction

    Returns:
        np.ndarray: Returns bias field corrected MRI volume
    """
    # Ref: https://medium.com/@alexandro.ramr777/how-to-do-bias-field-correction-with-python-156b9d51dd79
    # Ref: https://simpleitk.readthedocs.io/en/master/link_N4BiasFieldCorrection_docs.html
    # Convert the NumPy array to SimpleITK image
    vol_sitk = sitk.GetImageFromArray(vol)

    vol_sitk = sitk.Cast(vol_sitk, sitk.sitkFloat64)

    vol_sitk_transformed = sitk.RescaleIntensity(vol_sitk, 0, 255)

    vol_sitk_transformed = sitk.LiThreshold(vol_sitk_transformed, 0, 1)

    head_mask = vol_sitk_transformed

    shrink_factor = 4

    input_img = vol_sitk

    input_img = sitk.Shrink(vol_sitk, [shrink_factor] * input_img.GetDimension())
    mask_img = sitk.Shrink(head_mask, [shrink_factor] * input_img.GetDimension())

    # Perform bias field correction using N4BiasFieldCorrection
    bias_corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected = bias_corrector.Execute(input_img, mask_img)

    log_bias_field = bias_corrector.GetLogBiasFieldAsImage(vol_sitk)

    log_bias_field = sitk.Cast(log_bias_field, sitk.sitkFloat64)

    corrected_image_full_resolution = vol_sitk / sitk.Exp(log_bias_field)

    # Get the NumPy array representation of the bias-corrected volume
    bias_corrected_vol = sitk.GetArrayFromImage(corrected_image_full_resolution)

    return bias_corrected_vol


# In[8]:


def standardise_volume_pixels(vol):
    """Summary

    Args:
        vol (np.ndarray): MRI volume

    Returns:
        np.ndarray: Standardised MRI volume
    """
    # Calculate the mean and standard deviation
    mean_value = np.mean(vol)
    std_value = np.std(vol)

    # Standardise the data
    standardised_vol = (vol - mean_value) / std_value

    return standardised_vol


# In[9]:


def center_volume_pixels(vol):
    """Summary

    Args:
        vol (np.ndarray): MRI volume

    Returns:
        np.ndarray: Zero centered MRI volume
    """
    # Calculate the mean value
    mean_value = np.mean(vol)

    # Center the data
    centered_vol = vol - mean_value

    return centered_vol


# In[10]:


def normalise_volume_pixels(vol):
    """Summary

    Args:
        vol (np.ndarray): MRI volume

    Returns:
        np.ndarray: Normalised MRI volume
    """
    # Normalise the volume pixels to the range [0, 1]
    min_value = np.min(vol)
    max_value = np.max(vol)
    normalised_vol = (vol - min_value) / (max_value - min_value)

    return normalised_vol


# In[11]:


def random_rotation(vol, rotation_angles=[-2.0, -1.5, -1.0, 1.0, 1.5, 2.0, ]):
    """Summary

    Args:
        vol (np.ndarray): MRI volume
        rotation_angles (list, optional): List angles for random rotations

    Returns:
        np.ndarray: Returns randomly rotated MRI volume
    """
    rotation_angle = np.random.choice(rotation_angles)
    # print(f"Rotation by {rotation_angle} degrees.")
    rotated_vol = ndimage.rotate(vol, rotation_angle, reshape=False, mode='nearest')
    # print(rotated_vol.shape)
    return rotated_vol


# In[12]:


def random_horizontal_flip(vol):
    """Summary

    Args:
        vol (np.ndarray): MRI volume

    Returns:
        np.ndarray: Returns horizontally flipped MRI volume
    """
    flipped_vol = np.flip(vol, axis=2)
    return flipped_vol


# In[13]:


def preprocess_mri(mri_vol):
    """Summary

    Args:
        mri_vol (np.ndarray): MRI volume

    Returns:
        np.ndarray: Returns preprocessed MRI volume
    """
    mri_vol = resize_3D_volume(mri_vol)
    mri_vol = denoise_3D_volume(mri_vol)
    mri_vol = efficient_bias_field_correction_volume(mri_vol)
    mri_vol = normalise_volume_pixels(mri_vol)
    mri_vol = center_volume_pixels(mri_vol)
    mri_vol = standardise_volume_pixels(mri_vol)
    return mri_vol


# In[14]:


def get_correct_labels_mrnet(filenames, labels_dataframe):
    """Summary

    Args:
        filenames (list): List of filenames of the MRI scans
        labels_dataframe (pd.Dataframe): Dataframe with all MRNet cases and labels

    Returns:
        list: List of corresponding labels for given MRNet MRI filenames
    """
    labels = []
    for file in filenames:
        name = os.path.normpath(file).split(os.sep)[-1]
        case_name = name.split('.')[0]
        label = labels_dataframe.loc[labels_dataframe['Case'] == case_name, 'ACL'].tolist()[0]
        labels.append(label)
    return labels


# In[15]:


def get_correct_labels_kneemri(filenames, labels_dataframe):
    """Summary

    Args:
        filenames (list): List of filenames of the MRI scans
        labels_dataframe (pd.Dataframe): Dataframe with all KneeMRI metadata and labels

    Returns:
        list: List of corresponding labels for given KneeMRI case filenames
    """
    labels = []
    for file in filenames:
        name = os.path.normpath(file).split(os.sep)[-1]
        vol_file_name = name.split('.')[0] + '.pck'
        label = labels_dataframe.loc[labels_dataframe['volumeFilename'] == vol_file_name, 'aclDiagnosis'].tolist()[0]
        labels.append(label)
    return labels


# In[16]:


def batch_generator(filenames, labels, batch_size):
    '''
    This function loads the respective filenames and labels in the memory 
    based on the parameter batch size. It helps to control the amount of
    RAM being consumed as the datasets are large.

    Args:
        filenames (list): List of file paths to the MRI
        labels (list): List of corresponding labels of the MRI
        batch_size (int): Batch size

    Yields:
        tuple: Tuple of list of loaded MRI files and corresponding labels
    '''
    N = len(filenames)
    i = 0
    random_state_counter = 610
    filenames, labels = shuffle(filenames, labels, random_state=random_state_counter + 69)  # Shuffle at the start
    while True:
        batch_images = []
        batch_filenames = filenames[i:i + batch_size]
        for file in batch_filenames:
            mri_vol = np.load(file)
            mri_vol = np.expand_dims(mri_vol, axis=3)  # Adding extra axis for making it compatible for 3D Convolutions
            batch_images.append(mri_vol)
        batch_labels = labels[i:i + batch_size]
        batch_images = np.array(batch_images)
        batch_labels = np.array(batch_labels)
        yield (batch_images, batch_labels)
        i = i + batch_size
        if i + batch_size > N:
            i = 0
            random_state_counter += 1
            filenames, labels = shuffle(filenames, labels, random_state=random_state_counter + 69)  # Shuffle at the end of each epoch


# In[17]:


def predict_batch_generator(filenames, batch_size):
    '''
    This function loads the respective filenames and labels in the memory 
    based on the parameter batch size. It helps to control the amount of
    RAM being consumed as the datasets are large.

    Args:
        filenames (list): List of filenames of MRI
        batch_size (int): Batch size

    Yields:
        list: List of loaded MRIs
    '''
    N = len(filenames)
    i = 0
    while i < N:
        batch_images = []
        batch_filenames = filenames[i:i + batch_size]
        for file in batch_filenames:
            mri_vol = np.load(file)
            mri_vol = np.expand_dims(mri_vol, axis=3)  # Adding extra axis for making it compatible for 3D Convolutions
            batch_images.append(mri_vol)
        batch_images = np.array(batch_images)
        yield batch_images
        i = i + batch_size


# In[18]:


def extract_middle_three(mri_vol):
    """Summary

    Args:
        mri_vol (np.ndarray): MRI volume

    Returns:
        np.ndarray: Returns the extracted middle three slices reshaped to (256,256,3)
    """
    middle_index = int(len(mri_vol) / 2) - 1
    extracted_portion = mri_vol[middle_index - 1:middle_index + 2]
    extracted_portion = extracted_portion.reshape(256, 256, 3)
    return extracted_portion


# In[19]:


def batch_generator_tf_3(filenames, labels, batch_size):
    '''
    This function loads the respective filenames and labels in the memory
    based on the parameter batch size. It helps to control the amount of
    RAM being consumed as the datasets are large.

    Args:
        filenames (list): List of file paths to MRI
        labels (list): List of corresponding labels
        batch_size (int): Batch Size

    Yields:
        tuple: Tuple of list of loaded MRIs with middle three slices extracted and corresponding labels
    '''
    N = len(filenames)
    i = 0
    random_state_counter = 610
    filenames, labels = shuffle(filenames, labels, random_state=random_state_counter + 69)  # Shuffle at the start
    while True:
        batch_images = []
        batch_filenames = filenames[i:i + batch_size]
        for file in batch_filenames:
            mri_vol = np.load(file)
            extracted = extract_middle_three(mri_vol)
            # mri_vol = np.expand_dims(mri_vol, axis=3) # Adding extra axis for making it compatible for 3D Convolutions
            batch_images.append(extracted)
        batch_labels = labels[i:i + batch_size]
        batch_images = np.array(batch_images)
        batch_labels = np.array(batch_labels)
        yield (batch_images, batch_labels)
        i = i + batch_size
        if i + batch_size > N:
            i = 0
            random_state_counter += 1
            filenames, labels = shuffle(filenames, labels, random_state=random_state_counter + 69)  # Shuffle at the end of each epoch


# In[20]:


def predict_batch_generator_tf_3(filenames, batch_size):
    '''
    This function loads the respective filenames and labels in the memory
    based on the parameter batch size. It helps to control the amount of
    RAM being consumed as the datasets are large.

    Args:
        filenames (list): List of file paths to MRI
        batch_size (int): Batch Size

    Yields:
        list: List of loaded MRI with middle three slices extracted
    '''
    N = len(filenames)
    i = 0
    while i < N:
        batch_images = []
        batch_filenames = filenames[i:i + batch_size]
        for file in batch_filenames:
            mri_vol = np.load(file)
            extracted = extract_middle_three(mri_vol)
            # mri_vol = np.expand_dims(mri_vol, axis=3) # Adding extra axis for making it compatible for 3D Convolutions
            batch_images.append(extracted)
        batch_images = np.array(batch_images)
        yield batch_images
        i = i + batch_size


# In[21]:


def extract_middle_five(mri_vol):
    """Summary

    Args:
        mri_vol (np.ndarray): MRI volume

    Returns:
        np.ndarray: Returns the extracted middle five slices reshaped to (256,256,5)
    """
    middle_index = int(len(mri_vol) / 2) - 1
    extracted_portion = mri_vol[middle_index - 2:middle_index + 3]
    extracted_portion = extracted_portion.reshape(256, 256, 5)
    return extracted_portion


# In[22]:


def batch_generator_tf_5(filenames, labels, batch_size):
    '''
    This function loads the respective filenames and labels in the memory
    based on the parameter batch size. It helps to control the amount of
    RAM being consumed as the datasets are large.

    Args:
        filenames (list): List of file paths to MRI
        labels (list): List of corresponding labels
        batch_size (int): Batch Size

    Yields:
        tuple: Tuple of list of loaded MRIs with middle five slices extracted and corresponding labels
    '''
    N = len(filenames)
    i = 0
    random_state_counter = 610
    filenames, labels = shuffle(filenames, labels, random_state=random_state_counter + 69)  # Shuffle at the start
    while True:
        batch_images = []
        batch_filenames = filenames[i:i + batch_size]
        for file in batch_filenames:
            mri_vol = np.load(file)
            extracted = extract_middle_five(mri_vol)
            # mri_vol = np.expand_dims(mri_vol, axis=3) # Adding extra axis for making it compatible for 3D Convolutions
            batch_images.append(extracted)
        batch_labels = labels[i:i + batch_size]
        batch_images = np.array(batch_images)
        batch_labels = np.array(batch_labels)
        yield (batch_images, batch_labels)
        i = i + batch_size
        if i + batch_size > N:
            i = 0
            random_state_counter += 1
            filenames, labels = shuffle(filenames, labels, random_state=random_state_counter + 69)  # Shuffle at the end of each epoch


# In[23]:


def predict_batch_generator_tf_5(filenames, batch_size):
    '''
    This function loads the respective filenames and labels in the memory
    based on the parameter batch size. It helps to control the amount of
    RAM being consumed as the datasets are large.

    Args:
        filenames (list): List of file paths to MRI
        batch_size (int): Batch Size

    Yields:
        list: List of loaded MRI with middle five slices extracted
    '''
    N = len(filenames)
    i = 0
    while i < N:
        batch_images = []
        batch_filenames = filenames[i:i + batch_size]
        for file in batch_filenames:
            mri_vol = np.load(file)
            extracted = extract_middle_five(mri_vol)
            # mri_vol = np.expand_dims(mri_vol, axis=3) # Adding extra axis for making it compatible for 3D Convolutions
            batch_images.append(extracted)
        batch_images = np.array(batch_images)
        yield batch_images
        i = i + batch_size


# In[24]:


def compute_class_weights(y_train):
    """Summary

    Args:
        y_train (list): List of labels

    Returns:
        dict: A dictionary of labels and their corresponding class weights
    """
    class_weights = dict(zip(np.unique(y_train),
                             class_weight.compute_class_weight(class_weight='balanced',
                                                               classes=np.unique(y_train),
                                                               y=y_train)))
    return class_weights


# In[25]:


# MODEL CALLBACK FUNCTIONS

def model_lr_schedule(initial_lr=0.0001):
    """Summary

    Args:
        initial_lr (float, optional): Initial learning rate to be set

    Returns:
        TYPE: Learning rate scheduler for Keras
    """
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_lr,
                                                              decay_steps=100000,
                                                              decay_rate=0.96,
                                                              staircase=True)
    return lr_schedule


def model_callback_checkpoint(model_name, model_store_path='Models'):
    """Summary

    Args:
        model_name (str): Name of the model
        model_store_path (str, optional): Path to store the models

    Returns:
        TYPE: Keras checkpoint callback to store the best model
    """
    file_name = f"{model_store_path}/{model_name}/{model_name}.h5"

    # For running code on Windows
    if platform.system() == "Windows":
        file_name = file_name.replace('/', '\\')

    checkpoint_callback = keras.callbacks.ModelCheckpoint(file_name,
                                                          save_best_only=True)
    return checkpoint_callback


def model_callback_earlystopping():
    """Summary

    Returns:
        TYPE: Keras earlystopping callback for monitoring Validation Loss
    """
    earlystopping_callback = keras.callbacks.EarlyStopping(monitor="val_loss",
                                                           patience=10,
                                                           verbose=1,
                                                           restore_best_weights=True)
    return earlystopping_callback


# In[26]:


def store_model_history(model_name, model_history, model_history_path='Models'):
    """Summary

    Args:
        model_name (str): Model name
        model_history (TYPE): Keras model history
        model_history_path (str, optional): Path to store model history
    """
    file_name = f"{model_history_path}/{model_name}/{model_name}-history.pck"

    # For running code on Windows
    if platform.system() == "Windows":
        file_name = file_name.replace('/', '\\')

    parent_directory = os.path.dirname(file_name)
    if not os.path.exists(parent_directory):
        os.makedirs(parent_directory, exist_ok=True)

    with open(file_name, 'wb') as fh:
        pickle.dump(model_history, fh)


# In[27]:


def load_model_history(model_name, model_history_path='Models'):
    """Summary

    Args:
        model_name (str): Model name
        model_history_path (str, optional): Path where model history is stored

    Returns:
        TYPE: Keras model history
    """
    file_name = f"{model_history_path}/{model_name}/{model_name}-history.pck"

    # For running code on Windows
    if platform.system() == "Windows":
        file_name = file_name.replace('/', '\\')

    if os.path.exists(file_name):
        with open(file_name, 'rb') as fh:
            history = pickle.load(fh)
    else:
        print(f"ERROR: History file {file_name} not found.")
        return None

    return history


# In[28]:


def load_model_from_disk(model_name, model_store_path='Models'):
    """Summary

    Args:
        model_name (str): Model name
        model_store_path (str, optional): Path to load a model from

    Returns:
        TYPE: Description
    """
    file_name = f"{model_store_path}/{model_name}/{model_name}.h5"

    # For running code on Windows
    if platform.system() == "Windows":
        file_name = file_name.replace('/', '\\')

    if os.path.exists(file_name):
        model = load_model(file_name)
    else:
        print(f"ERROR: Model file {file_name} not found.")
        return None

    return model


# In[29]:


# Function for Model Evaluation

def evaluate_model(true_labels, predicted_labels, predicted_probs, label_names):
    """Summary

    Args:
        true_labels (list): List of true labels
        predicted_labels (list): List of predicted labels
        predicted_probs (list): List of predicted probabilities
        label_names (list): List of labels
    """
    print('\nEvaluation Metrics:\n')
    # Check for multi-class, else proceed for binary
    if len(label_names) > 2:
        print(f"Balanced Accuracy : {round(balanced_accuracy_score(true_labels, predicted_labels), 2)}")
        print(f"Precision : {round(precision_score(true_labels, predicted_labels, average='weighted'), 2)}")
        print(f"Recall : {round(recall_score(true_labels, predicted_labels, average='weighted'), 2)}")
        print(f"F1 Score: {round(f1_score(true_labels, predicted_labels, average='weighted'), 2)}")
        print(f"ROC AUC Score : {round(roc_auc_score(true_labels, predicted_probs, multi_class='ovr'), 2)}")
    else:
        print(f"Balanced Accuracy : {round(balanced_accuracy_score(true_labels, predicted_labels), 2)}")
        print(f"Precision : {round(precision_score(true_labels, predicted_labels), 2)}")
        print(f"Recall : {round(recall_score(true_labels, predicted_labels), 2)}")
        print(f"F1 Score: {round(f1_score(true_labels, predicted_labels), 2)}")
        print(f"ROC AUC Score : {round(roc_auc_score(true_labels, predicted_probs), 2)}")

    print("\nClassification report : ")
    print(classification_report(true_labels, predicted_labels, target_names=label_names))

    matrix = confusion_matrix(true_labels, predicted_labels)
    print("\nConfusion Matrix : ")
    # print(matrix)
    ConfusionMatrixDisplay(matrix, display_labels=label_names).plot(cmap=plt.cm.Blues)


# In[30]:


# Function to plot Accuracy and Loss of a model

def plot_acc_loss(model_history):
    """Summary

    Args:
        model_history (TYPE): Model history
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6), dpi=160)

    acc = model_history['accuracy']
    val_acc = model_history['val_accuracy']

    # Get number of epochs
    epochs = range(1, len(acc) + 1)

    # Plot training and validation accuracy per epoch
    ax1.plot(epochs, acc, label="Training Accuracy")
    ax1.plot(epochs, val_acc, label="Validation Accuracy")
    ax1.set_title('Training and Validation Accuracy')
    ax1.set_xlabel('EPOCHS')
    ax1.set_ylabel('ACCURACY')
    ax1.legend()

    loss = model_history['loss']
    val_loss = model_history['val_loss']
    
    # Plot training and validation loss per epoch
    ax2.plot(epochs, loss, label="Training Loss")
    ax2.plot(epochs, val_loss, label="Validation Loss")
    ax2.set_title('Training and Validation Loss')
    ax2.set_xlabel('EPOCHS')
    ax2.set_ylabel('LOSS')
    ax2.legend()
    # plt.show()


# In[32]:


def calculate_best_cutoff_threshold(labels, predicted_probs):
    """Summary

    Args:
        labels (list): List of true labels
        predicted_probs (list): List of predicted probabilities

    Returns:
        TYPE: Description
    """
    # Calculate PR Curve
    precision, recall, thresholds = precision_recall_curve(labels, predicted_probs)

    # Convert to f score
    fscore = (2 * precision * recall) / (precision + recall)

    # Locate the index of the largest f score
    ix = np.argmax(fscore)
    print('\nBest cutoff Threshold = %f, F-Score = %.3f' % (thresholds[ix], fscore[ix]))
    return thresholds[ix]

