#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.layers import (BatchNormalization, Conv3D, Dense,
                                     Dropout, Flatten, GlobalAveragePooling3D,
                                     MaxPooling3D)
from tensorflow.keras.models import Model, Sequential
from tensorflow.python.client import device_lib


# In[10]:


print(f"\nTensorflow version : {tf.__version__}")


# In[6]:


print(f"\nTensorflow devices available : \n {device_lib.list_local_devices()}")


# In[8]:


print(f"\nTensorflow physical devices available : \n {tf.config.list_physical_devices()}")


# In[11]:


def mri_model_1(model_name, num_classes, depth=30, width=256, height=256):
    """Summary

    Args:
        model_name (str): Model name
        num_classes (int): Number of classes
        depth (int, optional): Depth of MRI volume
        width (int, optional): Width of MRI volume
        height (int, optional): Height of MRI volume

    Returns:
        TYPE: Keras Model
    """
    # Determine the number of units and activation function of the last layer
    # based on the input number of classes
    if num_classes == 2:
        last_layer_units = 1
        last_layer_activation = 'sigmoid'
    elif num_classes > 2:
        last_layer_units = num_classes
        last_layer_activation = 'softmax'

    model = Sequential([
        Conv3D(64, kernel_size=(3, 3, 3), padding="same", activation='relu', input_shape=(depth, width, height, 1)),
        MaxPooling3D(pool_size=(2, 2, 2), padding="same"),
        BatchNormalization(),
        Conv3D(64, kernel_size=(3, 3, 3), padding="same", activation='relu'),
        MaxPooling3D(pool_size=(2, 2, 2), padding="same"),
        BatchNormalization(),
        Conv3D(128, kernel_size=(3, 3, 3), padding="same", activation='relu'),
        MaxPooling3D(pool_size=(2, 2, 2), padding="same"),
        BatchNormalization(),
        Conv3D(128, kernel_size=(3, 3, 3), padding="same", activation='relu'),
        MaxPooling3D(pool_size=(2, 2, 2), padding="same"),
        BatchNormalization(),
        Conv3D(256, kernel_size=(3, 3, 3), padding="same", activation='relu'),
        MaxPooling3D(pool_size=(2, 2, 2), padding="same"),
        BatchNormalization(),
        GlobalAveragePooling3D(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(last_layer_units, activation=last_layer_activation)
    ], name=model_name)

    return model


# In[1]:


def mri_model_2(model_name, num_classes, depth=30, width=256, height=256):
    """Summary

    Args:
        model_name (str): Model name
        num_classes (int): Number of classes
        depth (int, optional): Depth of MRI volume
        width (int, optional): Width of MRI volume
        height (int, optional): Height of MRI volume

    Returns:
        TYPE: Keras Model
    """
    # Determine the number of units and activation function of the last layer
    # based on the input number of classes
    if num_classes == 2:
        last_layer_units = 1
        last_layer_activation = 'sigmoid'
    elif num_classes > 2:
        last_layer_units = num_classes
        last_layer_activation = 'softmax'

    model = Sequential([
        Conv3D(16, kernel_size=(3, 3, 3), padding="same", activation='relu', input_shape=(depth, width, height, 1)),
        MaxPooling3D(pool_size=(2, 2, 2), padding="same"),
        BatchNormalization(),
        Conv3D(32, kernel_size=(3, 3, 3), padding="same", activation='relu'),
        MaxPooling3D(pool_size=(2, 2, 2), padding="same"),
        BatchNormalization(),
        Conv3D(32, kernel_size=(3, 3, 3), padding="same", activation='relu'),
        MaxPooling3D(pool_size=(2, 2, 2), padding="same"),
        BatchNormalization(),
        Conv3D(64, kernel_size=(3, 3, 3), padding="same", activation='relu'),
        MaxPooling3D(pool_size=(2, 2, 2), padding="same"),
        BatchNormalization(),
        Conv3D(64, kernel_size=(3, 3, 3), padding="same", activation='relu'),
        MaxPooling3D(pool_size=(2, 2, 2), padding="same"),
        BatchNormalization(),
        Conv3D(128, kernel_size=(3, 3, 3), padding="same", activation='relu'),
        MaxPooling3D(pool_size=(2, 2, 2), padding="same"),
        BatchNormalization(),
        Conv3D(128, kernel_size=(3, 3, 3), padding="same", activation='relu'),
        MaxPooling3D(pool_size=(2, 2, 2), padding="same"),
        BatchNormalization(),
        Conv3D(256, kernel_size=(3, 3, 3), padding="same", activation='relu'),
        MaxPooling3D(pool_size=(2, 2, 2), padding="same"),
        BatchNormalization(),
        GlobalAveragePooling3D(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(last_layer_units, activation=last_layer_activation)
    ], name=model_name)

    return model


# In[12]:


def mri_model_3(model_name, num_classes, depth=30, width=256, height=256):
    """Summary

    Args:
        model_name (str): Model name
        num_classes (int): Number of classes
        depth (int, optional): Depth of MRI volume
        width (int, optional): Width of MRI volume
        height (int, optional): Height of MRI volume

    Returns:
        TYPE: Keras Model
    """
    # Determine the number of units and activation function of the last layer
    # based on the input number of classes
    if num_classes == 2:
        last_layer_units = 1
        last_layer_activation = 'sigmoid'
    elif num_classes > 2:
        last_layer_units = num_classes
        last_layer_activation = 'softmax'

    model = Sequential([
        Conv3D(32, kernel_size=(3, 3, 3), padding="same", activation='relu', input_shape=(depth, width, height, 1)),
        MaxPooling3D(pool_size=(2, 2, 2), padding="same"),
        BatchNormalization(),
        Conv3D(32, kernel_size=(3, 3, 3), padding="same", activation='relu'),
        MaxPooling3D(pool_size=(2, 2, 2), padding="same"),
        BatchNormalization(),
        Conv3D(32, kernel_size=(3, 3, 3), padding="same", activation='relu'),
        MaxPooling3D(pool_size=(2, 2, 2), padding="same"),
        BatchNormalization(),
        Conv3D(64, kernel_size=(3, 3, 3), padding="same", activation='relu'),
        MaxPooling3D(pool_size=(2, 2, 2), padding="same"),
        BatchNormalization(),
        Conv3D(64, kernel_size=(3, 3, 3), padding="same", activation='relu'),
        MaxPooling3D(pool_size=(2, 2, 2), padding="same"),
        BatchNormalization(),
        Conv3D(64, kernel_size=(3, 3, 3), padding="same", activation='relu'),
        MaxPooling3D(pool_size=(2, 2, 2), padding="same"),
        BatchNormalization(),
        Conv3D(128, kernel_size=(3, 3, 3), padding="same", activation='relu'),
        MaxPooling3D(pool_size=(2, 2, 2), padding="same"),
        BatchNormalization(),
        Conv3D(128, kernel_size=(3, 3, 3), padding="same", activation='relu'),
        MaxPooling3D(pool_size=(2, 2, 2), padding="same"),
        BatchNormalization(),
        Conv3D(128, kernel_size=(3, 3, 3), padding="same", activation='relu'),
        MaxPooling3D(pool_size=(2, 2, 2), padding="same"),
        BatchNormalization(),
        Conv3D(256, kernel_size=(3, 3, 3), padding="same", activation='relu'),
        MaxPooling3D(pool_size=(2, 2, 2), padding="same"),
        BatchNormalization(),
        Conv3D(256, kernel_size=(3, 3, 3), padding="same", activation='relu'),
        MaxPooling3D(pool_size=(2, 2, 2), padding="same"),
        BatchNormalization(),
        GlobalAveragePooling3D(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dropout(0.2),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(last_layer_units, activation=last_layer_activation)
    ], name=model_name)

    return model


# In[13]:


def mri_model_4(model_name, num_classes, depth=30, width=256, height=256):
    """Summary

    Args:
        model_name (str): Model name
        num_classes (int): Number of classes
        depth (int, optional): Depth of MRI volume
        width (int, optional): Width of MRI volume
        height (int, optional): Height of MRI volume

    Returns:
        TYPE: Keras Model
    """
    # Determine the number of units and activation function of the last layer
    # based on the input number of classes
    if num_classes == 2:
        last_layer_units = 1
        last_layer_activation = 'sigmoid'
    elif num_classes > 2:
        last_layer_units = num_classes
        last_layer_activation = 'softmax'

    model = Sequential([
        Conv3D(32, kernel_size=(3, 3, 3), padding="same", activation='relu', input_shape=(depth, width, height, 1)),
        Conv3D(32, kernel_size=(3, 3, 3), padding="same", activation='relu'),
        MaxPooling3D(pool_size=(2, 2, 2), padding="same"),
        Conv3D(32, kernel_size=(3, 3, 3), padding="same", activation='relu'),
        Conv3D(64, kernel_size=(3, 3, 3), padding="same", activation='relu'),
        MaxPooling3D(pool_size=(2, 2, 2), padding="same"),
        Conv3D(64, kernel_size=(3, 3, 3), padding="same", activation='relu'),
        Conv3D(128, kernel_size=(3, 3, 3), padding="same", activation='relu'),
        MaxPooling3D(pool_size=(2, 2, 2), padding="same"),
        Dropout(0.6),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(last_layer_units, activation=last_layer_activation)
    ], name=model_name)

    return model


# In[14]:


def mri_model_5(model_name, num_classes, depth=30, width=256, height=256):
    """Summary

    Args:
        model_name (str): Model name
        num_classes (int): Number of classes
        depth (int, optional): Depth of MRI volume
        width (int, optional): Width of MRI volume
        height (int, optional): Height of MRI volume

    Returns:
        TYPE: Keras Model
    """
    # Determine the number of units and activation function of the last layer
    # based on the input number of classes
    if num_classes == 2:
        last_layer_units = 1
        last_layer_activation = 'sigmoid'
    elif num_classes > 2:
        last_layer_units = num_classes
        last_layer_activation = 'softmax'

    model = Sequential([
        Conv3D(16, kernel_size=(3, 3, 3), padding="same", activation='relu', input_shape=(depth, width, height, 1)),
        MaxPooling3D(pool_size=(2, 2, 2), padding="same"),
        Conv3D(16, kernel_size=(3, 3, 3), padding="same", activation='relu'),
        MaxPooling3D(pool_size=(2, 2, 2), padding="same"),
        Conv3D(32, kernel_size=(3, 3, 3), padding="same", activation='relu'),
        MaxPooling3D(pool_size=(2, 2, 2), padding="same"),
        Conv3D(32, kernel_size=(3, 3, 3), padding="same", activation='relu'),
        MaxPooling3D(pool_size=(2, 2, 2), padding="same"),
        Conv3D(32, kernel_size=(3, 3, 3), padding="same", activation='relu'),
        MaxPooling3D(pool_size=(2, 2, 2), padding="same"),
        Conv3D(64, kernel_size=(3, 3, 3), padding="same", activation='relu'),
        MaxPooling3D(pool_size=(2, 2, 2), padding="same"),
        Conv3D(64, kernel_size=(3, 3, 3), padding="same", activation='relu'),
        MaxPooling3D(pool_size=(2, 2, 2), padding="same"),
        Conv3D(128, kernel_size=(3, 3, 3), padding="same", activation='relu'),
        MaxPooling3D(pool_size=(2, 2, 2), padding="same"),
        Conv3D(128, kernel_size=(3, 3, 3), padding="same", activation='relu'),
        MaxPooling3D(pool_size=(2, 2, 2), padding="same"),
        Conv3D(256, kernel_size=(3, 3, 3), padding="same", activation='relu'),
        MaxPooling3D(pool_size=(2, 2, 2), padding="same"),
        Dropout(0.5),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.4),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.1),
        Dense(last_layer_units, activation=last_layer_activation)
    ], name=model_name)

    return model


# In[15]:


def mri_model_6(model_name, num_classes, depth=30, width=256, height=256):
    """Summary

    Args:
        model_name (str): Model name
        num_classes (int): Number of classes
        depth (int, optional): Depth of MRI volume
        width (int, optional): Width of MRI volume
        height (int, optional): Height of MRI volume

    Returns:
        TYPE: Keras Model
    """
    # Determine the number of units and activation function of the last layer
    # based on the input number of classes
    if num_classes == 2:
        last_layer_units = 1
        last_layer_activation = 'sigmoid'
    elif num_classes > 2:
        last_layer_units = num_classes
        last_layer_activation = 'softmax'

    model = Sequential([
        Conv3D(8, kernel_size=(3, 3, 3), padding="same", activation='relu', input_shape=(depth, width, height, 1)),
        Conv3D(16, kernel_size=(3, 3, 3), padding="same", activation='relu'),
        MaxPooling3D(pool_size=(2, 2, 2), padding="same"),
        Conv3D(16, kernel_size=(3, 3, 3), padding="same", activation='relu'),
        Conv3D(16, kernel_size=(3, 3, 3), padding="same", activation='relu'),
        MaxPooling3D(pool_size=(2, 2, 2), padding="same"),
        Conv3D(16, kernel_size=(3, 3, 3), padding="same", activation='relu'),
        Conv3D(16, kernel_size=(3, 3, 3), padding="same", activation='relu'),
        MaxPooling3D(pool_size=(2, 2, 2), padding="same"),
        BatchNormalization(),
        Conv3D(32, kernel_size=(3, 3, 3), padding="same", activation='relu'),
        Conv3D(32, kernel_size=(3, 3, 3), padding="same", activation='relu'),
        MaxPooling3D(pool_size=(2, 2, 2), padding="same"),
        Conv3D(32, kernel_size=(3, 3, 3), padding="same", activation='relu'),
        MaxPooling3D(pool_size=(2, 2, 2), padding="same"),
        BatchNormalization(),
        Conv3D(64, kernel_size=(3, 3, 3), padding="same", activation='relu'),
        MaxPooling3D(pool_size=(2, 2, 2), padding="same"),
        Conv3D(64, kernel_size=(3, 3, 3), padding="same", activation='relu'),
        MaxPooling3D(pool_size=(2, 2, 2), padding="same"),
        Conv3D(64, kernel_size=(3, 3, 3), padding="same", activation='relu'),
        MaxPooling3D(pool_size=(2, 2, 2), padding="same"),
        BatchNormalization(),
        Conv3D(128, kernel_size=(3, 3, 3), padding="same", activation='relu'),
        MaxPooling3D(pool_size=(2, 2, 2), padding="same"),
        Conv3D(128, kernel_size=(3, 3, 3), padding="same", activation='relu'),
        MaxPooling3D(pool_size=(2, 2, 2), padding="same"),
        Conv3D(128, kernel_size=(3, 3, 3), padding="same", activation='relu'),
        MaxPooling3D(pool_size=(2, 2, 2), padding="same"),
        BatchNormalization(),
        Conv3D(256, kernel_size=(3, 3, 3), padding="same", activation='relu'),
        MaxPooling3D(pool_size=(2, 2, 2), padding="same"),
        Conv3D(256, kernel_size=(3, 3, 3), padding="same", activation='relu'),
        MaxPooling3D(pool_size=(2, 2, 2), padding="same"),
        Conv3D(256, kernel_size=(3, 3, 3), padding="same", activation='relu'),
        MaxPooling3D(pool_size=(2, 2, 2), padding="same"),
        Dropout(0.5),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.1),
        Dense(last_layer_units, activation=last_layer_activation)
    ], name=model_name)

    return model


# In[1]:


def mri_model_tf_3_vgg(model_name, num_classes, depth=30, width=256, height=256):
    """Summary

    Args:
        model_name (str): Model name
        num_classes (int): Number of classes
        depth (int, optional): Depth of MRI volume
        width (int, optional): Width of MRI volume
        height (int, optional): Height of MRI volume

    Returns:
        TYPE: Keras VGG16 Model for Transfer Learning with 3 channels
    """
    # Determine the number of units and activation function of the last layer
    # based on the input number of classes
    if num_classes == 2:
        last_layer_units = 1
        last_layer_activation = 'sigmoid'
    elif num_classes > 2:
        last_layer_units = num_classes
        last_layer_activation = 'softmax'

    # Import VGG16 model with ImageNet weights and specified input shape with three channels
    vgg_model = VGG16(include_top=False, weights='imagenet', input_shape=(width, height, 3))

    # Set VGG16 model as non-trainable
    vgg_model.trainable = False

    model = Sequential([
        vgg_model,
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.4),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.1),
        Dense(last_layer_units, activation=last_layer_activation)
    ], name=model_name)

    return model


# In[4]:


def mri_model_tf_3_xception(model_name, num_classes, depth=30, width=256, height=256):
    """Summary

    Args:
        model_name (str): Model name
        num_classes (int): Number of classes
        depth (int, optional): Depth of MRI volume
        width (int, optional): Width of MRI volume
        height (int, optional): Height of MRI volume

    Returns:
        TYPE: Keras Xception Model for Transfer Learning with 3 channels
    """
    # Determine the number of units and activation function of the last layer
    # based on the input number of classes
    if num_classes == 2:
        last_layer_units = 1
        last_layer_activation = 'sigmoid'
    elif num_classes > 2:
        last_layer_units = num_classes
        last_layer_activation = 'softmax'

    # Import Xception model with ImageNet weights and specified input shape with three channels
    xception_model = Xception(include_top=False, weights='imagenet', input_shape=(width, height, 3))

    # Set Xception model as non-trainable
    xception_model.trainable = False

    model = Sequential([
        xception_model,
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.4),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.1),
        Dense(last_layer_units, activation=last_layer_activation)
    ], name=model_name)

    return model


# In[ ]:


def mri_model_tf_3_resnet(model_name, num_classes, depth=30, width=256, height=256):
    """Summary

    Args:
        model_name (str): Model name
        num_classes (int): Number of classes
        depth (int, optional): Depth of MRI volume
        width (int, optional): Width of MRI volume
        height (int, optional): Height of MRI volume

    Returns:
        TYPE: Keras ResNet50 Model for Transfer Learning with 3 channels
    """
    # Determine the number of units and activation function of the last layer
    # based on the input number of classes
    if num_classes == 2:
        last_layer_units = 1
        last_layer_activation = 'sigmoid'
    elif num_classes > 2:
        last_layer_units = num_classes
        last_layer_activation = 'softmax'

    # Import ResNet50 model with ImageNet weights and specified input shape with three channels
    resnet_model = ResNet50(include_top=False, weights='imagenet', input_shape=(width, height, 3))

    # Set ResNet50 model as non-trainable
    resnet_model.trainable = False

    model = Sequential([
        resnet_model,
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.4),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.1),
        Dense(last_layer_units, activation=last_layer_activation)
    ], name=model_name)

    return model


# In[ ]:


def avg_and_copy_wts(weights, num_channels_to_fill):
    """
    Function to calculate average weights along axes channels and copy to
    remaning n-channels that need to be filled

    Args:
        weights (TYPE): Keras model weights
        num_channels_to_fill (TYPE): Number of channels to fill with the average weights

    Returns:
        TYPE: Returns the new weights after filling n-channels with average of weights
    """
    average_weights = np.mean(weights, axis=-2).reshape(weights[:, :, -1:, :].shape)  # Average along the second to last channel axis channel
    wts_copied_to_mult_channels = np.tile(average_weights, (num_channels_to_fill, 1))  # Repeat the array n-times
    return(wts_copied_to_mult_channels)


# In[ ]:


def mri_model_tf_5_vgg(model_name, num_classes, depth=30, width=256, height=256):
    """Summary

    Args:
        model_name (str): Model name
        num_classes (int): Number of classes
        depth (int, optional): Depth of MRI volume
        width (int, optional): Width of MRI volume
        height (int, optional): Height of MRI volume

    Returns:
        TYPE: Keras VGG16 Model for Transfer Learning with 5 channels
    """
    # Determine the number of units and activation function of the last layer
    # based on the input number of classes
    if num_classes == 2:
        last_layer_units = 1
        last_layer_activation = 'sigmoid'
    elif num_classes > 2:
        last_layer_units = num_classes
        last_layer_activation = 'softmax'

    # Import VGG16 model without input shape
    vgg_model = VGG16(include_top=False, weights='imagenet')

    # Get config dictionary for VGG16
    vgg_config = vgg_model.get_config()

    # Set input shape to new desired shape
    h, w, c = height, width, 5
    vgg_config["layers"][0]["config"]["batch_input_shape"] = (None, h, w, c)

    # Get new model with the updated configuration
    vgg_updated = Model.from_config(vgg_config)

    # Get config. for the updated model
    vgg_updated_config = vgg_updated.get_config()
    # Extract layer names
    vgg_updated_layer_names = [vgg_updated_config['layers'][x]['name'] for x in range(len(vgg_updated_config['layers']))]

    # Grab the first conv block name
    first_conv_name = vgg_updated_layer_names[1]

    # Update weights for all layers, for the first conv layer average first three channel weights to the
    # remaning number out of total channels
    for layer in vgg_model.layers:
        if layer.name in vgg_updated_layer_names:

            if layer.get_weights() != []:  # Check if the layer has weights
                target_layer = vgg_updated.get_layer(layer.name)

                if layer.name in first_conv_name:  # First conv block
                    weights = layer.get_weights()[0]
                    biases = layer.get_weights()[1]

                    # Adjust weights of the extra channels
                    weights_extra_channels = np.concatenate((weights,
                                                             avg_and_copy_wts(weights, c - 3)),
                                                            axis=-2)
                    # Set weights for the first conv block
                    target_layer.set_weights([weights_extra_channels, biases])
                    target_layer.trainable = False  # Set the layer as non-trainable
                else:
                    # Set weights for other layers
                    target_layer.set_weights(layer.get_weights())  
                    target_layer.trainable = False  # Set the layer as non-trainable

    model = Sequential([
        vgg_updated,
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.4),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.1),
        Dense(last_layer_units, activation=last_layer_activation)
    ], name=model_name)

    return model


# In[ ]:


def mri_model_tf_5_resnet(model_name, num_classes, depth=30, width=256, height=256):
    """Summary

    Args:
        model_name (str): Model name
        num_classes (int): Number of classes
        depth (int, optional): Depth of MRI volume
        width (int, optional): Width of MRI volume
        height (int, optional): Height of MRI volume

    Returns:
        TYPE: Keras ResNet50 Model for Transfer Learning with 5 channels
    """
    # Determine the number of units and activation function of the last layer
    # based on the input number of classes
    if num_classes == 2:
        last_layer_units = 1
        last_layer_activation = 'sigmoid'
    elif num_classes > 2:
        last_layer_units = num_classes
        last_layer_activation = 'softmax'

    # Import ResNet50 model without input shape
    resnet_model = ResNet50(include_top=False, weights='imagenet')

    # Get config dictionary for ResNet50
    resnet_config = resnet_model.get_config()

    # Set input shape to new desired shape
    h, w, c = height, width, 5
    resnet_config["layers"][0]["config"]["batch_input_shape"] = (None, h, w, c)

    # Get new model with the updated configuration
    resnet_updated = Model.from_config(resnet_config)

    # Get config. for the updated model
    resnet_updated_config = resnet_updated.get_config()
    # Extract layer names
    resnet_updated_layer_names = [resnet_updated_config['layers'][x]['name'] for x in range(len(resnet_updated_config['layers']))]

    # Grab the first conv block name
    first_conv_name = resnet_updated_layer_names[2]

    # Update weights for all layers, for the first conv layer average first three channel weights to the
    # remaning number out of total channels
    for layer in resnet_model.layers:
        if layer.name in resnet_updated_layer_names:

            if layer.get_weights() != []:  # Check if the layer has weights
                target_layer = resnet_updated.get_layer(layer.name)

                if layer.name in first_conv_name:  # First conv block
                    weights = layer.get_weights()[0]
                    biases = layer.get_weights()[1]
                    
                    # Adjust weights of the extra channels
                    weights_extra_channels = np.concatenate((weights,
                                                             avg_and_copy_wts(weights, c - 3)),
                                                            axis=-2)

                    # Set weights for the first conv block
                    target_layer.set_weights([weights_extra_channels, biases])
                    target_layer.trainable = False  # Set the layer as non-trainable
                else:
                    # Set weights for other layers
                    target_layer.set_weights(layer.get_weights())
                    target_layer.trainable = False  # Set the layer as non-trainable

    model = Sequential([
        resnet_updated,
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.4),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.1),
        Dense(last_layer_units, activation=last_layer_activation)
    ], name=model_name)

    return model

