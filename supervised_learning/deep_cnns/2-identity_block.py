#!/usr/bin/env python3
"""
Identity Block
"""
from tensorflow import keras as K


def identity_block(A_prev, filters):
    """
    Builds an identity block as described in Deep
    Residual Learning for Image Recognition (2015)

    Arguments:
    A_prev -- output of the previous layer
    filters -- list or tuple containing F11, F3, F12, respectively:
               F11 is the number of filters in the first 1x1 convolution
               F3 is the number of filters in the 3x3 convolution
               F12 is the number of filters in the second 1x1 convolution

    Returns:
    Activated output of the identity block
    """
    F11, F3, F12 = filters

    init = K.initializers.he_normal(seed=0)

    conv1 = K.layers.Conv2D(
        filters=F11,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="valid",
        kernel_initializer=init,
    )(A_prev)
    bn1 = K.layers.BatchNormalization(axis=3)(conv1)
    relu1 = K.layers.Activation("relu")(bn1)

    conv2 = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        kernel_initializer=init,
    )(relu1)
    bn2 = K.layers.BatchNormalization(axis=3)(conv2)
    relu2 = K.layers.Activation("relu")(bn2)

    conv3 = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="valid",
        kernel_initializer=init,
    )(relu2)
    bn3 = K.layers.BatchNormalization(axis=3)(conv3)

    add = K.layers.Add()([bn3, A_prev])

    output = K.layers.Activation("relu")(add)

    return output
