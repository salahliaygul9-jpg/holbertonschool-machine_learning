#!/usr/bin/env python3
"""Task 3. Projection Block"""
from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """
    Builds a projection block as described in Deep
    Residual Learning for Image Recognition (2015).

    Arguments:
    A_prev -- output from the previous layer (tensor of shape (H, W, C))
    filters -- list or tuple containing F11, F3, F12:
        F11 is the number of filters in the first 1x1 convolution
        F3 is the number of filters in the 3x3 convolution
        F12 is the number of filters in the second 1x1
        convolution and the shortcut connection
    s -- stride of the first convolution in both the main
    path and the shortcut connection (default is 2)

    Returns:
    activated_output -- the activated output of the
    projection block (tensor of shape (H/s, W/s, F12))
    """
    F11, F3, F12 = filters

    C1x1_1a = K.layers.Conv2D(
        filters=F11,
        kernel_size=1,
        strides=s,
        kernel_initializer='he_normal',
        padding='same'
    )(A_prev)
    C1x1_1a_BN = K.layers.BatchNormalization()(C1x1_1a)
    C1x1_1a_relu = K.layers.Activation('relu')(C1x1_1a_BN)

    C3x3 = K.layers.Conv2D(
        filters=F3,
        kernel_size=3,
        padding='same',
        kernel_initializer='he_normal'
    )(C1x1_1a_relu)
    C3x3_BN = K.layers.BatchNormalization()(C3x3)
    C3x3_relu = K.layers.Activation('relu')(C3x3_BN)

    C1x1_2a = K.layers.Conv2D(
        filters=F12,
        kernel_size=1,
        kernel_initializer='he_normal',
        padding='same'
    )(C3x3_relu)
    C1x1_2a_BN = K.layers.BatchNormalization()(C1x1_2a)

    C1x1_1b = K.layers.Conv2D(
        filters=F12,
        kernel_size=1,
        strides=s,
        kernel_initializer='he_normal',
        padding='same'
    )(A_prev)
    C1x1_1b_BN = K.layers.BatchNormalization()(C1x1_1b)

    path_addition = K.layers.Add()([C1x1_2a_BN, C1x1_1b_BN])
    return K.layers.Activation('relu')(path_addition)
