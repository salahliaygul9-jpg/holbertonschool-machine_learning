#!/usr/bin/env python3
"""
14-batch_norm.py
Creates a batch normalization layer in TensorFlow
"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network

    prev: activated output of the previous layer
    n: number of nodes in the layer to be created
    activation: activation function to use
    Returns: tensor of the activated output for the layer
    """
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    x = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=init
    )(prev)

    bn = tf.keras.layers.BatchNormalization(
        epsilon=1e-7,
        gamma_initializer='ones',
        beta_initializer='zeros'
    )

    x = bn(x, training=True)

    return activation(x)
