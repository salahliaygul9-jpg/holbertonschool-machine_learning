#!/usr/bin/env python3
""" L2 regularization with tensorflow """

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """ create layer with l2 reg in tensorflow """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    reg = tf.contrib.layers.l2_regularizer(lambtha)
    layer = tf.layers.Dense(n, activation=activation, kernel_initializer=init,
                            name="layer", kernel_regularizer=reg)
    return layer(prev)
