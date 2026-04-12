#!/usr/bin/env python3
"""
Contrast
"""
import tensorflow as tf


def change_contrast(image, lower, upper):
    """
    """
    return tf.image.random_contrast(image, lower, upper)
