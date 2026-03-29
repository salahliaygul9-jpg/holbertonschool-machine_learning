#!/usr/bin/env python3
"""Script to convert a label vector to a one-hot matrix in Keras."""

import tensorflow.keras as K


def one_hot(labels, classes=None):
    """Converts a label vector into a one-hot matrix."""
    one_hot_matrix = K.utils.to_categorical(labels, num_classes=classes)
    return one_hot_matrix
