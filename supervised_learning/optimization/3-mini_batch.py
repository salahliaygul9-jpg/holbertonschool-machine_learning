#!/usr/bin/env python3
<<<<<<< HEAD
"""Module that creates mini-batches for mini-batch gradient descent."""

import numpy as np
=======
"""
<<<<<<< HEAD
Defines function that trains a loaded neural network model
using mini-batch gradient descent
"""


import tensorflow as tf
=======
Creates mini-batches for training a neural network with mini-batch
gradient descent.
"""

import numpy as np
>>>>>>> c1c2691c58dd678013e4ef20721791c1d9eaefd7
>>>>>>> 1108b125e88820ebfe01b9d115c3d04ed1d2481d
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
<<<<<<< HEAD
    """Creates mini-batches for training a neural network.

    Args:
        X: numpy.ndarray of shape (m, nx) representing input data.
        Y: numpy.ndarray of shape (m, ny) representing the labels.
        batch_size: number of data points in a batch.
=======
    """
<<<<<<< HEAD
    Creates mini-batches to be used for training
    a neural network using mini-batch gradient descent.

    Args:
        X (numpy.ndarray): Input data of shape (m, nx).
        Y (numpy.ndarray): Labels of shape (m, ny).
        batch_size (int): Number of data points in a batch.

    Returns:
        list: List of mini-batches containing tuples (X_batch, Y_batch).
    """
    m = X.shape[0]
    mini_batches = []

    # Shuffle the data
    X_shuffled, Y_shuffled = shuffle_data(X, Y)

    # Calculate the number of batches
    num_batches = m // batch_size

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size

        X_batch = X_shuffled[start_idx:end_idx]
        Y_batch = Y_shuffled[start_idx:end_idx]

        mini_batches.append((X_batch, Y_batch))

    # Handle the last batch
    # (if the total number of data points is not divisible by batch_size)
    if m % batch_size != 0:
        start_idx = num_batches * batch_size
        X_batch = X_shuffled[start_idx:]
        Y_batch = Y_shuffled[start_idx:]
        mini_batches.append((X_batch, Y_batch))

    return mini_batches
=======
    Creates mini-batches to be used for training a neural network using
    mini-batch gradient descent.

    Args:
        X: numpy.ndarray of shape (m, nx) representing input data
           m: number of data points
           nx: number of features in X
        Y: numpy.ndarray of shape (m, ny) representing the labels
           m: same number of data points as in X
           ny: number of classes for classification tasks.
        batch_size: number of data points in a batch
>>>>>>> 1108b125e88820ebfe01b9d115c3d04ed1d2481d

    Returns:
        List of mini-batches containing tuples (X_batch, Y_batch).
    """
<<<<<<< HEAD
    X_s, Y_s = shuffle_data(X, Y)
    m = X.shape[0]
    mini_batches = []

    for i in range(0, m, batch_size):
        X_batch = X_s[i:i + batch_size]
        Y_batch = Y_s[i:i + batch_size]
        mini_batches.append((X_batch, Y_batch))

    return mini_batches
=======
    m = X.shape[0]
    mini_batches = []
    
    shuffled_X, shuffled_Y = shuffle_data(X, Y)

    num_complete_batches = m // batch_size

    for i in range(num_complete_batches)
        start = i * batch_size
        end = start + batch_size
        mini_batch_X = shuffled_X[start:end]
        mini_batch_Y = shuffled_Y[start:end]
        mini_batches.append((mini_batch_X, mini_batch_Y))

   if m % batch_size != 0:
       start = num_complete_batches * batch_size
       mini_batch_X = shuffled_X[start:]
       mini_batch_Y = shuffled_Y[start:]
       mini_batches.append((mini_batch_X, mini_batch_Y))

  return mini_batches
>>>>>>> c1c2691c58dd678013e4ef20721791c1d9eaefd7
>>>>>>> 1108b125e88820ebfe01b9d115c3d04ed1d2481d
