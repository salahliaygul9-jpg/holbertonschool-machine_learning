#!/usr/bin/env python3
"""Module that performs forward propagation for a simple RNN."""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """Perform forward propagation for a simple RNN.

    Args:
        rnn_cell: Instance of RNNCell.
        X (numpy.ndarray): Input data of shape (t, m, i).
        h_0 (numpy.ndarray): Initial hidden state of shape (m, h).

    Returns:
        tuple: H, Y
            H contains all hidden states
            Y contains all outputs
    """
    t = X.shape[0]
    m, h = h_0.shape
    o = rnn_cell.by.shape[1]

    H = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, o))
    H[0] = h_0

    for step in range(t):
        H[step + 1], Y[step] = rnn_cell.forward(H[step], X[step])

    return H, Y
