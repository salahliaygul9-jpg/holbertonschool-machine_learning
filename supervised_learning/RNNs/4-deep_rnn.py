#!/usr/bin/env python3
"""Module that performs forward propagation for a deep RNN."""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """Perform forward propagation for a deep RNN.

    Args:
        rnn_cells (list): List of RNNCell instances.
        X (numpy.ndarray): Input data of shape (t, m, i).
        h_0 (numpy.ndarray): Initial hidden state of shape (l, m, h).

    Returns:
        tuple: H, Y
            H contains all hidden states
            Y contains all outputs
    """
    t, m, _ = X.shape
    l, _, h = h_0.shape
    o = rnn_cells[-1].by.shape[1]

    H = np.zeros((t + 1, l, m, h))
    Y = np.zeros((t, m, o))
    H[0] = h_0

    for step in range(t):
        x_t = X[step]
        for layer in range(l):
            h_next, y = rnn_cells[layer].forward(H[step, layer], x_t)
            H[step + 1, layer] = h_next
            x_t = h_next
        Y[step] = y

    return H, Y
