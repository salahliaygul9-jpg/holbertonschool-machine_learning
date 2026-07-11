#!/usr/bin/env python3
"""Module that defines an LSTM cell."""
import numpy as np


class LSTMCell:
    """Represents an LSTM unit."""

    def __init__(self, i, h, o):
        """Initialize the LSTM cell.

        Args:
            i (int): Dimensionality of the data.
            h (int): Dimensionality of the hidden state.
            o (int): Dimensionality of the output.
        """
        self.Wf = np.random.randn(i + h, h)
        self.Wu = np.random.randn(i + h, h)
        self.Wc = np.random.randn(i + h, h)
        self.Wo = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """Perform forward propagation for one time step.

        Args:
            h_prev (numpy.ndarray): Previous hidden state of shape (m, h).
            c_prev (numpy.ndarray): Previous cell state of shape (m, h).
            x_t (numpy.ndarray): Input data of shape (m, i).

        Returns:
            tuple: h_next, c_next, y
                h_next is the next hidden state
                c_next is the next cell state
                y is the output of the cell
        """
        hx = np.concatenate((h_prev, x_t), axis=1)

        f = 1 / (1 + np.exp(-(np.matmul(hx, self.Wf) + self.bf)))
        u = 1 / (1 + np.exp(-(np.matmul(hx, self.Wu) + self.bu)))
        c_bar = np.tanh(np.matmul(hx, self.Wc) + self.bc)
        o = 1 / (1 + np.exp(-(np.matmul(hx, self.Wo) + self.bo)))

        c_next = f * c_prev + u * c_bar
        h_next = o * np.tanh(c_next)

        y_linear = np.matmul(h_next, self.Wy) + self.by
        y_shifted = y_linear - np.max(y_linear, axis=1, keepdims=True)
        y_exp = np.exp(y_shifted)
        y = y_exp / np.sum(y_exp, axis=1, keepdims=True)

        return h_next, c_next, y
