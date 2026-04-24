#!/usr/bin/env python3
"""
Gradient Descent with L2 Regularization
"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates neural network parameters using gradient descent
    with L2 regularization.

    Args:
    Y: one-hot labels (classes, m)
    weights: model parameters
    cache: layer outputs
    alpha: learning rate
    lambtha: regularization strength
    L: number of layers
    """
    weights_copy = weights.copy()
    dz = cache["A" + str(L)] - Y
    for i in range(L, 0, -1):
        A = cache["A" + str(i - 1)]
        w = "W" + str(i)
        b = "b" + str(i)
        dw = (1 / len(Y[0])) * np.matmul(dz, A.T) + (
            lambtha * weights[w]) / len(Y[0])
        db = (1 / len(Y[0])) * np.sum(dz, axis=1, keepdims=True)
        weights[w] = weights[w] - alpha * dw
        weights[b] = weights[b] - alpha * db
        dz = np.matmul(weights_copy["W" + str(i)].T, dz) * (1 - A * A)
