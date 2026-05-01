#!/usr/bin/env python3
"""
13-batch_norm.py
Normalizes an unactivated layer using batch normalization
"""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes an unactivated output using batch normalization

    Z: numpy.ndarray of shape (m, n)
    gamma: numpy.ndarray of shape (1, n) (scale)
    beta: numpy.ndarray of shape (1, n) (offset)
    epsilon: small number to avoid division by zero
    Returns: normalized Z
    """
    mean = np.mean(Z, axis=0)
    variance = np.var(Z, axis=0)

    Z_norm = (Z - mean) / np.sqrt(variance + epsilon)
    return gamma * Z_norm + beta
