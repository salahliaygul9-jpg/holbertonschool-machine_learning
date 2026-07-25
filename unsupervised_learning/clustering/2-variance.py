#!/usr/bin/env python3
"""Intra-cluster variance module"""
import numpy as np


def variance(X, C):
    """Calculates the total intra-cluster variance for a data set."""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(C, np.ndarray) or C.ndim != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None

    distances = np.linalg.norm(X[:, np.newaxis] - C[np.newaxis], axis=2)
    min_distances = np.min(distances, axis=1)
    var = np.sum(min_distances ** 2)

    return var
