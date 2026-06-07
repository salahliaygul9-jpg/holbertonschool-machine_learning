#!/usr/bin/env python3
"""K-means clustering module"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """Performs K-means on a dataset."""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape
    if k > n:
        return None, None

    low = X.min(axis=0)
    high = X.max(axis=0)

    # Initialize centroids (uses numpy.random.uniform #1)
    C = np.random.uniform(low, high, size=(k, d))

    for _ in range(iterations):
        # Assign each point to nearest centroid
        # (n, 1, d) - (1, k, d) => (n, k, d) => (n, k)
        distances = np.linalg.norm(X[:, np.newaxis] - C[np.newaxis], axis=2)
        clss = np.argmin(distances, axis=1)

        C_new = np.zeros((k, d))
        for j in range(k):
            points = X[clss == j]
            if len(points) == 0:
                # Reinitialize empty cluster (uses numpy.random.uniform #2)
                C_new[j] = np.random.uniform(low, high, size=(d,))
            else:
                C_new[j] = points.mean(axis=0)

        if np.allclose(C, C_new):
            return C_new, clss

        C = C_new

    # Final assignment after last iteration
    distances = np.linalg.norm(X[:, np.newaxis] - C[np.newaxis], axis=2)
    clss = np.argmin(distances, axis=1)

    return C, clss
