import numpy as np


def initialize(X, k):
    """Initialize cluster centroids for K-means clustering."""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    n, d = X.shape
    if k > n:
        return None

    low = X.min(axis=0)
    high = X.max(axis=0)

    return np.random.uniform(low, high, size=(k, d))
