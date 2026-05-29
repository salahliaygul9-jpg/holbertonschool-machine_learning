#!/usr/bin/env python3
"""Maximization step for GMM EM algorithm"""
import numpy as np


def maximization(X, g):
    """Calculates the maximization step in the EM algorithm for a GMM."""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or g.ndim != 2:
        return None, None, None

    n, d = X.shape
    k = g.shape[0]

    if g.shape[1] != n:
        return None, None, None
    if not np.isclose(np.sum(g, axis=0), 1).all():
        return None, None, None

    pi = np.zeros(k)
    m = np.zeros((k, d))
    S = np.zeros((k, d, d))

    for i in range(k):
        n_k = np.sum(g[i])

        pi[i] = n_k / n
        m[i] = np.sum(g[i, :, np.newaxis] * X, axis=0) / n_k

        diff = X - m[i]
        S[i] = (g[i, :, np.newaxis] * diff).T @ diff / n_k

    return pi, m, S
