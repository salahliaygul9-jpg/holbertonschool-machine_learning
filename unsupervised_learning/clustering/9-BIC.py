#!/usr/bin/env python3
"""Bayesian Information Criterion for GMM module"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Finds the best number of clusters for a GMM using BIC."""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    n, d = X.shape

    if kmax is None:
        kmax = n
    if not isinstance(kmax, int) or kmax <= 0:
        return None, None, None, None
    if kmax < kmin or kmax > n:
        return None, None, None, None
    if kmax - kmin + 1 < 2:
        return None, None, None, None

    likelihoods = []
    bics = []
    results = []

    for k in range(kmin, kmax + 1):
        pi, m, S, g, lk = expectation_maximization(
            X, k, iterations, tol, verbose)
        if pi is None:
            return None, None, None, None

        results.append((pi, m, S))
        likelihoods.append(lk)

        # p = number of parameters:
        # k-1 priors + k*d means + k*d*(d+1)/2 covariance entries
        p = k - 1 + k * d + k * d * (d + 1) // 2
        bics.append(p * np.log(n) - 2 * lk)

    l = np.array(likelihoods)
    b = np.array(bics)

    best_idx = np.argmin(b)
    best_k = kmin + best_idx
    best_result = results[best_idx]

    return best_k, best_result, l, b
