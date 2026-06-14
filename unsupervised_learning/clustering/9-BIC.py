#!/usr/bin/env python3
"""Bayesian Information Criterion for GMM"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Finds the best number of clusters for a GMM using BIC.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        kmin: positive integer, minimum number of clusters (inclusive)
        kmax: positive integer, maximum number of clusters (inclusive)
        iterations: positive integer, maximum iterations for EM
        tol: non-negative float, tolerance for EM
        verbose: boolean, whether EM should print info

    Returns:
        best_k, best_result, l, b or None, None, None, None on failure
    """
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
    if kmax < kmin + 1:
        return None, None, None, None
    l_list = []
    b_list = []
    results = []
    for k in range(kmin, kmax + 1):
        pi, m, S, g, ll = expectation_maximization(
            X, k, iterations, tol, verbose)
        if pi is None:
            return None, None, None, None
        results.append((pi, m, S))
        l_list.append(ll)
        p = k * d * (d + 1) / 2 + d * k + (k - 1)
        bic = p * np.log(n) - 2 * ll
        b_list.append(bic)
    ll = np.array(l_list)
    b = np.array(b_list)
    best_idx = np.argmin(b)
    best_k = kmin + best_idx
    best_result = results[best_idx]

    return best_k, best_result, ll, b
