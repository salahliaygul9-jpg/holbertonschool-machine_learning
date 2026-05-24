#!/usr/bin/env python3
"""Expectation Maximization for GMM"""
import numpy as np


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Performs the expectation maximization for a GMM

    Parameters:
        X: numpy.ndarray of shape (n, d) containing the data set
        k: positive integer containing the number of clusters
        iterations: positive integer containing the maximum number of iterations
        tol: non-negative float containing tolerance of the log likelihood
        verbose: boolean that determines if you should print information

    Returns:
        pi, m, S, g, l or None, None, None, None, None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None

    initialize = __import__('4-initialize').initialize
    expectation = __import__('6-expectation').expectation
    maximization = __import__('7-maximization').maximization

    pi, m, S = initialize(X, k)
    if pi is None or m is None or S is None:
        return None, None, None, None, None

    l_prev = 0
    g = None
    l = None

    for i in range(iterations):
        # E-step
        g, l = expectation(X, pi, m, S)
        if g is None or l is None:
            return None, None, None, None, None

        # Print verbose output every 10 iterations
        if verbose and i % 10 == 0:
            print("Log Likelihood after {} iterations: {}".format(
                i, round(l, 5)))

        # Check for convergence
        if i > 0 and abs(l - l_prev) <= tol:
            break

        l_prev = l

        # M-step
        pi, m, S = maximization(X, g)
        if pi is None or m is None or S is None:
            return None, None, None, None, None

    # Final E-step after last maximization
    g, l = expectation(X, pi, m, S)
    if g is None or l is None:
        return None, None, None, None, None

    if verbose:
        print("Log Likelihood after {} iterations: {}".format(
            i + 1, round(l, 5)))

    return pi, m, S, g, l
