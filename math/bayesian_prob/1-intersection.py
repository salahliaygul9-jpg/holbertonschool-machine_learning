#!/usr/bin/env python3
"""1-intersection.py"""
import numpy as np


def likelihood(x, n, P):
    """Calculates likelihood of x given n and P (binomial)."""
    if x == 0 or x == n:
        comb = 1
    else:
        num = np.prod(np.arange(n, n-x, -1), dtype=object)
        den = np.prod(np.arange(1, x+1), dtype=object)
        comb = num // den

    likelihoods = comb * (P ** x) * ((1 - P) ** (n - x))
    return likelihoods.astype(float)


def intersection(x, n, P, Pr):
    """Calculates the intersection of obtaining data with prior probabilities."""
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if np.any(Pr < 0) or np.any(Pr > 1):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    likelihoods = likelihood(x, n, P)
    posterior = likelihoods * Pr
    return posterior
