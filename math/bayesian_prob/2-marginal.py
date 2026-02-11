#!/usr/bin/env python3
""" bayesian probability"""
import numpy as np
import math  

def marginal(x, n, P, Pr):
    """calculates the marginal probability"""
    return np.sum(intersection(x, n, P, Pr))

def intersection(x, n, P, Pr):
    """calculates the intersection"""
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(Pr, np.ndarray) or P.shape != Pr.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")
    if np.any(Pr < 0) or np.any(Pr > 1):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    comb = math.factorial(n) / (math.factorial(x) * math.factorial(n - x))
    likelihood = comb * (P ** x) * ((1 - P) ** (n - x))
    return likelihood * Pr
