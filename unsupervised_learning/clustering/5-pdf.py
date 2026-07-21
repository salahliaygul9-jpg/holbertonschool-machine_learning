#!/usr/bin/env python3
"""Probability density function module"""
import numpy as np


def pdf(X, m, S):
    """Calculates the PDF of a Gaussian distribution."""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(m, np.ndarray) or m.ndim != 1:
        return None
    if not isinstance(S, np.ndarray) or S.ndim != 2:
        return None

    n, d = X.shape
    if m.shape[0] != d or S.shape != (d, d):
        return None

    S_det = np.linalg.det(S)
    S_inv = np.linalg.inv(S)

    coeff = 1 / (np.sqrt(((2 * np.pi) ** d) * S_det))

    diff = X - m
    exponent = -0.5 * np.sum(diff @ S_inv * diff, axis=1)

    P = coeff * np.exp(exponent)
    P = np.maximum(P, 1e-300)

    return P
