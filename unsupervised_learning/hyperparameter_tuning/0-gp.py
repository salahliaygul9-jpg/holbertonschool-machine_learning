#!/usr/bin/env python3
"""Gaussian Process module"""
import numpy as np
 
 
class GaussianProcess:
    """Represents a noiseless 1D Gaussian process"""
 
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Class constructor
 
        X_init: numpy.ndarray of shape (t, 1) - inputs already sampled
        Y_init: numpy.ndarray of shape (t, 1) - outputs of the black-box function
        l: length parameter for the kernel
        sigma_f: standard deviation of the black-box function output
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)
 
    def kernel(self, X1, X2):
        """
        Calculates the covariance kernel matrix between two matrices
        using the Radial Basis Function (RBF)
 
        X1: numpy.ndarray of shape (m, 1)
        X2: numpy.ndarray of shape (n, 1)
 
        Returns: covariance kernel matrix as numpy.ndarray of shape (m, n)
        """
        # Squared Euclidean distance: ||x1 - x2||^2
        sqdist = (
            np.sum(X1 ** 2, axis=1).reshape(-1, 1)
            + np.sum(X2 ** 2, axis=1)
            - 2 * X1 @ X2.T
        )
        return self.sigma_f ** 2 * np.exp(-sqdist / (2 * self.l ** 2))
