#!/usr/bin/env python3
"""Gaussian Process module"""

import numpy as np


class GaussianProcess:
    """represents a noiseless 1D Gaussian process"""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Class constructor

        Parameters:
        X_init is a numpy.ndarray of shape (t, 1)
            representing the inputs already sampled
        Y_init is a numpy.ndarray of shape (t, 1)
            representing the outputs of the black-box function
            for each input in X_init
        l is the length parameter for the kernel
        sigma_f is the standard deviation given to the output
            of the black-box function
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """
        Calculates the covariance kernel matrix
        using a Gaussian kernel

        Parameters:
        X1 is a numpy.ndarray of shape (m, 1)
        X2 is a numpy.ndarray of shape (n, 1)

        Returns:
        covariance kernel matrix as a numpy.ndarray
        of shape (m, n)
        """
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + \
                 np.sum(X2**2, 1) - 2 * np.matmul(X1, X2.T)

        return self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqdist)

    def predict(self, X_s):
        """
        Predicts the mean and variance of points
        in a Gaussian process

        Parameters:
        X_s is a numpy.ndarray of shape (s, 1)
            containing all of the points whose mean
            and variance should be calculated

        Returns:
        mu, sigma
        mu is a numpy.ndarray of shape (s,)
            containing the mean for each point in X_s
        sigma is a numpy.ndarray of shape (s,)
            containing the variance for each point in X_s
        """
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K)

        mu_s = K_s.T @ K_inv @ self.Y

        cov_s = K_ss - K_s.T @ K_inv @ K_s

        mu = mu_s.reshape(-1)
        sigma = np.diag(cov_s)

        return mu, sigma
