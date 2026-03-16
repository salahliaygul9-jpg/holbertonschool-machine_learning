#!/usr/bin/env python3
""" new class """
import numpy as np


class Neuron():
    """ class neuron """
    def __init__(self, nx):
        """ init function """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """ getter W """
        return self.__W

    @property
    def b(self):
        """ getter b """
        return self.__b

    @property
    def A(self):
        """ getter A """
        return self.__A

    def forward_prop(self, X):
        """ forward function """
        nx, m = np.shape(X)
        x = np.matmul(self.__W, X) + self.__b
        self.__A = 1/(1 + (np.exp(-x)))
        return self.__A

    def cost(self, Y, A):
        """ cost function """
        m = Y.shape[1]
        cos = -(np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)))
        return cos/m

    def evaluate(self, X, Y):
        """ evaluate function """
        z = self.forward_prop(X)
        s = np.round(z)
        return s.astype(np.int), self.cost(Y, z)
        """def gradient_descent(self, X, Y, A, alpha=0.05):
        gradient descent function
        nx, m = np.shape(X)
        self.__W = self.__W - (alpha/m) * np.matmul(X, (A-Y).T).T
        self.__b = self.__b - (alpha/m) * np.sum(A-Y).T
        return self.__W, self.__b"""

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """ gradient descent function """
        nx, m = np.shape(X)
        db = np.sum(A-Y)/m
        dW = np.sum((A-Y)*X, axis=1)/m
        self.__W = self.__W - (alpha * dW)
        self.__b = self.__b - (alpha * db)
        return self.__W, self.__b
