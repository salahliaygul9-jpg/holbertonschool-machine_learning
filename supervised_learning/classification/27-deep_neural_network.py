#!/usr/bin/env python3
"""
DeepNeuralNetwork performing binary classification
"""


import numpy as np
import pickle


class DeepNeuralNetwork:
    """
    Class that represents a deep neural network for binary classification
    """

    def __init__(self, nx, layers):
        """
        Class constructor
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) < 1:
            raise TypeError("layers must be a list of positive integers")

        weights = {}
        previous = nx

        for index, layer in enumerate(layers, 1):

            if type(layer) is not int or layer < 0:
                raise TypeError("layers must be a list of positive integers")

            weights["b{}".format(index)] = np.zeros((layer, 1))
            weights["W{}".format(index)] = (
                np.random.randn(layer, previous) * np.sqrt(2 / previous))
            previous = layer

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = weights

    @property
    def L(self):
        """
        gets the private instance attribute __L
        """
        return self.__L

    @property
    def cache(self):
        """
        gets the private instance attribute __cache
        """
        return self.__cache

    @property
    def weights(self):
        """
        gets the private instance attribute __weights
        """
        return self.__weights

    def forward_prop(self, X):
        """
            method calculate forward propagation of neural network

            :param X: ndarray, shape(nx,m) input data

            :return: output neural network and cache
        """

        # store X in A0
        if 'A0' not in self.__cache:
            self.__cache['A0'] = X

        for i in range(1, self.__L + 1):
            # first layer
            if i == 1:
                W = self.__weights["W{}".format(i)]
                b = self.__weights["b{}".format(i)]
                # multiplication of weight and add bias
                Z = np.matmul(W, X) + b
            else:  # next layers
                W = self.__weights["W{}".format(i)]
                b = self.__weights["b{}".format(i)]
                X = self.__cache['A{}'.format(i - 1)]
                Z = np.matmul(W, X) + b

            # activation function :
            # for last use softmax for multiclass
            if i == self.__L:
                exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
                self.__cache["A{}".format(i)] = (
                        exp_Z / np.sum(exp_Z, axis=0, keepdims=True))
            else:
                self.__cache["A{}".format(i)] = 1 / (1 + np.exp(-Z))

        return self.__cache["A{}".format(i)], self.__cache

    def cost(self, Y, A):
        """
            Calculate cross-entropy cost for multiclass
        """

        # store m value
        m = Y.shape[1]

        # calculate log loss function
        log_loss = -(1 / m) * np.sum(Y * np.log(A))

        return log_loss

    def evaluate(self, X, Y):
        """
            Method to evaluate the network's prediction

            :param X: ndarray shape(nx,m) contains input data
            :param Y: ndarray shape (1,m) correct labels

            :return: network's prediction and cost of the network
        """

        # run forward propagation
        output, cache = self.forward_prop(X)

        # calculate cost
        cost = self.cost(Y, output)

        result = np.zeros_like(output)

        result[np.argmax(output, axis=0), np.arange(output.shape[1])] = 1

        return result, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
            Method calculate one pass of gradient descent
            on neural network

            :param Y: ndarray, shape(1,m), correct labels
            :param cache: dictionary containing all intermediary value of
             network
            :param alpha: learning rate

        """

        m = Y.shape[1]

        dZ_f = cache["A{}".format(self.L)] - Y

        for layer in range(self.L, 0, -1):

            A_p = cache["A{}".format(layer - 1)]

            dW = (1 / m) * np.matmul(dZ_f, A_p.T)
            db = (1 / m) * np.sum(dZ_f, axis=1, keepdims=True)

            A = self.weights['W{}'.format(layer)]
            dZ = np.matmul(A.T, dZ_f) * A_p * (1 - A_p)

            self.__weights["W{}".format(layer)] -= alpha * dW
            self.__weights["b{}".format(layer)] -= alpha * db

            dZ_f = dZ

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
            Method to train deep neural network

            :param X: ndarray, shape(nx,m), input data
            :param Y: ndarray, shapte(1,m), correct labels
            :param iterations: number of iterations to train over
            :param alpha: learning rate
            :param verbose: boolean print or not information
            :param graph: boolean print or not graph
            :param step: int

            :return: evaluation of training after iterations
        """

        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if not isinstance(verbose, bool):
            raise TypeError("verbose must be a boolean")
        if not isinstance(graph, bool):
            raise TypeError("graph must be a boolean")
        if verbose is True or graph is True:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        count = []

        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)

            if i != iterations:
                self.gradient_descent(Y, self.cache, alpha)

            cost = self.cost(Y, A)

            costs.append(cost)
            count.append(i)

            if verbose and (i % step == 0 or i == 0 or i == iterations):
                print("Cost after {} iterations: {}".format(i, cost))

        if graph:
            plt.plot(count, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """
            Method to saves instance object to a file in pickle format

            :param filename: file which the object should be saved

        """
        if not filename.endswith('.pkl'):
            filename += '.pkl'

        with open(filename, 'wb') as file:

            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        """
            method to load a pickled DeepNeuralNetwork object

            :param filename: file from which object should be loaded

            :return: loaded object
                    or None if filename doesn't exist
        """

        try:
            with open(filename, 'rb') as file:
                loaded_object = pickle.load(file)
            return loaded_object

        except FileNotFoundError:
            return None
