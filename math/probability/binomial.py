#!/usr/bin/env python3
"""Binomial class for distribution"""


class Binomial:
    """Binomial class for distribution"""
    e = 2.7182818285

    def __init__(self, data=None, n=1, p=0.5):
        self.n = int(n)
        self.p = float(p)
        if n < 1:
            raise ValueError("n must be a positive value")
        if p <= 0 or p >= 1:
            raise ValueError("p must be greater than 0 and less than 1")
        if data is not None:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = sum(data) / len(data)
            var = sum(map(lambda i: (i - mean) ** 2, data)) / len(data)
            self.p = 1 - ((var) / mean)
            self.n = int(round(mean / self.p))
            self.p = mean / self.n

    def pmf(self, k):
        """Probability Mass Function for binomial"""
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        n_factorial = 1
        for i in range(1, self.n + 1):
            n_factorial *= i
        x_factorial = 1
        for i in range(1, k + 1):
            x_factorial *= i
        op_factorial = 1
        for i in range(1, (self.n - k) + 1):
            op_factorial *= i
        combinatory = (n_factorial) / (x_factorial * op_factorial)
        pmf = combinatory * (self.p ** k) * ((1 - self.p) ** (self.n - k))
        return pmf

    def cdf(self, k):
        """
        Cumulative Distribution Function for binomial
        """
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        cdf = 0
        for i in range(k + 1):
            cdf += Binomial.pmf(self, i)
        return cdf
