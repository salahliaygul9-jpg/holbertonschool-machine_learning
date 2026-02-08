#!/usr/bin/env python3
"""Normal class for distribution"""


class Normal:
    """Normal class for distribution"""
    e = 2.7182818285
    pi = 3.1415926536

    def __init__(self, data=None, mean=0., stddev=1.):
        self.mean = float(mean)
        self.stddev = float(stddev)
        if self.stddev <= 0:
            raise ValueError("stddev must be a positive value")
        if data is not None:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data) / len(data)
            self.stddev = sum(map(lambda i: (i - self.mean) ** 2, data))
            self.stddev = (self.stddev / len(data)) ** (1 / 2)

    def z_score(self, x):
        """
        z-score of a given x-score
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        x-score of a given z-score
        """
        return z * self.stddev + self.mean

    def pdf(self, x):
        """
        Probability Density Function for normal
        """
        exponent = (- 1 / 2) * (((x - self.mean) / self.stddev) ** 2)
        coeficient = 1 / (self.stddev * (2 * Normal.pi) ** (1 / 2))
        pdf = coeficient * Normal.e ** exponent
        return pdf

    def cdf(self, x):
        """
        Cumulative Distribution Function for normal
        """
        val = (x - self.mean) / (self.stddev * (2 ** (1 / 2)))
        erf1 = (2 / Normal.pi ** (1 / 2))
        erf2 = (val - (val**3)/3 + (val**5)/10 - (val**7)/42 + (val**9)/216)
        cdf = (1 / 2) * (1 + erf1 * erf2)
        return cdf
