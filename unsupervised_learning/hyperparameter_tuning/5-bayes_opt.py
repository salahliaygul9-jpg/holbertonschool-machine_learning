#!/usr/bin/env python3
"""Bayesian Optimization module"""

import numpy as np
from scipy.stats import norm

GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """performs Bayesian optimization on a noiseless 1D GP"""

    def __init__(
        self,
        f,
        X_init,
        Y_init,
        bounds,
        ac_samples,
        l=1,
        sigma_f=1,
        xsi=0.01,
        minimize=True
    ):
        self.f = f

        self.gp = GP(
            X_init,
            Y_init,
            l=l,
            sigma_f=sigma_f
        )

        self.X_s = np.linspace(
            bounds[0],
            bounds[1],
            ac_samples
        ).reshape(-1, 1)

        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        mu, sigma = self.gp.predict(self.X_s)

        if self.minimize:
            best = np.min(self.gp.Y)
            imp = best - mu - self.xsi
        else:
            best = np.max(self.gp.Y)
            imp = mu - best - self.xsi

        Z = np.zeros_like(mu)
        nonzero = sigma > 0

        Z[nonzero] = imp[nonzero] / sigma[nonzero]

        EI = np.zeros_like(mu)
        EI[nonzero] = (
            imp[nonzero] * norm.cdf(Z[nonzero]) +
            sigma[nonzero] * norm.pdf(Z[nonzero])
        )

        X_next = self.X_s[np.argmax(EI)]
        return X_next, EI

    def optimize(self, iterations=100):
        """
        Optimizes the black-box function
        """
        for _ in range(iterations):
            X_next, _ = self.acquisition()

            # stop if already sampled
            if np.any(np.isclose(self.gp.X, X_next)).all():
                break

            Y_next = self.f(X_next)

            self.gp.update(X_next, Y_next)

        idx = np.argmin(self.gp.Y) if self.minimize else np.argmax(self.gp.Y)

        X_opt = self.gp.X[idx]
        Y_opt = self.gp.Y[idx]

        return X_opt, Y_opt
