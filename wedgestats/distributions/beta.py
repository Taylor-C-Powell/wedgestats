"""Beta distribution."""

import numpy as np
from scipy import stats as sp_stats

from wedgestats.distributions._base import ContinuousDistribution


class Beta(ContinuousDistribution):
    """
    Beta distribution on the interval [0, 1].

    Parameters
    ----------
    alpha : float
        First shape parameter (> 0).
    beta : float
        Second shape parameter (> 0).
    """

    def __init__(self, alpha: float, beta: float) -> None:
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if beta <= 0:
            raise ValueError("beta must be positive")
        self.alpha = alpha
        self.beta = beta
        self._dist = sp_stats.beta(alpha, beta)

    def pdf(self, x: float) -> float:
        return float(self._dist.pdf(x))

    def cdf(self, x: float) -> float:
        return float(self._dist.cdf(x))

    def ppf(self, q: float) -> float:
        return float(self._dist.ppf(q))

    def mean(self) -> float:
        return float(self._dist.mean())

    def variance(self) -> float:
        return float(self._dist.var())

    def rvs(self, size: int = 1, random_state: int | None = None) -> np.ndarray:
        return self._dist.rvs(size=size, random_state=random_state)

    def __repr__(self) -> str:
        return f"Beta(alpha={self.alpha}, beta={self.beta})"
