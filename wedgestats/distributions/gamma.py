"""Gamma distribution."""

import numpy as np
from scipy import stats as sp_stats

from wedgestats.distributions._base import ContinuousDistribution


class Gamma(ContinuousDistribution):
    """
    Gamma distribution.

    Parameters
    ----------
    alpha : float
        Shape parameter (> 0).
    beta : float
        Rate parameter (> 0). The scale is 1/beta.
    """

    def __init__(self, alpha: float, beta: float) -> None:
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if beta <= 0:
            raise ValueError("beta must be positive")
        self.alpha = alpha
        self.beta = beta
        self._dist = sp_stats.gamma(a=alpha, scale=1.0 / beta)

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
        return f"Gamma(alpha={self.alpha}, beta={self.beta})"
