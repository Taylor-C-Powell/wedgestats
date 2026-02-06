"""Exponential distribution."""

import numpy as np
from scipy import stats as sp_stats

from wedgestats.distributions._base import ContinuousDistribution


class Exponential(ContinuousDistribution):
    """
    Exponential distribution for the time between events in a Poisson process.

    Parameters
    ----------
    lam : float
        Rate parameter (> 0). The scale is 1/lam.
    """

    def __init__(self, lam: float) -> None:
        if lam <= 0:
            raise ValueError("lam must be positive")
        self.lam = lam
        self._dist = sp_stats.expon(scale=1.0 / lam)

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
        return f"Exponential(lam={self.lam})"
