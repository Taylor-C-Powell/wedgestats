"""Poisson distribution."""

import numpy as np
from scipy import stats as sp_stats

from wedgestats.distributions._base import DiscreteDistribution


class Poisson(DiscreteDistribution):
    """
    Poisson distribution for the number of events in a fixed interval.

    Parameters
    ----------
    lam : float
        Expected number of events (rate parameter, > 0).
    """

    def __init__(self, lam: float) -> None:
        if lam <= 0:
            raise ValueError("lam must be positive")
        self.lam = lam
        self._dist = sp_stats.poisson(mu=lam)

    def pmf(self, k: int) -> float:
        return float(self._dist.pmf(k))

    def cdf(self, x: float) -> float:
        return float(self._dist.cdf(x))

    def mean(self) -> float:
        return float(self._dist.mean())

    def variance(self) -> float:
        return float(self._dist.var())

    def rvs(self, size: int = 1, random_state: int | None = None) -> np.ndarray:
        return self._dist.rvs(size=size, random_state=random_state)

    def __repr__(self) -> str:
        return f"Poisson(lam={self.lam})"
