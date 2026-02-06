"""Negative binomial distribution."""

import numpy as np
from scipy import stats as sp_stats

from wedgestats.distributions._base import DiscreteDistribution


class NegativeBinomial(DiscreteDistribution):
    """
    Negative binomial distribution for the number of failures before *r*
    successes.

    Parameters
    ----------
    r : int
        Number of successes required (> 0).
    p : float
        Probability of success on each trial (0 < p <= 1).
    """

    def __init__(self, r: int, p: float) -> None:
        if r <= 0:
            raise ValueError("r must be positive")
        if not (0 < p <= 1):
            raise ValueError("p must satisfy 0 < p <= 1")
        self.r = r
        self.p = p
        self._dist = sp_stats.nbinom(r, p)

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
        return f"NegativeBinomial(r={self.r}, p={self.p})"
