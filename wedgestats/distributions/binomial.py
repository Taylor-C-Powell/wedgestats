"""Binomial distribution."""

import numpy as np
from scipy import stats as sp_stats

from wedgestats.distributions._base import DiscreteDistribution


class Binomial(DiscreteDistribution):
    """
    Binomial distribution for the number of successes in *n* independent
    Bernoulli trials, each with probability *p* of success.

    Parameters
    ----------
    n : int
        Number of trials (>= 0).
    p : float
        Probability of success on each trial (0 <= p <= 1).
    """

    def __init__(self, n: int, p: float) -> None:
        if n < 0:
            raise ValueError("n must be non-negative")
        if not (0 <= p <= 1):
            raise ValueError("p must be between 0 and 1")
        self.n = n
        self.p = p
        self._dist = sp_stats.binom(n, p)

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
        return f"Binomial(n={self.n}, p={self.p})"
