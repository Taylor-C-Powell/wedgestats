"""Hypergeometric distribution."""

import numpy as np
from scipy import stats as sp_stats

from wedgestats.distributions._base import DiscreteDistribution


class Hypergeometric(DiscreteDistribution):
    """
    Hypergeometric distribution for the number of successes in *N* draws
    (without replacement) from a population of size *M* containing *n*
    success states.

    Parameters
    ----------
    M : int
        Total population size (>= 0).
    n : int
        Number of success states in the population (0 <= n <= M).
    N : int
        Number of draws (0 <= N <= M).
    """

    def __init__(self, M: int, n: int, N: int) -> None:
        if M < 0:
            raise ValueError("M (population size) must be non-negative")
        if not (0 <= n <= M):
            raise ValueError("n must satisfy 0 <= n <= M")
        if not (0 <= N <= M):
            raise ValueError("N must satisfy 0 <= N <= M")
        self.M = M
        self.n = n
        self.N = N
        self._dist = sp_stats.hypergeom(M, n, N)

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
        return f"Hypergeometric(M={self.M}, n={self.n}, N={self.N})"
