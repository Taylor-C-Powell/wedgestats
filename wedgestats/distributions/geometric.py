"""Geometric distribution."""

import numpy as np
from scipy import stats as sp_stats

from wedgestats.distributions._base import DiscreteDistribution


class Geometric(DiscreteDistribution):
    """
    Geometric distribution for the number of trials until the first success.

    The PMF is defined for k = 1, 2, 3, ... (number of trials).

    Parameters
    ----------
    p : float
        Probability of success on each trial (0 < p <= 1).
    """

    def __init__(self, p: float) -> None:
        if not (0 < p <= 1):
            raise ValueError("p must satisfy 0 < p <= 1")
        self.p = p
        self._dist = sp_stats.geom(p)

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
        return f"Geometric(p={self.p})"
