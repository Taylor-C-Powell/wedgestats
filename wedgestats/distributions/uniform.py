"""Discrete and continuous uniform distributions."""

import numpy as np
from scipy import stats as sp_stats

from wedgestats.distributions._base import ContinuousDistribution, DiscreteDistribution


class DiscreteUniform(DiscreteDistribution):
    """
    Discrete uniform distribution on integers {low, low+1, ..., high}.

    Parameters
    ----------
    low : int
        Lower bound (inclusive).
    high : int
        Upper bound (inclusive, must be >= low).
    """

    def __init__(self, low: int, high: int) -> None:
        if high < low:
            raise ValueError("high must be >= low")
        self.low = low
        self.high = high
        self._dist = sp_stats.randint(low, high + 1)

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
        return f"DiscreteUniform(low={self.low}, high={self.high})"


class ContinuousUniform(ContinuousDistribution):
    """
    Continuous uniform distribution on the interval [low, high].

    Parameters
    ----------
    low : float
        Lower bound.
    high : float
        Upper bound (must be > low).
    """

    def __init__(self, low: float = 0.0, high: float = 1.0) -> None:
        if high <= low:
            raise ValueError("high must be > low")
        self.low = low
        self.high = high
        self._dist = sp_stats.uniform(loc=low, scale=high - low)

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
        return f"ContinuousUniform(low={self.low}, high={self.high})"
