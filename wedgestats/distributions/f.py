"""F-distribution."""

import numpy as np
from scipy import stats as sp_stats

from wedgestats.distributions._base import ContinuousDistribution


class FDistribution(ContinuousDistribution):
    """
    F-distribution with *df1* and *df2* degrees of freedom.

    Parameters
    ----------
    df1 : int | float
        Numerator degrees of freedom (> 0).
    df2 : int | float
        Denominator degrees of freedom (> 0).
    """

    def __init__(self, df1: int | float, df2: int | float) -> None:
        if df1 <= 0:
            raise ValueError("df1 must be positive")
        if df2 <= 0:
            raise ValueError("df2 must be positive")
        self.df1 = df1
        self.df2 = df2
        self._dist = sp_stats.f(df1, df2)

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
        return f"FDistribution(df1={self.df1}, df2={self.df2})"
