"""Chi-squared distribution."""

import numpy as np
from scipy import stats as sp_stats

from wedgestats.distributions._base import ContinuousDistribution


class ChiSquared(ContinuousDistribution):
    """
    Chi-squared distribution with *df* degrees of freedom.

    Parameters
    ----------
    df : int
        Degrees of freedom (> 0).
    """

    def __init__(self, df: int) -> None:
        if df <= 0:
            raise ValueError("df must be positive")
        self.df = df
        self._dist = sp_stats.chi2(df)

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
        return f"ChiSquared(df={self.df})"
