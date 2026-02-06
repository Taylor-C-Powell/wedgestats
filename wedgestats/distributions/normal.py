"""Normal (Gaussian) distribution."""

import numpy as np
from scipy import stats as sp_stats

from wedgestats.distributions._base import ContinuousDistribution


class Normal(ContinuousDistribution):
    """
    Normal (Gaussian) distribution.

    Parameters
    ----------
    mu : float
        Mean of the distribution.
    sigma : float
        Standard deviation (> 0).
    """

    def __init__(self, mu: float = 0.0, sigma: float = 1.0) -> None:
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self.mu = mu
        self.sigma = sigma
        self._dist = sp_stats.norm(loc=mu, scale=sigma)

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
        return f"Normal(mu={self.mu}, sigma={self.sigma})"
