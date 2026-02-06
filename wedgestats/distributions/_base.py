"""Abstract base classes for probability distributions."""

from abc import ABC, abstractmethod

import numpy as np
from scipy import stats as sp_stats


class Distribution(ABC):
    """Base class for all probability distributions."""

    @abstractmethod
    def mean(self) -> float:
        """Return the distribution mean."""

    @abstractmethod
    def variance(self) -> float:
        """Return the distribution variance."""

    def std_dev(self) -> float:
        """Return the distribution standard deviation."""
        return float(np.sqrt(self.variance()))

    @abstractmethod
    def cdf(self, x: float) -> float:
        """Cumulative distribution function evaluated at *x*."""

    def sf(self, x: float) -> float:
        """Survival function (1 - CDF) evaluated at *x*."""
        return 1.0 - self.cdf(x)

    @abstractmethod
    def rvs(self, size: int = 1, random_state: int | None = None) -> np.ndarray:
        """Generate random variates."""

    @abstractmethod
    def __repr__(self) -> str: ...


class DiscreteDistribution(Distribution, ABC):
    """Base class for discrete probability distributions."""

    @abstractmethod
    def pmf(self, k: int) -> float:
        """Probability mass function evaluated at integer *k*."""


class ContinuousDistribution(Distribution, ABC):
    """Base class for continuous probability distributions."""

    @abstractmethod
    def pdf(self, x: float) -> float:
        """Probability density function evaluated at *x*."""

    @abstractmethod
    def ppf(self, q: float) -> float:
        """Percent-point (inverse CDF) function evaluated at quantile *q*."""
