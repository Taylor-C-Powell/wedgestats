"""Distribution classes for wedgestats."""

from wedgestats.distributions._base import (
    ContinuousDistribution,
    DiscreteDistribution,
    Distribution,
)
from wedgestats.distributions.beta import Beta
from wedgestats.distributions.binomial import Binomial
from wedgestats.distributions.chi_squared import ChiSquared
from wedgestats.distributions.exponential import Exponential
from wedgestats.distributions.f import FDistribution
from wedgestats.distributions.gamma import Gamma
from wedgestats.distributions.geometric import Geometric
from wedgestats.distributions.hypergeometric import Hypergeometric
from wedgestats.distributions.negative_binomial import NegativeBinomial
from wedgestats.distributions.normal import Normal
from wedgestats.distributions.poisson import Poisson
from wedgestats.distributions.t import StudentT
from wedgestats.distributions.uniform import ContinuousUniform, DiscreteUniform

__all__ = [
    "Distribution",
    "DiscreteDistribution",
    "ContinuousDistribution",
    "Beta",
    "Binomial",
    "ChiSquared",
    "ContinuousUniform",
    "DiscreteUniform",
    "Exponential",
    "FDistribution",
    "Gamma",
    "Geometric",
    "Hypergeometric",
    "NegativeBinomial",
    "Normal",
    "Poisson",
    "StudentT",
]
