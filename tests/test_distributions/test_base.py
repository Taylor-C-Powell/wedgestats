"""ABC contract tests — verify every distribution implements the required API."""

import numpy as np
import pytest

from wedgestats.distributions import (
    Beta,
    Binomial,
    ChiSquared,
    ContinuousDistribution,
    ContinuousUniform,
    DiscreteDistribution,
    DiscreteUniform,
    Exponential,
    FDistribution,
    Gamma,
    Geometric,
    Hypergeometric,
    NegativeBinomial,
    Normal,
    Poisson,
    StudentT,
)

DISCRETE_INSTANCES = [
    Binomial(10, 0.5),
    Hypergeometric(50, 20, 10),
    Poisson(3.0),
    Geometric(0.3),
    NegativeBinomial(5, 0.4),
    DiscreteUniform(1, 6),
]

CONTINUOUS_INSTANCES = [
    Normal(0, 1),
    Exponential(1.0),
    ChiSquared(5),
    StudentT(10),
    FDistribution(5, 10),
    Beta(2, 5),
    Gamma(2, 1),
    ContinuousUniform(0, 1),
]

ALL_INSTANCES = DISCRETE_INSTANCES + CONTINUOUS_INSTANCES


class TestDiscreteContract:
    @pytest.mark.parametrize("dist", DISCRETE_INSTANCES, ids=repr)
    def test_has_pmf(self, dist):
        assert isinstance(dist, DiscreteDistribution)
        result = dist.pmf(1)
        assert isinstance(result, float)

    @pytest.mark.parametrize("dist", DISCRETE_INSTANCES, ids=repr)
    def test_has_cdf(self, dist):
        result = dist.cdf(1)
        assert isinstance(result, float)


class TestContinuousContract:
    @pytest.mark.parametrize("dist", CONTINUOUS_INSTANCES, ids=repr)
    def test_has_pdf(self, dist):
        assert isinstance(dist, ContinuousDistribution)
        result = dist.pdf(0.5)
        assert isinstance(result, float)

    @pytest.mark.parametrize("dist", CONTINUOUS_INSTANCES, ids=repr)
    def test_has_cdf(self, dist):
        result = dist.cdf(0.5)
        assert isinstance(result, float)

    @pytest.mark.parametrize("dist", CONTINUOUS_INSTANCES, ids=repr)
    def test_has_ppf(self, dist):
        result = dist.ppf(0.5)
        assert isinstance(result, float)


class TestUniversalContract:
    @pytest.mark.parametrize("dist", ALL_INSTANCES, ids=repr)
    def test_mean(self, dist):
        assert isinstance(dist.mean(), float)

    @pytest.mark.parametrize("dist", ALL_INSTANCES, ids=repr)
    def test_variance(self, dist):
        assert isinstance(dist.variance(), float)

    @pytest.mark.parametrize("dist", ALL_INSTANCES, ids=repr)
    def test_std_dev(self, dist):
        sd = dist.std_dev()
        assert isinstance(sd, float)
        assert sd == pytest.approx(dist.variance() ** 0.5, abs=1e-10)

    @pytest.mark.parametrize("dist", ALL_INSTANCES, ids=repr)
    def test_sf(self, dist):
        x = 2  # integer point safe for all distributions
        sf = dist.sf(x)
        cdf = dist.cdf(x)
        assert sf == pytest.approx(1.0 - cdf, abs=1e-10)

    @pytest.mark.parametrize("dist", ALL_INSTANCES, ids=repr)
    def test_rvs_shape(self, dist):
        samples = dist.rvs(size=100, random_state=42)
        assert isinstance(samples, np.ndarray)
        assert samples.shape == (100,)

    @pytest.mark.parametrize("dist", ALL_INSTANCES, ids=repr)
    def test_repr(self, dist):
        r = repr(dist)
        assert isinstance(r, str)
        assert len(r) > 0
