"""Tests for DiscreteUniform and ContinuousUniform distributions."""

import pytest
from scipy import stats as sp_stats

from tests.conftest import ABS_TOL
from wedgestats.distributions import ContinuousUniform, DiscreteUniform


class TestDiscreteUniform:
    def setup_method(self):
        self.dist = DiscreteUniform(low=1, high=6)
        self.ref = sp_stats.randint(1, 7)

    def test_pmf(self):
        for k in range(1, 7):
            assert self.dist.pmf(k) == pytest.approx(self.ref.pmf(k), abs=ABS_TOL)

    def test_pmf_outside(self):
        assert self.dist.pmf(0) == pytest.approx(0.0, abs=ABS_TOL)
        assert self.dist.pmf(7) == pytest.approx(0.0, abs=ABS_TOL)

    def test_cdf(self):
        for k in range(0, 8):
            assert self.dist.cdf(k) == pytest.approx(self.ref.cdf(k), abs=ABS_TOL)

    def test_mean(self):
        assert self.dist.mean() == pytest.approx(3.5, abs=ABS_TOL)

    def test_variance(self):
        assert self.dist.variance() == pytest.approx(self.ref.var(), abs=ABS_TOL)

    def test_repr(self):
        assert repr(self.dist) == "DiscreteUniform(low=1, high=6)"

    def test_invalid(self):
        with pytest.raises(ValueError):
            DiscreteUniform(6, 1)


class TestContinuousUniform:
    def setup_method(self):
        self.dist = ContinuousUniform(low=2, high=8)
        self.ref = sp_stats.uniform(loc=2, scale=6)

    def test_pdf(self):
        for x in [2, 3, 5, 7.5, 8]:
            assert self.dist.pdf(x) == pytest.approx(self.ref.pdf(x), abs=ABS_TOL)

    def test_cdf(self):
        for x in [1, 2, 5, 8, 9]:
            assert self.dist.cdf(x) == pytest.approx(self.ref.cdf(x), abs=ABS_TOL)

    def test_ppf(self):
        for q in [0.0, 0.25, 0.5, 0.75, 1.0]:
            assert self.dist.ppf(q) == pytest.approx(self.ref.ppf(q), abs=ABS_TOL)

    def test_mean(self):
        assert self.dist.mean() == pytest.approx(5.0, abs=ABS_TOL)

    def test_variance(self):
        assert self.dist.variance() == pytest.approx(3.0, abs=ABS_TOL)

    def test_repr(self):
        assert repr(self.dist) == "ContinuousUniform(low=2, high=8)"

    def test_invalid(self):
        with pytest.raises(ValueError):
            ContinuousUniform(5, 5)
        with pytest.raises(ValueError):
            ContinuousUniform(5, 3)
