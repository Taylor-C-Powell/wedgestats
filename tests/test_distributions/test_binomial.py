"""Tests for Binomial distribution."""

import pytest
from scipy import stats as sp_stats

from tests.conftest import ABS_TOL
from wedgestats.distributions import Binomial


class TestBinomial:
    def setup_method(self):
        self.dist = Binomial(n=10, p=0.3)
        self.ref = sp_stats.binom(10, 0.3)

    def test_pmf(self):
        for k in range(11):
            assert self.dist.pmf(k) == pytest.approx(self.ref.pmf(k), abs=ABS_TOL)

    def test_cdf(self):
        for k in range(11):
            assert self.dist.cdf(k) == pytest.approx(self.ref.cdf(k), abs=ABS_TOL)

    def test_mean(self):
        assert self.dist.mean() == pytest.approx(self.ref.mean(), abs=ABS_TOL)

    def test_variance(self):
        assert self.dist.variance() == pytest.approx(self.ref.var(), abs=ABS_TOL)

    def test_sf(self):
        assert self.dist.sf(3) == pytest.approx(self.ref.sf(3), abs=ABS_TOL)

    def test_repr(self):
        assert repr(self.dist) == "Binomial(n=10, p=0.3)"

    def test_invalid_p(self):
        with pytest.raises(ValueError):
            Binomial(10, 1.5)

    def test_negative_n(self):
        with pytest.raises(ValueError):
            Binomial(-1, 0.5)
