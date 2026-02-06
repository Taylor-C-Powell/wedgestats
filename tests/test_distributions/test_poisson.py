"""Tests for Poisson distribution."""

import pytest
from scipy import stats as sp_stats

from tests.conftest import ABS_TOL
from wedgestats.distributions import Poisson


class TestPoisson:
    def setup_method(self):
        self.dist = Poisson(lam=4.0)
        self.ref = sp_stats.poisson(mu=4.0)

    def test_pmf(self):
        for k in range(15):
            assert self.dist.pmf(k) == pytest.approx(self.ref.pmf(k), abs=ABS_TOL)

    def test_cdf(self):
        for k in range(15):
            assert self.dist.cdf(k) == pytest.approx(self.ref.cdf(k), abs=ABS_TOL)

    def test_mean(self):
        assert self.dist.mean() == pytest.approx(4.0, abs=ABS_TOL)

    def test_variance(self):
        assert self.dist.variance() == pytest.approx(4.0, abs=ABS_TOL)

    def test_repr(self):
        assert repr(self.dist) == "Poisson(lam=4.0)"

    def test_invalid_lam(self):
        with pytest.raises(ValueError):
            Poisson(0)
        with pytest.raises(ValueError):
            Poisson(-1)
